import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))
    
    def F_normalize(self, x):
        # RMSNorm 实现: x / sqrt(mean(x**2)) * g
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + 1e-8) * self.g
    
    def forward(self, x):
        return self.F_normalize(x)


class SimpleResampler(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=768, cond_len=224, latent_cond_len=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_cond_len = latent_cond_len
        
        self.input_norm = RMSNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.expand_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, latent_cond_len * hidden_dim)
        )
        
        self.output_norm = RMSNorm(hidden_dim)
    
    def forward(self, cond, cond_mask):
        B = cond.shape[0]
        
        x = self.input_norm(cond)
        x = self.input_proj(x)
        
        if cond_mask is not None:
            mask = cond_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            x = x.mean(dim=1)
        
        x = self.expand_proj(x)
        x = x.view(B, self.latent_cond_len, self.hidden_dim)
        
        return self.output_norm(x)


class PerceiverResampler(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=768, cond_len=224, latent_cond_len=64, num_heads=8, depth=4, dropout=0.1):
        """
        __init__:
            Args:
                input_dim: 输入特征维度 (例如 cond_dim=512)
                hidden_dim: 内部 Transformer 的隐藏层维度
                cond_len: 原始条件序列的长度
                latent_cond_len: 压缩后的 Latent Token 数量 (即 Queries 的数量)
                depth: Self-Attention (Transformer Encoder) 的层数
                dropout: 0.1
        forward:
            Args:
                cond: 输入条件特征 (B, cond_len=224, cond_dim=512)
                cond_mask: 输入条件的掩码 (B, 224), 0 表示 padding (将被忽略)
            Returns:
                latents: 重采样后的特征 (B, latent_cond_len=64, hidden_dim=768)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_cond_len = latent_cond_len
        
        # 1. 输入预处理
        self.input_norm = RMSNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.pos_embbed = nn.Parameter(torch.randn(1, cond_len, hidden_dim) * 0.02)
        
        # 2. 定义可学习的 Latent Queries
        self.latents = nn.Parameter(torch.randn(1, latent_cond_len, hidden_dim) * 0.02)
        self.pos_emb_latents = nn.Parameter(torch.randn(1, latent_cond_len, hidden_dim) * 0.02)
        
        # 3. Cross-Attention 层 (压缩: Input -> Latent)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.cross_ln_q = RMSNorm(hidden_dim)
        self.cross_ln_k = RMSNorm(hidden_dim)
        
        # 4. Self-Attention 堆叠层 (深度特征提取)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim*4, 
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
        
        self.final_norm = RMSNorm(hidden_dim)

    def forward(self, cond, cond_mask):
        """
        Args:
            cond: 输入条件特征 (B, cond_len=224, cond_dim=512)
            cond_mask: 输入条件的掩码 (B, cond_len=224), 0 表示 padding (将被忽略)
        Returns:
            latents: 重采样后的特征 (B, latent_cond_len=64, hidden_dim=768)
        """
        B = cond.shape[0]
        
        # --- Step 1: 输入投影与位置编码 ---
        x = self.input_norm(cond)                         # cond: (B, cond_len=224, input_dim=512)
        x = self.input_proj(x)                            # (B, cond_len=224, hidden_dim=768)
        x = x + self.pos_embbed[:, :cond.size(1), :]      # (B, cond_len=224, hidden_dim=768) - 加上位置编码
        
        # --- Step 2: 准备 Latent Queries ---
        latents = self.latents.repeat(B, 1, 1)            # (B, latent_cond_len=64, hidden_dim=768) - 复制到 Batch 维度
        latents = latents + self.pos_emb_latents          # (B, latent_cond_len=64, hidden_dim=768) - 加上 Latent 的位置编码
        
        # --- Step 3: Cross Attention (Latents 查询 Input) ---
        # Query = Latents, Key/Value = Input (x)
        # Mask 处理: PyTorch MultiheadAttention 的 key_padding_mask 中 True 表示忽略 (即 pad 位置)
        key_padding_mask = (cond_mask == 0)               # (B, cond_len=224)
        
        q = self.cross_ln_q(latents)                      # (B, latent_cond_len=64, hidden_dim=768)
        k = self.cross_ln_k(x)                            # (B, cond_len=224, hidden_dim=768)
        v = x                                             # (B, cond_len=224, hidden_dim=768) - Value 这里直接使用投影后的 x
        
        # latents_out: (B, latent_cond_len=64, hidden_dim=768)
        # 注意: 这里只更新了 latents 的信息，将其压缩到了固定的长度
        latents_out, _ = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask)
        latents = latents + latents_out                   # (B, latent_cond_len=64, hidden_dim=768) - 残差连接
        
        # --- Step 4: Self Attention Layers (Latents 之间的交互) ---
        for layer in self.layers:
            latents = layer(latents)                      # (B, latent_cond_len=64, hidden_dim=768)
        
        # --- Step 5: 最终归一化 ---
        return self.final_norm(latents)                   # (B, latent_cond_len=64, hidden_dim=768) - 归一化输出