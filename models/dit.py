import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    """
    调制函数: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    Args:
        x: 输入特征 (B, seq_len, hidden_dim)
        shift: 调制参数 (B, hidden_dim)
        scale: 调制参数 (B, hidden_dim)
    Returns:
        modulated_x: 调制后的特征 (B, seq_len, hidden_dim)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout), # MLP 内部 Dropout
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout)  # MLP 输出 Dropout
        )
        
        # AdaLN: t 调制 Norm1, Norm2, Norm3 (Self-Attn, Cross-Attn, MLP)
        # Shift, Scale, Gate for MSA (3) + Shift, Scale, Gate for Cross (3) + Shift, Scale, Gate for MLP (3) = 9
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim, bias=True)
        )

    def forward(self, x, t, cond_feats):
        """
        Args:
            x: 输入特征 (B, seq_len, hidden_dim)
            t: 时间步嵌入 (B, hidden_dim)
            cond_feats: 条件特征 (B, cond_len, hidden_dim)
        """
        # (B, 9 * hidden_dim) -> 9 * (B, hidden_dim)
        shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(9, dim=1)
        
        # 1. Self-Attention (自注意力)
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)               # (B, seq_len, hidden_dim)
        
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out                              # (B, seq_len, hidden_dim)
        
        # 2. Cross-Attention (交叉注意力)
        x_norm2 = modulate(self.norm2(x), shift_cross, scale_cross)           # (B, seq_len, hidden_dim)
        cross_out, _ = self.cross_attn(x_norm2, cond_feats, cond_feats)
        x = x + gate_cross.unsqueeze(1) * cross_out                           # (B, seq_len, hidden_dim)
        
        # 3. MLP (多层感知机)
        x_norm3 = modulate(self.norm3(x), shift_mlp, scale_mlp)               # (B, seq_len, hidden_dim)
        mlp_out = self.mlp(x_norm3)                                           # (B, seq_len, hidden_dim)
        x = x + gate_mlp.unsqueeze(1) * mlp_out                               # (B, seq_len, hidden_dim)
        
        return x

class DiT(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=768, depth=12, num_heads=8, max_len=66, mlp_ratio=4.0, dropout=0.1, ablation_structure=False):
        super().__init__()
        """
        __init__:
            Args:
                input_dim: 输入维度 (默认 128)
                hidden_dim: 隐藏层维度 (默认 768)
                depth: Transformer 块数 (默认 12)
                num_heads: 注意力头数 (默认 8)
                max_len: 最大序列长度 (默认 66)
                mlp_ratio: MLP 隐藏层维度比例 (默认 4.0)
                dropout: 0.1
                ablation_structure: 是否消融结构位置编码 (layer_id, matrix_id)
        forward:
            Args:
                x: 输入的噪声化 Token 序列 (B, seq_len, input_dim)
                t: 扩散时间步 (B,)
                cond_feats: 条件特征 (B, cond_len, hidden_dim)
                layer_ids: 每个 Token 对应的层索引 (B, seq_len)
                matrix_ids: 每个 Token 对应的矩阵索引 (B, seq_len)
            Returns:
                x: 去噪后的预测值 (B, seq_len, input_dim)
        """
        self.hidden_dim = hidden_dim
        self.ablation_structure = ablation_structure
        
        # Input Embeddings (输入嵌入)
        self.x_embedder = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        
        # Structure Embeddings (结构嵌入)
        self.layer_type_embed = nn.Embedding(2, hidden_dim)
        self.matrix_type_embed = nn.Embedding(2, hidden_dim)
        
        # Timestep Embedding (时间步嵌入)
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Blocks (Transformer 块)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        
        # Final Layer (最终输出层)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-8),
            nn.Linear(hidden_dim, input_dim, bias=True)
        )
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        # DiT 权重初始化方案
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.xavier_uniform_(self.t_embedder[0].weight)
        nn.init.xavier_uniform_(self.t_embedder[2].weight)
        
        # 将 adaLN 调制层的权重初始化为 0
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.adaLN_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_final[-1].bias, 0)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, x, t, cond_feats, layer_ids, matrix_ids):
        """
        Args:
            x: 输入的噪声化 Token 序列 (B, seq_len, input_dim=128)
            t: 扩散时间步 (B,)
            cond_feats: 来自 Resampler 的条件特征 (B, latent_cond_len, hidden_dim)
            layer_ids: 每个 Token 对应的层索引 (B, seq_len)
            matrix_ids: 每个 Token 对应的矩阵索引 (B, seq_len)
        Returns:
            x: 去噪后的预测值 (B, seq_len, input_dim=128)
        """
        # 1. Embeddings (嵌入层)
        x = self.x_embedder(x) + self.pos_embed[:, :x.shape[1], :]        # (B, seq_len, hidden_dim)
        if not self.ablation_structure:
            x = x + self.layer_type_embed(layer_ids)                          # (B, seq_len, hidden_dim)
            x = x + self.matrix_type_embed(matrix_ids)                        # (B, seq_len, hidden_dim)
        
        # 2. Time (时间步处理)
        t_freq = self.get_timestep_embedding(t, self.hidden_dim)           # (B, hidden_dim)
        t_embed = self.t_embedder(t_freq)                                  # (B, hidden_dim)
        
        # 3. Blocks (Transformer 块)
        for block in self.blocks:
            x = block(x, t_embed, cond_feats)                              # (B, seq_len, hidden_dim)
            
        # 4. Final (最终输出层)
        shift, scale = self.adaLN_final(t_embed).chunk(2, dim=1)          # (B, hidden_dim)
        x = modulate(self.final_layer[0](x), shift, scale)                # (B, seq_len, hidden_dim)
        x = self.final_layer[1](x)                                        # (B, seq_len, input_dim)
        
        return x
