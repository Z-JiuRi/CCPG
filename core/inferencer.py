import torch
import torch.nn as nn
import os
import numpy as np
from models.ddpm import GaussianDiffusion
from models.dit import DiT
from models.resampler import PerceiverResampler, SimpleResampler
from utils.tools import load_config, seed_everything

class Inferencer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.data.device)
        self.use_ema = cfg.inference.use_ema
        self.checkpoint_path = cfg.inference.checkpoint_path
        self.cond_path = cfg.inference.cond_path
        
        seed_everything(cfg.inference.seed)
        
        ablation_list = getattr(cfg, 'ablation', []) or []
        self.ablation_structure = 'structure' in ablation_list
        self.ablation_size_aware = 'size_aware' in ablation_list
        self.ablation_resampler = 'resampler' in ablation_list
        
        if ablation_list:
            print(f"[消融实验]: {ablation_list}")
        
        if self.ablation_resampler:
            self.resampler = SimpleResampler(
                input_dim=cfg.data.cond_shape[1],
                hidden_dim=cfg.resampler.hidden_dim,
                cond_len=cfg.data.cond_shape[0],
                latent_cond_len=cfg.resampler.latent_cond_len,
                dropout=cfg.resampler.get('dropout', 0.1)
            ).to(self.device)
        else:
            self.resampler = PerceiverResampler(
                input_dim=cfg.data.cond_shape[1],
                hidden_dim=cfg.resampler.hidden_dim,
                cond_len=cfg.data.cond_shape[0],
                latent_cond_len=cfg.resampler.latent_cond_len,
                num_heads=cfg.resampler.num_heads,
                depth=cfg.resampler.depth
            ).to(self.device)
        
        self.dit = DiT(
            input_dim=cfg.data.token_size,
            hidden_dim=cfg.resampler.hidden_dim,
            depth=cfg.dit.depth,
            num_heads=cfg.dit.num_heads,
            max_len=cfg.data.max_len,
            mlp_ratio=cfg.dit.mlp_ratio,
            ablation_structure=self.ablation_structure
        ).to(self.device)
        
        self.diffusion = GaussianDiffusion(
            denoiser=self.dit,
            timesteps=cfg.diffusion.timesteps,
            beta_kwargs=cfg.diffusion.betas,
            prediction_type=cfg.diffusion.prediction_type,
            snr_gamma=cfg.diffusion.snr_gamma,
            ablation_size_aware=self.ablation_size_aware
        ).to(self.device)

        # Null cond placeholder
        self.null_cond = nn.Parameter(torch.zeros(1, cfg.resampler.latent_cond_len, cfg.resampler.hidden_dim, device=self.device))
        
        if not os.path.exists(self.checkpoint_path):
             raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        print(f"Loading checkpoint from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 处理 EMA 权重加载
        if self.use_ema and 'ema_shadow' in ckpt:
            print("Loading EMA weights...")
            shadow = ckpt['ema_shadow']
            model_dict = {
                'resampler': self.resampler,
                'dit': self.dit,
                'null_cond_wrapper': nn.ParameterList([self.null_cond])
            }
            
            for module_name, module in model_dict.items():
                for name, param in module.named_parameters():
                    full_name = f"{module_name}.{name}"
                    if full_name in shadow:
                        param.data.copy_(shadow[full_name].to(self.device))
                    else:
                        print(f"Warning: {full_name} not found in EMA shadow")
        else:
            print("Loading standard weights...")
            self.resampler.load_state_dict(ckpt['resampler'])
            self.dit.load_state_dict(ckpt['dit'])
            # 兼容性处理：防止旧模型没有 null_cond
            if 'null_cond' in ckpt:
                self.null_cond.data = ckpt['null_cond'].data.to(self.device)
            
        self.diffusion.eval()
        self.resampler.eval()
        self.dit.eval()
        
        # 3. 加载统计量 (用于反归一化)
        if not os.path.exists(cfg.data.stats_path):
             raise FileNotFoundError(f"Stats file not found at {cfg.data.stats_path}")
        self.stats = torch.load(cfg.data.stats_path, map_location='cpu')
        
        # 4. 预计算 IDs
        self.matrix_configs = [
            {'name': 'a1', 'shape': tuple(cfg.data.original_shapes.a1), 'layer_id': 0, 'matrix_id': 0},
            {'name': 'b1', 'shape': tuple(cfg.data.original_shapes.b1), 'layer_id': 0, 'matrix_id': 1},
            {'name': 'a2', 'shape': tuple(cfg.data.original_shapes.a2), 'layer_id': 1, 'matrix_id': 0},
            {'name': 'b2', 'shape': tuple(cfg.data.original_shapes.b2), 'layer_id': 1, 'matrix_id': 1},
        ]
        self.token_size = cfg.data.token_size
        self.layer_ids, self.matrix_ids, self.num_tokens = self._make_fixed_ids()
        
        print(f"Inferencer initialized. Total generation tokens: {self.num_tokens}")

    def _make_fixed_ids(self):
        """
        构建固定的 layer_ids 和 matrix_ids，逻辑与 DataLoader 完全一致
        """
        layer_ids_list = []
        matrix_ids_list = []
        
        total_tokens = 0
        
        # 按顺序遍历矩阵 (A1, B1, A2, B2)
        for config in self.matrix_configs:
            rows, cols = config['shape']
            num_elements = rows * cols
            
            if num_elements % self.token_size != 0:
                raise ValueError(f"Matrix {config['name']} size {num_elements} not divisible by token_size {self.token_size}")
            
            num_chunks = num_elements // self.token_size
            total_tokens += num_chunks
            
            # 生成对应的 ID
            # shape: (num_chunks,)
            l_ids = torch.full((num_chunks,), config['layer_id'], dtype=torch.long)
            m_ids = torch.full((num_chunks,), config['matrix_id'], dtype=torch.long)
            
            layer_ids_list.append(l_ids)
            matrix_ids_list.append(m_ids)
            
        # 拼接
        layer_ids = torch.cat(layer_ids_list)
        matrix_ids = torch.cat(matrix_ids_list)
        
        return layer_ids, matrix_ids, total_tokens

    @torch.no_grad()
    def inference(self):
        """
        执行推理，支持单文件或目录遍历
        """
        self.resampler.eval()
        self.dit.eval()
        
        if os.path.isdir(self.cond_path):
            print(f"Searching for condition files in {self.cond_path}...")
            cond_files = []
            for root, dirs, files in os.walk(self.cond_path):
                for file in files:
                    if file.endswith('.pth'):
                        cond_files.append(os.path.join(root, file))
            
            cond_files.sort()
            print(f"Found {len(cond_files)} condition files.")
            
            for i, cond_file in enumerate(cond_files):
                print(f"[{i+1}/{len(cond_files)}] Processing {cond_file}...")
                self._inference_single(cond_file)
        else:
            print(f"Processing single file {self.cond_path}...")
            self._inference_single(self.cond_path)

    def _inference_single(self, cond_path):
        """
        单个文件的推理逻辑
        """
        cond_tensor = torch.load(cond_path, map_location=self.device)
        cond = cond_tensor['cond'].unsqueeze(0)
        cond_mask = cond_tensor['mask'].unsqueeze(0)

        # 1. 通过 Resampler 获取条件特征
        cond_feats = self.resampler(cond, cond_mask) # (1, 64, hidden_dim)
        
        # 2. 准备 Unconditional Condition (用于 CFG)
        # null_cond 是 (1, 64, hidden_dim)，需要 expand 到 batch size
        uncond_cond = self.null_cond.expand(cond_feats.shape[0], -1, -1)
        
        # 3. 准备采样所需的 IDs
        # 扩展到 Batch 维度 (Batch, seq_len)
        batch_size = cond.shape[0]
        layer_ids = self.layer_ids.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        matrix_ids = self.matrix_ids.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # 目标形状 (Batch, num_tokens, token_size)
        shape = (batch_size, self.num_tokens, self.token_size)
        
        # 4. 执行扩散采样
        # 注意：这里调用的是 self.diffusion.sample，它会处理 p_sample_loop 或 ddim_sample
        samples = self.diffusion.sample(
            cond=cond_feats,
            shape=shape,
            use_ddim=self.cfg.inference.use_ddim,
            ddim_steps=self.cfg.inference.ddim_steps,
            eta=self.cfg.inference.eta,
            layer_ids=layer_ids,
            matrix_ids=matrix_ids,
            cfg_scale=self.cfg.inference.cfg_scale,
            uncond_cond=uncond_cond
        ) # -> (Batch, num_tokens, token_size)
        
        # 5. 重建并反归一化矩阵
        # 目前只处理 Batch 中的第一个样本
        
        # Determine output path
        if os.path.isdir(self.cond_path):
            rel_path = os.path.relpath(cond_path, self.cond_path)
            output_path = os.path.join(self.cfg.inference.output_dir, rel_path)
        else:
            output_path = os.path.join(self.cfg.inference.output_dir, os.path.basename(cond_path))
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self._reconstruct_matrices(samples[0]), output_path)
        print(f"Output saved to {output_path}")

    def _reconstruct_matrices(self, flat_tokens):
        """
        将生成的 Token 序列还原为矩阵字典，并执行反归一化
        Args:
            flat_tokens: (num_tokens, token_size)
        """
        # 1. 展平所有 Token -> (total_elements,)
        flat_data = flat_tokens.flatten()
        
        results = {}
        ptr = 0
        
        # 按顺序切分 (A1, B1, A2, B2)
        for config in self.matrix_configs:
            name = config['name']
            rows, cols = config['shape'] # 这里读到的是原始形状，比如 B1: (2048, 2)
            num_elements = rows * cols
            
            # 切片
            matrix_flat = flat_data[ptr : ptr + num_elements]
            ptr += num_elements
            
            if 'b' in name: # 针对 B 矩阵 (b1, b2)
                # 1. 它是以 (Rank, Dim) 的顺序被 Flatten 的，也就是 (cols, rows)
                #    所以要先 Reshape 成转置后的形状
                matrix_data = matrix_flat.reshape(cols, rows) 
                
                # 2. 反归一化 (在转置状态下进行，因为 Stats 是基于转置计算的)
                mean = self.stats['means'][name]
                std = self.stats['stds'][name]
                matrix_data = matrix_data * std + mean
                
                # 3. 最后转置回原始形状 (2048, 2)
                matrix_data = matrix_data.T
                
            else: # 针对 A 矩阵 (a1, a2)，它们本来就是 (Rank, Dim)，无需特殊处理
                matrix_data = matrix_flat.reshape(rows, cols)
                
                mean = self.stats['means'][name]
                std = self.stats['stds'][name]
                matrix_data = matrix_data * std + mean
            
            results[name] = matrix_data.cpu()
            
        return results