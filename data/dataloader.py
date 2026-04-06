from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split


import logging
logger = logging.getLogger(__name__)

class LoRADataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, data_dir, token_size=128):
        self.data_dir = Path(data_dir)
        self.token_size = token_size
        self.samples = []
        
        self.seeds = sorted(p.name for p in self.data_dir.iterdir())
         
        for seed in self.seeds:
            files = sorted([p for p in (self.data_dir / seed).iterdir() if p.name.endswith('.pth')])
            for f in files:
                data = torch.load(
                    self.data_dir / seed / f,
                    weights_only=True,
                    map_location='cpu'
                )
                self.samples.append({
                    'a1': data['a1'].cpu(),
                    'b1': data['b1'].cpu(),
                    'a2': data['a2'].cpu(),
                    'b2': data['b2'].cpu(),
                    'cond': data['cond'].cpu(),
                    'cond_mask': data['mask'].cpu()
                })
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        
        cond = item['cond'] # (cond_len, cond_dim)
        cond_mask = item['cond_mask'] # (cond_len,) 0=pad
        a1 = item['a1']
        b1 = item['b1'].T
        a2 = item['a2']
        b2 = item['b2'].T

        # Flatten & Tokenize
        # 顺序: A1, B1, A2, B2
        tokens_list = []    # (token_len, token_size)
        layer_ids   = []    # 0 for A1, B1, 1 for A2, B2
        matrix_ids  = []    # 0 for A, 1 for B
        
        def process_mat(mat, layer_id, matrix_id):
            flat = mat.flatten() # (N,)
            # 这里 flat 的顺序已经是 [Rank0_row, Rank1_row...]
            if flat.size(0) % self.token_size != 0:
                raise ValueError(f"Matrix size {flat.size(0)} not divisible by token_size {self.token_size}")
            
            chunks = flat.view(-1, self.token_size) # (mat_token_len, mat_token_size)
            tokens_list.append(chunks)
            layer_ids.append(torch.full((chunks.size(0),), layer_id))
            matrix_ids.append(torch.full((chunks.size(0),), matrix_id))
        
        process_mat(a1, 0, 0)
        process_mat(b1, 0, 1)
        process_mat(a2, 1, 0)
        process_mat(b2, 1, 1)
        
        target_tokens     = torch.cat(tokens_list, dim=0)       # (token_len, token_size)
        target_layer_ids  = torch.cat(layer_ids, dim=0).long()  # (token_len,)
        target_matrix_ids = torch.cat(matrix_ids, dim=0).long() # (token_len,)
        
        return {
            'cond': cond,                   # (cond_len, cond_dim)
            'cond_mask': cond_mask,         # (cond_len,)
            'tokens': target_tokens,        # (token_len, token_size)
            'layer_ids': target_layer_ids,  # (token_len,)
            'matrix_ids': target_matrix_ids # (token_len,)
        }

def get_dataloaders(cfg):
    tot_datasets = LoRADataset(cfg.data.data_dir, cfg.data.token_size)
    
    tot_size = len(tot_datasets)
    train_size = int(cfg.data.train_ratio * tot_size)
    val_size  = tot_size - train_size
    
    train_datasets, val_datasets = random_split(
        tot_datasets, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.train.seed) 
    )
    
    train_loader = DataLoader(train_datasets, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_datasets, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    return train_loader, val_loader

# # 测试代码
# if __name__ == '__main__':
#     from omegaconf import OmegaConf
#     cfg = OmegaConf.load('../configs/config.yaml')
#     train_loader, val_loader = get_dataloaders(cfg)
    
#     # 测试训练集
#     for batch in tqdm(train_loader, desc='Training Batch'):
#         cond = batch['cond']
#         cond_mask = batch['cond_mask']
#         tokens = batch['tokens']
#         layer_ids = batch['layer_ids']
#         matrix_ids = batch['matrix_ids']
        
#         print(f"cond shape: {cond.shape}")
#         print(f"cond_mask shape: {cond_mask.shape}")
#         print(f"tokens shape: {tokens.shape}")
#         print(f"layer_ids shape: {layer_ids.shape}")
#         print(f"matrix_ids shape: {matrix_ids.shape}")
#         break
    
#     # 测试验证集
#     for batch in tqdm(val_loader, desc='Validation Batch'):
#         cond = batch['cond']
#         cond_mask = batch['cond_mask']
#         tokens = batch['tokens']
#         layer_ids = batch['layer_ids']
#         matrix_ids = batch['matrix_ids']
        
#         print(f"cond shape: {cond.shape}")
#         print(f"cond_mask shape: {cond_mask.shape}")
#         print(f"tokens shape: {tokens.shape}")
#         print(f"layer_ids shape: {layer_ids.shape}")
#         print(f"matrix_ids shape: {matrix_ids.shape}")
#         break
