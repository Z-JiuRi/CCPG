import torch
import numpy as np
import random
import os
import logging
from pathlib import Path

from omegaconf import OmegaConf
import sys

def load_config(config_path: str, cli_args = None,):
    """
    加载配置并合并命令行参数
    
    Args:
        config_path: 配置文件路径
        cli_args: 命令行参数列表，如果为None则使用sys.argv[1:]
    Returns:
        合并后的OmegaConf配置对象
    """
    # 加载基础配置
    config = OmegaConf.load(config_path)
    
    # 如果没有提供cli_args，使用系统参数
    if cli_args is None:
        cli_args = sys.argv[1:]
    
    if not cli_args:
        return config
    
    # 解析命令行参数
    cli_conf = OmegaConf.from_cli(cli_args)
    config = OmegaConf.merge(config, cli_conf)
    
    return config

def seed_everything(seed=42):
    """固定所有随机种子，保证可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir: str):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format=('[%(asctime)s|%(name)s]- %(message)s'),
        datefmt='%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / "output.log"),
            logging.StreamHandler()
        ]
    )

def create_exp_dirs(path: str):
    """创建目录，可以直接输入 dir1/dir2/dir3"""
    Path(path).mkdir(parents=True, exist_ok=True)
    (Path(path) / "logs").mkdir(parents=True, exist_ok=True)
    (Path(path) / "ckpts").mkdir(parents=True, exist_ok=True)
    (Path(path) / "results" / "hist").mkdir(parents=True, exist_ok=True)
    (Path(path) / "results" / "heatmap").mkdir(parents=True, exist_ok=True)
    (Path(path) / "results" / "diff").mkdir(parents=True, exist_ok=True)
    return Path(path)

def compute_model_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters())

def get_grad_norm(model):
    """计算模型所有参数的梯度范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def zscore(x, mean=None, std=None):
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / std

def inv_zscore(x, mean, std):
    return x * std + mean

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}