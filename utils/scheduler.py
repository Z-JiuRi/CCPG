import math
import torch
import bisect
from torch.optim.lr_scheduler import _LRScheduler
from typing import List

class ConstantLR(_LRScheduler):
    """
    保持学习率不变的调度器。
    """
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 始终返回初始的 base_lrs
        return self.base_lrs


class CosineAnnealingLR(_LRScheduler):
    """
    标准的余弦退火调度器。
    """
    def __init__(self, optimizer, max_steps, eta_min=0, last_epoch=-1):
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        
        # 防止 step 溢出
        step = min(self.last_epoch, self.max_steps)
        
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * step / self.max_steps)) / 2
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmup(_LRScheduler):
    """
    带线性预热 (Warmup) 的余弦退火调度器。
    阶段 1: 线性增加 lr 从 start_lr 到 base_lr
    阶段 2: 余弦衰减 lr 从 base_lr 到 eta_min
    """
    def __init__(self, optimizer, max_steps, warmup_ratio=0.0, start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.start_lr = start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # 阶段 1: Warmup
        if step < self.warmup_steps:
            # 进度 0.0 -> 1.0
            progress = step / max(1, self.warmup_steps) 
            return [
                self.start_lr + (base_lr - self.start_lr) * progress
                for base_lr in self.base_lrs
            ]
        
        # 阶段 2: Cosine Annealing
        else:
            # 计算退火阶段的进度
            cos_step = step - self.warmup_steps
            cos_total = self.max_steps - self.warmup_steps
            cos_total = max(1, cos_total)
            
            # 防止超出最大步数
            if cos_step > cos_total:
                return [self.eta_min for _ in self.base_lrs]

            progress = cos_step / cos_total
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class CosineAnnealingRestartAtPoints(_LRScheduler):
    """
    在指定百分比位置重启的余弦退火调度器。
    """
    def __init__(self, optimizer, restart_points, max_steps, decay_ratio=1.0, eta_min=0, last_epoch=-1):
        """
        Args:
            restart_points (list[float]): 重启点位比例，如 [0.1, 0.3, 0.8]
            max_steps (int): 总训练步数
            decay_ratio (float): 每次重启后 lr 上限衰减比例
        """
        self.max_steps = max_steps
        self.decay_ratio = decay_ratio
        self.eta_min = eta_min
        
        # 预处理 milestones
        points = [0.0] + sorted(restart_points) + [1.0]
        # 去重、排序、转为具体步数
        self.milestones = sorted(list(set([int(p * max_steps) for p in points])))
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        
        # 1. 确定当前处于哪个区间
        if current_step >= self.milestones[-1]:
             idx = len(self.milestones) - 2
        else:
             # bisect_right 返回插入点，减1即为左侧边界索引
             idx = bisect.bisect_right(self.milestones, current_step) - 1
             # 保护机制：防止 idx < 0
             idx = max(0, idx)
            
        # 2. 获取区间起止
        start_step = self.milestones[idx]
        end_step = self.milestones[idx + 1]
        
        # 3. 计算进度
        segment_len = end_step - start_step
        if segment_len <= 0:
            progress = 1.0
        else:
            progress = (current_step - start_step) / segment_len
            
        # 4. 计算当前周期的衰减倍率
        current_decay = self.decay_ratio ** idx

        # 5. 余弦计算
        return [
            self.eta_min + (base_lr * current_decay - self.eta_min) *
            (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


def get_lr_scheduler(optimizer, **kwargs):
    """
    统一获取 Scheduler 的入口。
    
    Args:
        optimizer: 优化器
        **kwargs: 对应各个 Scheduler 需要的参数
            - scheduler_type (str): 'const', 'cosine', 'cosine_warmup', 'cosine_restart'
            - max_steps (int): 总步数 (REQUIRED for all except const)
            - warmup_ratio (float): 预热比例
            - restart_points (list): 重启点比例
            - decay_ratio (float): 重启衰减
            - eta_min (float): 最小学习率
            - start_lr (float): 初始学习率
    """
    scheduler_type = kwargs.get("type", "const").lower()
    
    # 提取公共参数 (带默认值)
    max_steps = kwargs.get("max_steps", 100)
    eta_min = kwargs.get("eta_min", 0.0)
    
    if scheduler_type == "const":
        return ConstantLR(optimizer)
        
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer, 
            max_steps=max_steps, 
            eta_min=eta_min
        )
        
    elif scheduler_type == "cosine_warmup":
        warmup_ratio = kwargs.get("warmup_ratio", 0.0)
        start_lr = kwargs.get("start_lr", 0.0)
        return CosineAnnealingWarmup(
            optimizer, 
            max_steps=max_steps, 
            warmup_ratio=warmup_ratio, 
            start_lr=start_lr, 
            eta_min=eta_min
        )
    
    elif scheduler_type == "cosine_restart":
        restart_points = kwargs.get("restart_points", [0.5])
        decay_ratio = kwargs.get("decay_ratio", 1.0)
        return CosineAnnealingRestartAtPoints(
            optimizer, 
            restart_points=restart_points, 
            max_steps=max_steps, 
            decay_ratio=decay_ratio, 
            eta_min=eta_min
        )
    
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")