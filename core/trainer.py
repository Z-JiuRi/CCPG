import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import torch.nn as nn

from models.ddpm import GaussianDiffusion
from models.dit import DiT
from models.resampler import PerceiverResampler, SimpleResampler
from utils.scheduler import get_lr_scheduler
from utils.tools import seed_everything, setup_logger, create_exp_dirs, EMA
from utils.visualize import setup_global_fonts, plot_gaussian, plot_heatmap, plot_histogram
from data.dataloader import get_dataloaders

import logging
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg):
# ==========================================================================================
        self.cfg = cfg
        self.device = torch.device(cfg.data.device)
        print(f"using device: {self.device}")
        self.exp_dir = create_exp_dirs(cfg.exp_dir)
        OmegaConf.save(cfg, self.exp_dir / "logs" / "config.yaml")
        logger.info(f"configs:\n{OmegaConf.to_yaml(cfg)}")
# ==========================================================================================
        setup_global_fonts()
        seed_everything(cfg.train.seed)
        setup_logger(self.exp_dir / "logs")
        self.writer = SummaryWriter(self.exp_dir / "logs")
# ==========================================================================================
        self.train_loader, self.val_loader = get_dataloaders(cfg)
        logger.info(f"[训练]: {len(self.train_loader.dataset)}")
        logger.info(f"[验证]: {len(self.val_loader.dataset)}")        
# ==========================================================================================
        ablation_list = getattr(cfg, 'ablation', []) or []
        self.ablation_structure = 'structure' in ablation_list
        self.ablation_size_aware = 'size_aware' in ablation_list
        self.ablation_resampler = 'resampler' in ablation_list
        
        if ablation_list:
            logger.info(f"[消融实验]: {ablation_list}")
        
        if self.ablation_resampler:
            self.resampler = SimpleResampler(
                input_dim=cfg.data.cond_shape[1],
                hidden_dim=cfg.resampler.hidden_dim,
                cond_len=cfg.data.cond_shape[0],
                latent_cond_len=cfg.resampler.latent_cond_len,
                dropout=cfg.resampler.dropout
            ).to(self.device)
        else:
            self.resampler = PerceiverResampler(
                input_dim=cfg.data.cond_shape[1],
                hidden_dim=cfg.resampler.hidden_dim,
                cond_len=cfg.data.cond_shape[0],
                latent_cond_len=cfg.resampler.latent_cond_len,
                num_heads=cfg.resampler.num_heads,
                depth=cfg.resampler.depth,
                dropout=cfg.resampler.dropout
            ).to(self.device)
        
        self.dit = DiT(
            input_dim=cfg.data.token_size,
            hidden_dim=cfg.resampler.hidden_dim,
            depth=cfg.dit.depth,
            num_heads=cfg.dit.num_heads,
            max_len=cfg.data.max_len,
            mlp_ratio=cfg.dit.mlp_ratio,
            dropout=cfg.resampler.dropout,
            ablation_structure=self.ablation_structure
        ).to(self.device)
        
        self.diffusion = GaussianDiffusion(
            denoiser=self.dit,
            timesteps=cfg.diffusion.timesteps,
            beta_kwargs=cfg.diffusion.betas,
            prediction_type=cfg.diffusion.prediction_type,
            snr_gamma=cfg.diffusion.snr_gamma,
            small_weight=cfg.diffusion.small_weight,
            ablation_size_aware=self.ablation_size_aware
        ).to(self.device)
        
        self.null_cond = nn.Parameter(torch.zeros(1, cfg.resampler.latent_cond_len, cfg.resampler.hidden_dim, device=self.device), requires_grad=True)
        
        self.optimizer = optim.AdamW(
            list(self.resampler.parameters()) + list(self.dit.parameters()) + [self.null_cond],
            lr=cfg.lr_scheduler.max_lr,
            weight_decay=cfg.train.weight_decay
        )
        
        # 创建学习率调度器
        self.grad_accum_steps = cfg.train.grad_accum_steps
        steps_per_epoch = (len(self.train_loader) + self.grad_accum_steps - 1) // self.grad_accum_steps
        tot_steps = steps_per_epoch * cfg.train.epochs
        
        lr_cfg = dict(cfg.lr_scheduler)
        lr_cfg["max_steps"] = tot_steps
        from torch.optim import lr_scheduler
        self.scheduler = get_lr_scheduler(self.optimizer, **lr_cfg)
        
        
        self.start_epoch = 1 
        self.step = 0
        
        # EMA
        self.ema = EMA(
            nn.ModuleDict({
                'resampler': self.resampler,
                'dit': self.dit,
                'null_cond_wrapper': nn.ParameterList([self.null_cond])
            }),
            decay=cfg.train.get('ema_rate', 0.999)
        )

        pretrained = getattr(cfg.train, 'pretrained', None)
        resume = getattr(cfg.train, 'resume', False)
        if pretrained:
            self.load_checkpoint(pretrained, resume=resume)

    def train(self):
        logger.info("Starting training...")
        best_loss = float('inf')
        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            self.diffusion.train()
            self.resampler.train()
            self.dit.train()
            tot_loss = 0
            accum_count = 0
            trainable_params = list(self.resampler.parameters()) + list(self.dit.parameters()) + [self.null_cond]
            self.optimizer.zero_grad(set_to_none=True)
            
            pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                cond = batch['cond'].to(self.device)
                cond_mask = batch['cond_mask'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                layer_ids = batch['layer_ids'].to(self.device)
                matrix_ids = batch['matrix_ids'].to(self.device)
                
                cond_feats = self.resampler(cond, cond_mask)
                cond_feats = cond_feats + torch.randn_like(cond_feats) * self.cfg.train.cond_noise_factor
                if self.cfg.train.cfg_drop_rate > 0 and torch.rand(1).item() < self.cfg.train.cfg_drop_rate:
                    # 替换为 Null Condition (Broadcast 到 batch size)
                    current_cond = self.null_cond.expand(cond_feats.shape[0], -1, -1)
                else:
                    current_cond = cond_feats
                
                if epoch % self.cfg.train.val_interval == 0 and batch_idx == len(self.train_loader) - 1:
                    loss_dict = self.diffusion(tokens, current_cond, layer_ids=layer_ids, matrix_ids=matrix_ids, return_pred=True)
                    pred = loss_dict['pred']
                    target = loss_dict['target']
                    plot_heatmap(pred, target, self.exp_dir / "results" / "heatmap" / f"[Train]_{epoch}.png")
                    plot_histogram(pred, target, self.exp_dir / "results" / "hist" / f"[Train]_{epoch}.png")
                    plot_gaussian(pred - target, self.exp_dir / "results" / "diff" / f"[Train]_{epoch}.png")
                else:
                    loss_dict = self.diffusion(tokens, current_cond, layer_ids=layer_ids, matrix_ids=matrix_ids)
                
                loss = loss_dict['loss']
                cos_sim = loss_dict['cos_sim']
                norm_sim = loss_dict['norm_sim']
                cos_small = loss_dict['cos_small']
                cos_large = loss_dict['cos_large']
                norm_small = loss_dict['norm_small']
                norm_large = loss_dict['norm_large']
                
                # 2. Backpropagation
                loss.backward()
                accum_count += 1
                should_step = accum_count == self.grad_accum_steps or batch_idx == len(self.train_loader) - 1

                if should_step:
                    grad_scale = 1.0 / accum_count
                    for param in trainable_params:
                        if param.grad is not None:
                            param.grad.mul_(grad_scale)

                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.train.grad_clip)
                    self.optimizer.step()
                    self.ema.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    
                    # 4. Logging
                    self.step += 1
                    self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                    self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.step)
                    self.writer.add_scalar('Train/Cos_sim', cos_sim.item(), self.step)
                    self.writer.add_scalar('Train/Norm_sim', norm_sim.item(), self.step)
                    self.writer.add_scalar('Train/Cos_small', cos_small.item(), self.step)
                    self.writer.add_scalar('Train/Cos_large', cos_large.item(), self.step)
                    self.writer.add_scalar('Train/Norm_small', norm_small.item(), self.step)
                    self.writer.add_scalar('Train/Norm_large', norm_large.item(), self.step)
                    self.writer.add_scalar('Train/GradNorm', grad_norm, self.step)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4e}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.4e}"
                })
                
                tot_loss += loss.item()
            
            avg_loss = tot_loss / len(self.train_loader)
            
            # Logging
            logger.info(f"[Train] Epoch {epoch}: Loss={avg_loss:.4e}")
            
            if epoch % self.cfg.train.val_interval == 0:
                val_loss = self.validate(epoch)
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint('best', is_best=True)
            
            # if epoch % self.cfg.train.save_interval == 0:
            #     self.save_checkpoint(epoch)
            

    @torch.no_grad()
    def validate(self, epoch):
        self.ema.apply_shadow()
        self.diffusion.eval()
        self.resampler.eval()
        self.dit.eval()
        tot_loss = 0
        tot_cos_sim = 0
        tot_norm_sim = 0
        tot_cos_small = 0
        tot_cos_large = 0
        tot_norm_small = 0
        tot_norm_large = 0
        
        for batch_idx, batch in enumerate(self.val_loader):
            cond = batch['cond'].to(self.device)
            cond_mask = batch['cond_mask'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            layer_ids = batch['layer_ids'].to(self.device)
            matrix_ids = batch['matrix_ids'].to(self.device)
            
            cond_feats = self.resampler(cond, cond_mask)
            loss_dict = self.diffusion(tokens, cond_feats, layer_ids=layer_ids, matrix_ids=matrix_ids, return_pred=True)
            
            pred = loss_dict['pred']
            target = loss_dict['target']
            if batch_idx == len(self.val_loader) - 1:
                plot_heatmap(pred, target, self.exp_dir / "results" / "heatmap" / f"[Val]_{epoch}.png")
                plot_histogram(pred, target, self.exp_dir / "results" / "hist" / f"[Val]_{epoch}.png")
                plot_gaussian(pred - target, self.exp_dir / "results" / "diff" / f"[Val]_{epoch}.png")
            
            tot_loss += loss_dict['loss'].item()
            tot_cos_sim += loss_dict['cos_sim'].item()
            tot_norm_sim += loss_dict['norm_sim'].item()
            tot_cos_small += loss_dict['cos_small'].item()
            tot_cos_large += loss_dict['cos_large'].item()
            tot_norm_small += loss_dict['norm_small'].item()
            tot_norm_large += loss_dict['norm_large'].item()
            
        avg_loss = tot_loss / len(self.val_loader)
        avg_cos_sim = tot_cos_sim / len(self.val_loader)
        avg_norm_sim = tot_norm_sim / len(self.val_loader)
        avg_cos_small = tot_cos_small / len(self.val_loader)
        avg_cos_large = tot_cos_large / len(self.val_loader)
        avg_norm_small = tot_norm_small / len(self.val_loader)
        avg_norm_large = tot_norm_large / len(self.val_loader)
        
        logger.info(f"[Val] Epoch {epoch}: Loss={avg_loss:.4e}")
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Cos_sim', avg_cos_sim, epoch)
        self.writer.add_scalar('Val/Norm_sim', avg_norm_sim, epoch)
        self.writer.add_scalar('Val/Cos_small', avg_cos_small, epoch)
        self.writer.add_scalar('Val/Cos_large', avg_cos_large, epoch)
        self.writer.add_scalar('Val/Norm_small', avg_norm_small, epoch)
        self.writer.add_scalar('Val/Norm_large', avg_norm_large, epoch)
        self.ema.restore()
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        # 如果是 best，文件名叫 best.pth，但内部记录的 epoch 依然是当前的数字
        filename = "best.pth" if is_best else f"{epoch}.pth"
        path = self.exp_dir / "ckpts" / filename
        
        torch.save({
            'resampler': self.resampler.state_dict(),
            'dit': self.dit.state_dict(),
            'null_cond': self.null_cond,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema_shadow': self.ema.shadow,
            'epoch': epoch  # 始终保存数字类型的 epoch
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    
    def load_checkpoint(self, path, resume=False):
        checkpoint = torch.load(path, map_location=self.device) # 确保 map_location
        
        # 1. 始终加载模型参数
        self.resampler.load_state_dict(checkpoint['resampler'])
        self.dit.load_state_dict(checkpoint['dit'])
        
        # 兼容性处理：防止旧模型没有 null_cond
        if 'null_cond' in checkpoint:
            self.null_cond.data = checkpoint['null_cond'].data
            
        # 2. 根据 resume 参数决定是否加载训练状态
        if resume:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'scheduler' in checkpoint and hasattr(self, 'scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            if 'ema_shadow' in checkpoint:
                self.ema.shadow = checkpoint['ema_shadow']
                
            self.start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming training from {path}, start epoch: {self.start_epoch}")
        else:
            # 如果不 resume，说明是重新训练 (Fine-tune)，
            # 此时模型参数已经变为预训练参数，必须重置 EMA，
            # 否则 EMA 还会保留 __init__ 时的随机初始化参数，导致 EMA 效果极差。
            self.ema.register()
            
            self.start_epoch = 1
            logger.info(f"Loaded pretrained model from {path}, starting from epoch 1 (Training state not loaded, EMA reset to current weights)")
