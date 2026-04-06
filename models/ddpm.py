# models/ddpm.py
"""
DDPM扩散模型 - 更新版
支持位置编码和CFG训练
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def linear_beta_scheduler(timesteps, **kwargs):
    """线性beta调度"""
    beta_start = kwargs.get("beta_start", 0.0001)
    beta_end = kwargs.get("beta_end", 0.02)
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_scheduler(timesteps, **kwargs):
    """余弦beta调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    s = kwargs.get("s", 0.008)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_scheduler(timesteps, **kwargs):
    """Sigmoid beta调度"""
    beta_start = kwargs.get("beta_start", 0.0001)
    beta_end = kwargs.get("beta_end", 0.02)
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas

def get_beta_scheduler(timesteps, **kwargs):
    """获取beta调度"""
    scheduler_type = kwargs.get("scheduler_type", "linear")
    
    if scheduler_type == "linear":
        return linear_beta_scheduler(timesteps, **kwargs)
    elif scheduler_type == "cosine":
        return cosine_beta_scheduler(timesteps, **kwargs)
    elif scheduler_type == "sigmoid":
        return sigmoid_beta_scheduler(timesteps, **kwargs)
    else:
        raise ValueError("Unknown beta scheduler: {}".format(scheduler_type))


def extract(a, t, x_shape: Tuple[int, ...]):
    """从a中提取t对应的值，并reshape以便广播"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 1000,
        beta_kwargs: Dict[str, Any] = {},
        prediction_type: str = "eps",
        snr_gamma: Optional[float] = None,
        small_weight: float = 0.3,
        ablation_size_aware: bool = False,
    ):
        """
        高斯扩散模型，支持三种预测类型:
        - eps: 预测噪声
        - x: 预测原始数据 (x_0)
        - v: 预测速度 (v-prediction)
        
        __init__:
            Args:
                denoiser: 去噪模型，输入(x_t, t, cond, layer_ids, matrix_ids)输出预测
                timesteps: 扩散步数
                beta_kwargs: beta 调度器参数
                prediction_type: 预测类型，'eps', 'x', 'v'
                snr_gamma: Min-SNR 权重截断阈值，None表示不启用，推荐设置为 5.0
                small_weight: 小样本权重，默认0.3
        forward:
            Args:
                x_0: 原始数据       (batch_size, token_size, token_len)
                cond: 条件信息      (batch_size, latent_cond_len, hidden_cond_dim)
                noise: 噪声数据     (batch_size, token_size, token_len)
                layer_ids: 层ID    (batch_size, token_len)
                matrix_ids: 矩阵ID (batch_size, token_len)
                return_pred: 是否返回预测结果，默认False
            Returns:
                loss_dict: 损失字典
        """
        super().__init__()
        
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma
        self.small_weight = small_weight
        self.ablation_size_aware = ablation_size_aware

        # 计算扩散参数
        betas = get_beta_scheduler(timesteps, **beta_kwargs)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册缓冲区
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 前向过程参数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # 后验分布参数
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(self, x_0, t, noise=None):
        """
        前向扩散过程：q(x_t | x_0)
        
        Args:
            x_0: 原始数据 (batch_size, token_size, token_len)
            t: 时间步 (batch_size,)
            noise: 噪声 (batch_size, token_size, token_len)
        
        Returns:
            x_t: 扩散后的样本 (batch_size, token_size, token_len)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def predict_x0_from_v(self, x_t, t, v):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v
    
    def predict_eps_from_x0(self, x_t, t, x_0):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alphas_cumprod_t * x_0) / sqrt_one_minus_alphas_cumprod_t
    
    def get_v(self, x_0, noise, t):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_0
    
    def q_posterior_mean_variance(self, x_t, t, x_0):
        """计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差"""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def denoiser_predictions(self, x_t, t, cond, layer_ids=None, matrix_ids=None, cfg_scale=1.0, uncond_cond=None):
        """
        获取模型预测，支持 Classifier-Free Guidance (CFG)
        
        Args:
            x_t: 扩散后的样本 (batch_size, token_size, token_len)
            t: 时间步 (batch_size,)
            cond: 条件 (batch_size, latent_cond_len, hidden_cond_dim)
            layer_ids: 层ID (batch_size, token_len)
            matrix_ids: 矩阵ID (batch_size, token_len)
            cfg_scale: CFG 缩放系数，大于1.0启用引导
            uncond_cond: 无条件引导向量，当 cfg_scale > 1.0 时必须提供
        
        Returns:
            pred_noise: 预测的噪声 (batch_size, token_size, token_len)
            pred_x0: 预测的x_0 (batch_size, token_size, token_len)
        """
        # CFG 逻辑
        if cfg_scale > 1.0 and uncond_cond is not None:
            x_in = torch.cat([x_t] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([uncond_cond, cond]) # 拼接 [Uncond, Cond]
            
            l_in = torch.cat([layer_ids] * 2) if layer_ids is not None else None
            m_in = torch.cat([matrix_ids] * 2) if matrix_ids is not None else None
            
            # 显式传参给 DiT
            model_output = self.denoiser(x=x_in, t=t_in, cond_feats=c_in, layer_ids=l_in, matrix_ids=m_in)
            
            out_uncond, out_cond = model_output.chunk(2)
            denoiser_output = out_uncond + cfg_scale * (out_cond - out_uncond)
        else:
            # 普通训练/推理
            denoiser_output = self.denoiser(x=x_t, t=t, cond_feats=cond, layer_ids=layer_ids, matrix_ids=matrix_ids)
        
        if self.prediction_type == "eps":
            pred_noise = denoiser_output
            pred_x0 = self.predict_x0_from_eps(x_t, t, pred_noise)
        elif self.prediction_type == "v":
            v = denoiser_output
            pred_x0 = self.predict_x0_from_v(x_t, t, v)
            pred_noise = self.predict_eps_from_x0(x_t, t, pred_x0)
        elif self.prediction_type == "x":
            pred_x0 = denoiser_output
            pred_noise = self.predict_eps_from_x0(x_t, t, pred_x0)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
                
        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t, t, cond, layer_ids=None, matrix_ids=None, cfg_scale=1.0, uncond_cond=None):
        """计算预测分布 p(x_{t-1} | x_t, cond) 的均值和方差"""
        pred_noise, pred_x0 = self.denoiser_predictions(x_t, t, cond, layer_ids=layer_ids, matrix_ids=matrix_ids, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_t=x_t, t=t, x_0=pred_x0)
        return model_mean, posterior_variance, posterior_log_variance, pred_x0
    
    @torch.no_grad()
    def p_sample(self, x_t, t, cond, layer_ids=None, matrix_ids=None, cfg_scale=1.0, uncond_cond=None):
        """单步采样"""
        b = x_t.shape[0]
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            x_t=x_t, t=t, cond=cond, layer_ids=layer_ids, matrix_ids=matrix_ids, cfg_scale=cfg_scale, uncond_cond=uncond_cond
        )
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(b, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, layer_ids=None, matrix_ids=None, cfg_scale=1.0, uncond_cond=None):
        """完整的采样循环"""
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        iterator = tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps, leave=False)
        
        for t in iterator:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, cond, layer_ids=layer_ids, matrix_ids=matrix_ids, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
        return x
    
    @torch.no_grad()
    def ddim_sample(self, shape, cond, ddim_steps, eta, layer_ids=None, matrix_ids=None, cfg_scale=1.0, uncond_cond=None):
        """DDIM采样"""
        device = self.betas.device
        b = shape[0]
        times = torch.linspace(-1, self.timesteps - 1, steps=ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        x = torch.randn(shape, device=device)
        
        iterator = tqdm(time_pairs, desc='DDIM Sampling', total=len(time_pairs), leave=False)
        
        for time, time_prev in iterator:
            t = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise, pred_x0 = self.denoiser_predictions(x, t, cond, layer_ids=layer_ids, matrix_ids=matrix_ids, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
            
            if time_prev < 0:
                x = pred_x0
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_prev = self.alphas_cumprod[time_prev]
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            c = torch.sqrt(1 - alpha_prev - sigma ** 2)
            noise = torch.randn_like(x)
            
            x = torch.sqrt(alpha_prev) * pred_x0 + c * pred_noise + sigma * noise
            
        return x
    
    @torch.no_grad()
    def sample(self, cond, shape, use_ddim=False, ddim_steps=50, eta=0.0, layer_ids=None, matrix_ids=None, cfg_scale=1.0, uncond_cond=None):
        """
        采样接口
        
        Args:
            cond: 条件 (batch_size, latent_cond_len, hidden_cond_dim)
            shape: 样本形状 (batch_size, token_size, token_len)
            use_ddim: 是否使用DDIM采样
            ddim_steps: DDIM步数
            eta: DDIM随机性参数
            layer_ids: 层ID (batch_size, token_len)
            matrix_ids: 矩阵ID (batch_size, token_len)
            cfg_scale: CFG缩放系数
            uncond_cond: 无条件引导向量        
        Returns:
            x: 采样结果
        """
        if shape is None:
            raise ValueError("You must explicitly provide 'shape' tuple to sample().")
            
        if use_ddim:
            return self.ddim_sample(shape, cond, ddim_steps, eta, layer_ids=layer_ids, matrix_ids=matrix_ids, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
        else:
            return self.p_sample_loop(shape, cond, layer_ids=layer_ids, matrix_ids=matrix_ids, cfg_scale=cfg_scale, uncond_cond=uncond_cond)
    
    def loss_fn(self, x_0, t, cond, noise=None, layer_ids=None, matrix_ids=None, return_pred=False):
        """
        计算训练损失
        
        Args:
            x_0: 原始数据 (batch_size, token_size, token_len)
            t: 时间步 (batch_size,)
            cond: 条件 (batch_size, latent_cond_len, hidden_cond_dim)
            noise: 噪声 (batch_size, token_size, token_len)
            layer_ids: 层ID (batch_size, token_len)
            matrix_ids: 矩阵ID (batch_size, token_len)
            return_pred: 是否返回预测结果
        Returns:
            loss_dict: 损失字典
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise)
        denoiser_output = self.denoiser(x_t, t, cond, layer_ids=layer_ids, matrix_ids=matrix_ids)
        
        if self.prediction_type == "eps":
            target = noise
        elif self.prediction_type == "x":
            target = x_0
        elif self.prediction_type == "v":
            target = self.get_v(x_0, noise, t)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        if self.ablation_size_aware:
            loss_batch = F.mse_loss(denoiser_output, target, reduction='none').mean(dim=[1, 2])
            is_small = None
            is_large = None
        else:
            loss_per_token = F.mse_loss(denoiser_output, target, reduction='none').mean(dim=2)
            
            is_small = ((layer_ids == 0) & (matrix_ids == 0)) | \
                       ((layer_ids == 1) & (matrix_ids == 1))
                       
            is_large = ~is_small
            
            loss_small = (loss_per_token * is_small.float()).sum(dim=1) / (is_small.float().sum(dim=1) + 1e-8)
            loss_large = (loss_per_token * is_large.float()).sum(dim=1) / (is_large.float().sum(dim=1) + 1e-8)
            
            loss_batch = self.small_weight * loss_small + (1 - self.small_weight) * loss_large
        
        # Min-SNR 加权
        if self.snr_gamma is not None:
            snr = self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
            snr_clamped = torch.clamp(snr, max=self.snr_gamma)
            
            if self.prediction_type == 'eps':
                loss_weight = snr_clamped / snr
            elif self.prediction_type == 'v':
                loss_weight = snr_clamped / (snr + 1.0)
            else:
                # loss_weight = torch.ones_like(loss_batch)
                loss_weight = snr_clamped
            
            loss = (loss_batch * loss_weight).mean()
        else:
            loss = loss_batch.mean()
        
        
        # Metrics
        with torch.no_grad():
            flat_pred = denoiser_output.reshape(x_0.shape[0], -1)
            flat_target = target.reshape(x_0.shape[0], -1)
            cos_sim = F.cosine_similarity(flat_pred, flat_target, dim=1).mean()
            norm_sim = (torch.norm(flat_pred, dim=1) / (torch.norm(flat_target, dim=1) + 1e-8)).mean()
            
            if is_small is not None and is_large is not None:
                cos_per_token = F.cosine_similarity(denoiser_output, target, dim=2)
                cos_small = (cos_per_token * is_small.float()).sum() / (is_small.float().sum() + 1e-8)
                cos_large = (cos_per_token * is_large.float()).sum() / (is_large.float().sum() + 1e-8)
                
                pred_sq = denoiser_output.pow(2).sum(dim=2)
                target_sq = target.pow(2).sum(dim=2)
                
                pred_norm_small = (pred_sq * is_small.float()).sum(dim=1).sqrt()
                target_norm_small = (target_sq * is_small.float()).sum(dim=1).sqrt()
                
                pred_norm_large = (pred_sq * is_large.float()).sum(dim=1).sqrt()
                target_norm_large = (target_sq * is_large.float()).sum(dim=1).sqrt()
                
                norm_sim_small = (pred_norm_small / (target_norm_small + 1e-8)).mean()
                norm_sim_large = (pred_norm_large / (target_norm_large + 1e-8)).mean()
            else:
                cos_small = cos_sim
                cos_large = cos_sim
                norm_sim_small = norm_sim
                norm_sim_large = norm_sim
        
        loss_dict = {
            'loss': loss,
            'cos_sim': cos_sim,
            'norm_sim': norm_sim, 
            'cos_small': cos_small,
            'cos_large': cos_large,
            'norm_small': norm_sim_small,
            'norm_large': norm_sim_large
        }
        if return_pred:
            loss_dict['pred'] = denoiser_output
            loss_dict['target'] = target
        
        return loss_dict
    
    def forward(self, x_0, cond, noise=None, layer_ids=None, matrix_ids=None, return_pred=False):
        """
        前向传播（训练时调用）
        
        Args:
            x_0: 原始数据 (batch_size, token_size, token_len)
            cond: 条件 (batch_size, latent_cond_len, hidden_cond_dim)
            noise: 噪声 (batch_size, token_size, token_len)
            layer_ids: 层ID (batch_size, token_len)
            matrix_ids: 矩阵ID (batch_size, token_len)
            return_pred: 是否返回预测结果        
        Returns:
            loss_dict: 损失字典
        """
        b = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        return self.loss_fn(x_0, t, cond, noise, layer_ids=layer_ids, matrix_ids=matrix_ids, return_pred=return_pred)