import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import logging
logger = logging.getLogger(__name__)
import matplotlib
matplotlib.use('Agg')


def setup_global_fonts():
    """设置全局中文字体和英文字体"""
    
    # 字体文件路径配置
    chinese_font_path = os.path.expanduser('~/zxd/.envs/SongTi.ttf')
    english_font_path = os.path.expanduser('~/zxd/.envs/TimesNewRoman.ttf')
    
    def setup_chinese_font(font_path):
        """专门设置中文字体"""
        print("🔤 正在设置中文字体...")
        
        # 检查中文字体文件是否存在
        if not os.path.exists(font_path):
            print(f"❌ 中文字体文件不存在: {font_path}")
            # 尝试使用系统默认中文字体
            chinese_system_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
                                'Noto Sans CJK SC', 'Source Han Sans SC', 'PingFang SC']
            for font_name in chinese_system_fonts:
                if font_name in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    print(f"✅ 使用中文字体: {font_name}")
                    return True
            print("⚠️  未找到合适的中文字体，可能显示乱码")
            return False
        else:
            try:
                # 添加中文字体到matplotlib
                font_prop = fm.FontProperties(fname=font_path)
                chinese_font_name = font_prop.get_name()
                
                # 注册字体
                fm.fontManager.addfont(font_path)
                
                # 设置中文字体优先级
                plt.rcParams['font.sans-serif'] = [chinese_font_name] + plt.rcParams['font.sans-serif']
                
                # 验证字体是否加载成功
                available_fonts = [f.name for f in fm.fontManager.ttflist]
                if chinese_font_name in available_fonts:
                    # print(f"✅ 中文字体加载成功: {chinese_font_name}")
                    print(f"✅ 中文字体加载成功: 标准宋体")
                    return True
                else:
                    print(f"⚠️  中文字体 '{chinese_font_name}' 加载可能失败")
                    return False
                    
            except Exception as e:
                print(f"❌ 中文字体设置出错: {e}")
                return False

    def setup_english_font(font_path):
        """专门设置英文字体"""
        print("🔤 正在设置英文字体...")
        
        # 检查英文字体文件是否存在
        if not os.path.exists(font_path):
            print(f"❌ 英文字体文件不存在: {font_path}")
            # 使用常见的优质英文字体
            english_system_fonts = ['DejaVu Sans', 'Liberation Sans', 'Arial', 
                                    'Helvetica', 'Calibri', 'Segoe UI']
            for font_name in english_system_fonts:
                if font_name in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = [font_name] + plt.rcParams['font.family']
                    print(f"✅ 使用英文字体: {font_name}")
                    return True
            print("⚠️  未找到合适的英文字体，将使用默认字体")
            return False
        else:
            try:
                # 添加英文字体到matplotlib
                font_prop = fm.FontProperties(fname=font_path)
                english_font_name = font_prop.get_name()
                
                # 注册字体
                fm.fontManager.addfont(font_path)
                
                # 设置英文字体（主要用于非中文文本）
                current_family = plt.rcParams['font.family']
                if isinstance(current_family, str):
                    current_family = [current_family]
                plt.rcParams['font.family'] = [english_font_name] + current_family
                
                # 验证字体是否加载成功
                available_fonts = [f.name for f in fm.fontManager.ttflist]
                if english_font_name in available_fonts:
                    print(f"✅ 英文字体加载成功: {english_font_name}")
                    return True
                else:
                    print(f"⚠️  英文字体 '{english_font_name}' 加载可能失败")
                    return False
                    
            except Exception as e:
                print(f"❌ 英文字体设置出错: {e}")
                return False

    # 设置中文字体
    setup_chinese_font(chinese_font_path)
    
    # 设置英文字体
    setup_english_font(english_font_path)
    
    # 设置其他全局字体参数
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # plt.rcParams['figure.dpi'] = 100
    # plt.rcParams['savefig.dpi'] = 300
    plt.close('all')


def plot_gaussian(data, filename=None):
    if data.dim() == 3:
        data = data[0]
    plt.figure(figsize=(10, 5), layout='constrained')
    # 使用seaborn的distplot，自动处理缩放
    sns.histplot(data.detach().cpu().numpy().flatten(), bins=100, alpha=0.5, label='data', kde=False, stat='density')
    # 用高斯分布拟合data的分布
    mu, std = norm.fit(data.detach().cpu().numpy().flatten())
    x = np.linspace(min(data.detach().cpu().numpy().flatten()), max(data.detach().cpu().numpy().flatten()), 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', linewidth=2, label=f'data gaussian fit: μ={mu:.6f}, σ={std:.6f}')
    plt.title(f'Gaussian Distribution')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    if filename:
        plt.savefig(filename)
    plt.close()

def plot_histogram(pre, ori, filename=None):
    # 如果传入的是三维数据，只取最后一维
    if pre.dim() == 3:
        pre = pre[0]
    if ori.dim() == 3:
        ori = ori[0]
    
    plt.figure(figsize=(10, 5))
    # 使用seaborn绘制原始数据和重建数据的分布
    sns.histplot(ori.detach().cpu().numpy().flatten(), bins=100, alpha=0.5, label='Original', stat='density', color='blue')
    sns.histplot(pre.detach().cpu().numpy().flatten(), bins=100, alpha=0.5, label='Reconstructed', stat='density', color='orange')

    # 拟合原始数据的高斯分布
    mu_orig, std_orig = norm.fit(ori.detach().cpu().numpy().flatten())
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p_orig = norm.pdf(x, mu_orig, std_orig)
    plt.plot(x, p_orig, 'k', linewidth=2, label=f'Original Fit: μ={mu_orig:.2f}, σ={std_orig:.2f}')

    # 拟合重建数据的高斯分布
    mu_recon, std_recon = norm.fit(pre.detach().cpu().numpy().flatten())
    p_recon = norm.pdf(x, mu_recon, std_recon)
    plt.plot(x, p_recon, 'r', linewidth=2, label=f'Reconstructed Fit: μ={mu_recon:.2f}, σ={std_recon:.2f}')
    plt.title(f'Value Distribution')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    if filename:
        plt.savefig(filename)
    plt.close()

def plot_asymmetric_violin(pre, ori, labels=None, colors=None, title=None, filename=None):
    """
    绘制两个一维tensor的不对称小提琴对比图
    
    参数:
    pre, ori: 要比较的两个一维PyTorch tensor
    labels: 两个数据集的标签列表 [label1, label2]
    colors: 两个数据集的颜色列表 [color1, color2]
    title: 图表标题
    """
    # 转换为numpy数组
    data1 = pre.detach().flatten().cpu().numpy() if torch.is_tensor(pre) else np.array(pre)
    data2 = ori.detach().flatten().cpu().numpy() if torch.is_tensor(ori) else np.array(ori)
    
    # 设置默认参数
    if labels is None:
        labels = ['Tensor 1', 'Tensor 2']
    if colors is None:
        colors = ['skyblue', 'salmon']
    
    # 创建DataFrame
    df = pd.DataFrame({
        'value': np.concatenate([data1, data2]),
        'distribution': [labels[0]] * len(data1) + [labels[1]] * len(data2),
    })
    
    # 创建图形
    plt.figure(figsize=(10, 5))
    
    # 绘制小提琴图 - 使用split=True创建不对称效果
    ax = sns.violinplot(
        data=df, 
        x='distribution', 
        y='value', 
        hue='distribution', 
        split=True, 
        palette=colors, 
        inner='box',
        bw_method='scott',
        cut=0,
        # density_norm='area',
        # common_norm=True,
        legend=False
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('数值', fontsize=12)
    plt.xlabel('')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'{labels[0]}: 均值={data1.mean():.2f}, 标准差={data1.std():.2f}\n'
    stats_text += f'{labels[1]}: 均值={data2.mean():.2f}, 标准差={data2.std():.2f}'
    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(filename)
    plt.close()

def plot_heatmap(pre, ori, filename=None):
    # 如果传入的是三维数据，只取最后一维
    if pre.dim() == 3:
        pre = pre[0]
    if ori.dim() == 3:
        ori = ori[0]
    fig, axs = plt.subplots(3, 1, figsize=(20, 10))

    # 去除中间维度
    pre_data = pre.detach().cpu().squeeze()
    ori_data = ori.detach().cpu().squeeze()
    diff_data = pre_data - ori_data

    # 为每个子图设置独立的对称颜色范围
    im1 = axs[0].imshow(pre_data, aspect='auto', cmap='seismic', 
                        vmin=-pre_data.abs().max(), vmax=pre_data.abs().max())
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title('pre')

    im2 = axs[1].imshow(ori_data, aspect='auto', cmap='seismic', 
                        vmin=-ori_data.abs().max(), vmax=ori_data.abs().max())
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title('ori')

    im3 = axs[2].imshow(diff_data, aspect='auto', cmap='seismic', 
                        vmin=-diff_data.abs().max(), vmax=diff_data.abs().max())
    fig.colorbar(im3, ax=axs[2])
    axs[2].set_title('diff')

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

def plot_single_heatmap(data, filename=None):
    plt.figure(figsize=(10, 5))
    im = plt.imshow(data, aspect='auto', cmap='seismic', 
                    vmin=-data.abs().max(), vmax=data.abs().max())
    plt.colorbar(im)
    plt.title('Single Heatmap')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()
