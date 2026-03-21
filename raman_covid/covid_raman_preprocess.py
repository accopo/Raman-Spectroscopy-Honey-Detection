import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# 强烈建议安装 pybaselines 以获得最佳基线校正效果
try:
    from pybaselines.whittaker import asls
    PYBASELINES_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未检测到 pybaselines！强烈建议运行 'pip install pybaselines'")
    print("⚠️ 血清拉曼的荧光背景极强，缺少基线校正将严重影响诊断准确率！")
    PYBASELINES_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ==========================================
# 进阶版预处理模块 (拉曼光谱标准流水线)
# ==========================================
def remove_cosmic_rays(spectra, kernel_size=3):
    """1. 宇宙射线/尖峰剔除 (中值滤波)"""
    return np.array([medfilt(s, kernel_size=kernel_size) for s in spectra])

def apply_asls_baseline(spectra):
    """2. 非对称最小二乘基线校正 (AsLS - 黄金标准)"""
    if not PYBASELINES_AVAILABLE:
        return spectra
    processed = []
    for s in spectra:
        # lam: 惩罚参数 (通常在 1e4 到 1e6 之间)
        # p: 非对称性参数 (通常在 0.001 到 0.05 之间)
        baseline = asls(s, lam=1e5, p=0.01)[0]
        processed.append(s - baseline)
    return np.array(processed)

def apply_sg_smoothing(spectra, window=21, poly=3):
    """3. Savitzky-Golay 平滑 (消除高频热噪声)"""
    return np.array([savgol_filter(s, window, poly) for s in spectra])

def apply_snv(spectra):
    """4. 标准正态变量变换 (消除激光功率和样本浓度的绝对差异)"""
    processed = []
    for s in spectra:
        mean_val = np.mean(s)
        std_val = np.std(s)
        if std_val != 0:
            processed.append((s - mean_val) / std_val)
        else:
            processed.append(s)
    return np.array(processed)

# ==========================================
# 数据读取核心
# ==========================================
def load_and_transpose_txt(filepath, label, wavenumbers):
    print(f"📥 正在加载 {os.path.basename(filepath)} ...")
    df = pd.read_csv(filepath, sep='\t', header=None)
    df = df.dropna(axis=1, how='all')
    df_t = df.T
    df_t.columns = wavenumbers
    df_t.insert(0, 'Label', label)
    return df_t

# ==========================================
# 顶级期刊专用：2x2 预处理前后统计对比图
# ==========================================
def plot_professional_preprocessing_panel(wavenumbers_float, labels, raw_spectra, final_spectra, output_dir):
    print("🎨 正在渲染顶刊级 2x2 预处理统计学对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    color_map = {'Healthy': '#2ca02c', 'Suspected': '#ff7f0e', 'COVID': '#d62728'}
    class_order = ['Healthy', 'Suspected', 'COVID']

    # --- 图 A: 原始光谱 (Spaghetti Plot) ---
    ax = axes[0, 0]
    for label_name in class_order:
        idx = np.where(labels == label_name)[0]
        if len(idx) > 0:
            # 随机抽样 30 条画线以免过于拥挤
            sample_idx = np.random.choice(idx, min(30, len(idx)), replace=False)
            for i in sample_idx:
                ax.plot(wavenumbers_float, raw_spectra[i], color=color_map[label_name], alpha=0.15, linewidth=0.8)
            # 添加隐藏图例
            ax.plot([], [], color=color_map[label_name], label=label_name, linewidth=2)
            
    ax.set_title('(a) Raw Raman Spectra (Fluorescence Background)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Absolute Intensity (a.u.)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # --- 图 B: 原始光谱 (均值 ± 标准差) ---
    ax = axes[0, 1]
    for label_name in class_order:
        idx = np.where(labels == label_name)[0]
        if len(idx) > 0:
            class_mean = np.mean(raw_spectra[idx], axis=0)
            class_std = np.std(raw_spectra[idx], axis=0)
            ax.plot(wavenumbers_float, class_mean, color=color_map[label_name], label=label_name, linewidth=2)
            ax.fill_between(wavenumbers_float, class_mean - class_std, class_mean + class_std, 
                            color=color_map[label_name], alpha=0.15)
            
    ax.set_title('(b) Raw Spectra Statistical Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- 图 C: 预处理后光谱 (Spaghetti Plot) ---
    ax = axes[1, 0]
    for label_name in class_order:
        idx = np.where(labels == label_name)[0]
        if len(idx) > 0:
            sample_idx = np.random.choice(idx, min(30, len(idx)), replace=False)
            for i in sample_idx:
                ax.plot(wavenumbers_float, final_spectra[i], color=color_map[label_name], alpha=0.2, linewidth=0.8)
                
    ax.set_title('(c) Processed Spectra (AsLS + SG + SNV)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity (SNV)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- 图 D: 预处理后光谱 (均值 ± 标准差) ---
    ax = axes[1, 1]
    for label_name in class_order:
        idx = np.where(labels == label_name)[0]
        if len(idx) > 0:
            class_mean = np.mean(final_spectra[idx], axis=0)
            class_std = np.std(final_spectra[idx], axis=0)
            ax.plot(wavenumbers_float, class_mean, color=color_map[label_name], linewidth=2)
            ax.fill_between(wavenumbers_float, class_mean - class_std, class_mean + class_std, 
                            color=color_map[label_name], alpha=0.2)
            
    ax.set_title('(d) Processed Spectra Statistical Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'figure_preprocessing_evaluation.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ 顶级学术图表已保存至: {os.path.abspath(plot_path)}")

def main():
    # ⚠️ 请确认这里的数据集路径
    DATA_DIR = '/public/home/liuzhenfang/datasets' 
    OUTPUT_DIR = 'covid_model_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files_map = {'Healthy': 'raw_Helthy.txt', 'Suspected': 'raw_Suspected.txt', 'COVID': 'raw_COVID.txt'}
    
    file_wn = os.path.join(DATA_DIR, 'wave_number.txt')
    if not os.path.exists(file_wn):
        raise FileNotFoundError("找不到 wave_number.txt")

    with open(file_wn, 'r') as f:
        wavenumbers = [str(round(float(val), 2)) for val in f.read().split()]

    df_list = []
    for label, filename in files_map.items():
        filepath = os.path.join(DATA_DIR, filename)
        df_list.append(load_and_transpose_txt(filepath, label, wavenumbers))
        
    df_master = pd.concat(df_list, ignore_index=True)
    raw_spectra = df_master[wavenumbers].values
    labels = df_master['Label'].values
    wavenumbers_float = np.array([float(w) for w in wavenumbers])

    print("\n🚀 开始执行深层光谱清洗流水线...")
    
    # 步骤 1: 剔除异常尖峰
    print("  -> [1/4] 宇宙射线/异常峰剔除 (Median Filter)...")
    s1 = remove_cosmic_rays(raw_spectra)
    
    # 步骤 2: AsLS 基线校正 (核心)
    print("  -> [2/4] 自适应基线提取与校正 (AsLS)...")
    s2 = apply_asls_baseline(s1)
    
    # 步骤 3: 平滑
    print("  -> [3/4] 高频噪声平滑 (Savitzky-Golay)...")
    s3 = apply_sg_smoothing(s2)
    
    # 步骤 4: 归一化
    print("  -> [4/4] 样本浓度归一化 (SNV)...")
    final_spectra = apply_snv(s3)

    # 绘制顶刊评价图
    plot_professional_preprocessing_panel(wavenumbers_float, labels, raw_spectra, final_spectra, OUTPUT_DIR)

    # 覆盖并保存数据
    df_master[wavenumbers] = final_spectra
    output_file = 'covid_spectral_processed.csv'
    df_master.to_csv(output_file, index=False)
    print(f"\n✅ 干净的特征数据已准备完毕: {os.path.abspath(output_file)}")
    print("✅ 您现在可以使用这批高质量数据重新运行 CNN 训练代码了！")

if __name__ == "__main__":
    main()