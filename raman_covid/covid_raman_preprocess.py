import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
import warnings

try:
    from pybaselines import polynomial
    PYBASELINES_AVAILABLE = True
except ImportError:
    print("警告: pybaselines 未安装，将跳过基线校正步骤。")
    PYBASELINES_AVAILABLE = False

warnings.filterwarnings('ignore')

def load_and_transpose_txt(filepath, label, wavenumbers):
    """优化后的TXT矩阵读取函数"""
    print(f"正在读取 {os.path.basename(filepath)} ...")
    
    # 使用制表符 \t 分隔读取
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    # 关键修复：消除由于每行末尾多余 \t 导致的空列 (NaN)
    df = df.dropna(axis=1, how='all')
    
    # 转置矩阵: 变成 N行(样本) x 900列(波数)
    df_t = df.T
    
    if df_t.shape[1] != len(wavenumbers):
        print(f"  [警告] 样本特征数({df_t.shape[1]}) 与 波数点数({len(wavenumbers)}) 不匹配！")
        
    df_t.columns = wavenumbers
    
    # 将 Label 插入为第一列
    df_t.insert(0, 'Label', label)
    print(f"  -> 成功获取到 {len(df_t)} 个样本，标签: {label}")
    
    return df_t

def preprocess_spectra(spectra, wavenumbers, baseline_poly=5, sg_window=21, sg_poly=3):
    """光谱预处理流水线：基线校正 -> SG平滑 -> SNV"""
    processed = []
    wavenumbers_float = np.array([float(w) for w in wavenumbers])
    
    print("开始执行预处理: [基线校正] -> [SG平滑] -> [SNV归一化] ...")
    for s in spectra:
        # 1. 基线校正 (基于你的创新点)
        if PYBASELINES_AVAILABLE:
            base = polynomial.imodpoly(s, x_data=wavenumbers_float, poly_order=baseline_poly)[0]
            s = s - base
            
        # 2. SG平滑
        s = savgol_filter(s, sg_window, sg_poly)
        
        # 3. SNV 归一化
        mean_val = np.mean(s)
        std_val = np.std(s)
        if std_val != 0:
            s = (s - mean_val) / std_val
            
        processed.append(s)
        
    return np.array(processed)

def main():
    # ========================================================
    # ⚠️ 请在这里填入你存放 4 个 txt 文件的绝对文件夹路径 ⚠️
    # 例如: '/public/home/liuzhenfang/downloads/dataset/'
    # 如果就在当前代码所在目录，可以填 './'
    # ========================================================
    DATA_DIR = '/public/home/liuzhenfang/datasets' 
    
    # 修正了文件名的拼写 (raw_Helthy.txt)
    files_map = {
        'Healthy': 'raw_Helthy.txt',
        'Suspected': 'raw_Suspected.txt',
        'COVID': 'raw_COVID.txt'
    }
    
    file_wn = os.path.join(DATA_DIR, 'wave_number.txt')
    if not os.path.exists(file_wn):
        raise FileNotFoundError(f"找不到波数文件: {file_wn}，请检查 DATA_DIR 路径是否正确！")

    # 1. 极其强壮的波数读取方式：利用 split() 自动处理空格、制表符、换行符
    with open(file_wn, 'r') as f:
        wn_raw = f.read().split()
        wavenumbers = [str(round(float(val), 2)) for val in wn_raw]
    print(f"成功读取波数 (X轴): 共 {len(wavenumbers)} 个点。")

    # 2. 依次读取三个组别的数据
    df_list = []
    for label, filename in files_map.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到文件: {filepath}")
        
        df_group = load_and_transpose_txt(filepath, label, wavenumbers)
        df_list.append(df_group)
        
    # 合并为总表
    df_master = pd.concat(df_list, ignore_index=True)
    
    # 3. 提取光谱特征矩阵并执行预处理
    spectra_matrix = df_master[wavenumbers].values
    processed_matrix = preprocess_spectra(spectra_matrix, wavenumbers)
    
    # 4. 更新 DataFrame 里的数值并保存
    df_master[wavenumbers] = processed_matrix
    
    output_file = 'covid_spectral_processed.csv'
    df_master.to_csv(output_file, index=False)
    print(f"\n✅ 预处理全部完成！共合成 {len(df_master)} 个样本的结构化表格。")
    print(f"✅ 文件已保存至: {os.path.abspath(output_file)}")
    print("✅ 你现在可以运行 covid_dualhead_cnn.py 开始训练了！")

if __name__ == "__main__":
    main()