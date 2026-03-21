import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import combinations
import re
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 模块 1: 网络架构定义 (必须与训练时完全一致以匹配 .pth)
# ==========================================
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DualHeadSpectralCNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DualHeadSpectralCNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2), nn.BatchNorm1d(16), nn.LeakyReLU(0.01),
            SEBlock1D(16), nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.LeakyReLU(0.01),
            SEBlock1D(32), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            SEBlock1D(64)
        )
        
        # 动态推导 Flatten 尺寸
        bs = 1
        input_tensor = torch.autograd.Variable(torch.rand(bs, 1, input_size))
        output_feat = self.features(input_tensor)
        flatten_size = output_feat.data.view(bs, -1).size(1)
        
        self.head_multi = nn.Sequential(
            nn.Linear(flatten_size, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.01), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
        self.head_binary = nn.Sequential(
            nn.Linear(flatten_size, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.01), nn.Dropout(0.4),
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1) 
        out_multi = self.head_multi(features)
        out_bin = self.head_binary(features)
        return out_multi, out_bin


# ==========================================
# 模块 2: 数据还原器 (精准重构某一次交叉验证的环境)
# ==========================================
class DataReconstructor:
    def __init__(self, data_path='/public/home/liuzhenfang/datasets/honey_spectral_master.csv'):
        self.data_path = data_path
        self.wavenumber_cols = []
        self.valid_main_ids = []
        self.pure_samples = []
        self.target_df = None
        self.label_encoder = LabelEncoder()
        
    def prepare_global_data(self):
        df = pd.read_csv(self.data_path)
        self.wavenumber_cols = [col for col in df.columns if col.replace('.', '', 1).isdigit()]
        
        excluded_samples = ['4', '8', '11']
        self.valid_main_ids = [str(i) for i in range(1, 18) if str(i) not in excluded_samples]
        self.pure_samples = [f'H{i}' for i in self.valid_main_ids]
        
        def clean_id(x):
            if pd.isna(x): return 'nan'
            s = str(x).strip().upper()
            nums = re.findall(r'\d+', s)
            if nums: return str(int(nums[0]))
            return s
                
        df['Main_Honey_ID_Str'] = df['Main_Honey_ID'].apply(clean_id)
        type_col = df['Type'].astype(str)
        is_rice = type_col.str.contains('rice|大米', case=False, na=False)
        is_beet = type_col.str.contains('beet|甜菜', case=False, na=False)
        
        pure_df = df[df['Sample'].isin(self.pure_samples)].copy()
        rice_spiked_df = df[is_rice & (df['Main_Honey_ID_Str'].isin(self.valid_main_ids))].copy()
        beet_spiked_df = df[is_beet & (df['Main_Honey_ID_Str'].isin(self.valid_main_ids))].copy()

        mixed_spiked_df = pd.concat([rice_spiked_df, beet_spiked_df], ignore_index=True)
        
        target_samples = self.pure_samples + mixed_spiked_df['Sample'].tolist()
        self.target_df = df[df['Sample'].isin(target_samples)].copy()
        
        # 打标签
        self.target_df['Concentration_Target'] = 'Unknown'
        self.target_df.loc[self.target_df['Sample'].isin(self.pure_samples), 'Concentration_Target'] = 'Pure'
        for idx, row in mixed_spiked_df.iterrows():
            c = row.get('Mix_Concentration', 0)
            try: c = float(c)
            except: c = 0.0
            if pd.isna(c) or c == 0: target = 'Pure'
            elif c == 0.1 or c == 10: target = 'Mixed-10%'
            elif c == 0.2 or c == 20: target = 'Mixed-20%'
            elif c == 0.3 or c == 30: target = 'Mixed-30%'
            elif c == 0.5 or c == 50: target = 'Mixed-50%'
            else: target = 'Mixed-Unknown'
            self.target_df.loc[self.target_df['Sample'] == row['Sample'], 'Concentration_Target'] = target

        self.label_encoder.fit(self.target_df['Concentration_Target'].values)
        return len(self.wavenumber_cols), len(self.label_encoder.classes_)

    def get_fold_data(self, fold_idx):
        """ 根据折数 (1-91) 精准还原对应的训练集与测试集 """
        test_combinations = list(combinations(self.valid_main_ids, 2))
        test_ids = list(test_combinations[fold_idx - 1]) # 折数从1开始
        train_ids = [m for m in self.valid_main_ids if m not in test_ids]

        train_pure = [f"H{i}" for i in train_ids]
        test_pure = [f"H{i}" for i in test_ids]

        train_df = self.target_df[
            (self.target_df['Sample'].isin(train_pure)) | 
            ((self.target_df['Concentration_Target'] != 'Pure') & (self.target_df['Main_Honey_ID_Str'].isin(train_ids)))
        ]
        test_df = self.target_df[
            (self.target_df['Sample'].isin(test_pure)) | 
            ((self.target_df['Concentration_Target'] != 'Pure') & (self.target_df['Main_Honey_ID_Str'].isin(test_ids)))
        ]

        X_train = train_df[self.wavenumber_cols].values
        X_test = test_df[self.wavenumber_cols].values
        y_test = test_df['Concentration_Target'].values
        
        # 必须使用与训练时完全相同的缩放器！
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_test_scaled, y_test


# ==========================================
# 模块 3: XAI 解析引擎 (加载权重 + 画图)
# ==========================================
class XAI_Analyzer:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstructor = DataReconstructor()
        self.input_size, self.num_classes = self.reconstructor.prepare_global_data()
        self.model = DualHeadSpectralCNN1D(self.input_size, self.num_classes).to(self.device)
        
    def find_best_fold(self):
        """ 自动解析 CSV，找到硬错误最低、准确率最高的 Fold """
        csv_files = glob.glob(os.path.join(self.run_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"在 {self.run_dir} 中未找到 CSV 结果文件！")
            
        df_res = pd.read_csv(csv_files[0])
        
        # 筛选逻辑：纯判掺假(硬错误)最少 -> 掺假判纯最少 -> 准确率最高
        df_res = df_res.sort_values(by=['Pure_Misclassified_As_Spiked', 'Spiked_Misclassified_As_Pure', 'Accuracy'], 
                                    ascending=[True, True, False])
        best_fold = df_res.iloc[0]['Combination']
        best_acc = df_res.iloc[0]['Accuracy'] * 100
        print(f"📊 [自动寻优] 发现最佳 Fold: {best_fold} (准确率: {best_acc:.2f}%)")
        return int(best_fold), best_acc

    def load_weight_and_visualize(self, fold_idx=None):
        if fold_idx is None:
            fold_idx, acc = self.find_best_fold()
        else:
            acc = 0.0 # 自定义查看模式
            
        weight_path = os.path.join(self.run_dir, 'model_weights', f'fold_{fold_idx}_model.pth')
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"未找到权重文件: {weight_path}")
             
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        print(f"⚖️ [权重加载] 成功加载模型参数: {os.path.basename(weight_path)}")
        
        # 还原测试数据
        X_test_scaled, y_test = self.reconstructor.get_fold_data(fold_idx)
        
        # 开始生成热力图
        spiked_indices = [i for i, label in enumerate(y_test) if label != 'Pure']
        if not spiked_indices:
            print("❌ 该 Fold 测试集中无掺假样本，无法生成热力图。")
            return
            
        X_spiked = X_test_scaled[spiked_indices]
        X_tensor = torch.tensor(X_spiked, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_tensor.requires_grad_()
        
        _, out_bin = self.model(X_tensor)
        spiked_score = out_bin[:, 0].sum()
        
        self.model.zero_grad()
        spiked_score.backward()
        
        gradients = X_tensor.grad.squeeze().cpu().numpy()
        inputs = X_tensor.detach().squeeze().cpu().numpy()
        
        saliency = np.abs(gradients * inputs)
        mean_saliency = np.mean(saliency, axis=0) if saliency.ndim > 1 else saliency
        
        if mean_saliency.max() > 0:
            mean_saliency = (mean_saliency - mean_saliency.min()) / (mean_saliency.max() - mean_saliency.min())
            
        self._plot_and_save(mean_saliency, inputs, fold_idx, acc)

    def _plot_and_save(self, mean_saliency, inputs, fold_idx, acc):
        wavenumbers = [float(w) for w in self.reconstructor.wavenumber_cols]
        top_indices = np.argsort(mean_saliency)[-10:]
        
        mean_spectrum = np.mean(inputs, axis=0) if inputs.ndim > 1 else inputs
        mean_spectrum = (mean_spectrum - mean_spectrum.min()) / (mean_spectrum.max() - mean_spectrum.min())
        
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        ax1.plot(wavenumbers, mean_spectrum, color='gray', alpha=0.5, linewidth=2, label='Mean Spiked Spectrum')
        ax1.set_xlabel('Raman Shift Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Spectral Intensity', color='gray', fontsize=12, fontweight='bold')
        
        ax2 = ax1.twinx()
        ax2.plot(wavenumbers, mean_saliency, color='red', linewidth=2, alpha=0.8, label='SE-CNN Gradient Saliency')
        ax2.fill_between(wavenumbers, 0, mean_saliency, color='red', alpha=0.1)
        ax2.set_ylabel('Feature Contribution Score', color='red', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        
        for idx in top_indices:
            ax2.annotate(f"{wavenumbers[idx]:.1f}", 
                         xy=(wavenumbers[idx], mean_saliency[idx]), 
                         xytext=(0, 15), textcoords="offset points", 
                         ha='center', va='bottom', fontsize=9, rotation=90, 
                         color='darkred', weight='bold',
                         arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.5))

        title = f'XAI Saliency Map (Extracted from Fold {fold_idx})'
        if acc > 0: title += f' - Acc: {acc:.1f}%'
        plt.title(title, fontsize=15, fontweight='bold')
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # 专门创建一个独立的高清解析图文件夹
        xai_dir = os.path.join(self.run_dir, 'xai_visualizations')
        os.makedirs(xai_dir, exist_ok=True)
        
        save_path = os.path.join(xai_dir, f'Post_Training_XAI_Fold_{fold_idx}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   ✅ [生图成功] 绝美热力图已保存至: {save_path}")


def main():
    print("==================================================")
    print("🚀 独立 XAI 引擎：深度解析预训练模型权重")
    print("==================================================")
    
    # 1. 自动寻找最近运行的 V2_Optimal 结果文件夹
    results_base = 'ml_results'
    if not os.path.exists(results_base):
        print("未找到 ml_results 文件夹，请先运行 main.py 训练模型。")
        return
        
    all_runs = glob.glob(os.path.join(results_base, 'run_*_V2_Optimal'))
    if not all_runs:
        # 兼容其他命名格式
        all_runs = glob.glob(os.path.join(results_base, 'run_*_DualHead_Tuned'))
        if not all_runs:
            print("未找到包含训练权重的项目文件夹！")
            return
            
    # 按时间排序，获取最新的一次运行记录
    latest_run_dir = sorted(all_runs, reverse=True)[0]
    print(f"📂 锁定最新的训练工程资产: {latest_run_dir}")
    
    # 2. 启动分析器
    analyzer = XAI_Analyzer(latest_run_dir)
    
    # 3. 执行自动寻优与制图
    analyzer.load_weight_and_visualize(fold_idx=29)  # fold_idx=None 表示让代码自动找最强的那一轮
    
    print("\n💡 进阶提示：如果您想强制查看其它 Fold (例如 Fold 42) 的热力图，")
    print("   只需修改最后一行代码为 analyzer.load_weight_and_visualize(fold_idx=42)")

if __name__ == "__main__":
    main()