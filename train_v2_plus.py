import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from itertools import combinations
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import datetime
import random

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 模块 1: 深度学习核心架构 (V2版)
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction='mean'): 
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return torch.mean(focal_loss)
        elif self.reduction == 'sum': return torch.sum(focal_loss)
        return focal_loss

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
        flatten_size = self._get_conv_output((1, input_size))
        
        self.head_multi = nn.Sequential(
            nn.Linear(flatten_size, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.01), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        self.head_binary = nn.Sequential(
            nn.Linear(flatten_size, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.01), nn.Dropout(0.4),
            nn.Linear(128, 2) 
        )

    def _get_conv_output(self, shape):
        bs = 1
        input_tensor = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.features(input_tensor)
        return output_feat.data.view(bs, -1).size(1)

    def forward(self, x):
        features = self.features(x).view(x.size(0), -1) 
        return self.head_multi(features), self.head_binary(features)


# ==========================================
# 模块 2: 训练器封装
# ==========================================
class OptimizedCNN_Trainer:
    def __init__(self, epochs=65, batch_size=16, lr=0.001, pure_threshold=0.45):  
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pure_threshold = pure_threshold 
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.pure_idx = None, None

    def fit(self, X, y):
        if self.model is None:
            self.label_encoder.fit(y)
            self.pure_idx = list(self.label_encoder.classes_).index('Pure')
            self.model = DualHeadSpectralCNN1D(input_size=X.shape[1], num_classes=len(self.label_encoder.classes_)).to(self.device)
            
        y_encoded = self.label_encoder.transform(y)
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y_encoded, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion_multi = FocalLoss(gamma=1.0) 
        criterion_bin = nn.CrossEntropyLoss(label_smoothing=0.05)  
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                batch_X_noisy = batch_X + torch.randn_like(batch_X) * 0.005 
                batch_y_bin = (batch_y == self.pure_idx).long()
                
                optimizer.zero_grad()
                out_multi, out_bin = self.model(batch_X_noisy)
                
                loss = 0.75 * criterion_bin(out_bin, batch_y_bin) + 0.25 * criterion_multi(out_multi, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out_multi, out_bin = self.model(torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device))
            
            bin_pred = (F.softmax(out_bin, dim=1)[:, 1] >= self.pure_threshold).long() 
            final_pred = torch.argmax(out_multi, dim=1)   

            final_pred[bin_pred == 1] = self.pure_idx
            conflict_mask = (bin_pred == 0) & (final_pred == self.pure_idx)
            if conflict_mask.any():
                spiked_logits = out_multi.clone()
                spiked_logits[:, self.pure_idx] = -float('inf') 
                final_pred[conflict_mask] = torch.argmax(spiked_logits[conflict_mask], dim=1)
                
        return self.label_encoder.inverse_transform(final_pred.cpu().numpy())


# ==========================================
# 模块 3: 【极致修复版】数据读取与清洗
# ==========================================
def load_and_filter_data():
    print("\n[Data Loader] 开始加载和筛选数据...")
    try:
        df = pd.read_csv('/public/home/liuzhenfang/datasets/honey_spectral_master.csv')
    except FileNotFoundError:
        print("   ❌ 错误: 找不到文件 honey_spectral_processed_all.csv")
        return None

    wavenumber_cols = [col for col in df.columns if col.replace('.', '', 1).isdigit()]
    excluded_samples = ['4', '8', '11']
    valid_main_ids = [str(i) for i in range(1, 18) if str(i) not in excluded_samples]
    pure_samples = [f'H{i}' for i in valid_main_ids]
    
    # 1. 完美复刻您验证过的 clean_id
    def clean_id(x):
        if pd.isna(x): return 'nan'
        s = str(x).strip().upper()
        if s.startswith('H'): s = s[1:]
        if s.endswith('.0'): s = s[:-2]
        return s
            
    df['Main_Honey_ID_Str'] = df['Main_Honey_ID'].apply(clean_id)
    
    # 2. 完美复刻您的 Type 清洗机制
    if 'Type' in df.columns:
        df['Type_Clean'] = df['Type'].astype(str).str.strip().str.lower()
    else:
        print("   ❌ 错误: 数据集中找不到 'Type' 列！")
        return None

    # 3. 严格提取
    pure_df = df[df['Sample'].isin(pure_samples)].copy()
    rice_spiked_df = df[(df['Type_Clean'] == 'rice-spiked') & (df['Main_Honey_ID_Str'].isin(valid_main_ids))].copy()
    beet_spiked_df = df[(df['Type_Clean'] == 'beet-spiked') & (df['Main_Honey_ID_Str'].isin(valid_main_ids))].copy()

    mixed_spiked_df = pd.concat([rice_spiked_df, beet_spiked_df], ignore_index=True)
    
    print(f"   ✓ 提取纯蜂蜜样本: {len(pure_df)} 个")
    print(f"   ✓ 提取大米掺假样本: {len(rice_spiked_df)} 个")
    print(f"   ✓ 提取甜菜掺假样本: {len(beet_spiked_df)} 个")
    
    if len(mixed_spiked_df) == 0:
        print("   ❌ 致命错误：大米和甜菜的掺假样本读取为 0！请检查 CSV 中的 Type 列是否真的是 'rice-spiked' / 'beet-spiked'")
        return None
    
    filtered_df = pd.concat([pure_df, mixed_spiked_df], ignore_index=True)
    return filtered_df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df


def create_concentration_labels(df, spiked_df, pure_samples):
    df_with_labels = df.copy()
    df_with_labels['Concentration_Target'] = 'Unknown'
    df_with_labels['SampleType'] = 'Unknown'
    df_with_labels.loc[df_with_labels['Sample'].isin(pure_samples), 'Concentration_Target'] = 'Pure'
    df_with_labels.loc[df_with_labels['Sample'].isin(pure_samples), 'SampleType'] = 'Pure'

    for idx, row in spiked_df.iterrows():
        # 【极其重要修复】：防止浮点数精度爆炸 (0.1 变成 0.100000001 导致 == 判断失败)
        # 兼容大小写字段名
        c_val = row.get('Mix_Concentration', row.get('mix_concentration', 0))
        try: 
            c = float(c_val)
        except: 
            c = 0.0

        if pd.isna(c) or c == 0: target = 'Pure'
        # 使用区间容错判断，彻底消灭浮点误差漏判！
        elif 0.05 < c < 0.15 or c == 10: target = 'Mixed-10%'
        elif 0.15 < c < 0.25 or c == 20: target = 'Mixed-20%'
        elif 0.25 < c < 0.35 or c == 30: target = 'Mixed-30%'
        elif 0.40 < c < 0.60 or c == 50: target = 'Mixed-50%'
        else: target = 'Mixed-Unknown'

        mask = df_with_labels['Sample'] == row['Sample']
        df_with_labels.loc[mask, 'Concentration_Target'] = target
        df_with_labels.loc[mask, 'SampleType'] = 'Mixed-Spiked'

    return df_with_labels


# ==========================================
# 模块 4 & 5: 交叉验证核心管线与评估图表
# ==========================================
def calculate_error_metrics(y_true, y_pred, sample_types):
    accuracy = accuracy_score(y_true, y_pred)
    pure_indices = [i for i, st in enumerate(sample_types) if st == 'Pure']
    spiked_indices = [i for i, st in enumerate(sample_types) if st != 'Pure']
    
    pure_err = sum(1 for i in pure_indices if y_true[i] == 'Pure' and y_pred[i] != 'Pure')
    spiked_err = sum(1 for i in spiked_indices if y_true[i] != 'Pure' and y_pred[i] == 'Pure')
    soft_err = sum(1 for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != 'Pure' and p != 'Pure' and t != p)

    return {
        'accuracy': accuracy, 'pure_misclassified_as_spiked': pure_err,
        'spiked_misclassified_as_pure': spiked_err, 'soft_errors': soft_err,
        'total_pure': len(pure_indices), 'total_spiked': len(spiked_indices)
    }

def run_cnn_mixed_experiment(df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df, output_dir):
    target_samples = pure_samples + mixed_spiked_df['Sample'].tolist()
    target_df = df[df['Sample'].isin(target_samples)].copy()
    target_df = create_concentration_labels(target_df, mixed_spiked_df, pure_samples)

    trainer = OptimizedCNN_Trainer(epochs=65, batch_size=16, lr=0.001, pure_threshold=0.45)
    test_combinations = list(combinations(valid_main_ids, 2))
    results, all_y_true, all_y_pred = [], [], []

    weights_dir = os.path.join(output_dir, 'model_weights')
    os.makedirs(weights_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"🔬 开始 V2 双头门控 CNN: 91轮交叉验证验证")
    print(f"=======================================================")

    for combo_idx, test_ids in enumerate(test_combinations):
        test_ids = list(test_ids)
        train_ids = [m for m in valid_main_ids if m not in test_ids]

        train_pure, test_pure = [f"H{i}" for i in train_ids], [f"H{i}" for i in test_ids]
        
        train_df = target_df[
            (target_df['Sample'].isin(train_pure)) | 
            ((target_df['SampleType'] != 'Pure') & (target_df['Main_Honey_ID_Str'].isin(train_ids)))
        ]
        test_df = target_df[
            (target_df['Sample'].isin(test_pure)) | 
            ((target_df['SampleType'] != 'Pure') & (target_df['Main_Honey_ID_Str'].isin(test_ids)))
        ]

        # 【终极报警系统】：精准打印出到底为什么被跳过！
        if len(train_df) == 0:
            print(f"   ⚠️ [警告] 轮次 {combo_idx+1}: 训练集完全为空，跳过！")
            continue
            
        classes_in_train = set(train_df['Concentration_Target'].values)
        if len(classes_in_train) < 2:
            print(f"   ⚠️ [警告] 轮次 {combo_idx+1}: 分类标签不足 (仅含有 {classes_in_train})，跳过！")
            continue
            
        if 'Pure' not in classes_in_train:
            print(f"   ⚠️ [致命] 轮次 {combo_idx+1}: 训练集中没有找到纯蜂蜜标签！跳过！")
            continue

        X_train, y_train = train_df[wavenumber_cols].values, train_df['Concentration_Target'].values
        X_test, y_test = test_df[wavenumber_cols].values, test_df['Concentration_Target'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        try:
            trainer.fit(X_train_scaled, y_train)
            torch.save(trainer.model.state_dict(), os.path.join(weights_dir, f'fold_{combo_idx+1}_model.pth'))
            
            y_pred = trainer.predict(X_test_scaled)
            metrics = calculate_error_metrics(y_test, y_pred, test_df['SampleType'].values)
            
            results.append({
                'Combination': combo_idx + 1, 'Model': 'Dual-Head CNN (V2 Tuned)',
                'Accuracy': metrics['accuracy'],
                'Pure_Misclassified_As_Spiked': metrics['pure_misclassified_as_spiked'],
                'Spiked_Misclassified_As_Pure': metrics['spiked_misclassified_as_pure'],
                'Soft_Errors': metrics['soft_errors'],
                'Total_Pure': metrics['total_pure'], 'Total_Spiked': metrics['total_spiked']
            })
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            if (combo_idx + 1) % 10 == 0 or (combo_idx + 1) == 91:
                print(f"   ✓ 稳步推进中: 已完成 {combo_idx + 1}/91 轮...")
                
        except Exception as e:
            print(f"   ❌ 轮次 {combo_idx+1} 代码执行崩溃: {e}")

    return pd.DataFrame(results), all_y_true, all_y_pred


def plot_global_confusion_matrix(y_true, y_pred, output_dir, rand_id):
    if len(y_true) == 0: return
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    ideal_order = ['Pure', 'Mixed-10%', 'Mixed-20%', 'Mixed-30%', 'Mixed-50%', 'Mixed-Unknown']
    labels_order = [lbl for lbl in ideal_order if lbl in all_labels]
    for lbl in all_labels:
        if lbl not in labels_order: labels_order.append(lbl)
            
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order,
                linewidths=1, linecolor='black', annot_kws={"size": 12, "weight": "bold"})
    plt.title('Global Confusion Matrix (Aggregated across 91 Folds)', fontsize=15, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Global_Confusion_Matrix_{rand_id}.png'), dpi=300)
    plt.close()

def summarize_cnn_results(results_df, output_dir, rand_id):
    if len(results_df) == 0: 
        print("\n❌ 警告：结果DataFrame为空，未能生成统计图表。")
        return
        
    print(f"\n=== V2 门控 CNN 混合模型终极指标 ===")
    t_pure = results_df['Total_Pure'].sum()       
    t_spiked = results_df['Total_Spiked'].sum()   
    t_samples = t_pure + t_spiked    

    h_err_p = results_df['Pure_Misclassified_As_Spiked'].sum() / t_pure * 100 
    h_err_s = results_df['Spiked_Misclassified_As_Pure'].sum() / t_spiked * 100 
    s_err = results_df['Soft_Errors'].sum() / t_spiked * 100 
    total_err = (results_df['Pure_Misclassified_As_Spiked'].sum() + results_df['Spiked_Misclassified_As_Pure'].sum() + results_df['Soft_Errors'].sum()) / t_samples * 100 

    print("-" * 65)
    print(f" 纯判掺假(%) : {h_err_p:>8.2f}% | 掺假判纯(%) : {h_err_s:>8.2f}%")
    print(f" 软错误率(%) : {s_err:>8.2f}% | 【CNN 全局总错误率】: {total_err:>8.2f}%")
    print("-" * 65)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.pie([results_df['Pure_Misclassified_As_Spiked'].sum(), results_df['Spiked_Misclassified_As_Pure'].sum(), results_df['Soft_Errors'].sum()], 
            labels=['Pure -> Spiked', 'Spiked -> Pure', 'Concentration Error'], colors=['#ff9999','#66b3ff','#99ff99'], autopct='%1.2f%%')
    ax1.add_patch(plt.Circle((0,0),0.70,fc='white'))
    ax1.set_title('Distribution of CNN Errors')
    
    rates, bars = [h_err_p, h_err_s, s_err, total_err], ['Pure Error', 'Spiked Error', 'Soft Error', 'Total Error']
    ax2.bar(bars, rates, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    for i, v in enumerate(rates): ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')
    ax2.set_ylabel('Error Rate (%)'); ax2.set_title('Error Rates Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'V2_CNN_Report_{rand_id}.png'), dpi=300)
    plt.close()


def main():
    if not torch.cuda.is_available(): print("💡 提示: 当前使用 CPU 训练。")
    data_res = load_and_filter_data()
    if data_res is None: return
    df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df = data_res

    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    rand_id = f"{random.randint(10000, 99999)}"  
    output_dir = f'ml_results/run_{run_time}_{rand_id}_V2_Optimal'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 本次运行资产将保存在: {output_dir}")

    cnn_results, all_y_true, all_y_pred = run_cnn_mixed_experiment(
        df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df, output_dir
    )
    
    plot_global_confusion_matrix(all_y_true, all_y_pred, output_dir, rand_id)
    summarize_cnn_results(cnn_results, output_dir, rand_id)
    if len(cnn_results) > 0:
        cnn_results.to_csv(os.path.join(output_dir, f'V2_CNN_91_rounds_{rand_id}.csv'), index=False)
        print(f"\n✅ V2 模型已成功运行完毕并归档！")

if __name__ == "__main__":
    main()
#sbatch run_main_v2.sh
#tail -f train_log_v2.txt