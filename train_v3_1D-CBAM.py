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
# 【V3 核心创新 1】: 1D CBAM 双域注意力机制
# (同时包含 Channel Attention 与 Spatial Attention)
# ==========================================
class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock1D(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAMBlock1D, self).__init__()
        self.ca = ChannelAttention1D(channel, ratio=ratio)
        self.sa = SpatialAttention1D(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


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


# ==========================================
# 【V3 核心创新 2】: RamanDualHeadCNN_V3 (引入特征投影头)
# ==========================================
class RamanDualHeadCNN_V3(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RamanDualHeadCNN_V3, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2), nn.BatchNorm1d(16), nn.LeakyReLU(0.01),
            CBAMBlock1D(16), nn.MaxPool1d(2, 2),
            
            nn.Conv1d(16, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.LeakyReLU(0.01),
            CBAMBlock1D(32), nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            CBAMBlock1D(64)
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
        
        # 特征投影头 (Projection Head): 为未来的对比学习做准备
        self.proj_head = nn.Sequential(
            nn.Linear(flatten_size, 64), nn.ReLU(), nn.Linear(64, 32)
        )

    def _get_conv_output(self, shape):
        bs = 1
        input_tensor = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.features(input_tensor)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1) 
        
        out_multi = self.head_multi(features_flat)
        out_bin = self.head_binary(features_flat)
        
        proj_feat = self.proj_head(features_flat)
        proj_feat = F.normalize(proj_feat, dim=1) # 归一化用于相似度计算
        
        return out_multi, out_bin, proj_feat


class OptimizedCNN_Wrapper_V3:
    def __init__(self, epochs=65, batch_size=16, lr=0.001, pure_threshold=0.45):  
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pure_threshold = pure_threshold 
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pure_idx = None 

    def fit(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        self.pure_idx = list(self.label_encoder.classes_).index('Pure')

        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = RamanDualHeadCNN_V3(input_size=X.shape[1], num_classes=num_classes).to(self.device)
        
        # FocalLoss 设为 none，方便后续做动态门控加权
        criterion_multi = FocalLoss(gamma=1.0, reduction='none') 
        criterion_bin = nn.CrossEntropyLoss(label_smoothing=0.05)  
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                noise = torch.randn_like(batch_X) * 0.005 
                batch_X_noisy = batch_X + noise
                
                batch_y_bin = (batch_y == self.pure_idx).long()
                
                optimizer.zero_grad()
                
                # 接收 V3 架构的三个输出
                out_multi, out_bin, proj_feat = self.model(batch_X_noisy)
                
                # 1. 主头硬分类损失
                loss_bin = criterion_bin(out_bin, batch_y_bin)
                
                # 2. 【V3核心创新：Gated Focal Loss】
                focal_loss_multi_unreduced = criterion_multi(out_multi, batch_y)
                
                # 获取二分类主头认为该样本是“掺假(Spiked)”的概率
                bin_probs = F.softmax(out_bin, dim=1)
                spiked_prob = bin_probs[:, 0]  # 假设 1 是 Pure，0 是 Spiked
                
                # 动态加权：如果判定为纯品(spiked_prob很小)，则多分类损失接近0，梯度解耦！
                gated_focal_loss = torch.mean(spiked_prob * focal_loss_multi_unreduced)
                
                # 最终组合损失
                loss = 0.75 * loss_bin + 0.25 * gated_focal_loss
                
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            out_multi, out_bin, _ = self.model(X_tensor)
            
            bin_probs = F.softmax(out_bin, dim=1) 
            pure_probs = bin_probs[:, 1] 
            bin_pred = (pure_probs >= self.pure_threshold).long() 
            
            multi_pred = torch.argmax(out_multi, dim=1)   
            final_pred = multi_pred.clone()

            final_pred[bin_pred == 1] = self.pure_idx
            
            conflict_mask = (bin_pred == 0) & (multi_pred == self.pure_idx)
            if conflict_mask.any():
                spiked_logits = out_multi.clone()
                spiked_logits[:, self.pure_idx] = -float('inf') 
                final_pred[conflict_mask] = torch.argmax(spiked_logits[conflict_mask], dim=1)
                
        return self.label_encoder.inverse_transform(final_pred.cpu().numpy())


# ==========================================
# 数据加载与图表绘制工具
# ==========================================
def load_and_filter_data():
    print("开始加载和筛选数据...")
    try:
        df = pd.read_csv('/public/home/liuzhenfang/datasets/honey_spectral_processed_all.csv')
    except FileNotFoundError:
        print("   - 错误: 找不到文件 honey_spectral_processed_all.csv")
        return None

    wavenumber_cols = [col for col in df.columns if col.replace('.', '', 1).isdigit()]
    excluded_samples = ['4', '8', '11']
    valid_main_ids = [str(i) for i in range(1, 18) if str(i) not in excluded_samples]
    pure_samples = [f'H{i}' for i in valid_main_ids]
    
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
    
    pure_df = df[df['Sample'].isin(pure_samples)].copy()
    rice_spiked_df = df[is_rice & (df['Main_Honey_ID_Str'].isin(valid_main_ids))].copy()
    beet_spiked_df = df[is_beet & (df['Main_Honey_ID_Str'].isin(valid_main_ids))].copy()

    mixed_spiked_df = pd.concat([rice_spiked_df, beet_spiked_df], ignore_index=True)
    
    print(f"   - 剩余纯蜂蜜样本数量: {len(pure_df)} 个")
    print(f"   - 混合掺假样本总数量 (Rice+Beet): {len(mixed_spiked_df)} 个")
    
    if len(mixed_spiked_df) == 0: return None
    filtered_df = pd.concat([pure_df, mixed_spiked_df], ignore_index=True)
    return filtered_df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df


def create_concentration_labels(df, spiked_df, pure_samples):
    df_with_labels = df.copy()
    df_with_labels['Concentration_Target'] = 'Unknown'
    df_with_labels['SampleType'] = 'Unknown'
    df_with_labels.loc[df_with_labels['Sample'].isin(pure_samples), 'Concentration_Target'] = 'Pure'
    df_with_labels.loc[df_with_labels['Sample'].isin(pure_samples), 'SampleType'] = 'Pure'

    for idx, row in spiked_df.iterrows():
        c = row.get('Mix_Concentration', 0)
        try: c = float(c)
        except: c = 0.0

        if pd.isna(c) or c == 0: target = 'Pure'
        elif c == 0.1 or c == 10: target = 'Mixed-10%'
        elif c == 0.2 or c == 20: target = 'Mixed-20%'
        elif c == 0.3 or c == 30: target = 'Mixed-30%'
        elif c == 0.5 or c == 50: target = 'Mixed-50%'
        else: target = 'Mixed-Unknown'

        mask = df_with_labels['Sample'] == row['Sample']
        df_with_labels.loc[mask, 'Concentration_Target'] = target
        df_with_labels.loc[mask, 'SampleType'] = 'Mixed-Spiked'

    return df_with_labels


def calculate_error_metrics(y_true, y_pred, sample_types):
    accuracy = accuracy_score(y_true, y_pred)
    pure_indices = [i for i, st in enumerate(sample_types) if st == 'Pure']
    pure_misclassified_as_spiked = sum(1 for i in pure_indices if y_true[i] == 'Pure' and y_pred[i] != 'Pure')
    spiked_indices = [i for i, st in enumerate(sample_types) if st != 'Pure']
    spiked_misclassified_as_pure = sum(1 for i in spiked_indices if y_true[i] != 'Pure' and y_pred[i] == 'Pure')
    soft_errors = sum(1 for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != 'Pure' and p != 'Pure' and t != p)

    return {
        'accuracy': accuracy, 'pure_misclassified_as_spiked': pure_misclassified_as_spiked,
        'spiked_misclassified_as_pure': spiked_misclassified_as_pure, 'soft_errors': soft_errors,
        'total_pure': len(pure_indices), 'total_spiked': len(spiked_indices)
    }

def run_cnn_mixed_experiment(df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df, output_dir):
    target_samples = pure_samples + mixed_spiked_df['Sample'].tolist()
    target_df = df[df['Sample'].isin(target_samples)].copy()
    target_df = create_concentration_labels(target_df, mixed_spiked_df, pure_samples)

    # 启用 V3 终极引擎
    model = OptimizedCNN_Wrapper_V3(epochs=65, batch_size=16, lr=0.001, pure_threshold=0.45)

    test_combinations = list(combinations(valid_main_ids, 2))
    results, all_y_true_global, all_y_pred_global = [], [], []
    weights_dir = os.path.join(output_dir, 'model_weights')
    os.makedirs(weights_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"🔥 开始 V3 终极架构验证: CBAM注意力 + Gated Loss 混合门控")
    print(f"=======================================================")

    last_X_test, last_y_test, last_scaler = None, None, None

    for combo_idx, test_ids in enumerate(test_combinations):
        test_ids = list(test_ids)
        train_ids = [m for m in valid_main_ids if m not in test_ids]

        train_pure, test_pure = [f"H{i}" for i in train_ids], [f"H{i}" for i in test_ids]
        
        train_df = target_df[(target_df['Sample'].isin(train_pure)) | ((target_df['SampleType'] != 'Pure') & (target_df['Main_Honey_ID_Str'].isin(train_ids)))]
        test_df = target_df[(target_df['Sample'].isin(test_pure)) | ((target_df['SampleType'] != 'Pure') & (target_df['Main_Honey_ID_Str'].isin(test_ids)))]

        if len(train_df) == 0 or len(set(train_df['Concentration_Target'].values)) < 2: continue

        X_train, y_train = train_df[wavenumber_cols].values, train_df['Concentration_Target'].values
        X_test, y_test = test_df[wavenumber_cols].values, test_df['Concentration_Target'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        sample_types_test = test_df['SampleType'].values

        try:
            model.fit(X_train_scaled, y_train)
            torch.save(model.model.state_dict(), os.path.join(weights_dir, f'fold_{combo_idx+1}_model.pth'))
            
            y_pred = model.predict(X_test_scaled)
            metrics = calculate_error_metrics(y_test, y_pred, sample_types_test)
            
            results.append({
                'Combination': combo_idx + 1, 'Model': 'RamanDualHeadCNN_V3',
                'Accuracy': metrics['accuracy'],
                'Pure_Misclassified_As_Spiked': metrics['pure_misclassified_as_spiked'],
                'Spiked_Misclassified_As_Pure': metrics['spiked_misclassified_as_pure'],
                'Soft_Errors': metrics['soft_errors'],
                'Total_Pure': metrics['total_pure'], 'Total_Spiked': metrics['total_spiked']
            })
            
            all_y_true_global.extend(y_test)
            all_y_pred_global.extend(y_pred)
            last_X_test, last_y_test, last_scaler = X_test_scaled, y_test, scaler

            if (combo_idx + 1) % 10 == 0 or (combo_idx + 1) == 91:
                print(f"   ✓ V3 网络稳步推进中: 已完成 {combo_idx + 1}/91 轮...")
                
        except Exception as e:
            print(f"轮次 {combo_idx+1} 失败: {e}")

    return pd.DataFrame(results), model, last_X_test, last_y_test, all_y_true_global, all_y_pred_global

def summarize_cnn_results(results_df, output_dir, rand_id):
    if len(results_df) == 0: return
        
    print(f"\n=== V3 网络 (CBAM + Gated Focal Loss) 终极指标 ===")
    total_pure, total_spiked = results_df['Total_Pure'].sum(), results_df['Total_Spiked'].sum()
    total_samples = total_pure + total_spiked    

    hard_err_pure = results_df['Pure_Misclassified_As_Spiked'].sum() / total_pure * 100 
    hard_err_spiked = results_df['Spiked_Misclassified_As_Pure'].sum() / total_spiked * 100 
    soft_err = results_df['Soft_Errors'].sum() / total_spiked * 100 
    total_err_rate = (results_df['Pure_Misclassified_As_Spiked'].sum() + results_df['Spiked_Misclassified_As_Pure'].sum() + results_df['Soft_Errors'].sum()) / total_samples * 100 

    print("-" * 65)
    print(f" 样本总数 : {total_samples} (纯品 {total_pure} / 掺假 {total_spiked})")
    print(f" 纯判掺假(%) (硬错误) : {hard_err_pure:>8.2f}%")
    print(f" 掺假判纯(%) (硬错误) : {hard_err_spiked:>8.2f}%")
    print(f" 软错误率(%) (浓度错判): {soft_err:>8.2f}%")
    print(f" => 【V3 全局总错误率】: {total_err_rate:>8.2f}%")
    print("-" * 65)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    errors = [results_df['Pure_Misclassified_As_Spiked'].sum(), results_df['Spiked_Misclassified_As_Pure'].sum(), results_df['Soft_Errors'].sum()]
    labels = ['Pure -> Spiked (Hard)', 'Spiked -> Pure (Hard)', 'Concentration Error (Soft)']
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    ax1.pie(errors, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90, pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    ax1.add_patch(centre_circle)
    ax1.set_title('Distribution of V3 CNN Errors')
    
    rates = [hard_err_pure, hard_err_spiked, soft_err, total_err_rate]
    bars = ['Pure Error', 'Spiked Error', 'Soft Error', 'Total Error']
    ax2.bar(bars, rates, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    for i, v in enumerate(rates): ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')
    ax2.set_ylabel('Error Rate (%)')
    ax2.set_title('V3 CNN Error Rates Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'V3_CNN_Report_{rand_id}.png'), dpi=300)
    plt.close()

def plot_tsne_visualization(X, y, output_dir, rand_id):
    print(f"🌌 生成高维光谱 t-SNE 降维可视化...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    unique_labels = sorted(list(set(y)))
    colors = sns.color_palette("Set2", len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = (y == label)
        if label == 'Pure':
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, color='red', marker='*', s=150, edgecolors='black', zorder=5)
        else:
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, color=colors[i], alpha=0.7, edgecolors='w', s=60)

    plt.title('t-SNE Visualization of SORS Spectra (Mixed Spiked Dataset)', fontsize=15, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Concentration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tSNE_Visualization_{rand_id}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_global_confusion_matrix(y_true, y_pred, output_dir, rand_id):
    print(f"📊 生成全局混淆矩阵 (Confusion Matrix)...")
    labels_order = ['Pure', 'Mixed-10%', 'Mixed-20%', 'Mixed-30%', 'Mixed-50%']
    labels_order = [lbl for lbl in labels_order if lbl in set(y_true)]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order,
                linewidths=1, linecolor='black', annot_kws={"size": 12, "weight": "bold"})
    plt.title('Global Confusion Matrix (Aggregated across 91 Folds)', fontsize=15, fontweight='bold')
    plt.ylabel('True Label (Actual)', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label (Model Output)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Global_Confusion_Matrix_{rand_id}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_cnn_saliency_map(model_wrapper, X_test, y_test, wavenumber_cols, output_dir, rand_id):
    print(f"🧠 启动 CNN 可解释性分析引擎 (Gradient Saliency Map)...")
    device, model = model_wrapper.device, model_wrapper.model
    model.eval()
    
    spiked_indices = [i for i, label in enumerate(y_test) if label != 'Pure']
    if not spiked_indices: return
        
    X_tensor = torch.tensor(X_test[spiked_indices], dtype=torch.float32).unsqueeze(1).to(device)
    X_tensor.requires_grad_()
    
    _, out_bin, _ = model(X_tensor) # V3 返回三个变量
    spiked_score = out_bin[:, 0].sum()
    model.zero_grad()
    spiked_score.backward()
    
    gradients = X_tensor.grad.squeeze().cpu().numpy()
    inputs = X_tensor.detach().squeeze().cpu().numpy()
    mean_saliency = np.mean(np.abs(gradients * inputs), axis=0)
    
    if mean_saliency.max() > 0:
        mean_saliency = (mean_saliency - mean_saliency.min()) / (mean_saliency.max() - mean_saliency.min())
        
    wavenumbers = [float(w) for w in wavenumber_cols]
    top_indices = np.argsort(mean_saliency)[-10:]
    mean_spectrum = np.mean(inputs, axis=0)
    mean_spectrum = (mean_spectrum - mean_spectrum.min()) / (mean_spectrum.max() - mean_spectrum.min())
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(wavenumbers, mean_spectrum, color='gray', alpha=0.5, linewidth=2, label='Mean Spiked Spectrum')
    ax1.set_xlabel('Raman Shift Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spectral Intensity', color='gray', fontsize=12, fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.plot(wavenumbers, mean_saliency, color='red', linewidth=2, alpha=0.8, label='CBAM Gradient Saliency')
    ax2.fill_between(wavenumbers, 0, mean_saliency, color='red', alpha=0.1)
    ax2.set_ylabel('Feature Contribution Score', color='red', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    
    for idx in top_indices:
        ax2.annotate(f"{wavenumbers[idx]:.1f}", xy=(wavenumbers[idx], mean_saliency[idx]), 
                     xytext=(0, 15), textcoords="offset points", ha='center', va='bottom', fontsize=9, rotation=90, 
                     color='darkred', weight='bold', arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.5))

    plt.title('Explainable AI: Key Biomarkers Driving V3 CNN Detection', fontsize=15, fontweight='bold')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'V3_Saliency_Map_{rand_id}.png'), dpi=300)
    plt.close()

def main():
    if not torch.cuda.is_available(): print("💡 提示: 当前使用 CPU 训练。")
    data_res = load_and_filter_data()
    if data_res is None: return
    df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df = data_res

    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    rand_id = f"{random.randint(10000, 99999)}"  
    output_dir = f'ml_results/run_{run_time}_{rand_id}_V3_Architecture'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 本次 V3 运行资产将全部分类保存在: {output_dir}")

    target_samples = pure_samples + mixed_spiked_df['Sample'].tolist()
    target_df = df[df['Sample'].isin(target_samples)].copy()
    target_df = create_concentration_labels(target_df, mixed_spiked_df, pure_samples)
    X_all_scaled = StandardScaler().fit_transform(target_df[wavenumber_cols].values)
    
    plot_tsne_visualization(X_all_scaled, target_df['Concentration_Target'].values, output_dir, rand_id)

    cnn_results, trained_model_wrapper, last_X_test, last_y_test, all_y_true, all_y_pred = run_cnn_mixed_experiment(
        df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df, output_dir
    )
    
    plot_global_confusion_matrix(all_y_true, all_y_pred, output_dir, rand_id)
    summarize_cnn_results(cnn_results, output_dir, rand_id)
    cnn_results.to_csv(os.path.join(output_dir, f'V3_CNN_91_rounds_{rand_id}.csv'), index=False)
    
    if trained_model_wrapper is not None and last_X_test is not None:
        generate_cnn_saliency_map(trained_model_wrapper, last_X_test, last_y_test, wavenumber_cols, output_dir, rand_id)
    
    print(f"\n✅ V3 终极流水线已执行完毕！")

if __name__ == "__main__":
    main()
#sbatch run_main_v3.sh
#tail -f train_log_v3.txt