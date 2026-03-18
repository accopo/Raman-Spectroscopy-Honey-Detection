import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# ==========================================
# 创新点 1: 1D 通道注意力机制 (保持核心创新)
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

# ==========================================
# 创新点 2: Focal Loss (保持核心创新)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==========================================
# 创新点 3: 双头解耦模型架构 (优化升级版: 加宽加深网络容量)
# ==========================================
class MedicalDualHeadCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MedicalDualHeadCNN, self).__init__()
        
        # 优化1: 增加通道数 (16->32, 32->64, 64->128)，提升高维光谱特征提取能力
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.LeakyReLU(0.01),
            SEBlock1D(32), nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            SEBlock1D(64), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.01),
            SEBlock1D(128), nn.AdaptiveAvgPool1d(16) # 压缩长度以防参数爆炸
        )
        
        flatten_dim = 128 * 16 # 2048
        
        # 多分类头：Healthy, Suspected, COVID
        # 优化2: 加大 Dropout 防过拟合 (0.4 -> 0.5)
        self.head_multi = nn.Sequential(
            nn.Linear(flatten_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.01), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 二分类头：Healthy (1) vs Abnormal (0)
        self.head_binary = nn.Sequential(
            nn.Linear(flatten_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.01), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.head_multi(feat), self.head_binary(feat)

# ==========================================
# 论文临床评估核心逻辑
# ==========================================
def calculate_clinical_metrics(y_true, y_pred, pos_label, neg_label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = (y_true == pos_label) | (y_true == neg_label)
    yt = y_true[mask]
    yp = y_pred[mask]
    
    TP = np.sum((yt == pos_label) & (yp == pos_label))
    FN = np.sum((yt == pos_label) & (yp != pos_label))
    TN = np.sum((yt == neg_label) & (yp == neg_label))
    FP = np.sum((yt == neg_label) & (yp != neg_label))
    
    acc = (TP + TN) / len(yt) * 100 if len(yt) > 0 else 0
    sens = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0
    
    return acc, sens, spec

def plot_paper_clinical_bars(results_df, output_dir):
    print("-> 正在渲染临床指标对比直方图...")
    plt.figure(figsize=(10, 6))
    
    melted_df = pd.melt(results_df, id_vars=['Comparison Pair'], 
                        value_vars=['Accuracy (%)', 'Sensitivity (%)', 'Specificity (%)'],
                        var_name='Metric', value_name='Percentage')
    
    sns.barplot(x='Comparison Pair', y='Percentage', hue='Metric', data=melted_df, 
                palette=['#4c72b0', '#55a868', '#c44e52'], edgecolor='black', linewidth=1)
    
    plt.ylim(0, 115)
    plt.title('Clinical Diagnostic Performance by Pairwise Comparison', fontsize=15, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.xlabel('', fontsize=12)
    plt.xticks(fontsize=11, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clinical_performance_bars.png'), dpi=300)
    plt.close()

def plot_tsne_visualization(X_scaled, y_labels, output_dir):
    print("-> 正在生成 t-SNE 血清光谱流形分布图...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(9, 7))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_labels, 
                    palette={'Healthy':'#2ca02c', 'Suspected':'#ff7f0e', 'COVID':'#d62728'}, 
                    s=60, alpha=0.8, edgecolor='w')
    plt.title('t-SNE Visualization of Serum Raman Spectra', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'), dpi=300)
    plt.close()

def plot_global_confusion_matrix(y_true, y_pred, labels, output_dir):
    print("-> 正在生成全局混淆矩阵热力图...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    plt.title('Medical Diagnosis Global Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Disease State', fontsize=12)
    plt.xlabel('AI Predicted State', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'global_confusion_matrix.png'), dpi=300)
    plt.close()

def plot_saliency_map(model, X_tensor, y_test, wavenumbers, output_dir):
    print("-> 正在生成 CNN 显著性热力图 (寻找 COVID 生物标志物)...")
    model.eval()
    
    covid_idx = [i for i, label in enumerate(y_test) if label == 'COVID']
    if len(covid_idx) == 0: return
    
    X_covid = X_tensor[covid_idx].requires_grad_()
    _, out_bin = model(X_covid)
    
    abnormal_score = out_bin[:, 0].sum() 
    model.zero_grad()
    abnormal_score.backward()
    
    gradients = np.abs(X_covid.grad.squeeze().cpu().numpy())
    inputs = X_covid.detach().squeeze().cpu().numpy()
    saliency = np.mean(gradients * inputs, axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    mean_spectrum = np.mean(inputs, axis=0)
    mean_spectrum = (mean_spectrum - mean_spectrum.min()) / (mean_spectrum.max() - mean_spectrum.min() + 1e-8)
    wavenumbers_float = [float(w) for w in wavenumbers]
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(wavenumbers_float, mean_spectrum, color='gray', alpha=0.6, label='Mean COVID Spectrum')
    ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Normalized Intensity', fontsize=12)
    
    ax2 = ax1.twinx()
    ax2.plot(wavenumbers_float, saliency, color='red', alpha=0.8, label='AI Biomarker Importance')
    ax2.fill_between(wavenumbers_float, 0, saliency, color='red', alpha=0.2)
    ax2.set_ylabel('Saliency Score', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Explainable AI: Key Raman Shifts Driving COVID-19 Detection', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'saliency_map_covid.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data_path = 'covid_spectral_processed.csv'
    if not os.path.exists(data_path):
        print("错误: 找不到 covid_spectral_processed.csv，请先运行预处理脚本。")
        return
        
    output_dir = "covid_model_results"
    os.makedirs(output_dir, exist_ok=True)
        
    df = pd.read_csv(data_path)
    wavenumbers = [c for c in df.columns if c != 'Label']
    
    X = df[wavenumbers].values
    X_scaled = StandardScaler().fit_transform(X)
    
    y_raw = df['Label'].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    healthy_idx = list(le.classes_).index('Healthy')
    
    plot_tsne_visualization(X_scaled, y_raw, output_dir)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true = []
    all_y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 优化3: 增加整体训练轮数 60 -> 150
    EPOCHS = 150 
    
    print(f"\n>>> 开始进行 5-Fold 医学模型交叉验证训练 (Dual-Head CNN | {EPOCHS} Epochs) <<<")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        y_train_bin = (y_train == healthy_idx).astype(int)
        
        train_ds = TensorDataset(torch.FloatTensor(X_train).unsqueeze(1), torch.LongTensor(y_train), torch.LongTensor(y_train_bin))
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        
        model = MedicalDualHeadCNN(input_size=len(wavenumbers), num_classes=3).to(device)
        criterion_multi = FocalLoss(gamma=2.0)
        criterion_bin = nn.CrossEntropyLoss()
        
        # 优化4: 加入 Weight Decay(L2正则化) 防止过长轮数下的过拟合
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        # 优化5: 引入余弦退火学习率调度器 (平滑收敛的终极武器)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        model.train()
        for epoch in range(EPOCHS):
            for bx, by_m, by_b in train_loader:
                bx, by_m, by_b = bx.to(device), by_m.to(device), by_b.to(device)
                
                # 优化6: 增强的混合数据增强 (平移 + 高斯噪声)
                shift = random.uniform(-0.02, 0.02)
                bx_aug = bx + shift + torch.randn_like(bx) * 0.01 
                
                optimizer.zero_grad()
                out_m, out_b = model(bx_aug)
                
                loss_m = criterion_multi(out_m, by_m)
                loss_b = criterion_bin(out_b, by_b)
                loss = 0.5 * loss_m + 0.5 * loss_b # 平衡损失权重
                loss.backward()
                optimizer.step()
                
            # 每轮结束后更新学习率
            scheduler.step()
                
        # 评估阶段
        model.eval()
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(device)
        with torch.no_grad():
            out_m, out_b = model(X_val_tensor)
            
            prob_b = F.softmax(out_b, dim=1)
            is_healthy_pred = (prob_b[:, 1] > 0.5)
            
            final_pred = torch.argmax(out_m, dim=1)
            final_pred[is_healthy_pred] = healthy_idx
            
            all_y_true.extend(le.inverse_transform(y_val))
            all_y_pred.extend(le.inverse_transform(final_pred.cpu().numpy()))
            
        print(f"  ✓ Fold {fold+1} 训练及推理完成 (已完成 {EPOCHS} 轮平滑收敛)")
        
    last_X_val_tensor = X_val_tensor
    last_y_val_raw = le.inverse_transform(y_val)

    print("\n==================================================")
    print("📊 论文 Table 2 格式临床评估结果")
    print("==================================================")
    
    pairs = [
        {'name': 'COVID-19 vs Suspected', 'pos': 'COVID', 'neg': 'Suspected'},
        {'name': 'COVID-19 vs Healthy', 'pos': 'COVID', 'neg': 'Healthy'},
        {'name': 'Suspected vs Healthy', 'pos': 'Suspected', 'neg': 'Healthy'}
    ]
    
    results_data = []
    
    print(f"{'Comparison Pair':<25} | {'Accuracy (%)':<15} | {'Sensitivity (%)':<15} | {'Specificity (%)':<15}")
    print("-" * 78)
    
    for p in pairs:
        acc, sens, spec = calculate_clinical_metrics(all_y_true, all_y_pred, p['pos'], p['neg'])
        results_data.append({
            'Comparison Pair': p['name'],
            'Accuracy (%)': round(acc, 2),
            'Sensitivity (%)': round(sens, 2),
            'Specificity (%)': round(spec, 2)
        })
        print(f"{p['name']:<25} | {acc:<15.2f} | {sens:<15.2f} | {spec:<15.2f}")
        
    print("-" * 78)
    
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, 'Table2_Clinical_Results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ 临床指标表格已导出至: {csv_path}")
    
    plot_paper_clinical_bars(results_df, output_dir)
    print(f"✅ 临床指标直方图已生成: {os.path.join(output_dir, 'clinical_performance_bars.png')}")
    
    plot_global_confusion_matrix(all_y_true, all_y_pred, list(le.classes_), output_dir)
    print(f"✅ 全局混淆矩阵已生成: {os.path.join(output_dir, 'global_confusion_matrix.png')}")
    
    plot_saliency_map(model, last_X_val_tensor, last_y_val_raw, wavenumbers, output_dir)
    print(f"✅ XAI可解释性特征峰图已生成: {os.path.join(output_dir, 'saliency_map_covid.png')}")

if __name__ == "__main__":
    main()