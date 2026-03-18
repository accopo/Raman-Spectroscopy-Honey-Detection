import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torch.nn.functional as F

# 导入你的模型
from raman_model_v3 import RamanDualHeadCNN_V3

# 设置全局字体和画图风格 (SCI 风格)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] # 顶刊常用字体
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def load_data_and_model(csv_path, weight_path, device):
    print("📥 正在加载数据与模型...")
    df = pd.read_csv(csv_path)
    
    # 提取拉曼位移 (波段 X 轴)
    # 假设从第13列开始是拉曼光谱的波数(例如 '585.97', '587.57')
    wavenumbers = df.columns[12:].astype(float).values
    features = df.iloc[:, 12:].values
    
    # 提取标签
    labels_bin = df['Is_Mixed'].astype(int).values
    unique_concentrations = sorted(df['Mix_Concentration'].unique())
    # nan 表示纯蜂蜜，其他为掺假浓度
    conc_to_idx = {val: idx for idx, val in enumerate(unique_concentrations)}
    labels_multi = df['Mix_Concentration'].map(conc_to_idx).values
    
    # 构建人类可读的标签名，用于画图
    class_names = []
    for val in unique_concentrations:
        if pd.isna(val): class_names.append("Pure")
        else: class_names.append(f"{int(val*100)}%")
        
    X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1).to(device)
    
    # 加载模型
    model = RamanDualHeadCNN_V3(input_length=features.shape[1], num_multiclass=len(class_names)).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    return X_tensor, labels_bin, labels_multi, wavenumbers, class_names, model, features

def plot_confusion_matrices(labels_bin, preds_bin, labels_multi, preds_multi, class_names):
    print("🎨 正在绘制混淆矩阵热力图...")
    
    # 1. 二分类混淆矩阵
    cm_bin = confusion_matrix(labels_bin, preds_bin)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 18},
                xticklabels=['Pure', 'Spiked'], yticklabels=['Pure', 'Spiked'])
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Binary Classification (Pure vs Spiked)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Fig1_CM_Binary.png', dpi=300)
    plt.close()
    
    # 2. 多分类混淆矩阵
    cm_multi = confusion_matrix(labels_multi, preds_multi)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Oranges', annot_kws={"size": 16},
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Concentration', fontsize=14, fontweight='bold')
    plt.ylabel('True Concentration', fontsize=14, fontweight='bold')
    plt.title('Multi-class Classification (Concentration)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Fig2_CM_Multi.png', dpi=300)
    plt.close()

def plot_tsne(proj_features, labels_multi, class_names):
    print("🎨 正在绘制 t-SNE 潜空间流形图...")
    # 使用 t-SNE 将 64 维特征降维到 2 维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(proj_features)
    
    plt.figure(figsize=(9, 7))
    # 选一组好看的配色方案
    palette = sns.color_palette("Set1", len(class_names))
    
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], 
                    hue=[class_names[i] for i in labels_multi], 
                    hue_order=class_names,
                    palette=palette, s=100, alpha=0.8, edgecolor='k')
    
    plt.xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    plt.title('Latent Space Feature Manifold (Projection Head)', fontsize=16, fontweight='bold')
    plt.legend(title='Concentration', title_fontsize='13', fontsize='12', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig3_tSNE_LatentSpace.png', dpi=300)
    plt.close()

def plot_attention_map(model, X_tensor, features_raw, wavenumbers, labels_bin, device):
    print("🎨 正在提取并绘制 1D-CBAM 空间注意力机制...")
    
    # 找到一个被正确识别的掺假样本 (比如找第一个掺假样本)
    idx = np.where(labels_bin == 1)[0][0]
    sample_tensor = X_tensor[idx:idx+1]
    sample_raw = features_raw[idx]
    
    # 注册一个 Hook 来截获网络中第一层 CBAM 的空间注意力权重 (Shape: [1, 1, 700])
    attention_weights = []
    def get_attention(module, input, output):
        # output is the sigmoid attention map from SpatialAttention1D
        attention_weights.append(output.detach().cpu().numpy())
        
    # 将 hook 挂载到第一个 CBAM 的空间注意力(sa)的最后一个激活层上
    hook_handle = model.shared_features[1].sa.sigmoid.register_forward_hook(get_attention)
    
    # 前向传播 (触发 hook)
    _ = model(sample_tensor)
    hook_handle.remove() # 用完拆除
    
    # 提取权重，形状从 [1, 1, 700] 变成 [700]
    attn = attention_weights[0].flatten()
    
    # 为了视觉效果平滑，我们可以对 attention 稍微进行一点平滑处理 (可选)
    window = 5
    attn_smooth = np.convolve(attn, np.ones(window)/window, mode='same')
    
    # --- 开始画图 ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 绘制原始拉曼光谱
    color1 = 'tab:blue'
    ax1.plot(wavenumbers, sample_raw, color=color1, linewidth=2, label='Raman Spectrum (Spiked)')
    ax1.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim([wavenumbers[0], wavenumbers[-1]])
    
    # 创建共用 x 轴的第二个 y 轴，用于画注意力权重
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    # 绘制注意力曲线，并在下方填充红色以突出显示
    ax2.plot(wavenumbers, attn_smooth, color=color2, linewidth=2, linestyle='--', label='CBAM Spatial Attention')
    ax2.fill_between(wavenumbers, 0, attn_smooth, color=color2, alpha=0.2)
    ax2.set_ylabel('Attention Weight (0~1)', fontsize=14, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 1.05])
    
    # 标出注意力最高的前 2 个波峰对应的波数 (向物理意义致敬)
    top_indices = attn_smooth.argsort()[-3:][::-1] # 找前三大值
    for i in top_indices:
        # 为了避免标得太密，可以加点距离判断，这里简化处理标最高点
        pass
    max_idx = np.argmax(attn_smooth)
    ax2.axvline(x=wavenumbers[max_idx], color='k', linestyle=':', alpha=0.7)
    ax2.text(wavenumbers[max_idx]+5, attn_smooth[max_idx]-0.1, f'{wavenumbers[max_idx]:.1f} cm$^{{-1}}$', 
             fontsize=12, fontweight='bold')
    
    plt.title('Explainable AI: 1D-CBAM Focusing on Key Raman Biomarkers', fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig('Fig4_CBAM_Attention.png', dpi=300)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = 'honey_spectral_processed_all.csv'
    weight_path = 'best_raman_v3.pth'
    
    # 1. 加载所有数据和训练好的模型
    X_tensor, labels_bin, labels_multi, wavenumbers, class_names, model, features_raw = load_data_and_model(csv_path, weight_path, device)
    
    # 2. 获取整个数据集的预测结果
    print("⏳ 正在进行全量数据推理...")
    with torch.no_grad():
        bin_logits, multi_logits, proj_feat = model(X_tensor)
        
        preds_bin = (torch.sigmoid(bin_logits).view(-1) > 0.5).cpu().numpy()
        preds_multi = torch.argmax(multi_logits, dim=1).cpu().numpy()
        proj_features_np = proj_feat.cpu().numpy()
        
    # 3. 逐个生成论文图表
    plot_confusion_matrices(labels_bin, preds_bin, labels_multi, preds_multi, class_names)
    plot_tsne(proj_features_np, labels_multi, class_names)
    plot_attention_map(model, X_tensor, features_raw, wavenumbers, labels_bin, device)
    
    print("\n🎉 大功告成！四张 SCI 级别插图已全部生成在当前目录下，请下载查看！")

if __name__ == "__main__":
    main()
#sbatch run_test.sh