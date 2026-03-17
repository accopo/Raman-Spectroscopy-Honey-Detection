import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import combinations
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import datetime
import random
import logging

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 【论文创新 1】: 1D 通道注意力机制模块
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


# ==========================================
# 【论文创新 2】: 双头任务解耦 CNN
# ==========================================
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
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1) 
        
        out_multi = self.head_multi(features)
        out_bin = self.head_binary(features)
        
        return out_multi, out_bin


class OptimizedCNN_Wrapper:
    """
    无损微调特化版 Wrapper
    """
    def __init__(self, epochs=65, batch_size=16, lr=0.001, pure_threshold=0.35):  
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        # 【无损微调利器 1】：置信度阈值 (Threshold)
        # 默认 0.5。当我们设为 0.45 时，意味着只要"纯蜂蜜"的概率大于 45%，我们就无脑相信它是纯的。
        # 这个值越低，纯判假(误杀)越少，但假判纯(漏网)可能会微增。可以根据需要微调 (0.40 ~ 0.55 之间)！
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

        self.model = DualHeadSpectralCNN1D(input_size=X.shape[1], num_classes=num_classes).to(self.device)
        
        criterion_multi = FocalLoss(gamma=1.0) 
        
        # 【无损微调利器 2】：标签平滑 (Label Smoothing)
        # 防止二分类主头过于自信，软化决策边界，增强泛化稳定性！
        criterion_bin = nn.CrossEntropyLoss(label_smoothing=0.05)  
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 【无损微调利器 3】：微量高斯噪声注入 (Data Augmentation)
                # 0.5% 的强度刚好模拟仪器的本底噪声，使得卷积核被迫学习真实的糖类光谱波峰
                noise = torch.randn_like(batch_X) * 0.005 
                batch_X_noisy = batch_X + noise
                
                batch_y_bin = (batch_y == self.pure_idx).long()
                
                optimizer.zero_grad()
                out_multi, out_bin = self.model(batch_X_noisy)
                
                loss_multi = criterion_multi(out_multi, batch_y)
                loss_bin = criterion_bin(out_bin, batch_y_bin)
                
                loss = 0.75 * loss_bin + 0.25 * loss_multi
                
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            out_multi, out_bin = self.model(X_tensor)
            
            # 【应用无损微调利器 1】：概率阈值偏移
            # 不再使用死板的 argmax，而是获取具体的预测概率 (Softmax)
            bin_probs = F.softmax(out_bin, dim=1) 
            pure_probs = bin_probs[:, 1] # 获取预测为 Pure (索引1) 的置信度
            
            # 如果 Pure 的概率大于我们设置的保护阈值，就判定为 Pure
            bin_pred = (pure_probs >= self.pure_threshold).long() 
            
            multi_pred = torch.argmax(out_multi, dim=1)   
            
            final_pred = multi_pred.clone()

            # 门控决策机制 Boundary Gating
            # 1. 绝对拦截"纯判假" (受 pure_threshold 保护)
            final_pred[bin_pred == 1] = self.pure_idx
            
            # 2. 绝对拦截"假判纯"
            conflict_mask = (bin_pred == 0) & (multi_pred == self.pure_idx)
            if conflict_mask.any():
                spiked_logits = out_multi.clone()
                spiked_logits[:, self.pure_idx] = -float('inf') 
                final_pred[conflict_mask] = torch.argmax(spiked_logits[conflict_mask], dim=1)
                
        return self.label_encoder.inverse_transform(final_pred.cpu().numpy())


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
        if nums:
            return str(int(nums[0]))
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
    
    if len(mixed_spiked_df) == 0:
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

    soft_errors = 0
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != 'Pure' and pred_label != 'Pure' and true_label != pred_label:
            soft_errors += 1

    return {
        'accuracy': accuracy,
        'pure_misclassified_as_spiked': pure_misclassified_as_spiked,
        'spiked_misclassified_as_pure': spiked_misclassified_as_pure,
        'soft_errors': soft_errors,
        'total_pure': len(pure_indices),
        'total_spiked': len(spiked_indices)
    }


def setup_logging(output_dir):
    """设置日志记录"""
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_cnn_mixed_experiment(df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df, output_dir):
    logger = setup_logging(output_dir)
    
    target_samples = pure_samples + mixed_spiked_df['Sample'].tolist()
    target_df = df[df['Sample'].isin(target_samples)].copy()
    target_df = create_concentration_labels(target_df, mixed_spiked_df, pure_samples)

    # 默认 pure_threshold=0.45，为纯品提供 5% 的偏袒保护！
    model = OptimizedCNN_Wrapper(epochs=65, batch_size=16, lr=0.001, pure_threshold=0.45)

    test_combinations = list(combinations(valid_main_ids, 2))
    results = []

    weights_dir = os.path.join(output_dir, 'model_weights')
    os.makedirs(weights_dir, exist_ok=True)

    logger.info(f"🔬 开始无损微调: Dual-Head CNN (标签平滑 + 概率偏移)")

    for combo_idx, test_ids in enumerate(test_combinations):
        test_ids = list(test_ids)
        train_ids = [m for m in valid_main_ids if m not in test_ids]

        train_pure = [f"H{i}" for i in train_ids]
        test_pure = [f"H{i}" for i in test_ids]

        train_df = target_df[
            (target_df['Sample'].isin(train_pure)) | 
            ((target_df['SampleType'] != 'Pure') & (target_df['Main_Honey_ID_Str'].isin(train_ids)))
        ]
        test_df = target_df[
            (target_df['Sample'].isin(test_pure)) | 
            ((target_df['SampleType'] != 'Pure') & (target_df['Main_Honey_ID_Str'].isin(test_ids)))
        ]

        if len(train_df) == 0 or len(set(train_df['Concentration_Target'].values)) < 2:
            continue

        X_train, y_train = train_df[wavenumber_cols].values, train_df['Concentration_Target'].values
        X_test, y_test = test_df[wavenumber_cols].values, test_df['Concentration_Target'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        sample_types_test = test_df['SampleType'].values

        try:
            model.fit(X_train_scaled, y_train)
            
            weight_path = os.path.join(weights_dir, f'fold_{combo_idx+1}_model.pth')
            torch.save(model.model.state_dict(), weight_path)
            
            y_pred = model.predict(X_test_scaled)
            metrics = calculate_error_metrics(y_test, y_pred, sample_types_test)
            
            results.append({
                'Combination': combo_idx + 1,
                'Model': 'Dual-Head CNN (Tuned)',
                'Accuracy': metrics['accuracy'],
                'Pure_Misclassified_As_Spiked': metrics['pure_misclassified_as_spiked'],
                'Spiked_Misclassified_As_Pure': metrics['spiked_misclassified_as_pure'],
                'Soft_Errors': metrics['soft_errors'],
                'Total_Pure': metrics['total_pure'],
                'Total_Spiked': metrics['total_spiked']
            })
            
            if (combo_idx + 1) % 10 == 0 or (combo_idx + 1) == 91:
                logger.info(f"   ✓ 稳步推进中: 已完成 {combo_idx + 1}/91 轮交叉验证并保存权重...")
                
        except Exception as e:
            logger.error(f"轮次 {combo_idx+1} 失败: {e}")

    return pd.DataFrame(results)


def summarize_cnn_results(results_df, output_dir):
    logger = logging.getLogger(__name__)
    
    if len(results_df) == 0:
        logger.warning("结果为空，无法生成总结")
        return
        
    logger.info("=== 无损微调 Dual-Head 门控 CNN 混合模型终极指标 ===")
    
    total_pure = results_df['Total_Pure'].sum()       
    total_spiked = results_df['Total_Spiked'].sum()   
    total_samples = total_pure + total_spiked    

    hard_err_pure = results_df['Pure_Misclassified_As_Spiked'].sum() / total_pure * 100 
    hard_err_spiked = results_df['Spiked_Misclassified_As_Pure'].sum() / total_spiked * 100 
    soft_err = results_df['Soft_Errors'].sum() / total_spiked * 100 
    
    total_errors = results_df['Pure_Misclassified_As_Spiked'].sum() + results_df['Spiked_Misclassified_As_Pure'].sum() + results_df['Soft_Errors'].sum()
    total_err_rate = total_errors / total_samples * 100 

    logger.info("-" * 65)
    logger.info(f" 测试集样本总数 : {total_samples} (纯蜂蜜 {total_pure} / 混合掺假 {total_spiked})")
    logger.info("-" * 65)
    logger.info(f" 纯判掺假(%) (硬错误) : {hard_err_pure:>8.2f}%")
    logger.info(f" 掺假判纯(%) (硬错误) : {hard_err_spiked:>8.2f}%")
    logger.info(f" 软错误率(%) (浓度错判): {soft_err:>8.2f}%")
    logger.info(f" => 【CNN 全局总错误率】: {total_err_rate:>8.2f}%")
    logger.info("-" * 65)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    errors = [results_df['Pure_Misclassified_As_Spiked'].sum(), 
              results_df['Spiked_Misclassified_As_Pure'].sum(), 
              results_df['Soft_Errors'].sum()]
    labels = ['Pure -> Spiked (Hard)', 'Spiked -> Pure (Hard)', 'Concentration Error (Soft)']
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    ax1.pie(errors, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    ax1.add_patch(centre_circle)
    ax1.set_title('Distribution of Fine-Tuned CNN Errors')
    
    rates = [hard_err_pure, hard_err_spiked, soft_err, total_err_rate]
    bars = ['Pure Error', 'Spiked Error', 'Soft Error', 'Total Error']
    ax2.bar(bars, rates, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    for i, v in enumerate(rates):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
    ax2.set_ylabel('Error Rate (%)')
    ax2.set_title('Fine-Tuned CNN Error Rates Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fine_Tuned_CNN_Mixed_Model_Report.png'), dpi=300)
    plt.close()


def generate_random_id(length=4):
    """生成随机ID"""
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.choice(chars) for _ in range(length))


def main():
    if not torch.cuda.is_available():
        print("💡 提示: 当前使用 CPU 训练。")
        
    data_res = load_and_filter_data()
    if data_res is None: 
        print("数据加载失败")
        return
    df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df = data_res

    # 动态时间戳加随机编号建包
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    random_id = generate_random_id()
    output_dir = f'ml_results/run_{run_time}_ID{random_id}_DualHead_Tuned'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 本次运行资产将全部分类保存在: {output_dir}")
    print(f"   📋 运行编号: {random_id}")

    cnn_results = run_cnn_mixed_experiment(df, wavenumber_cols, pure_samples, valid_main_ids, mixed_spiked_df, output_dir)
    
    summarize_cnn_results(cnn_results, output_dir)
    results_file = os.path.join(output_dir, f'Dual_Head_CNN_Mixed_91_rounds_Tuned_ID{random_id}.csv')
    cnn_results.to_csv(results_file, index=False)
    
    print(f"\n✅ 无损微调版运行完毕并归档！")
    print(f"   📊 结果文件: {results_file}")
    print(f"   📝 日志文件: {os.path.join(output_dir, 'training.log')}")


if __name__ == "__main__":
    main()
#sbatch run_test.sh 
