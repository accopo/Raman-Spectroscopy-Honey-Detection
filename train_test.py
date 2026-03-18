import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix

# 导入我们刚刚写好的 V3 模型和损失函数
from raman_model_v3 import RamanDualHeadCNN_V3, RamanCompositeLoss

# ==========================================
# 🔴 第一步：完美适配你 CSV 格式的数据集加载器
# ==========================================
class RamanDataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        super(RamanDataset, self).__init__()
        
        print(f"Loading data from: {csv_path} ...")
        # 读取 CSV 文件
        df = pd.read_csv(csv_path) 
        
        # 为了保证训练稳健，我们固定随机种子打乱整个数据集
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # -------------------------------------------------------------------
        # 1. 解析光谱特征 (根据你的 CSV，前 12 列是元数据，后面全是拉曼光谱)
        # -------------------------------------------------------------------
        # iloc[:, 12:] 意思是：取所有行，从第 12 列（索引12）开始一直到最后一列
        features = df.iloc[:, 12:].values 
        
        # -------------------------------------------------------------------
        # 2. 解析标签
        # -------------------------------------------------------------------
        # 二分类标签：Is_Mixed 列 (假设 0代表纯，1代表掺假)
        labels_bin_np = df['Is_Mixed'].astype(int).values
        
        # 多分类标签：Mix_Concentration 列
        # 神经网络分类器需要类别索引是 0, 1, 2... 
        # 我们自动提取所有出现的浓度，并从小到大映射为整数索引
        self.unique_concentrations = sorted(df['Mix_Concentration'].unique())
        self.num_classes = len(self.unique_concentrations)
        conc_to_idx = {val: idx for idx, val in enumerate(self.unique_concentrations)}
        
        # 将实际浓度(如10, 20)映射为索引(如1, 2)
        labels_multi_np = df['Mix_Concentration'].map(conc_to_idx).values

        # -------------------------------------------------------------------
        # 3. 划分训练集 (80%) 和验证集 (20%)
        # -------------------------------------------------------------------
        total_samples = len(df)
        split_idx = int(0.8 * total_samples)
        
        if is_train:
            features = features[:split_idx]
            labels_bin_np = labels_bin_np[:split_idx]
            labels_multi_np = labels_multi_np[:split_idx]
            print(f" => 初始化训练集: {len(features)} 个样本")
        else:
            features = features[split_idx:]
            labels_bin_np = labels_bin_np[split_idx:]
            labels_multi_np = labels_multi_np[split_idx:]
            print(f" => 初始化验证集: {len(features)} 个样本")

        # 将 NumPy 数组转换为 PyTorch 的 Tensor
        self.data = torch.tensor(features, dtype=torch.float32).unsqueeze(1) # 形状变为 (N, 1, Seq_Len)
        self.labels_bin = torch.tensor(labels_bin_np, dtype=torch.float32)   # BCE Loss 需要 float
        self.labels_multi = torch.tensor(labels_multi_np, dtype=torch.long)  # CE Loss 需要 long

        # 记录光谱的真实长度 (比如可能有 1024 或 800 维)
        self.input_length = self.data.shape[2]
        self.num_samples = len(self.labels_bin)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels_bin[idx], self.labels_multi[idx]

# ==========================================
# 🟢 第二步：训练主函数
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 当前使用的设备: {device}")

    # 1. 准备数据
    csv_file_path = '/public/home/liuzhenfang/datasets/honey_spectral_processed_all.csv'
    
    train_dataset = RamanDataset(csv_path=csv_file_path, is_train=True)
    val_dataset = RamanDataset(csv_path=csv_file_path, is_train=False) 
    
    # 打印映射关系，方便你核对结果
    conc_mapping = dict(enumerate(train_dataset.unique_concentrations))
    print(f"💡 浓度映射关系: {conc_mapping}")
    print(f"💡 光谱特征维度: {train_dataset.input_length}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. 动态初始化 V3 模型并送入显卡
    model = RamanDualHeadCNN_V3(
        input_length=train_dataset.input_length, 
        num_multiclass=train_dataset.num_classes
    ).to(device)

    # 3. 初始化复合损失函数和优化器
    criterion = RamanCompositeLoss(alpha=0.75) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 记录最佳权重的变量
    best_combined_acc = 0.0
    save_path = 'best_raman_v3.pth'

    # 4. 开始训练循环
    num_epochs = 70
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, (inputs, labels_bin, labels_multi) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels_bin = labels_bin.to(device)
            labels_multi = labels_multi.to(device)

            # 前向传播
            bin_logits, multi_logits, proj_feat = model(inputs)

            # 计算损失
            loss, loss_bin, loss_multi = criterion(
                bin_logits, multi_logits, proj_feat, labels_bin, labels_multi
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_loss:.4f}")

        # ------------------------------------------
        # 验证阶段 (评估模型)
        # ------------------------------------------
        model.eval()
        correct_bin = 0
        correct_multi = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels_bin, labels_multi in val_loader:
                inputs = inputs.to(device)
                labels_bin = labels_bin.to(device)
                labels_multi = labels_multi.to(device)

                bin_logits, multi_logits, _ = model(inputs)

                # 计算二分类准确率 (纯/伪)
                bin_preds = (torch.sigmoid(bin_logits).view(-1) > 0.5).long()
                correct_bin += (bin_preds == labels_bin.long()).sum().item()

                # 计算多分类准确率 (浓度)
                multi_preds = torch.argmax(multi_logits, dim=1)
                correct_multi += (multi_preds == labels_multi).sum().item()

                total_val += labels_bin.size(0)

        if total_val > 0:
            val_acc_bin = 100. * correct_bin / total_val
            val_acc_multi = 100. * correct_multi / total_val
            print(f"   => Val Bin Acc (真假判别): {val_acc_bin:.2f}% | Val Multi Acc (浓度预测): {val_acc_multi:.2f}%")
            
            # --- 新增：保存最优模型逻辑 ---
            # 我们用 真假判断 和 浓度预测 的综合准确率来衡量模型好坏
            combined_acc = val_acc_bin + val_acc_multi
            if combined_acc > best_combined_acc:
                best_combined_acc = combined_acc
                torch.save(model.state_dict(), save_path)
                print(f"   🌟 发现更优模型！已保存权重至: {save_path} (综合得分: {combined_acc:.2f})")

    # ==========================================
    # 🔴 训练结束：加载最佳模型，输出详细评估报告
    # ==========================================
    print("\n" + "="*50)
    print("🎯 训练结束，正在使用最佳模型生成详细评估报告...")
    print("="*50)
    
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    all_bin_targets = []
    all_bin_preds = []
    all_multi_targets = []
    all_multi_preds = []
    
    with torch.no_grad():
        for inputs, labels_bin, labels_multi in val_loader:
            inputs = inputs.to(device)
            bin_logits, multi_logits, _ = model(inputs)
            
            bin_preds = (torch.sigmoid(bin_logits).view(-1) > 0.5).long()
            multi_preds = torch.argmax(multi_logits, dim=1)
            
            all_bin_targets.extend(labels_bin.cpu().numpy())
            all_bin_preds.extend(bin_preds.cpu().numpy())
            all_multi_targets.extend(labels_multi.cpu().numpy())
            all_multi_preds.extend(multi_preds.cpu().numpy())

    print("\n【任务一】纯伪二分类报告 (0=纯蜂蜜, 1=掺假):")
    print(confusion_matrix(all_bin_targets, all_bin_preds))
    print(classification_report(all_bin_targets, all_bin_preds, target_names=["Pure (0)", "Spiked (1)"]))

    print(f"\n【任务二】浓度多分类报告 (映射关系: {conc_mapping}):")
    print(confusion_matrix(all_multi_targets, all_multi_preds))
    print(classification_report(all_multi_targets, all_multi_preds))
    
    print("✅ 详细报告生成完毕！你可以将这些数据用于论文中的混淆矩阵图表绘制。")

if __name__ == "__main__":
    main()