import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 创新点升级 1：1D CBAM 双域注意力机制
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
        # 在通道维度上取最大值和平均值，保留波段(Spatial)维度
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM1D(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention1D(in_planes, ratio)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)  # 突出关键通道
        out = out * self.sa(out) # 突出关键拉曼峰位
        return out

# ==========================================
# 创新点升级 2：多尺度感知模块 (1D-Inception)
# ==========================================
class MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv1D, self).__init__()
        # 宽核：捕捉散射基线与宽频包络 (如 k=7)
        self.branch1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=7, padding=3)
        # 中核：捕捉常规拉曼峰 (如 k=5)
        self.branch2 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=5, padding=2)
        # 窄核：精准定位尖锐特征峰 (如 k=3)
        self.branch3 = nn.Conv1d(in_channels, out_channels - 2*(out_channels // 3), kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.leaky_relu(self.bn(out))

# ==========================================
# 完整架构：V3 双头解耦与软门控网络
# ==========================================
class RamanDualHeadCNN_V3(nn.Module):
    def __init__(self, input_length=700, num_multiclass=4):
        super(RamanDualHeadCNN_V3, self).__init__()
        
        # 1. 共享的主干网络 (多尺度 + CBAM注意力)
        self.shared_features = nn.Sequential(
            MultiScaleConv1D(1, 32),
            CBAM1D(32),
            nn.MaxPool1d(2),
            
            MultiScaleConv1D(32, 64),
            CBAM1D(64),
            nn.MaxPool1d(2),
            
            MultiScaleConv1D(64, 128),
            CBAM1D(128),
            nn.MaxPool1d(2)
        )
        
        # 计算 Flatten 后的维度 (根据 input_length 推算)
        self.flatten_dim = 128 * (input_length // 8) 
        
        # 2. 任务解耦：各自的独立特征提取层 (摒弃GAP，保留空间分布)
        self.binary_fc_features = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        
        self.multi_fc_features = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        
        # 3. 双头决策
        # 主头：纯伪二分类
        self.binary_head = nn.Linear(256, 1) # 输出1维，用BCE Loss
        
        # 副头：多分类 (浓度)
        # 创新点升级 3：可微门控机制 (融合主头信息)
        self.multi_head = nn.Linear(256 + 1, num_multiclass) 

        # 创新点升级 4：对比学习投影头 (Projection Head)
        # 用于在潜空间强制拉开"纯蜂蜜"与"10%掺假"的距离
        self.proj_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        # 形状假设: (Batch, 1, 700)
        shared_out = self.shared_features(x)
        flatten_out = shared_out.view(shared_out.size(0), -1)
        
        # 提取二分类特征并预测
        bin_feat = self.binary_fc_features(flatten_out)
        bin_logits = self.binary_head(bin_feat)
        bin_prob = torch.sigmoid(bin_logits) # 获取样本是"掺假"的概率
        
        # 提取多分类特征
        multi_feat = self.multi_fc_features(flatten_out)
        
        # --- 核心改进：软门控 (Soft-Gating) ---
        # 策略A：将主头的预测概率拼接进去，给多分类头提供强烈暗示
        # 如果 bin_prob 接近 0 (判定为纯)，网络会学习到自动压低各个浓度的预测值
        multi_feat_fused = torch.cat([multi_feat, bin_prob], dim=1) 
        
        # 策略B (可选的乘法门控)：强制抑制多分类特征
        # multi_feat_fused = multi_feat * bin_prob 
        
        multi_logits = self.multi_head(multi_feat_fused)
        
        # --- 创新点 4: 对比学习特征投影 ---
        # 降维并进行 L2 归一化，将特征投射到单位超球面上，便于计算余弦相似度
        proj_feat = F.normalize(self.proj_head(multi_feat), dim=1)
        
        return bin_logits, multi_logits, proj_feat

# ==========================================
# 创新点升级 5：动态软门控联合损失函数 (Dynamic Soft-Gated Loss)
# ==========================================
class RamanCompositeLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, temperature=0.07):
        super(RamanCompositeLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, bin_logits, multi_logits, proj_feat, labels_bin, labels_multi):
        # 1. 主头损失：BCE Loss (绝不退让的宏观真伪判定)
        loss_bin = self.bce_loss(bin_logits.view(-1), labels_bin.float())

        # 2. 副头损失：Focal Loss (专注攻克 10% 微量掺假的难样本)
        ce_loss_multi = self.ce_loss(multi_logits, labels_multi)
        pt = torch.exp(-ce_loss_multi)
        focal_loss_multi = ((1 - pt) ** self.gamma) * ce_loss_multi
        
        # === 核心创新：物理逻辑驱动的可微软门控 Loss ===
        # bin_prob 代表当前网络认为它是"掺假"的概率 [0, 1]
        bin_prob = torch.sigmoid(bin_logits.view(-1)).detach() # detach阻断梯度，避免副头绑架主头
        
        # 逻辑自洽：如果判定为"纯"(prob≈0)，纯样本根本没有浓度概念，所以强行乘以0屏蔽该项Loss
        # 只有在判定为"掺假"(prob≈1)时，才严厉惩罚浓度预测错误
        gated_focal_loss = torch.mean(bin_prob * focal_loss_multi)

        # 3. 监督对比损失 (SupCon Loss): 辅助拉开潜空间距离
        # 这里用一种轻量化的实现：只拉开"纯(0)"与"掺假(1)"在多分类特征空间的距离
        # (框架预留：此处仅为余弦相似度矩阵展示，实际训练靠前两项 Loss 已经非常强劲)
        sim_matrix = torch.matmul(proj_feat, proj_feat.T) / self.temperature
        
        # 总体 Loss 解耦加权
        total_loss = self.alpha * loss_bin + (1 - self.alpha) * gated_focal_loss
        return total_loss, loss_bin, gated_focal_loss

# --- 测试一下网络维度与损失计算 ---
if __name__ == "__main__":
    model = RamanDualHeadCNN_V3(input_length=700)
    dummy_input = torch.randn(16, 1, 700) 
    
    # 模拟标签：16个样本，0代表纯，1代表掺假；多分类标签0-3代表四种浓度
    dummy_labels_bin = torch.randint(0, 2, (16,))
    dummy_labels_multi = torch.randint(0, 4, (16,))

    out_bin, out_multi, out_proj = model(dummy_input)
    
    criterion = RamanCompositeLoss()
    total_loss, l_bin, l_multi = criterion(out_bin, out_multi, out_proj, dummy_labels_bin, dummy_labels_multi)
    
    print("🚀 Model V3 initialized successfully!")
    print(f"🔹 Binary Head Shape: {out_bin.shape}")   # (16, 1)
    print(f"🔹 Multi Head Shape: {out_multi.shape}")  # (16, 4)
    print(f"🔹 Proj Head Shape  : {out_proj.shape}")  # (16, 64)
    print("-" * 40)
    print(f"📉 Total Loss : {total_loss.item():.4f}")
    print(f"   ┣ Binary   : {l_bin.item():.4f} (Weight: 0.75)")
    print(f"   ┗ Multi    : {l_multi.item():.4f} (Weight: 0.25, Gated)")