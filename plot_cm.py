import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 绘制多分类混淆矩阵
cm = np.array([
    [7, 1, 0, 0, 0],
    [0, 5, 0, 0, 0],
    [0, 0, 4, 0, 0],
    [1, 0, 0, 11, 0],
    [0, 0, 0, 0, 7]
])
classes = ['Pure', '10%', '20%', '30%', '50%']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 16}) # 字体调大
plt.ylabel('True Concentration', fontsize=14)
plt.xlabel('Predicted Concentration', fontsize=14)
plt.title('Multi-class Confusion Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('confusion_matrix_multi.png', dpi=300) # 保存为高清图
print("图片已保存为 confusion_matrix_multi.png")