# 为什么会出现线性不可分？
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

mask = (y == 1) | (y == 2)
X_hard = X[mask, 2:4]  # 花瓣长度和花瓣宽度
y_hard = y[mask] - 1  # 调整为0和1，方便处理

# 深入分析两类样本的统计特性
print("\n=== 两类样本的统计分析 ===")

# 创建两个类别的子数据集
versicolor_data = X_hard[y_hard == 0]
virginica_data = X_hard[y_hard == 1]

print("Versicolor (类别0) 统计:")
print(f"  花瓣长度: 均值={versicolor_data[:, 0].mean():.2f}, 标准差={versicolor_data[:, 0].std():.2f}")
print(f"  花瓣宽度: 均值={versicolor_data[:, 1].mean():.2f}, 标准差={versicolor_data[:, 1].std():.2f}")

print("\nVirginica (类别1) 统计:")
print(f"  花瓣长度: 均值={virginica_data[:, 0].mean():.2f}, 标准差={virginica_data[:, 0].std():.2f}")
print(f"  花瓣宽度: 均值={virginica_data[:, 1].mean():.2f}, 标准差={virginica_data[:, 1].std():.2f}")

# 计算两个类别中心点的距离
center_versicolor = versicolor_data.mean(axis=0)
center_virginica = virginica_data.mean(axis=0)
distance = np.linalg.norm(center_versicolor - center_virginica)
print(f"\n两类中心点距离: {distance:.2f}")


# 计算重叠程度
def calculate_overlap(feature1, feature2, feature_name):
    """计算两个分布在指定特征上的重叠程度"""
    min1, max1 = feature1.min(), feature1.max()
    min2, max2 = feature2.min(), feature2.max()

    # 重叠区间
    overlap_start = max(min1, min2)
    overlap_end = min(max1, max2)

    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
        total_span = max(max1, max2) - min(min1, min2)
        overlap_ratio = overlap_length / total_span
        return overlap_ratio
    return 0


overlap_length = calculate_overlap(versicolor_data[:, 0], virginica_data[:, 0], "花瓣长度")
overlap_width = calculate_overlap(versicolor_data[:, 1], virginica_data[:, 1], "花瓣宽度")

print(f"花瓣长度重叠比例: {overlap_length:.2f}")
print(f"花瓣宽度重叠比例: {overlap_width:.2f}")

# 绘制直方图展示重叠
plt.subplot(1, 2, 1)
plt.hist(versicolor_data[:, 0], alpha=0.5, color='orange', label='versicolor', bins=15)
plt.hist(virginica_data[:, 0], alpha=0.5, color='green', label='virginica', bins=15)
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('频数')
plt.title('花瓣长度分布重叠')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(versicolor_data[:, 1], alpha=0.5, color='orange', label='versicolor', bins=15)
plt.hist(virginica_data[:, 1], alpha=0.5, color='green', label='virginica', bins=15)
plt.xlabel('花瓣宽度 (cm)')
plt.ylabel('频数')
plt.title('花瓣宽度分布重叠')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
