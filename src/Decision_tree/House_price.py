import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# 1. 准备数据
# 房屋面积（平方米）作为特征X
X = np.array([[50], [80], [100], [120], [150]])
# 房价（万元）作为目标y
y = np.array([100, 180, 220, 250, 300])


# 2. 创建并训练决策树回归模型
tree_model = DecisionTreeRegressor(
    criterion='squared_error',  # 使用平方误差作为分裂标准，结果为我们之前手动求的SSE值/样本数
    max_depth=3,  # 限制最大深度为3
    min_samples_split=2,  # 节点最少样本数为2才能分裂
    min_samples_leaf=1,  # 叶节点最少样本数为1
    random_state=42  # 设置随机种子以确保结果可复现
)

# 训练模型
tree_model.fit(X, y)

# 3. 可视化决策树
plt.figure(figsize=(12, 8))

# 4.绘制决策树
plot_tree(tree_model,
          feature_names=["area"],
          filled=True,  # 填充颜色表示类别/值
          rounded=True,  # 圆角节点
          precision=1,  # 显示1位小数
          fontsize=10)

plt.title("CART", fontsize=14, pad=20)
plt.show()
