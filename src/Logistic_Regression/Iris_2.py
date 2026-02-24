# 案例2：选择两个线性不可分的类别
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 同样是versicolor(1)和virginica(2)，使用相同的两个特征

# 选取versicolor(1)和virginica(2)两类
mask = (y == 1) | (y == 2)
X_hard = X[mask, 2:4]  # 花瓣长度和花瓣宽度
y_hard = y[mask] - 1  # 调整为0和1，方便处理

# 可视化这两个类别的分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_hard[y_hard == 0, 0], X_hard[y_hard == 0, 1],
           c='orange', label='versicolor', alpha=0.7, edgecolors='black')
plt.scatter(X_hard[y_hard == 1, 0], X_hard[y_hard == 1, 1],
           c='green', label='virginica', alpha=0.7, edgecolors='black')
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('花瓣宽度 (cm)')
plt.title('Versicolor vs Virginica: 线性不可分的典型')
plt.legend()
plt.grid(True, alpha=0.3)

# 尝试线性分类器（逻辑回归）
X_train, X_test, y_train, y_test = train_test_split(
    X_hard, y_hard, test_size=0.3, random_state=42
)

log_reg_hard = LogisticRegression()
log_reg_hard.fit(X_train, y_train)

# 绘制决策边界
xx, yy = np.meshgrid(
    np.linspace(X_hard[:, 0].min()-0.5, X_hard[:, 0].max()+0.5, 200),
    np.linspace(X_hard[:, 1].min()-0.5, X_hard[:, 1].max()+0.5, 200)
)

Z = log_reg_hard.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_hard[y_hard == 0, 0], X_hard[y_hard == 0, 1],
           c='orange', label='versicolor', alpha=0.7, edgecolors='black')
plt.scatter(X_hard[y_hard == 1, 0], X_hard[y_hard == 1, 1],
           c='green', label='virginica', alpha=0.7, edgecolors='black')
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('花瓣宽度 (cm)')
plt.title(f'逻辑回归决策边界 (准确率: {log_reg_hard.score(X_test, y_test):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

print(f"逻辑回归在Versicolor vs Virginica上的准确率: {log_reg_hard.score(X_test, y_test):.3f}")
print("观察: 准确率显著下降，说明两类样本存在重叠区域!")
