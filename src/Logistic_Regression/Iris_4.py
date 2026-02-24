# 非线性模型或特征工程
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

mask = (y == 1) | (y == 2)
X_hard = X[mask, 2:4]  # 花瓣长度和花瓣宽度
y_hard = y[mask] - 1  # 调整为0和1，方便处理

X_train, X_test, y_train, y_test = train_test_split(
    X_hard, y_hard, test_size=0.3, random_state=42
)

log_reg_hard = LogisticRegression()
log_reg_hard.fit(X_train, y_train)

xx, yy = np.meshgrid(
    np.linspace(X_hard[:, 0].min()-0.5, X_hard[:, 0].max()+0.5, 200),
    np.linspace(X_hard[:, 1].min()-0.5, X_hard[:, 1].max()+0.5, 200)
)

# 方案1：使用非线性模型（决策树）
from sklearn.tree import DecisionTreeClassifier

# 使用决策树处理线性不可分问题
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

# 绘制决策树的决策边界
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
Z_dt = dt_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z_dt = Z_dt.reshape(xx.shape)
plt.contourf(xx, yy, Z_dt, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_hard[y_hard == 0, 0], X_hard[y_hard == 0, 1],
           c='orange', label='versicolor', alpha=0.7, edgecolors='black')
plt.scatter(X_hard[y_hard == 1, 0], X_hard[y_hard == 1, 1],
           c='green', label='virginica', alpha=0.7, edgecolors='black')
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('花瓣宽度 (cm)')
plt.title(f'决策树决策边界 (准确率: {dt_classifier.score(X_test, y_test):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

# 方案2：特征工程 - 创建多项式特征
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 创建多项式特征 + 逻辑回归的管道
poly_log_reg = make_pipeline(
    PolynomialFeatures(degree=2),
    LogisticRegression()
)
poly_log_reg.fit(X_train, y_train)

# 绘制多项式特征的决策边界
plt.subplot(1, 2, 2)
Z_poly = poly_log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z_poly = Z_poly.reshape(xx.shape)
plt.contourf(xx, yy, Z_poly, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_hard[y_hard == 0, 0], X_hard[y_hard == 0, 1],
           c='orange', label='versicolor', alpha=0.7, edgecolors='black')
plt.scatter(X_hard[y_hard == 1, 0], X_hard[y_hard == 1, 1],
           c='green', label='virginica', alpha=0.7, edgecolors='black')
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('花瓣宽度 (cm)')
plt.title(f'多项式特征+逻辑回归 (准确率: {poly_log_reg.score(X_test, y_test):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 对比所有方法
print("\n=== 不同方法在Versicolor vs Virginica上的表现 ===")
print(f"1. 简单逻辑回归: {log_reg_hard.score(X_test, y_test):.3f}")
print(f"2. 决策树(max_depth=3): {dt_classifier.score(X_test, y_test):.3f}")
print(f"3. 多项式特征(2次)+逻辑回归: {poly_log_reg.score(X_test, y_test):.3f}")
