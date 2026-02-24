# 案例1：选择两个线性可分的类别
import numpy as np
import pandas as pd
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

# 创建一个DataFrame以便分析
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("鸢尾花数据集前5行：")
print(df.head())
print("\n各类别样本数量：")
print(df['species'].value_counts())

# 我们只使用两个特征：花瓣长度和花瓣宽度，以及两个类别：setosa(0)和versicolor(1)

# 选取setosa(0)和versicolor(1)两类，以及两个特征
mask = (y == 0) | (y == 1)
X_binary = X[mask, 2:4]  # 只使用花瓣长度和花瓣宽度
y_binary = y[mask]

# 可视化这两个类别的分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_binary[y_binary == 0, 0], X_binary[y_binary == 0, 1],
           c='blue', label='setosa', alpha=0.7, edgecolors='black')
plt.scatter(X_binary[y_binary == 1, 0], X_binary[y_binary == 1, 1],
           c='red', label='versicolor', alpha=0.7, edgecolors='black')
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('花瓣宽度 (cm)')
plt.title('Setosa vs Versicolor: 明显线性可分')
plt.legend()
plt.grid(True, alpha=0.3)

# 使用逻辑回归（线性分类器）
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 绘制决策边界
xx, yy = np.meshgrid(
    np.linspace(X_binary[:, 0].min()-0.5, X_binary[:, 0].max()+0.5, 200),
    np.linspace(X_binary[:, 1].min()-0.5, X_binary[:, 1].max()+0.5, 200)
)

Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_binary[y_binary == 0, 0], X_binary[y_binary == 0, 1],
           c='blue', label='setosa', alpha=0.7, edgecolors='black')
plt.scatter(X_binary[y_binary == 1, 0], X_binary[y_binary == 1, 1],
           c='red', label='versicolor', alpha=0.7, edgecolors='black')
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('花瓣宽度 (cm)')
plt.title(f'逻辑回归决策边界 (准确率: {log_reg.score(X_test, y_test):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"逻辑回归在Setosa vs Versicolor上的准确率: {log_reg.score(X_test, y_test):.3f}")
print("结论: Setosa和Versicolor两类是完全线性可分的!")
