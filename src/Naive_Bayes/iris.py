# 朴素贝叶斯分类示例
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer, Binarizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 尝试三种朴素贝叶斯分类器
classifiers = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(binarize=0.5)  # 需要二值化处理
}

# 对于MultinomialNB和BernoulliNB，需要对连续特征进行预处理
# 将连续特征转换为计数特征（通过分箱）或二值特征

# 创建离散化转换器（用于MultinomialNB）
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_train_discrete = discretizer.fit_transform(X_train)
X_test_discrete = discretizer.transform(X_test)

# 创建二值化转换器（用于BernoulliNB）
binarizer = Binarizer(threshold=0.5)
X_train_binary = binarizer.fit_transform(X_train)
X_test_binary = binarizer.transform(X_test)

# 存储结果
results = []
models = {}  # 存储训练好的模型

# 训练和评估每个分类器
for name, clf in classifiers.items():
    if name == 'GaussianNB':
        # 高斯朴素贝叶斯直接使用原始数据
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        data_type = "原始连续数据"
        models[name] = {'model': clf, 'type': 'original', 'X_test': X_test, 'y_pred': y_pred}
    elif name == 'MultinomialNB':
        # 多项式朴素贝叶斯使用离散化数据
        clf.fit(X_train_discrete, y_train)
        y_pred = clf.predict(X_test_discrete)
        accuracy = accuracy_score(y_test, y_pred)
        data_type = "离散化数据（5个分箱）"
        models[name] = {'model': clf, 'type': 'discrete', 'X_test': X_test_discrete, 'y_pred': y_pred}
    else:  # BernoulliNB
        # 伯努利朴素贝叶斯使用二值化数据
        clf.fit(X_train_binary, y_train)
        y_pred = clf.predict(X_test_binary)
        accuracy = accuracy_score(y_test, y_pred)
        data_type = "二值化数据"
        models[name] = {'model': clf, 'type': 'binary', 'X_test': X_test_binary, 'y_pred': y_pred}

    results.append({
        '分类器': name,
        '准确率': f'{accuracy:.4f}',
        '使用数据': data_type
    })

# 显示结果
results_df = pd.DataFrame(results)
print("\n三种朴素贝叶斯在鸢尾花数据集上的表现：")
print(results_df.to_string(index=False))

print("\n结论：对于鸢尾花这种连续特征的数据集，")
print("GaussianNB表现最好，因为它直接利用了特征的连续性。")
print("MultinomialNB和BernoulliNB需要特征转换，可能损失信息。")

# ============== 可视化部分 ==============
# 1. 创建一个大图
fig = plt.figure(figsize=(20, 12))
fig.suptitle('朴素贝叶斯分类器性能可视化', fontsize=16, fontweight='bold')

# 设置颜色
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
class_colors = ['#FF9999', '#66B2FF', '#99FF99']

# 2. 准确率对比条形图
ax1 = plt.subplot(2, 4, 1)
accuracies = [float(r['准确率']) for r in results]
classifier_names = [r['分类器'] for r in results]

bars = ax1.bar(classifier_names, accuracies, color=colors, edgecolor='black')
ax1.set_ylabel('准确率', fontsize=12)
ax1.set_title('不同分类器准确率对比', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3)

# 在每个条形上添加准确率数值
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=11)

# 3. 混淆矩阵可视化
for i, (name, model_info) in enumerate(models.items()):
    ax = plt.subplot(2, 4, i + 2)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, model_info['y_pred'])

    # 显示混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{name}混淆矩阵', fontsize=13, fontweight='bold')

    # 在每个单元格中添加数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)

# 4. 决策边界可视化（使用前两个特征）
# 由于原始数据是4维的，我们只取前两个特征进行可视化
X_2d = X[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, random_state=42
)

# 为每个分类器训练2D版本用于可视化决策边界
classifiers_2d = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(binarize=0.5)
}

# 准备决策边界数据
h = 0.02  # 网格步长
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 创建子图
axes = [plt.subplot(2, 4, 5), plt.subplot(2, 4, 6), plt.subplot(2, 4, 7)]

for idx, (name, clf) in enumerate(classifiers_2d.items()):
    ax = axes[idx]

    # 准备训练数据
    if name == 'GaussianNB':
        X_train_vis = X_train_2d
        clf.fit(X_train_vis, y_train_2d)
    elif name == 'MultinomialNB':
        # 离散化
        discretizer_2d = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        X_train_vis = discretizer_2d.fit_transform(X_train_2d)
        clf.fit(X_train_vis, y_train_2d)
    else:  # BernoulliNB
        # 二值化
        binarizer_2d = Binarizer(threshold=0.5)
        X_train_vis = binarizer_2d.fit_transform(X_train_2d)
        clf.fit(X_train_vis, y_train_2d)

    # 预测整个网格
    if name == 'GaussianNB':
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    elif name == 'MultinomialNB':
        # 网格数据也需要离散化
        Z = clf.predict(discretizer_2d.transform(np.c_[xx.ravel(), yy.ravel()]))
    else:  # BernoulliNB
        Z = clf.predict(binarizer_2d.transform(np.c_[xx.ravel(), yy.ravel()]))

    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    # 绘制训练点
    for i, color in zip(range(3), class_colors):
        idx = np.where(y_train_2d == i)
        ax.scatter(X_train_2d[idx, 0], X_train_2d[idx, 1],
                   c=color, edgecolor='black', s=50,
                   label=f'{target_names[i]} (训练)', alpha=0.7)

    # 绘制测试点
    for i, color in zip(range(3), class_colors):
        idx = np.where(y_test_2d == i)
        ax.scatter(X_test_2d[idx, 0], X_test_2d[idx, 1],
                   c=color, edgecolor='black', s=100, marker='^',
                   label=f'{target_names[i]} (测试)', alpha=1.0)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f'{name}决策边界', fontsize=13, fontweight='bold')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.legend(loc='upper right', fontsize=9)

# 5. 添加总结信息
ax_summary = plt.subplot(2, 4, 8)
ax_summary.axis('off')

summary_text = f"""
数据集信息:
- 样本数: {len(X)}
- 特征数: {len(feature_names)}
- 类别数: {len(target_names)}
- 训练集大小: {len(X_train)}
- 测试集大小: {len(X_test)}

最佳模型: {classifier_names[np.argmax(accuracies)]}
最高准确率: {max(accuracies):.4f}

特征名称:
{chr(10).join([f'{i + 1}. {name}' for i, name in enumerate(feature_names)])}

目标类别:
{chr(10).join([f'{i}. {name}' for i, name in enumerate(target_names)])}
"""

ax_summary.text(0.1, 0.5, summary_text, fontsize=11,
                verticalalignment='center', linespacing=1.5)
ax_summary.set_title('实验总结', fontsize=13, fontweight='bold')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 显示图表
plt.show()
