# 单科决策树、AdaBoosting与随机森林进行预测

import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se
import matplotlib.pyplot as mp
import numpy as np

# 使用糖尿病数据集
diabetes = sd.load_diabetes()  # 加载糖尿病数据集
print("特征名称:", diabetes.feature_names)

random_seed = 7  # 随机种子，计算随机值，相同的随机种子得到的随机值一样
x, y = su.shuffle(diabetes.data,
                  diabetes.target,
                  random_state = random_seed)
# 计算训练数据的数量
train_size = int(len(x) * 0.8) # 以data中80%的数据作为训练数据
# 构建训练数据、测试数据
train_x = x[:train_size]  # 训练输入, x前面80%的数据
test_x = x[train_size:]   # 测试输入, x后面20%的数据
train_y = y[:train_size]  # 训练输出
test_y = y[train_size:]   # 测试输出

######## 单棵树进行预测 ########
# 模型
model = st.DecisionTreeRegressor(max_depth=4)  # 决策回归器

# 训练
model.fit(train_x, train_y)
# 预测
pre_test_y = model.predict(test_x)
# 打印预测输出和实际输出的R2值
print("单棵树R2得分:", sm.r2_score(test_y, pre_test_y))

######## AdaBoosting进行预测 ########
# 模型
model2 = se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth=4),
                              n_estimators=400,   # 决策树数量
                              random_state=random_seed) # 随机种子

# 训练
model2.fit(train_x, train_y)
# 预测
pre_test_y2 = model2.predict(test_x)
# 打印预测输出和实际输出的R2值
print("AdaBoosting方法R2得分:", sm.r2_score(test_y, pre_test_y2))

######## 随机森林进行预测 ########
# 模型
model3 = se.RandomForestRegressor(max_depth=10,  # 最大深度
                                 n_estimators=1000,  # 树数量
                                 min_samples_split=2)  # 最小样本数量，小于该数就不再划分子节点

# 训练
model3.fit(train_x, train_y)
# 预测
pre_test_y3 = model3.predict(test_x)
# 打印预测输出和实际输出的R2值
print("随机森林R2得分:", sm.r2_score(test_y, pre_test_y3))


# 决策树可视化
mp.figure("Decision Tree", figsize=(30, 15), facecolor="white")
mp.title("Decision Tree Structure", fontsize=20)

# 绘制决策树
st.plot_tree(model,
             feature_names=diabetes.feature_names,
             filled=True,           # 填充颜色
             rounded=True,          # 圆角矩形
             fontsize=8,            # 减小字体大小
             impurity=False,        # 不显示不纯度
             precision=2,           # 数值精度
             proportion=True,       # 节点大小按样本比例显示
             node_ids=True,         # 显示节点ID
            )

mp.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)


# 获取特征重要性
fi = model.feature_importances_  # 获取特征重要性
print("特征重要性:", fi)

# 特征重要性可视化
mp.figure("Feature importances", facecolor="lightgray")
mp.plot()
mp.title("DT Feature", fontsize=16)
mp.ylabel("Feature importances", fontsize=14)
mp.grid(linestyle=":", axis="y")
x = np.arange(fi.size)
sorted_idx = fi.argsort()[::-1]  # 重要性排序(倒序)
fi = fi[sorted_idx]  # 根据排序索引重新排特征值
feature_names = np.array(diabetes.feature_names)
mp.xticks(x, feature_names[sorted_idx])
mp.bar(x, fi, 0.4, color="dodgerblue", label="DT Feature importances")

mp.legend()
mp.tight_layout()
mp.show()
