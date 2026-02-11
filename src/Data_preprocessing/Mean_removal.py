# 均值移除
import numpy as np
import sklearn.preprocessing as sp

# 样本数据
raw_samples = np.array([
    [1.0, 2.0, 3.0],
    [5.0, 4.0, 3.0],
    [6.0, 8.0, 2.0]
])
print(raw_samples)

# 手动标准移除
std_samples = raw_samples.copy()  # 复制样本数据
for col in std_samples.T:  # 遍历每列
    col_mean = col.mean()  # 计算平均数
    col_std = col.std()  # 求标准差
    col -= col_mean  # 减平均值
    col /= col_std  # 除标准差
print(std_samples)

# 使用sklearn
std_samples = sp.scale(raw_samples) # 求标准移除
print(std_samples)