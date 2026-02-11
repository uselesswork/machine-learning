# 范围缩放
import numpy as np
import sklearn.preprocessing as sp

# 样本数据
raw_samples = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]]).astype("float64")

# 手动数据缩放
mms_samples = raw_samples.copy()  # 复制样本数据

for col in mms_samples.T:
    col_min = col.min()
    col_max = col.max()
    col -= col_min
    col /= (col_max - col_min)
print(mms_samples)

# 使用sklearn数据缩放
mms = sp.MinMaxScaler(feature_range=(0, 1)) # 创建数据缩放对象
mms_samples = mms.fit_transform(raw_samples) # 数据缩放
print(mms_samples)
