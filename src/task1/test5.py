# 独热编码
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([[1, 3, 2],
                        [7, 5, 4],
                        [1, 8, 6],
                        [7, 3, 9]])

one_hot_encoder = sp.OneHotEncoder(
    sparse_output=False, # 是否采用稀疏格式
    dtype="int32",
    categories="auto")# 自动编码
oh_samples = one_hot_encoder.fit_transform(raw_samples) # 执行独热编码
print(oh_samples)

print(one_hot_encoder.inverse_transform(oh_samples)) # 解码
