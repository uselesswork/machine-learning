# 标签编码
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array(['audi', 'ford', 'audi','bmw','ford', 'bmw', 'abc'])

lb_encoder = sp.LabelEncoder() # 定义标签编码对象
lb_samples = lb_encoder.fit_transform(raw_samples) # 执行标签编码
print(lb_samples)

print(lb_encoder.inverse_transform(lb_samples)) # 逆向转换
