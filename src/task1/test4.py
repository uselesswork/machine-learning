# 二值化
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([[65.5, 89.0, 73.0],
                        [55.0, 99.0, 98.5],
                        [45.0, 22.5, 60.0]])
bin_samples = raw_samples.copy()  # 复制数组
# 生成掩码数组
mask1 = (bin_samples < 60)
mask2 = (bin_samples >= 60)
# 通过掩码进行二值化处理
bin_samples[mask1] = 0
bin_samples[mask2] = 1

print(bin_samples)  # 打印结果

bin = sp.Binarizer(threshold=59) # 创建二值化对象(注意边界值)
bin_samples = bin.transform(raw_samples) # 二值化预处理
print(bin_samples)
