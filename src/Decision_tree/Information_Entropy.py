# 信息熵计算
import math
import numpy as np
import matplotlib.pyplot as mp

class_num = 10  # 类别最大数量


def entropy_calc(n):
    p = 1.0 / n  # 计算每个类别的概率
    entropy_value = 0.0  # 信息熵

    for i in range(n):
        p_i = p * math.log(p)
        entropy_value += p_i

    return -entropy_value  # 返回熵值


entropies = []
for i in range(1, class_num + 1):
    entropy = entropy_calc(i)  # 计算类别为i的熵值
    entropies.append(entropy)

print(entropies)

# 可视化回归曲线
mp.figure('Entropy', facecolor='lightgray')
mp.title('Entropy', fontsize=20)
mp.xlabel('Class Num', fontsize=14)
mp.ylabel('Entropy', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle='-')
x = np.arange(1, 11, 1)
mp.plot(x, entropies, c='orangered', label='entropy')

mp.legend()
mp.show()