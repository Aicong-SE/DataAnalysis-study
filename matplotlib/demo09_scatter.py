'''
demo09_scatter.py 散点图
'''
import numpy as np
import matplotlib.pyplot as mp

n = 200
x = np.random.normal(175, 7, n)
y = np.random.normal(65, 10, n)

mp.figure('Points Chart', facecolor='lightgray')
mp.title('Points Chart', fontsize=18)
mp.xlabel('Height', fontsize=14)
mp.ylabel('weight', fontsize=14)
mp.grid(linestyle=':')
# 定义d数组，存储200个数字，定义每个样本点的颜色
d = np.sqrt((x - 175) ** 2 + (y - 65)**2)
mp.scatter(x, y, marker='o', s=80, alpha=0.8,
           c=d, cmap='jet_r', label='Points')
mp.legend()
mp.show()