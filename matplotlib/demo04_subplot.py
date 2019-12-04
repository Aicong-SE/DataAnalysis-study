'''
demo04_subplot.py 矩阵式布局
'''
import numpy as np
import matplotlib.pyplot as mp

mp.figure('Subplot',facecolor='gray')
# 绘制9宫格矩阵式子图,每个子图写一个数字
for i in range(1,10):
    mp.subplot(3,3,i)
    # 子图中写一个数字
    mp.text(0.5,0.5,i,ha='center',va='center',size=36,alpha=0.5,withdash=False)
    mp.xticks([])
    mp.yticks([])

mp.tight_layout()
mp.show()



