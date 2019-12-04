"""
demo05_imshow.py 热成像图
"""
import numpy as np
import matplotlib.pyplot as mp

n = 500
x, y = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))
# 通过x与y 得到每个坐标点的高度
z = (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)
# 画图
mp.figure('Imshow', facecolor='lightgray')
mp.title('Imshow', fontsize=18)
mp.imshow(z, cmap='jet', origin='lower')
mp.colorbar()
mp.show()
