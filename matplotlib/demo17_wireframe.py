# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_wireframe.py 
"""
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n = 500
x, y = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))
# 通过x与y 得到每个坐标点的高度
z = (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)
# 画图
mp.figure('Imshow', facecolor='lightgray')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')
ax3d.plot_wireframe(
    x, y, z, rstride=30, cstride=30,
    linewidth=0.5, color='dodgerblue')
mp.tight_layout()
mp.show()
