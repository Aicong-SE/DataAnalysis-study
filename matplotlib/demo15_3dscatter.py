# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_3dscatter.py  三维散点图
"""
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as mp

n = 300
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
z = np.random.normal(0, 1, n)

mp.figure('3D Scatter', facecolor='lightgray')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('x', fontsize=14)
ax3d.set_ylabel('y', fontsize=14)
ax3d.set_zlabel('z', fontsize=14)

ax3d.scatter(x, y, z, marker='o', s=100,
             color='dodgerblue', alpha=0.5)
mp.tight_layout()
mp.show()
