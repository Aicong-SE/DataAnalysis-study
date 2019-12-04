'''
demo05_gridspec.py 网格布局
'''
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.gridspec as mg
mp.figure('Grid Layout', facecolor='lightgray')
# 调用GridSpec方法拆分网格式布局
gs = mg.GridSpec(3, 3)
# 合并0行0-1列为一个子图
mp.subplot(gs[0, :2])
mp.text(0.5, 0.5, 1, ha='center', va='center', size=36)
mp.xticks([])
mp.yticks([])
mp.show()


