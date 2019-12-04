import numpy as np
import matplotlib.pyplot as mp

xarray = np.array([1,2,3,4,5,6,7,8])
yarray = np.array([12,14,35,35,46,34,6,4])
# 绘制折线图
mp.plot(xarray,yarray)
# 绘制水平线
mp.hlines(30,2,7)
# 绘制垂直线
mp.vlines([2,3,7],[5,10,15],20)
# 显示
mp.show()
