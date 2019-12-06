# 插值器
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as mp

min_val = -50
max_val = 50

x = np.linspace(min_val,max_val,15)
y = np.sinc(x)

mp.grid(linestyle=':')
mp.scatter(x, y, color='blue',s=80,label='Samples')

# 通过一系列的散点设计出符合一定规律插值器函数，使用线性插值（kind缺省值）
func = si.interp1d(x,y)
lin_x = np.linspace(min_val,max_val,1000)
lin_y = func(lin_x)
mp.plot(lin_x,lin_y)

# 三次样条插值 （CUbic Spline Interpolation） 获得一条光滑曲线
cubic = si.interp1d(x, y, kind='cubic')
cub_x = np.linspace(min_val, max_val, 200)
cub_y = cubic(cub_x)
mp.plot(cub_x,cub_y)

mp.legend()
mp.show()


