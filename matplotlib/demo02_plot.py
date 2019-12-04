'''
demo02_plot.py 绘图
'''
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-np.pi,np.pi,1000)
sinx = np.sin(x)
cosx = np.cos(x)/2

mp.plot(x,sinx,linestyle=':',linewidth=1,color='red',alpha=0.5,label=r'$sin(x)$')
mp.plot(x,cosx,label=r'$\frac{1}{2}cos(x)$')

# 设置坐标轴范围
# mp.xlim(0,np.pi)
# mp.ylim(0,1)

# 设置坐标轴刻度
vals = [-np.pi,-np.pi/2,0,np.pi/2,np.pi]
texts = [r'$-\pi$',r'$-\frac{\pi}{2}$','0',r'$\frac{\pi}{2}$',r'$\pi$']
mp.xticks(vals,texts)
mp.yticks([-1,-0.5,0,0.5,1])

# 设置坐标轴
ax = mp.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))

# 设置特殊点
px = [np.pi/2,np.pi/2]
py = [1,0]
mp.scatter(px,py,marker='o',s=40,edgecolors='red',facecolor='orange',zorder=3)


# 设置备注
mp.annotate(
    r'$\frac{\pi}{2}$',			#备注中显示的文本内容
    xycoords='data',			#备注目标点所使用的坐标系（data表示数据坐标系）
    xy=(np.pi/2, 1),	 				#备注目标点的坐标
    textcoords='offset points',	#备注文本所使用的坐标系（offset points表示参照点的偏移坐标系）
    xytext=(30, 30),				#备注文本的坐标
    fontsize=16,				#备注文本的字体大小
    arrowprops=dict(arrowstyle='fancy',connectionstyle='angle3')#使用字典定义文本指向目标点的箭头样式
)

mp.legend()
mp.show()


