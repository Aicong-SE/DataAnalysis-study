'''
demo03_figure.py 窗口操作
'''
import matplotlib.pyplot as mp

mp.figure('Figure A',facecolor='gray')
mp.plot([1,2],[3,4],color='orange',linewidth=3,label='Figure A')
# 设置图表标题　显示在图表上方
mp.title('Figure A',fontsize=20)
# 设置水平轴的文本
mp.xlabel('x',fontsize=16)
# 设置垂直轴的文本
mp.ylabel('y',fontsize=16)
# 设置刻度参数   labelsize设置刻度字体大小
mp.tick_params(labelsize=8)
# 设置图表网格线  linestyle设置网格线的样式
mp.grid(linestyle='--')
# 设置紧凑布局，把图表相关参数都显示在窗口中
mp.tight_layout()

mp.figure('Figure B',facecolor='gray')
mp.plot([1,2],[4,3],color=(0.3,0.5,0.6),linewidth=3,label='Figure B')
mp.title('Figure B',fontsize=20)

mp.show()



