'''
饼图
'''
import numpy as np
import matplotlib.pyplot as mp

values = [26, 17, 21, 29, 11]
spaces = [0.05, 0.01, 0.01, 0.01, 0.01]
labels = ['Python', 'JavaScript',
          'C++', 'Java', 'PHP']
colors = ['dodgerblue', 'orangered',
          'limegreen', 'violet', 'gold']

mp.figure('Pie Chart', facecolor='lightgray')
mp.axis('equal')  # 设置等轴比例绘制饼状图
mp.pie(values, spaces, labels, colors,
       '%.1f%%', shadow=True)
mp.legend()
mp.show()

