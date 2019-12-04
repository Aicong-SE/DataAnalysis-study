import numpy as np
import matplotlib.pyplot as mp
import matplotlib.animation as ma

n = 100
balls = np.zeros(n, dtype=[
    ('position', 'f8', 2),
    ('size', 'f8', 1),
    ('growth', 'f8', 1),
    ('color', 'f8', 4)])

# 初始化balls的字段值
# print(balls['position'].shape)
balls['position'] = np.random.uniform(0, 1, (n, 2))
balls['size'] = np.random.uniform(40, 80, n)
balls['growth'] = np.random.uniform(10, 30, n)
balls['color'] = np.random.uniform(0, 1, (n, 4))

# 画图
mp.figure('Points', facecolor='lightgray')
mp.xticks([])
mp.yticks([])
sc = mp.scatter(balls['position'][:, 0],
                balls['position'][:, 1],
                s=balls['size'],
                color=balls['color'])


def update(number):
    # 让每个球size自增，重新绘制界面
    balls['size'] += balls['growth']
    index = number % n
    balls['size'][index] = 60
    balls['position'][index] = \
        np.random.uniform(0, 1, (1, 2))

    sc.set_sizes(balls['size'])
    sc.set_offsets(balls['position'])

    # 执行动画
a = ma.FuncAnimation(mp.gcf(), update, interval=33)


mp.show()
