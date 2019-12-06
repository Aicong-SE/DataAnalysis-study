# 将25个好球和1个坏球放在一起，每次模3个球，全为好球加1分，只要摸到了坏球减6分，求100轮的过程中分值的变化。

import numpy as np
import matplotlib.pyplot as mp

a = np.random.hypergeometric(25, 1, 3, 100)
s = a-(3-a)*6

mp.scatter(
    np.arange(100), 					# x轴坐标数组
    s,					# y轴坐标数组
    marker='o', 			# 点型
    s=10,				# 大小
    color='orange',			# 颜色
)

mp.show()







