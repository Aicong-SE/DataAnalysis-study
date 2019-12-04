'''
demo04_shape.py 维度操作
'''
import numpy as np

ary = np.arange(1,9)
print(ary) # [1 2 3 4 5 6 7 8]
# 视图变维（数据共享） reshape() ravel()
print('======视图变维======')
ary2 = ary.reshape(2,4) # 维度变成(2,4)
print(ary2) # [[1 2 3 4] [5 6 7 8]]　
ary3 = ary2.ravel() # 变成１维数组
print(ary3) # [1 2 3 4 5 6 7 8]

# 复制变维 (数据肚子) flatten() copy()
print('======复制变维=====')
ary4 = ary3.flatten()
print(ary4)

# 就地变维　直接改变原数组对象的维度，不返回新数组
print('=====就地变维=====')
ary.shape = (2,4)
print(ary,'<-shape')
ary.resize(2,2,2)
print(ary,'<-resize')




