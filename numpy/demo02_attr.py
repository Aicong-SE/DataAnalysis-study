'''
demo02_attr.py 属性基本操作
'''

import numpy as np

ary = np.arange(1,9)
print(ary)
# shape: 维度
print(ary.shape) # (8,)
ary.shape = (2,4)
print(ary.shape)
# dtype: 元素数据类型
print(ary, ary.dtype)
# ary.dtype='float32'
# print(ary,ary.dtype)
ary = ary.astype('float32') # 建议使用
print(ary)

# size: 元素个数
print(ary, 'size:', ary.size, 'len():', len(ary))

# 索引访问
print(ary[0],'<-ary[0]',ary[0][1],'<-ary[0][1]')