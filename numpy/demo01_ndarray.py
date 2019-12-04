import numpy as np
ary = np.array([1,2,3,4,5,6])
print(ary,type(ary),ary[0])

# 数组中的每一个元素进行数值运算
print(ary + 2) # [3 4 5 6 7 8]
print(ary * 3) # [ 3  6  9 12 15 18]
print(ary > 3) # [False False False  True  True  True]
# 数组对应相加
print(ary + ary) # [ 2  4  6  8 10 12]

# 创建数组 arange()
ary = np.arange(10)
print(ary) # [0 1 2 3 4 5 6 7 8 9]
print(np.arange(1,5)) # [1 2 3 4]

# zeros()  ones()
ary = np.zeros(10,dtype='int32')
print(ary) # [0 0 0 0 0 0 0 0 0 0]
ary = np.ones((2,3),dtype='int32')
print(ary) # [1 1 1 1 1 1 1 1 1 1]

# zeros_like() ones_like()
# 生成同维度的数组
ary = np.array([[1,2,3],[4,5,6]])
ary = np.zeros_like(ary,dtype='int32')
print(ary,'<-zeros_like()')
ary = np.ones_like(ary,dtype='int32')
print(ary,'<-ones_like()')

