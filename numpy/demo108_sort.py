import numpy as np

names = np.array(
    ['Mi', 'Huawei', 'Oppo', 'Vivo', 'Apple'])
prices = np.array([2999, 4999, 3999, 3999, 8888])
volumes = np.array([80, 110, 60, 70, 30])

# 排序　按价格升序排列，输出品牌列表
print(np.msort(prices))
print(names[np.argsort(prices)])

# 联合间接排序
print(names[np.lexsort((-volumes, prices))])

# 插入排序
a = np.array([1,3,5,7,8])
b = np.array([4,6])
indices = np.searchsorted(a, b)
a = np.insert(a,indices,b)
print(a)



