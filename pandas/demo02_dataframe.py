import numpy as np
import pandas as pd

f = pd.DataFrame()
print(f)

# 通过列表创建DataFrame
data = ['a','b','c','d']
f = pd.DataFrame(data)
print('-'*10)
print(f)
data = [['zs', 10],['ls',12]]
f = pd.DataFrame(data, columns=['name','age'],index=[1,2], dtype='float')
print('-'*10)
print(f)
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print('-'*10)
print(df)
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['s1','s2','s3','s4'])
print('-'*10)
print(df)