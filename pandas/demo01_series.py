import pandas as pd
import numpy as np

# 创建Series对象
s = pd.Series()
print(s, type(s))

data = np.array(['zs',123,'c'])
s = pd.Series(data,index=['name','a','b'])
print('-'*10)
print(s)
print('-'*10)
print(s[1],'<-s[1]')
print('-'*10)
print(s['name'],'<-s["name"]')

# 通过字典创建Series
data = {'001':'zs','002':'ls'}
s = pd.Series(data)
print('-'*10)
print(s)

# 通过标量创建Series
s = pd.Series(5, index=[0,1,2])
print('-'*10)
print(s)

# 处理pandas的日期
dates = pd.Series(['2011', '2011-02', '2011-03-01', '2011/04/01', '2011/05/01 01:01:01', '01 Jun 2011'])
dates = pd.to_datetime(dates)
print('-'*10)
print(dates)

# 日期计算
delta = dates - pd.to_datetime('1970-1-1')
print(delta)
print(delta.dt.days)

# DateTimeIndex
dates = pd.date_range('2019/12/01', periods=5)
print('-'*10)
print(dates)
datelist = pd.bdate_range('2011/11/03', periods=5)
print('-'*10)
print(datelist)

