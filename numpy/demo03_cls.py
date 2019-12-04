import numpy as np

# 自定义复合类型
data = [('zs', [90, 80, 85], 15),
        ('ls', [92, 81, 83], 16),
        ('ww', [95, 85, 95], 15)]

#第一种设置dtype方式
ary = np.array(data,dtype='U2,3i8,i8')
print(ary)
print('==============')

# 第二种设置dtype
ary = np.array(data,dtype=[('name','U2',1),
                           ('scores','i8',3),
                           ('age','i8',1)])
print(ary['name'])
print('==============')


#第三种设置dtype方式
ary = np.array(data,dtype={
    'names':['name','scores','age'],
    'formats':['U2','3i8','i8']
})
print(ary['name'])
print('==============')

# 第四中设置dtype方式
ary = np.array(data,dtype={'names':('U2',0),
                         'scores':('3i8',16),
                         'ages':('i8',40)})
print(ary['names'])
print('==============')

# 第五种设置dtype方式
e = np.array([0x1234, 0x5667],
             dtype=('u2', {'lowc': ('u1', 0),
                            'hignc': ('u1', 1)}))
print('%x' % e[0])
print('%x %x' % (e['lowc'][0], e['hignc'][0]))
print('==============')

# 日期类型
dates = np.array(['2011','2011-02','2011-03-01'])
print(dates,dates.dtype)
dates = dates.astype('M8[D]')  # 转为时间类型
print(dates,dates.dtype)
print(dates[-1] - dates[0])
print('==============')



