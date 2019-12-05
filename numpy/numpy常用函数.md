[toc]

# numpy常用函数

### 加载文件

numpy提供了函数用于加载逻辑上可被解释为二维数组的文本文件，格式如下：

```
数据项1 <分隔符> 数据项2 <分隔符> ... <分隔符> 数据项n
例如：
AA,AA,AA,AA,AA
BB,BB,BB,BB,BB
...
或：
AA:AA:AA:AA:AA
BB:BB:BB:BB:BB
...


```

调用numpy.loadtxt()函数可以直接读取该文件并且获取ndarray数组对象：

```python
import numpy as np
# 直接读取该文件并且获取ndarray数组对象 
# 返回值：
#     unpack=False：返回一个二维数组
#     unpack=True： 多个一维数组
np.loadtxt(
    '../aapl.csv',			# 文件路径
    delimiter=',',			# 分隔符
    usecols=(1, 3),			# 读取1、3两列 （下标从0开始）
    unpack=False,			# 是否按列拆包
    dtype='U10, f8',		# 制定返回每一列数组中元素的类型
    converters={1:func}		# 转换器函数字典
)    

```

案例：读取aapl.csv文件，得到文件中的信息：

```python
import numpy as np
import datetime as dt
# 日期转换函数
def dmy2ymd(dmy):
	dmy = str(dmy, encoding='utf-8')
	time = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
	t = time.strftime('%Y-%m-%d')
	return t
dates, opening_prices,highest_prices, \
	lowest_prices, closeing_pric es  = np.loadtxt(
    '../data/aapl.csv',		# 文件路径
    delimiter=',',			# 分隔符
    usecols=(1, 3, 4, 5, 6),			# 读取1、3两列 （下标从0开始）
    unpack=True,
    dtype='M8[D], f8, f8, f8, f8',		# 制定返回每一列数组中元素的类型
    converters={1:dmy2ymd})

```

案例：使用matplotlib绘制K线图

1. 绘制dates与收盘价的折线图：

```python
import numpy as np
import datetime as dt
import matplotlib.pyplot as mp
import matplotlib.dates as md

# 绘制k线图，x为日期
mp.figure('APPL K', facecolor='lightgray')
mp.title('APPL K')
mp.xlabel('Day', fontsize=12)
mp.ylabel('Price', fontsize=12)

#拿到坐标轴
ax = mp.gca()
#设置主刻度定位器为周定位器（每周一显示主刻度文本）
ax.xaxis.set_major_locator( md.WeekdayLocator(byweekday=md.MO) )
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
#设置次刻度定位器为日定位器 
ax.xaxis.set_minor_locator(md.DayLocator())
mp.tick_params(labelsize=8)
dates = dates.astype(md.datetime.datetime)

mp.plot(dates, opening_prices, color='dodgerblue',
		linestyle='-')
mp.gcf().autofmt_xdate()
mp.show()


```

1. 绘制每一天的蜡烛图：

```python
#绘制每一天的蜡烛图
#填充色：涨为白色，跌为绿色
rise = closeing_prices >= opening_prices
color = np.array([('white' if x else 'limegreen') for x in rise])
#边框色：涨为红色，跌为绿色
edgecolor = np.array([('red' if x else 'limegreen') for x in rise])

#绘制线条
mp.bar(dates, highest_prices - lowest_prices, 0.1,
	lowest_prices, color=edgecolor)
#绘制方块
mp.bar(dates, closeing_prices - opening_prices, 0.8,
	opening_prices, color=color, edgecolor=edgecolor)

```

### 保存文件

将运算结果（array类型）保存到本地

```python
np.savetxt('/homt/spcluster.txt', # 保存地址
           spcluster,             # 数组
           fmt="%.18f,%.18f",     # 保存格式
           delimiter="\n") 		  # 分割符
```



### 算数平均值

```
S = [s1, s2, ..., sn]
```

样本中的每个值都是真值与误差的和。

```
算数平均值：
m = (s1 + s2 + ... + sn) / n
```

算数平均值表示对真值的无偏估计。

```python
m = np.mean(array)
m = array.mean()
```

案例：计算收盘价的算术平均值。

```python
import numpy as np
closing_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(6), unpack=True)
mean = 0
for closing_price in closing_prices:
    mean += closing_price
mean /= closing_prices.size
print(mean)
mean = np.mean(closing_prices)
print(mean)

```

### 加权平均值

样本：$S = [s_1, s_2, s_3 ... s_n]$

权重：$W =[w_1, w_2, w_3 ... w_n]$

加权平均值：$a = \frac{s_1w_1 + s_2w_2 + ... + s_nw_n}{w_1+w_2+...+w_n}$

```python
a = np.average(closing_prices, weights=volumes)

```



VWAP - 成交量加权平均价格（成交量体现了市场对当前交易价格的认可度，成交量加权平均价格将会更接近这支股票的真实价值）

```python
import numpy as np
closing_prices, volumes = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(6, 7), unpack=True)
vwap, wsum = 0, 0
for closing_price, volume in zip(
        closing_prices, volumes):
    vwap += closing_price * volume
    wsum += volume
vwap /= wsum
print(vwap)
vwap = np.average(closing_prices, weights=volumes)
print(vwap)

```

TWAP - 时间加权平均价格（时间越晚权重越高，参考意义越大）

```python
import datetime as dt
import numpy as np

def dmy2days(dmy):
    dmy = str(dmy, encoding='utf-8')
    date = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
    days = (date - dt.date.min).days
    return days

days, closing_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(1, 6), unpack=True,
    converters={1: dmy2days})
twap = np.average(closing_prices, weights=days)
print(twap)

```

### 最值

**np.max()  np.min() np.ptp()：** 返回一个数组中最大值/最小值/极差

```python
import numpy as np
# 产生9个介于[10, 100)区间的随机数
a = np.random.randint(10, 100, 9)
print(a)
print(np.max(a), np.min(a), np.ptp(a))
```

**np.argmax() np.argmin()：** 返回一个数组中最大/最小元素的下标

```python
print(np.argmax(a), np.argmin(a))
```

**np.maximum() np.minimum()：** 将两个同维数组中对应元素中最大/最小元素构成一个新的数组

```python
print(np.maximum(a, b), np.minimum(a, b), sep='\n')
```

案例：评估AAPL股票的波动性。

```python
import numpy as np
highest_prices, lowest_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(4, 5), dtype='f8, f8', unpack=True)
max_price = np.max(highest_prices)
min_price = np.min(lowest_prices)
print(min_price, '~', max_price)
```

查看AAPL股票最大最小值的日期，分析为什么这一天出现最大最小值。

```python
import numpy as np
dates, highest_prices, lowest_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(1, 4, 5), dtype='U10, f8, f8',
    unpack=True)
max_index = np.argmax(highest_prices)
min_index = np.argmin(lowest_prices)
print(dates[min_index], dates[max_index])
```

观察最高价与最低价的**波动范围**，分析这支股票底部是否坚挺。  

```python
import numpy as np
dates, highest_prices, lowest_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(1, 4, 5), dtype='U10, f8, f8',
    unpack=True)
highest_ptp = np.ptp(highest_prices)
lowest_ptp = np.ptp(lowest_prices)
print(lowest_ptp, highest_ptp)
```

### 中位数

将多个样本按照大小排序，取中间位置的元素。

**若样本数量为奇数，中位数为最中间的元素**

$[1, 2000, 3000, 4000, 10000000]$

**若样本数量为偶数，中位数为最中间的两个元素的平均值**

$[1,2000,3000,4000,5000,10000000]$

案例：分析中位数的算法，测试numpy提供的中位数API：

```python
import numpy as np
closing_prices = np.loadtxt( '../../data/aapl.csv', 
	delimiter=',', usecols=(6), unpack=True)
size = closing_prices.size
sorted_prices = np.msort(closing_prices) # 排序
median = (sorted_prices[int((size - 1) / 2)] + sorted_prices[int(size / 2)]) / 2 
print(median)
# 中位数函数
median = np.median(closing_prices)
print(median)
```

### 标准差

样本：$S = [s_1, s_2, s_3, ..., s_n]$

平均值：$m = \frac{s_1 + s_2 + s_3 + ... + s_n}{n}$

离差：$D = [d_1, d_2, d_3, ..., d_n]; d_i = S_i-m$

离差方：$Q = [q_1, q_2, q_3, ..., q_n]; q_i=d_i^2$

总体方差：$v = \frac{(q_1+q_2+q_3 + ... + q_n)}{n}$

总体标准差：$s = \sqrt{v}$

样本方差：$v' = \frac{(q_1+q_2+q_3 + ... + q_n)}{n-1}$

样本标准差：$s' = \sqrt{v'}$

```python
import numpy as np
closing_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',', usecols=(6), unpack=True)
mean = np.mean(closing_prices)         # 算数平均值
devs = closing_prices - mean           # 离差
dsqs = devs ** 2                       # 离差方
pvar = np.sum(dsqs) / dsqs.size        # 总体方差
pstd = np.sqrt(pvar)                   # 总体标准差
svar = np.sum(dsqs) / (dsqs.size - 1)  # 样本方差
sstd = np.sqrt(svar)                   # 样本标准差
print(pstd, sstd)
pstd = np.std(closing_prices)          # 总体标准差
sstd = np.std(closing_prices, ddof=1)  # 样本标准差
print(pstd, sstd)
```

### 数组的轴向汇总

案例：汇总每周的最高价，最低价，开盘价，收盘价。

```python
def func(data):
    pass
#func 	处理函数
#axis 	轴向 [0,1]
#array 	数组
np.apply_along_axis(func, axis, array)

```

沿着数组中所指定的轴向，调用处理函数，并将每次调用的返回值重新组织成数组返回。

```python
wdays, opening_prices, highest_prices, \
    lowest_prices, closing_prices = np.loadtxt(
        '../data/aapl.csv',
        delimiter=',', usecols=(1, 3, 4, 5, 6),
        unpack=True, converters={1: dmy2wday})

first_mon = np.where(wdays==0)[0][0]
last_fri = np.where(wdays==4)[0][-1]

wdays = wdays[first_mon:last_fri+1]
indices = np.arange(first_mon, last_fri+1)

#把周一至周五每天的indices值统计为5个数组
mon_indices = indices[wdays==0]
tue_indices = indices[wdays==1]
wen_indices = indices[wdays==2]
thu_indices = indices[wdays==3]
fri_indices = indices[wdays==4]
max_len = np.max((mon_indices.size, tue_indices.size, wen_indices.size, thu_indices.size, fri_indices.size))
mon_indices = np.pad(mon_indices, pad_width=(0, max_len-mon_indices.size), mode='constant', constant_values=-1)
indices = np.vstack((mon_indices,tue_indices,wen_indices,thu_indices,fri_indices))

# numpy将会把每一行的indices传入summary函数执行业务
def summary(indices):
    indices = indices[indices!=-1]
    opening_price = opening_prices[indices[0]]
    highest_price = highest_prices[indices].max()
    lowest_price = lowest_prices[indices].min()
    closing_price = closing_prices[indices[-1]]
    return opening_price, highest_price, lowest_price, closing_price
	
r = np.apply_along_axis(summary, 1, indices)
print(r)

np.savetxt('../../data/summary.csv', summaries, delimiter=',', fmt='%g')

	
```

### 移动均线

收盘价5日均线：从第五天开始，每天计算最近五天的收盘价的平均值所构成的一条线。

移动均线算法：

```python
a b c d e f g h i j ....
(a+b+c+d+e)/5
(b+c+d+e+f)/5
(c+d+e+f+g)/5
...
(f+g+h+i+j)/5

```

在K线图中绘制5日均线图

```python
import datetime as dt
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.dates as md

def dmy2ymd(dmy):
    dmy = str(dmy, encoding='utf-8')
    date = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
    ymd = date.strftime('%Y-%m-%d')
    return ymd

dates, closing_prices = np.loadtxt('../data/aapl.csv', delimiter=',',
    usecols=(1, 6), unpack=True, dtype='M8[D], f8', converters={1: dmy2ymd})
sma51 = np.zeros(closing_prices.size - 4)
for i in range(sma51.size):
    sma51[i] = closing_prices[i:i + 5].mean()
# 开始绘制5日均线
mp.figure('Simple Moving Average', facecolor='lightgray')
mp.title('Simple Moving Average', fontsize=20)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Price', fontsize=14)
ax = mp.gca()
# 设置水平坐标每个星期一为主刻度
ax.xaxis.set_major_locator(md.WeekdayLocator( byweekday=md.MO))
# 设置水平坐标每一天为次刻度
ax.xaxis.set_minor_locator(md.DayLocator())
# 设置水平坐标主刻度标签格式
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices, c='lightgray', label='Closing Price')
mp.plot(dates[4:], sma51, c='orangered', label='SMA-5(1)')
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()

```

### 卷积

先理解卷积运算的过程：

```python
a = [1, 2, 3, 4, 5]  源数组
b = [8, 7, 6]        卷积核

使用b作为卷积核对a数组执行卷积运算的过程如下：

               44  65  86          - 有效卷积 (valid)
           23  44  65  86  59      - 同维卷积 (same)
        8  23  44  65  86  59  30  - 完全卷积 (full)
0   0   1   2   3   4   5   0   0
6   7   8
    6   7   8
        6   7   8
            6   7   8
                6   7   8
                    6   7   8
                        6   7   8

c = numpy.convolve(a, b, 卷积类型)

```

**5日移动均线序列可以直接使用卷积实现**

```python
a = [a, b, c, d, e, f, g, h, i, j] 
b = [1/5, 1/5, 1/5, 1/5, 1/5]

```

**使用卷积函数numpy.convolve(a, b, 卷积类型)实现5日均线**

```python
sma52 = np.convolve( closing_prices, np.ones(5) / 5, 'valid')
mp.plot(dates[4:], sma52, c='limegreen', alpha=0.5,
        linewidth=6, label='SMA-5(2)')

```

**使用卷积函数numpy.convolve(a, b, 卷积类型)实现10日均线**

```python
sma10 = np.convolve(closing_prices, np.ones(10) / 10, 'valid')
mp.plot(dates[9:], sma10, c='dodgerblue', label='SMA-10')

```

**使用卷积函数numpy.convolve(a, b, 卷积类型)实现加权5日均线**

```python
weights = np.exp(np.linspace(-1, 0, 5))
weights /= weights.sum()
ema5 = np.convolve(closing_prices, weights[::-1], 'valid')
mp.plot(dates[4:], sma52, c='limegreen', alpha=0.5,
        linewidth=6, label='SMA-5')

```

### 布林带

布林带由三条线组成：

中轨：移动平均线

上轨：中轨+2x5日收盘价标准差	（顶部的压力）

下轨：中轨-2x5日收盘价标准差 	（底部的支撑力）

布林带收窄代表稳定的趋势，布林带张开代表有较大的波动空间的趋势。

**绘制5日均线的布林带**

```python
weights = np.exp(np.linspace(-1, 0, 5))
weights /= weights.sum()
em5 = np.convolve(closing_prices, weights[::-1], 'valid')
stds = np.zeros(em5.size)
for i in range(stds.size):
    stds[i] = closing_prices[i:i + 5].std()
stds *= 2
lowers = medios - stds
uppers = medios + stds

mp.plot(dates, closing_prices, c='lightgray', label='Closing Price')
mp.plot(dates[4:], medios, c='dodgerblue', label='Medio')
mp.plot(dates[4:], lowers, c='limegreen', label='Lower')
mp.plot(dates[4:], uppers, c='orangered', label='Upper')
```



## 线性模型

什么是线性关系？
$$
x=1 \quad \rarr \quad y=60 \\
x=2 \quad \rarr \quad y=65 \\
x=3 \quad \rarr \quad y=70 \\
x=4 \quad \rarr \quad y=75 \\
x=5 \quad \rarr \quad y= ??? \\
$$

### 线性预测

假设一组数据符合一种线型规律，那么就可以预测未来将会出现的数据。

```python
a	b	c	d	e	f	?
    b   c   d   e   f   g   ?
```

$$
\begin{cases}
aw_0 + bw_1 + cw_2 = d \\
bw_0 + cw_1 + dw_2 = e \\
cw_0 + dw_1 + ew_2 = f \\
\end{cases}
$$

线型方程组转换为矩阵相乘的形式：
$$
\left[ \begin{array}{ccc}
a & b & c\\
b & c & d\\
c & d & e\\
\end{array} 
\right ]
\times
\left[ \begin{array}{ccc}
w_0\\
w_1\\
w_2\\
\end{array} 
\right ]=
\left[ 
\begin{array}{ccc}
d\\
e\\
f\\
\end{array} 
\right ]
\\
\quad \quad  A \quad \quad \quad \quad \quad   x\quad \quad \quad  B  \quad
$$

```python
x = np.linalg.lstsq(A, B)[0]
```



根据线性模型的特点可以通过一组历史数据求出线性关系系数x, y, z，从而预测d、e、f下的一个数据是多少。

**线性预测需要使用历史数据进行检验，让预测结果可信度更高**

案例：使用线性预测，预测下一天的收盘价。

```python
# 整理五元一次方程组    最终获取一组股票走势预测值
N = 5
pred_prices = np.zeros(closing_prices.size - 2 * N + 1)
for i in range(pred_prices.size):
    a = np.zeros((N, N))
    for j in range(N):
        a[j, ] = closing_prices[i + j:i + j + N]
    b = closing_prices[i + N:i + N * 2]
    x = np.linalg.lstsq(a, b)[0]
    pred_prices[i] = b.dot(x)
# 由于预测的是下一天的收盘价，所以想日期数组中追加一个元素，为下一个工作日的日期
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices, 'o-', c='lightgray', label='Closing Price')
dates = np.append(dates, dates[-1] + pd.tseries.offsets.BDay())
mp.plot(dates[2 * N:], pred_prices, 'o-',c='orangered', 
        linewidth=3,label='Predicted Price')
mp.legend()
mp.gcf().autofmt_xdate() 
mp.show()
```

### 线性拟合

线性拟合可以寻求与一组散点走向趋势规律相适应的线型表达式方程。

有一组散点描述时间序列下的股价：

```python
[x1, y1]
[x2, y2]
[x3, y3] 
...
[xn, yn]
```

根据线型 y=kx + b 方程可得：

```python
kx1 + b = y1
kx2 + b = y2
kx3 + b = y3
...
kxn + b = yn
```


$$
\left[ \begin{array}{ccc}
x{_1} & 1\\
x{_2} & 1\\
x{_3} & 1 \\
x{_n} & 1 \\
\end{array} 
\right ]
\times
\left[ \begin{array}{ccc}
k\\
b\\
\end{array} 
\right ]
=
\left[ \begin{array}{ccc}
y{_1}\\
y{_2}\\
y{_3}\\
y{_n}\\
\end{array} 
\right ]
$$

样本过多，每两组方程即可求得一组k与b的值。np.linalg.lstsq(a, b) 可以通过最小二乘法求出所有结果中拟合误差最小的k与b的值。

案例：利用线型拟合画出股价的趋势线

1. 绘制趋势线（趋势可以表示为最高价、最低价、收盘价的均值）：

```python
dates, opening_prices, highest_prices, \
    lowest_prices, closing_prices = np.loadtxt('../data/aapl.csv', delimiter=',',
        usecols=(1, 3, 4, 5, 6), unpack=True,dtype='M8[D], f8, f8, f8, f8',
        converters={1: dmy2ymd})
trend_points = (highest_prices + lowest_prices + closing_prices) / 3
days = dates.astype(int)
a = np.column_stack((days, np.ones_like(days)))
x = np.linalg.lstsq(a, trend_points)[0]
trend_line = days * x[0] + x[1]
mp.figure('Trend', facecolor='lightgray')
mp.title('Trend', fontsize=20)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Price', fontsize=14)
ax = mp.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
dates = dates.astype(md.datetime.datetime)
rise = closing_prices - opening_prices >= 0.01
fall = opening_prices - closing_prices >= 0.01
fc = np.zeros(dates.size, dtype='3f4')
ec = np.zeros(dates.size, dtype='3f4')
fc[rise], fc[fall] = (1, 1, 1), (0.85, 0.85, 0.85)
ec[rise], ec[fall] = (0.85, 0.85, 0.85), (0.85, 0.85, 0.85)
mp.bar(dates, highest_prices - lowest_prices, 0,lowest_prices, color=fc, edgecolor=ec)
mp.bar(dates, closing_prices - opening_prices, 0.8,opening_prices, color=fc, 
       edgecolor=ec)
mp.scatter(dates, trend_points, c='dodgerblue',alpha=0.5, s=60, zorder=2)
mp.plot(dates, trend_line, linestyle='o-', c='dodgerblue',linewidth=3, label='Trend')
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

1. 绘制顶部压力线（趋势线+(最高价 - 最低价)）

```python
trend_points = (highest_prices + lowest_prices + closing_prices) / 3
spreads = highest_prices - lowest_prices
resistance_points = trend_points + spreads
days = dates.astype(int)
x = np.linalg.lstsq(a, resistance_points)[0]
resistance_line = days * x[0] + x[1]
mp.scatter(dates, resistance_points, c='orangered', alpha=0.5, s=60, zorder=2)
mp.plot(dates, resistance_line, c='orangered', linewidth=3, label='Resistance')
```

1. 绘制底部支撑线（趋势线-(最高价 - 最低价)）

```python
trend_points = (highest_prices + lowest_prices + closing_prices) / 3
spreads = highest_prices - lowest_prices
support_points = trend_points - spreads
days = dates.astype(int)
x = np.linalg.lstsq(a, support_points)[0]
support_line = days * x[0] + x[1]
mp.scatter(dates, support_points, c='limegreen', alpha=0.5, s=60, zorder=2)
mp.plot(dates, support_line, c='limegreen', linewidth=3, label='Support')
```

## 协方差、相关矩阵、相关系数

通过两组统计数据计算而得的协方差可以评估这两组统计数据的相似程度。

**样本**：

```python
A = [a1, a2, ..., an]
B = [b1, b2, ..., bn]
```

**平均值**：

```python
ave_a = (a1 + a2 +...+ an)/n
ave_b = (b1 + b2 +...+ bn)/n

```

**离差**（用样本中的每一个元素减去平均数，求得数据的误差程度）：

```python
dev_a = [a1, a2, ..., an] - ave_a
dev_b = [b1, b2, ..., bn] - ave_b

```

**协方差**

协方差可以简单反映两组统计样本的相关性，值为正，则为正相关；值为负，则为负相关，绝对值越大相关性越强。

```
cov_ab = mean(dev_a x dev_b)
cov_ba = mean(dev_b x dev_a)
```

案例：计算两组数据的协方差，并绘图观察。

```python
import numpy as np
import matplotlib.pyplot as mp

a = np.random.randint(1, 30, 10)
b = np.random.randint(1, 30, 10)
#平均值
ave_a = np.mean(a)
ave_b = np.mean(b)
#离差
dev_a = a - ave_a
dev_b = b - ave_b
#协方差
cov_ab = np.mean(dev_a*dev_b)
cov_ba = np.mean(dev_b*dev_a)
print('a与b数组：', a, b)
print('a与b样本方差：', np.sum(dev_a**2)/(len(dev_a)-1), np.sum(dev_b**2)/(len(dev_b)-1))
print('a与b协方差：',cov_ab, cov_ba)
#绘图，查看两条图线的相关性
mp.figure('COV LINES', facecolor='lightgray')
mp.title('COV LINES', fontsize=16)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
x = np.arange(0, 10)
#a,b两条线
mp.plot(x, a, color='dodgerblue', label='Line1')
mp.plot(x, b, color='limegreen', label='Line2')
#a,b两条线的平均线
mp.plot([0, 9], [ave_a, ave_a], color='dodgerblue', linestyle='--', alpha=0.7, linewidth=3)
mp.plot([0, 9], [ave_b, ave_b], color='limegreen', linestyle='--', alpha=0.7, linewidth=3)

mp.grid(linestyle='--', alpha=0.5)
mp.legend()
mp.tight_layout()
mp.show()

```

**相关系数**

协方差除去两组统计样本标准差的乘积是一个[-1, 1]之间的数。该结果称为统计样本的相关系数。

```python
# a组样本 与 b组样本做对照后的相关系数
cov_ab/(std_a x std_b)
# b组样本 与 a组样本做对照后的相关系数
cov_ba/(std_b x std_a)
# a样本与a样本作对照   b样本与b样本做对照   二者必然相等
cov_ab/(std_a x std_b)=cov_ba/(std_b x std_a)

```

通过相关系数可以分析两组数据的相关性：

```python
若相关系数越接近于0，越表示两组样本越不相关。
若相关系数越接近于1，越表示两组样本正相关。
若相关系数越接近于-1，越表示两组样本负相关。

```

案例：输出案例中两组数据的相关系数。

```python
print('相关系数：', cov_ab/(np.std(a)*np.std(b)), cov_ba/(np.std(a)*np.std(b)))

```

**相关矩阵**


$$
\left[ \begin{array}{c}
\frac{var\_a}{std\_a \times std\_a} & \frac{cov\_ab}{std\_a \times std\_b} \\
\frac{cov\_ba}{std\_b \times std\_a} & \frac{var\_b}{std\_b \times std\_b}\\
\end{array} 
\right ]
$$
矩阵正对角线上的值都为1。（同组样本自己相比绝对正相关）
$$
\left[ \begin{array}{ccc}
1 & \frac{cov\_ab}{std\_a \times std\_b} \\
\frac{cov\_ba}{std\_b \times std\_a} & 1\\
\end{array} 
\right ]
$$
numpy提供了求得相关矩阵的API：

```python
# 相关矩阵
numpy.corrcoef(a, b)	
# 相关矩阵的分子矩阵 
# [[a方差，ab协方差], [ba协方差, b方差]]
numpy.cov(a, b)

```

## 多项式拟合

多项式的一般形式：
$$
y=p_{0}x^n + p_{1}x^{n-1} + p_{2}x^{n-2} + p_{3}x^{n-3} +...+p_{n}
$$
多项式拟合的目的是为了找到一组  $p_0, p_1, ...,  p_n$，使得拟合方程尽可能的与实际样本数据相符合。

假设拟合得到的多项式如下：
$$
f(x)=p_{0}x^n + p_{1}x^{n-1} + p_{2}x^{n-2} + p_{3}x^{n-3} +...+p_{n}
$$
则拟合函数与真实结果的差方如下
$$
loss = (y_1-f(x_1))^2 + (y_2-f(x_2))^2 + ... + (y_n-f(x_n))^2
$$
那么多项式拟合的过程即为求取一组$p_0, p_1, ...,  p_n$, 使得loss的值最小。

**多项式拟合相关API：**

```
根据一组样本，并给出最高次幂，求出拟合系数
np.polyfit(X, Y, 最高次幂)->P
```

**多项式运算相关API：**

```
根据拟合系数与自变量求出拟合值, 由此可得拟合曲线坐标样本数据 [X, Y']
np.polyval(P, X)->Y'

多项式函数求导，根据拟合系数求出多项式函数导函数的系数
np.polyder(P)->Q 

已知多项式系数Q 求多项式函数的根（与x轴交点的横坐标）
xs = np.roots(Q)

两个多项式函数的差函数的系数（可以通过差函数的根求取两个曲线的交点）
Q = np.polysub(P1, P2)
```

案例：求多项式 y = 4x<sup>3</sup> + 3x<sup>2</sup> - 1000x + 1曲线驻点的坐标。

```python
'''
1. 求出多项式的导函数
2. 求出导函数的根，若导函数的根为实数，则该点则为曲线拐点。
导函数＝点斜率
求出导函数=0 的根,得到曲线拐点的x轴
'''
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-20, 20, 1000)
y = 4*x**3 + 3*x**2  - 1000*x + 1
Q = np.polyder([4,3,-1000,1])
xs = np.roots(Q)
ys =  4*xs**3 + 3*xs**2  - 1000*xs + 1
mp.plot(x, y)
mp.scatter(xs, ys, s=80, c='orangered')
mp.show()

```

案例：使用多项式函数拟合两只股票bhp、vale的差价函数：

```python
'''
1. 计算两只股票的差价
2. 利用多项式拟合求出与两只股票差价相近的多项式系数，最高次为4
3. 把该曲线的拐点都标出来。
'''
dates, bhp_closing_prices = np.loadtxt('../../data/bhp.csv', 
                                       delimiter=',',usecols=(1, 6), unpack=True, dtype='M8[D], f8', converters={1: dmy2ymd})
vale_closing_prices = np.loa dtxt('../../data/vale.csv', delimiter=',',
                                 usecols=(6), unpack=True)
diff_closing_prices = bhp_closing_prices - vale_closing_prices
days = dates.astype(int)
p = np.polyfit(days, diff_closing_prices, 4)
poly_closing_prices = np.polyval(p, days)
q = np.polyder(p)
roots_x = np.roots(q)
roots_y = np.polyval(p, roots_x)
mp.figure('Polynomial Fitting', facecolor='lightgray')
mp.title('Polynomial Fitting', fontsize=20)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Difference Price', fontsize=14)
ax = mp.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, poly_closing_prices, c='limegreen',
        linewidth=3, label='Polynomial Fitting')
mp.scatter(dates, diff_closing_prices, c='dodgerblue',
           alpha=0.5, s=60, label='Difference Price')
roots_x = roots_x.astype(int).astype('M8[D]').astype(
    		md.datetime.datetime)
mp.scatter(roots_x, roots_y, marker='^', s=80,
           c='orangered', label='Peek', zorder=4)
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()

```

## 数据平滑

数据的平滑处理通常包含有降噪、拟合等操作。降噪的功能意在去除额外的影响因素，拟合的目的意在数学模型化，可以通过更多的数学方法识别曲线特征。

案例：绘制两只股票收益率曲线。收益率 =（后一天收盘价-前一天收盘价） / 前一天收盘价。

1. 使用卷积完成数据降噪。

```python
dates, bhp_closing_prices = np.loadtxt( '../data/bhp.csv', delimiter=',', usecols=(1,6), dtype='M8[D], f8',converters={1:dmy2ymd}, unpack=True)
vale_closing_prices = np.loadtxt( '../data/vale.csv', delimiter=',', usecols=(6), dtype='f8',converters={1:dmy2ymd}, unpack=True)

bhp_returns = np.diff(bhp_closing_prices) / bhp_closing_prices[:-1]
vale_returns = np.diff(vale_closing_prices) / vale_closing_prices[:-1]
dates = dates[:-1]

#卷积降噪
convolve_core = np.hanning(8) 
convolve_core /= convolve_core.sum()
bhp_returns_convolved = np.convolve(bhp_returns, convolve_core, 'valid')
vale_returns_convolved = np.convolve(vale_returns, convolve_core, 'valid')
#绘制这条曲线
mp.figure('BHP VALE RETURNS', facecolor='lightgray')
mp.title('BHP VALE RETURNS', fontsize=20)
mp.xlabel('Date')
mp.ylabel('Price')
ax = mp.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(md.DateFormatter('%Y %m %d'))
dates = dates.astype('M8[D]')
#绘制收益线
mp.plot(dates, bhp_returns, color='dodgerblue', linestyle='--', label='bhp_returns', alpha=0.3)
mp.plot(dates, vale_returns, color='orangered', linestyle='--', label='vale_returns', alpha=0.3)
#绘制卷积降噪线
mp.plot(dates[7:], bhp_returns_convolved, color='dodgerblue', label='bhp_returns_convolved', alpha=0.5)
mp.plot(dates[7:], vale_returns_convolved, color='orangered', label='vale_returns_convolved', alpha=0.5)

mp.show()


```

1. 对处理过的股票收益率做多项式拟合。

```python
#拟合这两条曲线，获取两组多项式系数
dates = dates.astype(int)
bhp_p = np.polyfit(dates[7:], bhp_returns_convolved, 3)
bhp_polyfit_y = np.polyval(bhp_p, dates[7:])
vale_p = np.polyfit(dates[7:], vale_returns_convolved, 3)
vale_polyfit_y = np.polyval(vale_p, dates[7:])
#绘制拟合线
mp.plot(dates[7:], bhp_polyfit_y, color='dodgerblue', label='bhp_returns_polyfit')
mp.plot(dates[7:], vale_polyfit_y, color='orangered', label='vale_returns_polyfit')

```

1. 通过获取两个函数的焦点可以分析两只股票的投资收益比。

```python
#求两条曲线的交点  f(bhp) = f(vale)的根
sub_p = np.polysub(bhp_p, vale_p)
roots_x = np.roots(sub_p)	# 让f(bhp) - f(vale) = 0  函数的两个根既是两个函数的焦点
roots_x = roots_x.compress( (dates[0] <= roots_x) & (roots_x <= dates[-1]))
roots_y = np.polyval(bhp_p, roots_x)
#绘制这些点
mp.scatter(roots_x, roots_y, marker='D', color='green', s=60, zorder=3)

```

## 符号数组

sign函数可以把样本数组的变成对应的符号数组，正数变为1，负数变为-1，0则变为0。

```python
ary = np.sign(源数组)
```

**净额成交量（OBV）**

成交量可以反映市场对某支股票的人气，而成交量是一只股票上涨的能量。一支股票的上涨往往需要较大的成交量。而下跌时则不然。

若相比上一天的收盘价上涨，则为正成交量；若相比上一天的收盘价下跌，则为负成交量。

 绘制OBV柱状图

```python
dates, closing_prices, volumes = np.loadtxt(
    '../../data/bhp.csv', delimiter=',',
    usecols=(1, 6, 7), unpack=True,
    dtype='M8[D], f8, f8', converters={1: dmy2ymd})
diff_closing_prices = np.diff(closing_prices)
sign_closing_prices = np.sign(diff_closing_prices)
obvs = volumes[1:] * sign_closing_prices
mp.figure('On-Balance Volume', facecolor='lightgray')
mp.title('On-Balance Volume', fontsize=20)
mp.xlabel('Date', fontsize=14)
mp.ylabel('OBV', fontsize=14)
ax = mp.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
dates = dates[1:].astype(md.datetime.datetime)
mp.bar(dates, obvs, 1.0, color='dodgerblue',
       edgecolor='white', label='OBV')
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

**数组处理函数**

```python
ary = np.piecewise(源数组, 条件序列, 取值序列)
```

针对源数组中的每一个元素，检测其是否符合条件序列中的每一个条件，符合哪个条件就用取值系列中与之对应的值，表示该元素，放到目标 数组中返回。

条件序列: [a < 0, a == 0, a > 0]

取值序列: [-1, 0, 1]     

```python
a = np.array([70, 80, 60, 30, 40])
d = np.piecewise(
    a, 
    [a < 60, a == 60, a > 60],
    [-1, 0, 1])
# d = [ 1  1  0 -1 -1]
```

## 矢量化

矢量化指的是用数组代替标量来操作数组里的每个元素。

numpy提供了vectorize函数，可以把处理标量的函数矢量化，返回的函数可以直接处理ndarray数组。

```python
import math as m
import numpy as np

def foo(x, y):
    return m.sqrt(x**2 + y**2)

x, y = 1, 4
print(foo(x, y))
X, Y = np.array([1, 2, 3]), np.array([4, 5, 6])
vectorized_foo = np.vectorize(foo)
print(vectorized_foo(X, Y))
print(np.vectorize(foo)(X, Y))
```

numpy还提供了frompyfuc函数，也可以完成与vectorize相同的功能：

```python
# 把foo转换成矢量函数，该矢量函数接收2个参数，返回一个结果 
fun = np.frompyfunc(foo, 2, 1)
fun(X, Y)
```

案例：定义一种买进卖出策略，通过历史数据判断这种策略是否值得实施。

```python
dates, opening_prices, highest_prices, \
    lowest_prices, closing_prices = np.loadtxt(
        '../../data/bhp.csv', delimiter=',',
        usecols=(1, 3, 4, 5, 6), unpack=True,
        dtype='M8[D], f8, f8, f8, f8',
        converters={1: dmy2ymd})
    
# 定义一种投资策略
def profit(opening_price, highest_price,
           lowest_price, closing_price):
    buying_price = opening_price * 0.99
    if lowest_price <= buying_price <= highest_price:
        return (closing_price - buying_price) * \
            100 / buying_price
    return np.nan  # 无效值

# 矢量化投资函数
profits = np.vectorize(profit)(opening_prices, 
       highest_prices, lowest_prices, closing_prices)
nan = np.isnan(profits)
dates, profits = dates[~nan], profits[~nan]
gain_dates, gain_profits = dates[profits > 0], profits[profits > 0]
loss_dates, loss_profits = dates[profits < 0], profits[profits < 0]
mp.figure('Trading Simulation', facecolor='lightgray')
mp.title('Trading Simulation', fontsize=20)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Profit', fontsize=14)
ax = mp.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
if dates.size > 0:
    dates = dates.astype(md.datetime.datetime)
    mp.plot(dates, profits, c='gray',
            label='Profit')
    mp.axhline(y=profits.mean(), linestyle='--',
               color='gray')
if gain_dates.size > 0:
    gain_dates = gain_dates.astype(md.datetime.datetime)
    mp.plot(gain_dates, gain_profits, 'o',
            c='orangered', label='Gain Profit')
    mp.axhline(y=gain_profits.mean(), linestyle='--',
               color='orangered')
if loss_dates.size > 0:
    loss_dates = loss_dates.astype(md.datetime.datetime)
    mp.plot(loss_dates, loss_profits, 'o',
            c='limegreen', label='Loss Profit')
    mp.axhline(y=loss_profits.mean(), linestyle='--',
               color='limegreen')
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

## 矩阵

矩阵是numpy.matrix类类型的对象，该类继承自numpy.ndarray，任何针对多维数组的操作，对矩阵同样有效，但是作为子类矩阵又结合其自身的特点，做了必要的扩充，比如：乘法计算、求逆等。

**矩阵对象的创建**

```python
# 如果copy的值为True(缺省)，所得到的矩阵对象与参数中的源容器各自拥有独立的数据拷贝。
numpy.matrix(
    ary,		# 任何可被解释为矩阵的二维容器
  	copy=True	# 是否复制数据(缺省值为True，即复制数据)
)

np.matrix(ary)
np.mat(ary)
```

```python
# 等价于：numpy.matrix(..., copy=False)
# 由该函数创建的矩阵对象与参数中的源容器一定共享数据，无法拥有独立的数据拷贝
numpy.mat(任何可被解释为矩阵的二维容器)

```

```python
# 该函数可以接受字符串形式的矩阵描述：
# 数据项通过空格分隔，数据行通过分号分隔。例如：'1 2 3; 4 5 6'
numpy.mat(拼块规则)
```

**矩阵的乘法运算**

```python
# 矩阵的乘法：乘积矩阵的第i行第j列的元素等于
# 被乘数矩阵的第i行与乘数矩阵的第j列的点积
#
#           1   2   6
#    X----> 3   5   7
#    |      4   8   9
#    |
# 1  2  6   31  60  74
# 3  5  7   46  87 116
# 4  8  9   64 120 161
e = np.mat('1 2 6; 3 5 7; 4 8 9')
print(e * e)

```

**矩阵的逆矩阵**

若两个矩阵A、B满足：AB = E （E为单位矩阵），则B为A的逆矩阵。

```python
e = np.mat('1 2 6; 3 5 7; 4 8 9')
print(e.I)
print(e * e.I)

```

ndarray提供了方法让多维数组替代矩阵的运算： 

```python
a = np.array([
    [1, 2, 6],
    [3, 5, 7],
    [4, 8, 9]])
# 点乘法求ndarray的点乘结果，与矩阵的乘法运算结果相同
k = a.dot(a)
print(k)
# linalg模块中的inv方法可以求取a的逆矩阵
l = np.linalg.inv(a)
print(l)

```

案例：假设一帮孩子和家长出去旅游，去程坐的是bus，小孩票价为3元，家长票价为3.2元，共花了118.4；回程坐的是Train，小孩票价为3.5元，家长票价为3.6元，共花了135.2。分别求小孩和家长的人数。使用矩阵求解。
$$
\left[ \begin{array}{ccc}
	3 & 3.2 \\
	3.5 & 3.6 \\
\end{array} \right]
\times
\left[ \begin{array}{ccc}
	x \\
    y \\
\end{array} \right]
=
\left[ \begin{array}{ccc}
	118.4 \\
	135.2 \\
\end{array} \right]
$$

```python
import numpy as np

prices = np.mat('3 3.2; 3.5 3.6')
totals = np.mat('118.4; 135.2')

persons = prices.I * totals
print(persons)

```

把逆矩阵的概念推广到非方阵，即称为**广义逆矩阵**。

案例：斐波那契数列

1	1	 2	 3	5	8	13	21	34 ...

```python
X      1   1    1   1    1   1
       1   0    1   0    1   0
    --------------------------------
1  1   2   1    3   2    5   3
1  0   1   1    2   1    3   2
 F^1    F^2      F^3 	  F^4  ...  f^n

```

**代码**

```python
import numpy as np
n = 35

# 使用递归实现斐波那契数列
def fibo(n):
    return 1 if n < 3 else fibo(n - 1) + fibo(n - 2)
print(fibo(n))

# 使用矩阵实现斐波那契数列
print(int((np.mat('1. 1.; 1. 0.') ** (n - 1))[0, 0]))

```

## 通用函数

### 裁剪、压缩

**数组的裁剪**

```python
# 将调用数组中小于和大于下限和上限的元素替换为下限和上限，返回裁剪
# 后的数组，调用数组保持不变。
ndarray.clip(min=下限, max=上限)
```

**数组的压缩**

```python
# 返回由调用数组中满足条件的元素组成的新数组。
ndarray.compress(条件)
```

案例：

```python
from __future__ import unicode_literals
import numpy as np
a = np.array([10, 20, 30, 40, 50])
print(a)
b = a.clip(min=15, max=45)
print(b)
c = a.compress((15 <= a) & (a <= 45))
print(c)

```

### 加法与乘法通用函数

```python
np.add(a, a) 					# 两数组相加
np.add.reduce(a) 				# a数组元素累加和
np.add.accumulate(a) 			# 累加和过程
np.add.outer([10, 20, 30], a)	# 外和
np.prod(a)		# 累乘
np.cumprod(a)		# 累乘的过程
np.outer([10, 20, 30], a)

```

案例：

```python
a = np.arange(1, 7)
print(a)
b = a + a
print(b)
b = np.add(a, a)
print(b)
c = np.add.reduce(a)
print(c)
d = np.add.accumulate(a)
print(d)
#  +  	 1  2  3  4  5  6   
#	   --------------------
# 10   |11 12 13 14 15 16 |
# 20   |21 22 23 24 25 26 |
# 30   |31 32 33 34 35 36 |
       --------------------
f = np.add.outer([10, 20, 30], a)
print(f)
#  x  	 1  2  3  4  5  6   
#	   -----------------------
# 10   |10 20 30  40  50  60 |
# 20   |20 40 60  80 100 120 |
# 30   |30 60 90 120 150 180 |
       -----------------------
g = np.outer([10, 20, 30], a)
print(g)

```

### 除法与取整通用函数

```python
np.divide(a, b) 	# a 真除 b

np.floor(a / b)		# （真除的结果向下取整）
np.ceil(a / b) 		# （真除的结果向上取整）
np.trunc(a / b)		# （真除的结果截断取整）
np.round(a / b)		# （真除的结果四舍五入取整）
```

案例：

```python
import numpy as np

a = np.array([20, 20, -20, -20])
b = np.array([3, -3, 6, -6])
# 真除
c = np.true_divide(a, b)
c = np.divide(a, b)
c = a / b
print('array:',c)
# 对ndarray做floor操作
d = np.floor(a / b)
print('floor_divide:',d)
# 对ndarray做ceil操作
e = np.ceil(a / b)
print('ceil ndarray:',e)
# 对ndarray做trunc操作
f = np.trunc(a / b)
print('trunc ndarray:',f)
# 对ndarray做around操作
g = np.around(a / b)
print('around ndarray:',g)

```

### 位运算通用函数

```python
位异或：
c = a ^ b
c = np.bitwise_xor(a, b)
位与：
e = a & b
e = np.bitwise_and(a, b)
位或：
e = a | b
e = np.bitwise_or(a, b)
位反：
e = ~a
e = np.bitwise_not(a, b)
移位：
<<		__lshift__		left_shift
>>		__rshift__		right_shift
```

按位异或操作可以很方便的判断两个数据是否同号。

```
-8	1000
-7	1001
-6	1010
-5	1011
-4	1100
-3	1101
-2	1110
-1	1111
0	0000
1	0001
2	0010
3	0011
4	0100
5	0101
6	0110
7	0111

0111 1111 1111 1111 1111 1111 1111 1111 1111 1111 
0110 1101 0110 1101 0110 1101 0110 1101 0110 1101 


0 ^ 0 = 0
0 ^ 1 = 1
1 ^ 0 = 1
1 ^ 1 = 0

```

```python
a = np.array([0, -1, 2, -3, 4, -5])
b = np.array([0, 1, 2, 3, 4, 5])
print(a, b)
c = a ^ b
# c = a.__xor__(b)
# c = np.bitwise_xor(a, b)
print(np.where(c < 0)[0])
```



利用位与运算计算某个数字是否是2的幂

```python
#  1 2^0 00001   0 00000
#  2 2^1 00010   1 00001
#  4 2^2 00100   3 00011
#  8 2^3 01000   7 00111
# 16 2^4 10000  15 01111
# ...

d = np.arange(1, 21)
print(d)
e = d & (d - 1)
e = d.__and__(d - 1)
e = np.bitwise_and(d, d - 1)
print(e)

```

### 三角函数通用函数

```python
numpy.sin()

```

傅里叶定理：

法国科学家傅里叶提出傅里叶定理，任何一条周期曲线，无论多么跳跃或不规则，都能表示成一组光滑正弦曲线叠加之和。

**合成方波**

一个方波由如下参数的正弦波叠加而成：
$$
y = 4\pi \times sin(x) \\
y = \frac{4\pi}{3} \times sin(3x) \\
...\\
...\\
y = \frac{4\pi}{2n-1} \times sin((2n-1)x)
$$
曲线叠加的越多，越接近方波。所以可以设计一个函数，接收曲线的数量n作为参数，返回一个矢量函数，该函数可以接收x坐标数组，返回n个正弦波叠加得到的y坐标数组。

```python
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.zeros(1000)
n = 1000
for i in range(1， n+1):
	y += 4 / ((2 * i - 1) * np.pi) * np.sin((2 * i - 1) * x)
mp.plot(x, y, label='n=1000')
mp.legend()
mp.show()

```

## 特征值和特征向量

对于n阶方阵A，如果存在数a和非零n维列向量x，使得Ax=ax，则称a是矩阵A的一个特征值，x是矩阵A属于特征值a的特征向量

```python
#已知n阶方阵A， 求特征值与特征数组
# eigvals: 特征值数组
# eigvecs: 特征向量数组 
eigvals, eigvecs = np.linalg.eig(A)
#已知特征值与特征向量，求方阵
S = np.mat(eigvecs) * np.mat(np.diag(eigvals)) * np.mat(eigvecs.I) 
```

案例：

```python
import numpy as np
A = np.mat('3 -2; 1 0')
print(A)
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)
print(A * eigvecs[:, 0])	# 方阵*特征向量
print(eigvals[0] * eigvecs[:, 0])	#特征值*特征向量
S = np.mat(eigvecs) * np.mat(np.diag(eigvals)) * np.mat(eigvecs.I)
```

案例：读取图片的亮度矩阵，提取特征值与特征向量，保留部分特征值，重新生成新的亮度矩阵，绘制图片。

```python
"""
demo105_eigva.py 图像特征
"""
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as mp

# 读取图片   True: 读取灰度图像
img = sm.imread('../data/da_data/lily.jpg', True)
print(img.shape)

# 提取特征值
img = np.mat(img)
eigvals, eigvecs = np.linalg.eig(img)
print(eigvals.shape, eigvecs.shape)

# 生成原图像
eigvals[50:] = 0
img2 = eigvecs * np.diag(eigvals) * eigvecs.I
mp.subplot(221)
mp.imshow(img, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(222)
mp.imshow(img2.real, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.tight_layout()
mp.show()

```

## 奇异值分解

有一个矩阵M，可以分解为3个矩阵U、S、V，使得U x S x V等于M。U与V都是正交矩阵（乘以自身的转置矩阵结果为单位矩阵）。那么S矩阵主对角线上的元素称为矩阵M的奇异值，其它元素均为0。

```python
import numpy as np
M = np.mat('4 11 14; 8 7 -2')
print(M)
U, sv, V = np.linalg.svd(M, full_matrices=False)
print(U * U.T)
print(V * V.T)
print(sv)
S = np.diag(sv)
print(S)
print(U * S * V)
```

案例：读取图片的亮度矩阵，提取奇异值与两个正交矩阵，保留部分奇异值，重新生成新的亮度矩阵，绘制图片。

```python
original = sm.imread('../data/lily.jpg', True)
#提取奇异值  sv 	
U, sv, V = np.linalg.svd(original)
print(U.shape, sv.shape, V.shape)
sv[50:] = 0
original2 = np.mat(U) * np.mat(np.diag(sv)) * np.mat(V)
mp.figure("Lily Features")
mp.subplot(221)
mp.xticks([])
mp.yticks([])
mp.imshow(original, cmap='gray')

mp.subplot(222)
mp.xticks([])
mp.yticks([])
mp.imshow(original2, cmap='gray')
mp.tight_layout()

```

## 快速傅里叶变换(fft)

什么是傅里叶变换？

法国科学家傅里叶提出傅里叶定理，任何一条周期曲线，无论多么跳跃或不规则，都能表示成一组光滑正弦曲线叠加之和。傅里叶变换即是将不规则曲线拆解为一组光滑正弦曲线的过程。

傅里叶变换的目的是可将时域（即时间域）上的信号转变为频域（即频率域）上的信号，随着域的不同，对同一个事物的了解角度也就随之改变，因此在时域中某些不好处理的地方，在频域就可以较为简单的处理。这就可以大量减少处理信号存储量。



例如：弹钢琴

假设有一时间域函数：**y = f(x)**，根据傅里叶的理论它可以被分解为一系列正弦函数的叠加，他们的振幅A，频率&omega;或初相位&phi;不同：
$$
y = A_1sin(\omega_1x+\phi_1) +  A_2sin(\omega_2x+\phi_2) +  A_2sin(\omega_2x+\phi_2) + R
$$
所以傅里叶变换可以把一个比较复杂的函数转换为多个简单函数的叠加，看问题的角度也从时间域转到了频率域，有些的问题处理起来就会比较简单。

#### **傅里叶变换相关函数**

导入快速傅里叶变换所需模块

```python
import numpy.fft as nf
```

通过采样数与采样周期求得傅里叶变换分解所得曲线的**频率序列**

```python
freqs = nf.fftfreq(采样数量, 采样周期)
```

通过原函数值的序列j经过快速傅里叶变换得到一个**复数数组**，复数的模代表的是**振幅**，复数的辐角代表**初相位**

```python
nf.fft(原函数数组) -> 复数数组(表示一组正弦函数)
```

通过 **复数数组** 经过逆向傅里叶变换得到**合成的函数值数组**

```python
nf.ifft(复数数组)->原函数值数组
```

案例：针对方波，绘制时域图与频域图。

```python
import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as mp
times = np.linspace(0, 2 * np.pi, 201)
sigs1 = 4 / (1 * np.pi) * np.sin(1 * times)
sigs2 = 4 / (3 * np.pi) * np.sin(3 * times)
sigs3 = 4 / (5 * np.pi) * np.sin(5 * times)
sigs4 = 4 / (7 * np.pi) * np.sin(7 * times)
sigs5 = 4 / (9 * np.pi) * np.sin(9 * times)
sigs6 = sigs1 + sigs2 + sigs3 + sigs4 + sigs5

mp.subplot(121)
mp.title('Time Domain', fontsize=16)
mp.xlabel('Time', fontsize=12)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times, sigs1, label=r'$\omega$='+str(round(1 / (2 * np.pi),3)))
mp.plot(times, sigs2, label=r'$\omega$='+str(round(3 / (2 * np.pi),3)))
mp.plot(times, sigs3, label=r'$\omega$='+str(round(5 / (2 * np.pi),3)))
mp.plot(times, sigs4, label=r'$\omega$='+str(round(7 / (2 * np.pi),3)))
mp.plot(times, sigs5, label=r'$\omega$='+str(round(9 / (2 * np.pi),3)))
mp.plot(times, sigs6, label=r'$\omega$='+str(round(1 / (2 * np.pi),3)))
mp.legend()
mp.show()

```

案例：针对合成波做快速傅里叶变换，得到一组复数序列；再针对该复数序列做逆向傅里叶变换得到新的合成波并绘制。

```python
ffts = nf.fft(sigs6)
sigs7 = nf.ifft(ffts).real
mp.plot(times, sigs7, label=r'$\omega$='+str(round(1 / (2 * np.pi),3)), alpha=0.5, linewidth=6)
```

案例：针对合成波做快速傅里叶变换，得到分解波数组的频率、振幅、初相位数组，并绘制频域图像。

```python
# 得到分解波的频率序列
freqs = nf.fftfreq(times.size, times[1] - times[0])
# 复数的模为信号的振幅（能量大小）
ffts = nf.fft(sigs6)
pows = np.abs(ffts)

mp.subplot(122)
mp.title('Frequency Domain', fontsize=16)
mp.xlabel('Frequency', fontsize=12)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(freqs[freqs >= 0], pows[freqs >= 0], c='orangered', label='Frequency Spectrum')
mp.legend()
mp.tight_layout()
mp.show()

```

#### **基于傅里叶变换的频域滤波**

 含噪信号是高能信号与低能噪声叠加的信号，可以通过傅里叶变换的频域滤波实现降噪。

通过FFT使含噪信号转换为含噪频谱，去除低能噪声，留下高能频谱后再通过IFFT留下高能信号。

案例：基于傅里叶变换的频域滤波为音频文件去除噪声。

1. 读取音频文件，获取音频文件基本信息：采样个数，采样周期，与每个采样的声音信号值。绘制音频时域的：时间/位移图像。

```python
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp

sample_rate, noised_sigs = wf.read('../data/noised.wav')
noised_sigs = noised_sigs / 2 ** 15
times = np.arange(len(noised_sigs)) / sample_rate
mp.figure('Filter', facecolor='lightgray')
mp.subplot(221)
mp.title('Time Domain', fontsize=16)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], noised_sigs[:178],c='orangered', label='Noised')
mp.legend()
mp.show()

```

1. 基于傅里叶变换，获取音频频域信息，绘制音频频域的：频率/能量图像。

```python
freqs = nf.fftfreq(times.size, 1 / sample_rate)
noised_ffts = nf.fft(noised_sigs)
noised_pows = np.abs(noised_ffts)
mp.subplot(222)
mp.title('Frequency Domain', fontsize=16)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.semilogy(freqs[freqs >= 0],noised_pows[freqs >= 0], c='limegreen',label='Noised')
mp.legend()

```

1. 将低频噪声去除后绘制音频频域的：频率/能量图像。

```python
fund_freq = freqs[noised_pows.argmax()]
noised_indices = np.where(freqs != fund_freq)
filter_ffts = noised_ffts.copy()
filter_ffts[noised_indices] = 0
filter_pows = np.abs(filter_ffts)

mp.subplot(224)
mp.xlabel('Frequency', fontsize=12)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(freqs[freqs >= 0], filter_pows[freqs >= 0],c='dodgerblue', label='Filter')
mp.legend() 

```

1. 基于逆向傅里叶变换，生成新的音频信号，绘制音频时域的：时间/位移图像。

```python
filter_sigs = nf.ifft(filter_ffts).real
mp.subplot(223)
mp.xlabel('Time', fontsize=12)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], filter_sigs[:178],c='hotpink', label='Filter')
mp.legend()

```

1. 重新生成音频文件。

```python
wf.write('../../data/filter.wav',sample_rate,(filter_sigs * 2 ** 15).astype(np.int16))

```

## 随机数模块(random)

生成服从特定统计规律的随机数序列。

```python
np.random.normal()

[1,2,3,4,3,2,4,3,2,1,2,3,4,3,2,1,4,5,3,2,3,5]
[0,1,2,1,2,2,1,0,1,2,1,2,1,2]

random.normal(6, 1, 10)
[1.13, 2.345, 6.234, 6.435, 2.234, 8.345, ....]
```



#### 二项分布（binomial）

二项分布就是重复n次独立事件的伯努利试验。在每次试验中只有两种可能的结果，而且两种结果发生与否互相对立，并且相互独立，事件发生与否的概率在每一次独立试验中都保持不变。

```python
# 产生size个随机数，每个随机数来自n次尝试中的成功次数，其中每次尝试成功的概率为p。
np.random.binomial(n, p, size)

```

二项分布可以用于求如下场景的概率的近似值：

1. 某人投篮命中率为0.3，投10次，进5个球的概率。

```python
sum(np.random.binomial(10, 0.3, 200000) == 5) / 200000

```

1. 某人打客服电话，客服接通率是0.6，一共打了3次，都没人接的概率。

```python
sum(np.random.binomial(3, 0.6, 200000) == 0) / 200000

```

#### 超几何分布(hypergeometric)

```python
# 产生size个随机数，每个随机数t为在总样本中随机抽取nsample个样本后好样本的个数，总样本由ngood个好样本和nbad个坏样本组成
np.random.hypergeometric(ngood, nbad, nsample, size)

#
7个好苹果   3个坏苹果    摸3个苹果， 求有两个好苹果的概率。
```

模球游戏：将25个好球和1个坏球放在一起，每次模3个球，全为好球加1分，只要摸到了坏球减6分，求100轮的过程中分值的变化。

```python
import numpy as np
import matplotlib.pyplot as mp
outcomes = np.random.hypergeometric(25, 1, 3, 100)
scores = [0]
for outcome in outcomes:
    if outcome == 3:
        scores.append(scores[-1] + 1)
    else:
        scores.append(scores[-1] - 6)
scores = np.array(scores)
mp.figure('Hypergeometric Distribution', facecolor='lightgray')
mp.title('Hypergeometric Distribution', fontsize=20)
mp.xlabel('Round', fontsize=14)
mp.ylabel('Score', fontsize=14)
mp.tick_params(labelsize=12)
mp.grid(linestyle=':')
o, h, l, c = 0, scores.argmax(), scores.argmin(), scores.size-1
if scores[o] < scores[c]:
    color = 'orangered'
elif scores[c] < scores[o]:
    color = 'limegreen'
else:
    color = 'dodgerblue'
mp.plot(scores, c=color, label='Score')
mp.axhline(y=scores[o], linestyle='--',color='deepskyblue', linewidth=1)
mp.axhline(y=scores[h], linestyle='--',color='crimson', linewidth=1)
mp.axhline(y=scores[l], linestyle='--',color='seagreen', linewidth=1)
mp.axhline(y=scores[c], linestyle='--',color='orange', linewidth=1)
mp.legend()
mp.show()

```

#### 正态分布(normal)

```python
# 产生size个随机数，服从标准正态(期望=0, 标准差=1)分布。
np.random.normal(size)
# 产生size个随机数，服从正态分布(期望=1, 标准差=10)。
np.random.normal(loc=1, scale=10, size)

```

$$
标准正态分布概率密度: \frac{e^{-\frac{x^2}{2}}}{\sqrt{2\pi}}
$$

案例：生成10000个服从正态分布的随机数并绘制随机值的频数直方图。

```python
import numpy as np
import matplotlib.pyplot as mp
samples = np.random.normal(size=10000)
mp.figure('Normal Distribution',facecolor='lightgray')
mp.title('Normal Distribution', fontsize=20)
mp.xlabel('Sample', fontsize=14)
mp.ylabel('Occurrence', fontsize=14)
mp.tick_params(labelsize=12)
mp.grid(axis='y', linestyle=':')
mp.hist(samples, 100, normed=True,
               edgecolor='steelblue',
               facecolor='deepskyblue',
               label='Normal')[1]
mp.legend()
mp.show()


```

随机数的使用场景：

数据分析过程中做数据清洗时常用随机数填充空白值，修正异常值。



## 杂项功能

#### 排序

**联合间接排序**

联合间接排序支持为待排序列排序，若待排序列值相同，则利用参考序列作为参考继续排序。最终返回排序过后的有序索引序列。

```python
indices = numpy.lexsort((参考序列, 主序列))

```

案例：先按价格排序，再按销售量倒序排列。

```python
import numpy as np
prices = np.array([92,83,71,92,40,12,64])
volumes = np.array([100,251,4,12,709,34,75])
print(volumes)
names = ['Product1','Product2','Product3','Product4','Product5','Product6','Product7']
ind = np.lexsort((volumes*-1, prices)) 
print(ind)
for i in ind:
	print(names[i], end=' ')

```

**复数数组排序**

按照实部的升序排列，对于实部相同的元素，参考虚部的升序，直接返回排序后的结果数组。

```python
numpy.sort_complex(复数数组)

```

**插入排序**

若有需求需要向有序数组中插入元素，使数组依然有序，numpy提供了searchsorted方法查询并返回可插入位置数组。

```python
indices = numpy.searchsorted(有序数组, 待插入数据数组)

```

调用numpy提供了insert方法将待插入元素数组中的元素，按照位置数组中的位置，插入到目标数组中，返回结果数组。

```python
numpy.insert(A, indices, B) # 向A数组中的indices位置插入B数组中的元素

```

案例：

```python
import numpy as np
#             0  1  2  3  4  5  6
a = np.array([1, 2, 4, 5, 6, 8, 9])
b = np.array([7, 3])
c = np.searchsorted(a, b)
print(c)
d = np.insert(a, c, b)
print(d)


```

#### 插值

需求：统计各小区彩民买彩票的情况：

| 彩民数量 | 彩票购买量 |
| -------- | ---------- |
| 30       | 100注      |
| 40       | 120注      |
| 50       | 135注      |
| 60       | 155注      |
| 45       | -          |
| 65       | 170注      |

scipy提供了常见的插值算法可以通过一组离散样本生成符合一定规律插值器函数。若我们给插值器函数更多的散点x坐标序列，该函数将会返回相应的y坐标序列。

```python
func = si.interp1d(
    离散水平坐标, 
    离散垂直坐标,
    kind=插值算法(缺省为线性插值)
)

```

案例：

```python
# scipy.interpolate
import scipy.interpolate as si

# 原始数据 11组数据
min_x = -50
max_x = 50
dis_x = np.linspace(min_x, max_x, 11)
dis_y = np.sinc(dis_x)

# 通过一系列的散点设计出符合一定规律插值器函数，使用线性插值（kind缺省值）
linear = si.interp1d(dis_x, dis_y)
lin_x = np.linspace(min_x, max_x, 200)
lin_y = linear(lin_x)

# 三次样条插值 （CUbic Spline Interpolation） 获得一条光滑曲线
cubic = si.interp1d(dis_x, dis_y, kind='cubic')
cub_x = np.linspace(min_x, max_x, 200)
cub_y = cubic(cub_x)

```

#### 积分

直观地说，对于一个给定的正实值函数，在一个实数区间上的定积分可以理解为坐标平面上由曲线、直线以及轴围成的曲边梯形的面积值（一种确定的实数值）。

利用微元法认识什么是积分。

案例：

1. 在[-5, 5]区间绘制二次函数y=2x<sup>2</sup>+3x+4的曲线：

```python
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.patches as mc

def f(x):
    return 2 * x ** 2 + 3 * x + 4

a, b = -5, 5
x1 = np.linspace(a, b, 1001)
y1 = f(x1)
mp.figure('Integral', facecolor='lightgray')
mp.title('Integral', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(x1, y1, c='orangered', linewidth=6,label=r'$y=2x^2+3x+4$', zorder=0)
mp.legend()
mp.show()

```

1. 微分法绘制函数在与x轴还有[-5, 5]所组成的闭合区域中的小梯形。

```python
n = 50
x2 = np.linspace(a, b, n + 1)
y2 = f(x2)
area = 0
for i in range(n):
    area += (y2[i] + y2[i + 1]) * (x2[i + 1] - x2[i]) / 2
print(area)
for i in range(n):
    mp.gca().add_patch(mc.Polygon([
        [x2[i], 0], [x2[i], y2[i]],
        [x2[i + 1], y2[i + 1]], [x2[i + 1], 0]],
        fc='deepskyblue', ec='dodgerblue',
        alpha=0.5))

```



调用scipy.integrate模块的quad方法计算积分：

```python
import scipy.integrate as si
# 利用quad求积分 给出函数f，积分下限与积分上限[a, b]   返回(积分值，最大误差)
area = si.quad(f, a, b)[0]
print(area)
```

#### 图像

scipy.ndimage中提供了一些简单的图像处理，如高斯模糊、任意角度旋转、边缘识别等功能。

```python
import numpy as np
import scipy.misc as sm
import scipy.ndimage as sn
import matplotlib.pyplot as mp
#读取文件
original = sm.imread('../../data/head.jpg', True)

# pip3 install opencv-python==3.4.1 -i http://xx
# import cv2 
# cv2.imread('../xxx.jpg', 0)

#高斯模糊
median = sn.median_filter(original, 21)
#角度旋转
rotate = sn.rotate(original, 45)
#边缘识别
prewitt = sn.prewitt(original)
mp.figure('Image', facecolor='lightgray')
mp.subplot(221)
mp.title('Original', fontsize=16)
mp.axis('off')
mp.imshow(original, cmap='gray')
mp.subplot(222)
mp.title('Median', fontsize=16)
mp.axis('off')
mp.imshow(median, cmap='gray')
mp.subplot(223)
mp.title('Rotate', fontsize=16)
mp.axis('off')
mp.imshow(rotate, cmap='gray')
mp.subplot(224)
mp.title('Prewitt', fontsize=16)
mp.axis('off')
mp.imshow(prewitt, cmap='gray')
mp.tight_layout()
mp.show()

```

#### 金融相关

```python
import numpy as np
# 终值 = np.fv(利率, 期数, 每期支付, 现值)
# 将1000元以1%的年利率存入银行5年，每年加存100元，
# 到期后本息合计多少钱？
fv = np.fv(0.01, 5, -100, -1000)
print(round(fv, 2))
# 现值 = np.pv(利率, 期数, 每期支付, 终值)
# 将多少钱以1%的年利率存入银行5年，每年加存100元，
# 到期后本息合计fv元？
pv = np.pv(0.01, 5, -100, fv)
print(pv)
# 净现值 = np.npv(利率, 现金流)
# 将1000元以1%的年利率存入银行5年，每年加存100元，
# 相当于一次性存入多少钱？
npv = np.npv(0.01, [
    -1000, -100, -100, -100, -100, -100])
print(round(npv, 2))
fv = np.fv(0.01, 5, 0, npv)
print(round(fv, 2))
# 内部收益率 = np.irr(现金流)
# 将1000元存入银行5年，以后逐年提现100元、200元、
# 300元、400元、500元，银行利率达到多少，可在最后
# 一次提现后偿清全部本息，即净现值为0元？
irr = np.irr([-1000, 100, 200, 300, 400, 500])
print(round(irr, 2))
npv = np.npv(irr, [-1000, 100, 200, 300, 400, 500])
print(npv)
# 每期支付 = np.pmt(利率, 期数, 现值)
# 以1%的年利率从银行贷款1000元，分5年还清，
# 平均每年还多少钱？
pmt = np.pmt(0.01, 5, 1000)
print(round(pmt, 2))
# 期数 = np.nper(利率, 每期支付, 现值)
# 以1%的年利率从银行贷款1000元，平均每年还pmt元，
# 多少年还清？
nper = np.nper(0.01, pmt, 1000)
print(int(nper))
# 利率 = np.rate(期数, 每期支付, 现值, 终值)
# 从银行贷款1000元，平均每年还pmt元，nper年还清，
# 年利率多少？
rate = np.rate(nper, pmt, 1000, 0)
print(round(rate, 2))

```

