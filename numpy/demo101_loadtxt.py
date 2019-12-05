"""
demo101_loadtxt.py  加载文件
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt


def dmy2ymd(dmy):
    # 日期转换函数
    dmy = str(dmy, encoding='utf-8')
    time = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
    t = time.strftime('%Y-%m-%d')
    return t

dates, opening_prices, highest_prices, \
    lowest_prices, closing_prices = \
    np.loadtxt(
        '../data/da_data/aapl.csv', delimiter=',',
        usecols=(1, 3, 4, 5, 6),
        dtype='M8[D], f8, f8, f8, f8',
        unpack=True, converters={1: dmy2ymd})

# 画图
mp.figure('AAPL', facecolor='lightgray')
mp.title('AAPL', fontsize=18)
mp.grid(linestyle=':')
mp.xlabel('Date', fontsize=14)
mp.ylabel('Closing Price', fontsize=14)
# 设置刻度定位器
import matplotlib.dates as md
ax = mp.gca()
ax.xaxis.set_major_locator(
    md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(
    md.DateFormatter('%Y/%m/%d'))
# 把dates转成适合mp绘图的格式
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices, color='dodgerblue',
        linewidth=2, linestyle='--', label='closing')



color = [((1,1,1,1) if x else 'green') for x in closing_prices>opening_prices]
edgecolor = [('red' if x else 'green') for x in closing_prices>opening_prices]

mp.bar(
	dates,				# 水平坐标数组
    closing_prices-opening_prices,# 柱状图高度数组
    0.8,			# 柱子的宽度
    opening_prices, # 柱子位置高度
    color=color, 	# 填充颜色
    edgecolor = edgecolor,  # 边框颜色
    zorder = 3      # 图层设置
)

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
