"""
demo05_polyfit.py   多项式拟合
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
# bhp
dates, bhp_closing_prices = np.loadtxt(
    '../data/da_data/bhp.csv', delimiter=',',
    usecols=(1, 6), dtype='M8[D], f8',
    unpack=True, converters={1: dmy2ymd})
# vale
vale_closing_prices = np.loadtxt(
    '../data/da_data/vale.csv', delimiter=',',
    usecols=(6,))


# 画图
mp.figure('Polyfit', facecolor='lightgray')
mp.title('Polyfit', fontsize=18)
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

# 计算差价，绘制曲线
diff_prices = bhp_closing_prices - vale_closing_prices
mp.plot(dates, diff_prices, color='dodgerblue',
        linewidth=2, label='Diff Prices')
# 多项式拟合
times = dates.astype('M8[D]').astype('i4')
P = np.polyfit(times, diff_prices, 4)
pred_y = np.polyval(P, times)
mp.plot(dates, pred_y, color='orangered',
        linewidth=2, label='Polyfit Line')


mp.legend()
mp.gcf().autofmt_xdate()
mp.show()