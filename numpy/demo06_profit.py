"""
demo06_profit.py  数据平滑
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md


def dmy2ymd(dmy):
    # 日期转换函数
    dmy = str(dmy, encoding='utf-8')
    time = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
    t = time.strftime('%Y-%m-%d')
    return t

dates, bhp_closing_prices = np.loadtxt(
    '../data/da_data/bhp.csv', delimiter=',',
    usecols=(1, 6), dtype='M8[D], f8',
    converters={1: dmy2ymd}, unpack=True)

vale_closing_prices = np.loadtxt(
    '../data/da_data/vale.csv', delimiter=',',
    usecols=(6,))

# 求收益率，绘制收益率曲线
bhp_returns = np.diff(bhp_closing_prices) /\
    bhp_closing_prices[:-1]
vale_returns = np.diff(vale_closing_prices) /\
    vale_closing_prices[:-1]
# 画图
mp.figure('BHP VALE RETURNS', facecolor='lightgray')
mp.title('BHP VALE RETURNS', fontsize=20)
mp.xlabel('Date')
mp.ylabel('Price')
mp.grid(linestyle=':')
ax = mp.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(md.DateFormatter('%Y %m %d'))
dates = dates.astype(md.datetime.datetime)[1:]
# 绘制收益线
mp.plot(dates, bhp_returns, color='dodgerblue',
        linestyle='--', label='bhp_returns', alpha=0.3)
mp.plot(dates, vale_returns, color='orangered',
        linestyle='--', label='vale_returns', alpha=0.3)

# 卷积降噪
kernel = np.hanning(8)
kernel /= kernel.sum()
bhp_returns_convolved = np.convolve(
    bhp_returns, kernel, 'valid')
vale_returns_convolved = np.convolve(
    vale_returns, kernel, 'valid')
mp.plot(dates[7:], bhp_returns_convolved, color='dodgerblue',
        linestyle='-', label='bhp_returns_convolved',
        alpha=0.3)
mp.plot(dates[7:], vale_returns_convolved, color='orangered',
        linestyle='-', label='vale_returns_convolved',
        alpha=0.3)

# 多项式拟合
times = dates[7:].astype('M8[D]').astype('i4')
bhp_p = np.polyfit(times, bhp_returns_convolved, 3)
vale_p = np.polyfit(times, vale_returns_convolved, 3)
bhp_polyvals = np.polyval(bhp_p, times)
vale_polyvals = np.polyval(vale_p, times)
mp.plot(dates[7:], bhp_polyvals, color='dodgerblue',
        linestyle='-', label='bhp_polyvals')
mp.plot(dates[7:], vale_polyvals, color='orangered',
        linestyle='-', label='vale_polyvals')

# 计算两多项式函数的交点
diff_p = np.polysub(bhp_p, vale_p)
xs = np.roots(diff_p)
print(xs.astype('M8[D]'))


mp.legend()
mp.show()