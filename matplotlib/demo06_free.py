'''
demo06_free.py 自由布局
'''
import matplotlib.pyplot as mp

mp.figure('Flow Layout', facecolor='lightgray')
mp.axes([0.03, 0.03, 0.94, 0.94])
mp.text(0.5, 0.5, '1', ha='center', va='center', size=36)
mp.plot([1,2],[3,4])

mp.show()

