import pandas as pd

#用户表读取用户信息
users = pd.read_table('./users.dat',sep='::',names=['UserID','Gender','Age','Occupation','Zip-code'],engine='python')
# 电影评分表
ratings = pd.read_table('./ratings.dat',sep='::',names=['UserID','MovieID','Rating','Timestamp'],engine='python')
# 电影数据表
movies = pd.read_table('./movies.dat',sep='::',names=['MovieID','Title','Genres'],engine='python')
# 合并数据表
data = pd.merge(pd.merge(users,ratings),movies)
# 对数据初步描述分析
# print(data.describe().head(5))
# print(data.info())
# 查看每一部电影不同性别的平均评分并计算分歧差值，之后排序
data_gender = data.pivot_table(index=['Title','UserID'],values='MovieID')
print(data_gender)

