import pandas as pd
import matplotlib.pyplot as mp

#用户表读取用户信息
users = pd.read_table('./users.dat',sep='::',names=['UserID','Gender','Age','Occupation','Zip-code'],engine='python')
# 电影评分表
ratings = pd.read_table('./ratings.dat',sep='::',names=['UserID','MovieID','Rating','Timestamp'],engine='python')
# 电影数据表
movies = pd.read_table('./movies.dat',sep='::',names=['MovieID','Title','Genres'],engine='python')
# 合并数据表
data = pd.merge(pd.merge(users,ratings),movies)
# 对数据初步描述分析
print(data.describe().head())
print(data.info())
# 查看每一部电影不同性别的平均评分并计算分歧差值，之后排序
data_gender = data.pivot_table(index='Title',values='Rating',columns='Gender')
data_gender['difference'] = data_gender['F']-data_gender['M']
data_gender = data_gender.sort_values(by='difference',ascending=True)
print(data_gender.head())
# 算出每部电影平均得分并对其进行排序
data_rating = data.pivot_table(index='Title',values='Rating').sort_values(by='Rating',ascending=False)
print(data_rating)
# 查看评分次数多的电影并进行排序
data_rating_number = pd.crosstab(data.Title,data.Rating,margins=True).sort_values(by='All',ascending=False)
print(data_rating_number.head())
# 过滤掉评分条目数不足250条的电影
data_rating_number = data_rating_number[data_rating_number.All>=250]
print(data_rating_number.head())
# 评分最高的十部电影
print(data_rating.head(10))
# 查看不同年龄的分布情况并且采用直方图进行可视化
labels=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79']
data['age_group']=pd.cut(data.Age,range(0,81,10),labels=labels)
data.head()
data['age_group'].value_counts().plot(kind='bar')
mp.xticks(rotation=45)
mp.show()



