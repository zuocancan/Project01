import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np


# 读取 CSV 文件并选取需要处理的列
data = pd.read_csv(r"D:\Project01\data-test\data\test.csv", low_memory=False)

# 创建LabelEncoder对象
le = LabelEncoder()

# 将错误值中国替换成0000/00/00格式
data['birth_date'] = data['birth_date'].replace(['中国'], '0000/00/00')
# 将birth_date和 recent_flt_day中的年份分离开来
data['birth_date'] = data['birth_date'].str.split('/').str[0]
data['recent_flt_day'] = data['recent_flt_day'].str.split('/').str[0]

# 将取得的年份变成整数类型
data['birth_date'] = data['birth_date'].astype(int)

data['recent_flt_day'] = data['recent_flt_day'].astype(int)

# 对特征列进行编码转换
# 指定保留的列
cols_to_keep = ['birth_date', 'seg_dep_time', 'recent_flt_day']

# 获取要处理的列名列表并转换
for col in data.columns:
    if col not in cols_to_keep and data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# 将老的字段名替换成新的字段名
data.rename(columns={'ffp_nbr': 'ffp_cfm'}, inplace=True)

# 找到要替换的列，并把其中除0以外的其他数字都替换成1
data['ffp_cfm'] = pd.to_numeric(data['ffp_cfm'], errors='coerce')
data['ffp_cfm'] = np.where(data['ffp_cfm'] != 0, 1, data['ffp_cfm'])

# 将seg_dep_time中的时间段分离开来
data['seg_dep_time'] = data['seg_dep_time'].str.split(' ').str[1].str.split(':').str[0]

# 将取得的时间变成整数类型
data['seg_dep_time'] = data['seg_dep_time'].astype(int)

# 查看未去重前的行数和列数
print(data.shape)

# 去重
data.drop_duplicates(inplace=True)

# 查看去重之后的行数和列数
print(data.shape)

# 将处理后的数据写入新的 CSV 文件中
data.to_csv(r"D:\Project01\data-test\data\test-cleaned.csv", index=False)



