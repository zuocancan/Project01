import pandas as pd


# 读取 CSV 文件并选取需要处理的列
df = pd.read_csv(r"D:\Project01\data\.ipynb_checkpoints\test6.csv", low_memory=False)

# 将seg_dep_time中的时间段分离开来
df['seg_dep_time'] = df['seg_dep_time'].str.split(' ').str[1].str.split(':').str[0]

# 将取得的时间变成整数类型
df['seg_dep_time'] = df['seg_dep_time'].astype(int)

# 将处理后的数据写入新的 CSV 文件中
df.to_csv(r"D:\Project01\data\.ipynb_checkpoints\test7.csv", index=False)
