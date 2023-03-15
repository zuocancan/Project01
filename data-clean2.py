import pandas as pd

# 读取 CSV 文件并选取需要处理的列
df = pd.read_csv("C:\\Users\\19678\\Desktop\\data\\.ipynb_checkpoints\\test5.csv", low_memory=False)

# 将birth_date和 recent_flt_day中的年份分离开来
df['birth_date'] = df['birth_date'].str.split('/').str[0]

df['recent_flt_day'] = df['recent_flt_day'].str.split('/').str[0]

# 将取得的年份变成整数类型
df['birth_date'] = df['birth_date'].astype(int)

df['recent_flt_day'] = df['recent_flt_day'].astype(int)
# 将处理后的数据写入新的 CSV 文件中
df.to_csv("C:\\Users\\19678\\Desktop\\data\\.ipynb_checkpoints\\test6.csv", index=False)
