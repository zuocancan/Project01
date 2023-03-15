import pandas as pd


# 读取 CSV 文件并选取需要处理的列
df = pd.read_csv(r"D:\Project01\data\.ipynb_checkpoints\test7.csv", low_memory=False)

# 查看未去重前的行数和列数
print(df.shape)

# 去重
df.drop_duplicates(inplace=True)

# 查看去重之后的行数和列数
print(df.shape)
# 将处理后的数据写入新的 CSV 文件中
df.to_csv(r"D:\Project01\data\.ipynb_checkpoints\test8.csv", index=False)