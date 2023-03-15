import pandas as pd
import numpy as np

# 读取数据集
data = pd.read_csv(r'C:\Users\19678\Desktop\data\.ipynb_checkpoints\test4.csv')

# 将老的字段名替换成新的字段名
data.rename(columns={'ffp_nbr': 'ffp_cfm'}, inplace=True)

# 找到要替换的列，并把其中除0以外的其他数字都替换成1
data['ffp_cfm'] = pd.to_numeric(data['ffp_cfm'], errors='coerce')
data['ffp_cfm'] = np.where(data['ffp_cfm'] != 0, 1, data['ffp_cfm'])

# 更新csv
data.to_csv(r"C:\Users\19678\Desktop\data\.ipynb_checkpoints\test5.csv", index=False)
