import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
# 读取 CSV 文件并选取需要处理的列范围
data = pd.read_csv("C:\\Users\\19678\\Desktop\\data\\.ipynb_checkpoints\\train-checkpoint.csv", low_memory=False)


# 创建LabelEncoder对象
le = LabelEncoder()

# 对特征列进行编码转换
# 选择object类型的列（包括字符串类型）
obj_cols = data.select_dtypes(include=['object']).columns

# 将选择的列应用LabelEncoder转换
for col in obj_cols:
    data[col] = le.fit_transform(data[col])


# 打印转换后的数据
print(data.head())

# 将处理后的数据写入新的 CSV 文件中
data.to_csv("C:\\Users\\19678\\Desktop\\data\\.ipynb_checkpoints\\test.csv", index=False)
