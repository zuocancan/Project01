from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# 设置TensorFlow后端使用GPU加速
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 读取数据集
data = pd.read_csv("../data/Version_zyl_0.0.1.csv")
# data = pd.read_csv("../data/version_0.0.1.csv")

# 定义输入和输出
X = data.drop(['emd_lable2'], axis=1).values
y = data['emd_lable2'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据类型转换
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

# 打印训练集和测试集的形状
print("训练集的形状：", X_train.shape, y_train.shape)
print("测试集的形状：", X_test.shape, y_test.shape)

model = Sequential()
model.add(Dense(512, input_dim=656, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 128# 设置为GPU内存容量所允许的最大值
history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2)

# 输出训练进度
print(history.history)

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print('测试集上的准确率:', test_acc)

# 保存模型
model.save('model_657.h5')

#
# 2023-03-10 14:02:59.562294: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2023-03-10 14:03:00.194061: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2132 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1650 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5
# 2023-03-10 14:03:00.301205: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
# Epoch 1/100