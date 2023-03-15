from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# 读取数据集
data = pd.read_csv("../Filter/AllTrainData_ywc_64_features.csv")

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

# 创建神经网络模型
model = Sequential()
model.add(Dense(64, input_dim=63, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
batch_size = 64
history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_split=0.2)

# 输出训练进度
print(history.history)

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print('测试集上的准确率:', test_acc)

# 绘制训练进度的折线图
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# 绘制训练集和测试集的准确率
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 绘制训练集和测试集的损失值
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense
# import matplotlib.pyplot as plt
#
# # 读取数据集
# data = pd.read_csv("data/train.csv")
#
# # 定义输入和输出
# X = data.drop(['emd_lable2'], axis=1).values
# y = data['emd_lable2'].values
#
# # 创建神经网络模型
# model = Sequential()
# model.add(Dense(64, input_dim=51, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# # 编译模型
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # 训练模型
# history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
#
# # 输出训练进度
# print(history.history)
#
# # 绘制训练和验证集的损失和准确率曲线
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
