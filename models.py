import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


# 加载数据
data = pd.read_csv(r'D:\Project01\ywc-data\encoded_train.csv')

# 划分训练集和测试集
X = data.drop(['emd_lable2', 'emd_lable2'], axis=1).values
y = data['emd_lable2'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

"""
param = {'n_estimators': np.arange(100, 200, 10)}
dac = xgb.XGBClassifier(learning_rate=0.02,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0.1, subsample=1.0)
gs = GridSearchCV(dac, param_grid=param, scoring='roc_auc', cv=5)
gs.fit(X_train, y_train)
print('Best score:%0.3f' % gs.best_score_)
print('Best parameters set:%s' % gs.best_params_)
"""
param = {'max_depth': np.arange(3, 10, 1), 'min_child_weight': np.arange(0.1, 1, 0.1)}
model = xgb.XGBClassifier(n_estimators=190,
                          learning_rate=0.01,
                          gamma=0.1,
                          subsample=1.0)
gs = GridSearchCV(model, param_grid=param, scoring='roc_auc', cv=5)
gs.fit(X_train, y_train)
print("Best score:%0.3f" % gs. best_score_)
print("Best parameters set:%s" % gs. best_params_)

# 定义模型
# model = Sequential()
# model.add(Dense(64, input_dim=656, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# 编译模型
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 评估模型
# score = model.evaluate(X_test, y_test)
# print("测试的损失和准确率为", score)

# 保存模型
# model.save('model-ywc.joblib')

# 预测新乘客的选座概率
# new_passenger = np.random.rand(1, 656)
# prediction = model.predict(new_passenger)
# print("新乘客的选座概率为", prediction)