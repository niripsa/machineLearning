import pandas as pd
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatched
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils
# from sklearn.decomposition import PCA


# 训练集
data = pd.read_csv("../train1.csv")
data['Embarked'] = pd.Categorical(data['Embarked']).codes
data['Sex'] = pd.Categorical(data['Sex']).codes
# print(type(data['Age'][5]))
# print(data['Age'][5])
# exit(0)
# <class 'numpy.float64'>
data['Age'] = data['Age'].apply(lambda x: -1 if str(x) == 'nan' else x)
data['Fare'] = data['Fare'].apply(lambda x: (x/512.3292))
# x = data.iloc[:, [0, 2, 4, 5, 6, 7, 9, 10, 11]]
x = data.iloc[:, [2, 4, 5, 6, 7, 9, 11]]
y = data.iloc[:, [1]]
y = np_utils.to_categorical(y, num_classes=2)

# 测试集
data_t = pd.read_csv("../train2.csv")
data_t['Embarked'] = pd.Categorical(data_t['Embarked']).codes
data_t['Sex'] = pd.Categorical(data_t['Sex']).codes
data_t['Age'] = data_t['Age'].apply(lambda x: -1 if str(x) == 'nan' else x)
data_t['Fare'] = data_t['Fare'].apply(lambda x: (x/512.3292))
xt = data_t.iloc[:, [2, 4, 5, 6, 7, 9, 11]]
yt = data_t.iloc[:, [1]]
yt = np_utils.to_categorical(yt, num_classes=2)
# print(xt.info())
# print(x.info())
# exit(0)
# print(xt)
# print(x)
# exit(0)

# 预测
data_test = pd.read_csv("../test.csv")
# print(data_test.info())
# exit(0)
data_test['Embarked'] = pd.Categorical(data_test['Embarked']).codes
data_test['Sex'] = pd.Categorical(data_test['Sex']).codes
data_test['Age'] = data_test['Age'].apply(lambda x: -1 if str(x) == 'nan' else x)
data_test['Fare'] = data_test['Fare'].apply(lambda x: -1 if str(x) == 'nan' else x)
data_test['Fare'] = data_test['Fare'].apply(lambda x: (x/512.3292))
x_test = data_test.iloc[:, [1, 3, 4, 5, 6, 8, 10]]
# y_test = pd.read_csv("../gender_submission.csv").iloc[:, [1]]
# y_test = np_utils.to_categorical(y_test, num_classes=2)

print(x_test.iloc[1:2].shape)
x = x.values
xt = xt.values
# y = y.values
x_test = x_test.values
# y_test = y_test.values

model = Sequential()
model.add(Dense(32, input_dim=7, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print('start ---------------')
model.fit(x, y, epochs=250, batch_size=64)

print('testing ---------------')
loss, accuracy = model.evaluate(xt, yt)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

if accuracy >= 0.80:
    print("success!")
    data_predict = list(model.predict(x_test))
    # print(data_predict[0])
    # print(type(data_predict[0]))
    # print(data_predict[0][1])
    # exit(0)
    predict = []
    for x in range(data_predict.__len__()):
        if data_predict[x][0] > data_predict[x][1]:
            predict.append([x + 892, 0])
        else:
            predict.append([x + 892, 1])
    # print(pre_dict)
    dp = pd.DataFrame(predict, columns=['PassengerId', 'Survived'])
    # print(dp)
    dp.to_csv("../result.csv", index=False)
else:
    print("fail!")



# PCA: Principal component analysis 主成分分析
# pca = PCA(n_components=2)
# x = pca.fit_transform(x)
# print("各方向方差", pca.explained_variance_)
# print("方差所占比例", pca.explained_variance_ratio_)
#
# cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
# cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
# mpl.rcParams['font.sans-serif'] = u'SimHei'
# mpl.rcParams['axes.unicode_minus'] = False
#
# plt.figure(facecolor='w')
# plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='*', cmap=cm_dark)
# plt.grid(b=True, ls=":")
# plt.xlabel(u'组分1', fontsize=14)
# plt.ylabel(u'组分2', fontsize=14)
# plt.title(u'泰坦尼克号数据降维', fontsize=18)
# plt.show()
# 降维失败_(:з」∠)_


