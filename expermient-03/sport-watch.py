import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#  编写数据导入函数，设置传入两个参数，分别是特征文件的列表feature_paths和标签
# 文件的列表label_paths。
#  定义feature数组变量，列数量和特征维度一致为41；定义空的标签变量，列数量与标
# 签维度一致为1。
def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0,41))
    label = np.ndarray(shape=(0,1))
    #  使用pandas库的read_table函数读取一个特征文件的内容，其中指定分隔符为逗号、缺失值为问号且
    # 文件不包含表头行。
    #  使用Imputer函数，通过设定strategy参数为‘mean’，使用平均值对缺失数据进行补全。fit()
    # 函数
    # 用于训练预处理器，transform()
    # 函数用于生成预处理结果。
    #  将预处理后的数据加入feature，依次遍历完所有特征文件
    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))
    #  遵循与处理特征文件相同的思想，我们首先使用pandas库的read_table函数读取一个标签文件的内容，
    # 其中指定分隔符为逗号且文件不包含表头行。
    #  由于标签文件没有缺失值，所以直接将读取到的新数据加入label集合，依次遍历完所有标签文件，得
    # 到标签集合label。
    #  最后函数将特征集合feature与标签集合label返回。
    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))

    label = np.ravel(label)
    return feature, label

#  设置数据路径feature_paths和label_paths。
#  使用python的分片方法，将数据路径中的前4个值作为训练集，并作为参数传入load_dataset()函数中，
# 得到训练集合的特征x_train，训练集的标签y_train。
#  将最后一个值对应的数据作为测试集，送入load_dataset()函数中，得到测试集合的特征x_test，测试
# 集的标签y_test。
#
# 使用train_test_split()函数，通过设置测试集比例test_size为0，将数据随机打乱，便于后续分类
# 器的初始化和训练
if __name__ == '__main__':
    ''' 数据路径 '''
    featurePaths = ['A/A.feature','B/B.feature','C/C.feature','D/D.feature','E/E.feature']
    labelPaths = ['A/A.label','B/B.label','C/C.label','D/D.label','E/E.label']
    ''' 读入数据  '''
    x_train,y_train = load_datasets(featurePaths[:4],labelPaths[:4])
    x_test,y_test = load_datasets(featurePaths[4:],labelPaths[4:])
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 0.0)
    #  使用默认参数创建K近邻分类器，并将训练集x_train和y_train送入fit()
    # 函数进行训练，训练后的分类
    # 器保存到变量knn中。
    #  使用测试集x_test，进行分类器预测，得到分类结果answer_knn
    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')
    #  使用默认参数创建决策树分类器dt，并将训练集x_train和y_train送入fit()
    # 函数进行训练。训练后的分
    # 类器保存到变量dt中。
    #  使用测试集x_test，进行分类器预测，得到分类结果answer_dt。
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print('Prediction done')
    #  使用默认参数创建贝叶斯分类器，并将训练集x_train和y_train送入fit()
    # 函数进行训练。训练后的分类
    # 器保存到变量gnb中。
    #  使用测试集x_test，进行分类器预测，得到分类结果answer_gnb。
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))