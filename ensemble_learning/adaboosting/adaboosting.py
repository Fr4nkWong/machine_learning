# -*- coding: UTF-8 -*-
import numpy as np
import boosting as bst
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# 配置项
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'


def load_data(dir):
    """
    加载数据集
    :param dir - 数据集的存放路径
    :return dataset - 处理好的样本列表
    """
    fr = open(dir)
    line_list = fr.readlines()
    labels = []
    dataset = []
    for line in line_list:
        line = line.strip()
        attr_list = line.split('\t')
        labels.append(float(attr_list[-1]))
        attr_list = [float(attr) for attr in attr_list[:-1]]
        dataset.append(attr_list)
    return dataset, labels


def main():
    # sklear实现adaboost
    dataset, labels = bst.load_data(trainning_dir)  # 训练集
    testset, test_labels = bst.load_data(trainning_dir) # 测试集
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = "SAMME", n_estimators = 10)
    bdt.fit(dataset, labels)
    predictions = bdt.predict(dataset)
    err_arr = np.mat(np.ones((len(dataset), 1)))
    print('训练集的错误率:%.3f%%' % float(err_arr[predictions != labels].sum() / len(dataset) * 100))
    bdt.fit(testset, test_labels)
    predictions = bdt.predict(testset)
    err_arr = np.mat(np.ones((len(testset), 1)))
    print('测试集的错误率:%.3f%%' % float(err_arr[predictions != test_labels].sum() / len(testset) * 100))

    # 手撕adaboost


if __name__ == '__main__':
    main()