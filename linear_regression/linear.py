import numpy as np
import matplotlib.pyplot as plt
import torch


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


def linear_regression(dataset, labels):
    """
    计算回归
    :param dataset - 权重向量, w0 = b
    :param labels - 属性向量, 常量xo=1
    :return w_list - 结果 n*1
    """
    x_mat = np.mat(dataset) # m*(n+1)
    y_mat = np.mat(labels).T    # m*1
    xTx = x_mat.T * x_mat   # (n+1)*(n+1)
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，无法求逆")
        return
    w_vec = xTx.I * (x_mat.T * y_mat)   # (n+1)*1 analytical solution
    return w_vec


def lwlr(dataset, labels):
    """
    局部加权线性回归
    """
    pass


def plot_dataset(dataset, labels):
    """
    绘制数据集
    :param dataset - 数据集
    :param labels - 标记列表
    """
    data_mat = np.mat(dataset)  # m*(n+1)
    w_vec = linear_regression(dataset, labels) # (n+1)*1
    m = len(dataset)
    x_list = []
    y_list = []
    for i in range(m):
        x_list.append(dataset[i][-1])
        y_list.append(labels[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_list, y_list, s = 20, c = 'blue',alpha = .5)
    ax.plot(x_list, np.dot(data_mat, w_vec), c = 'red')
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()
    