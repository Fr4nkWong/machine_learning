import numpy as np


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


def adaboost():
    pass