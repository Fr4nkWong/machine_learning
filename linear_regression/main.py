# -*- coding: UTF-8 -*-
import numpy as np
import linear


# 配置项
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'


def main():
    # diy
    dataset, labels = linear.load_data(trainning_dir)  # 训练集
    linear.plot_dataset(dataset, labels)


if __name__ == '__main__':
    main()