# -*- coding: UTF-8 -*-
import numpy as np
import svm


# 配置项
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'


def main():
    dataset, labels = svm.load_data(trainning_dir)
    b, alphas = svm.smo(dataset, labels, 0.6, 0.001, 40)
    w = svm.clac_w(dataset, labels, alphas)
    svm.plot_data(dataset, labels, w, b, alphas)


if __name__ == '__main__':
    main()