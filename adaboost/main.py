# -*- coding: UTF-8 -*-
import numpy as np
import boosting as bst


# 配置项
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'


def main():
    dataset, labels = bst.load_data(trainning_dir)


if __name__ == '__main__':
    main()