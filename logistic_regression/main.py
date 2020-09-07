# -*- coding: UTF-8 -*-
import numpy as np
import logistic_regression as lr


# 配置项
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'


def main():
    # train model
	dataset, labels = lr.load_data(trainning_dir)
	weight_vec = lr.grad_ascent(dataset, labels)
	# plot
	lr.plot_data(dataset, labels, weight_vec)
	# test
	testset = lr.load_data(testing_dir)
	res = lr.classifier(testset[0], weight_vec)
	print(res)


if __name__ == '__main__':
    main()