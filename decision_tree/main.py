# -*- coding: UTF-8 -*-
import numpy as np
import decision_tree as dt

# 配置项
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'
attr_list = ['年龄', '有工作', '有自己的房子', '信贷情况']

def main():
	dataset = dt.load_data(trainning_dir)
	decision_tree = dt.generate_tree(dataset, attr_list)
	print(decision_tree)
	

if __name__ == '__main__':
	main()
