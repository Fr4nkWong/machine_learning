# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(dir):
	"""
	加载&预处理数据集
	:param dir - 数据集的存放路径
	:return dataset - 数据集
	"""
	# 加载&处理数据 (根据需求动态修改)
	fr = open(dir)
	line_list = fr.readlines()
	dataset = []
	labels = []
	index = 0
	for line in line_list:
		attr_list = line.strip().split('\t')
		dataset.append([1.0, float(attr_list[0]), float(attr_list[1])])	# (1,x)
		labels.append(attr_list[-1])
		index += 1
	return dataset, labels


def plot_data(dataset, labels, weight_vec):
	"""
	绘制样本点和决策边界
	:param dataset - 数据集的存放路径
	:param labels - 标记列表
	:param weight_vec - 权重向量
	:return matrix - np矩阵
	"""          
	# 绘制正、负样本点   
	data_matrix = np.mat(dataset)              
	n = np.shape(data_matrix)[0]
	xcord1 = []; ycord1 = []	#正样本
	xcord2 = []; ycord2 = []	#负样本
	for i in range(n):                                                    
		if int(labels[i]) == 1:
			xcord1.append(data_matrix[i,1]); ycord1.append(data_matrix[i,2])
		else:
			xcord2.append(data_matrix[i,1]); ycord2.append(data_matrix[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)	#绘制正样本
	ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)	#绘制负样本
	plt.title('DataSet')                                              
	plt.xlabel('x1'); plt.ylabel('x2')       
	# 绘制决策边界
	x1 = np.arange(-3.0, 3.0, 0.1)	# x1点数组
	weight_list = weight_vec.transpose().tolist()[0]
	x2 = (-weight_list[0] - weight_list[1] * x1) / weight_list[2]	# x2点数组 w0*x0+w1*x1+x2*x2 = y
	ax.plot(x1,x2)

	plt.show()	# 渲染


def sigmoid(x):
    """
	sigmoid函数
	:param x - 结果数据值
	:return sigmoid函数
	"""
    return 1/(1+np.exp(-x))


def grad_ascent(dataset, labels):
	"""
	梯度上升算法
	:param dataset - 数据集 m*n
	:param lables - 标记列表 m*1
	:return weight_vector - 权重向量 n*1 w0 w1 w2
	"""
	data_mat = np.mat(dataset)	# m*n
	label_vec = np.mat(labels, dtype='float').transpose() # m*1
	m, n = np.shape(data_mat)
	alpha = 0.001
	max_cycles = 500
	weight_vector = np.ones((n,1))	# 权重向量 n*1
	for i in range(max_cycles):
		res_vector = sigmoid(np.dot(data_mat, weight_vector))	# 结果向量 m*1
		error_vector = label_vec - res_vector	# 误差向量 m*1
		weight_vector = weight_vector + alpha * data_mat.transpose() * error_vector
	return weight_vector # numerical solution


def stoc_grad_ascent(dataset, labels, max_cycles=150):
	"""
	随机梯度上升
	:param dataset - 数据集 m*n
	:param labels - 标记列表 n*1
	:param max_cycles - 迭代次数
	:return weight_vector - 权重向量 n*1
	"""
	data_matrix = np.mat(dataset)
	m, n = np.shape(data_matrix)
	weight_vec = np.ones((n,1))
	for j in range(max_cycles):
		sample_index_list = list(range(m))
		for i in range(m):
			alpha = 4/(i+j+1.0) + 0.01
			random_index = int(random.uniform(0, len(sample_index_list)))
			sample_index = sample_index_list[random_index]	# 随机获取样本点的下标
			y = sigmoid(sum(np.dot(data_matrix[sample_index], weight_vec)))
			error = float(labels[sample_index]) - y.tolist()[0][0]
			print('error', error)
			weight_vec = weight_vec + alpha * error * data_matrix[sample_index].transpose()
			del(sample_index_list[random_index])
	return weight_vec


def classifier(dataset, weighet_vec):
	"""
	分类器
	:param dataset - 数据矩阵 m*n
	:param weighet_vec - 标记列表 n*1
	:return labels - 标记列表
	"""
	data_mat = np.mat(dataset)
	res_arr = np.array(np.dot(data_mat, weighet_vec))
	labels = []
	for i in range(len(res_arr)):
		if res_arr[0][i] >= 0.5:
			labels.append(1)
		else:
			labels.append(0)
	return labels


# 配置项
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'


if __name__ == '__main__':
	# logistic regression
    # train model
	dataset, labels = lr.load_data(trainning_dir)
	weight_vec = lr.stoc_grad_ascent(dataset, labels)
	# plot
	lr.plot_data(dataset, labels, weight_vec)
	# test
	testset = lr.load_data(testing_dir)
	res = lr.classifier(testset[0], weight_vec)
	print(weight_vec)
	print(res)