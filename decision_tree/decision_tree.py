# -*- coding: UTF-8 -*-
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import math

def load_data(dir):
	"""
	加载数据集
	:param dir - 数据集的存放路径
	:return dataset - 处理好的样本列表
	"""
	fr = open(dir)
	line_list = fr.readlines()
	dataset = []
	for line in line_list:
		line = line.strip()
		attr_vector = line.split(' ')
		dataset.append(attr_vector)
	data_matrix = np.array(dataset)
	return data_matrix


def draw_dataset(matrix, labels):
	"""
	数据可视化
	:param matrix - 数据文本的路径 m*n
	:param labels - 分类标注向量
	"""
	pass


def split_data(dataset, axis):
	"""
	划分数据集
	:param dataset - 样例数据集
	:param attr_list - 键表
	:return attr_val2data - 
	"""
	attr_val2data = {}
	for example in dataset:
		example = list(example)
		if example[axis] not in attr_val2data.keys():
			attr_val2data[example[axis]] = []
		attr_val2data[example[axis]].append(example)
	return attr_val2data


def calc_gini(dataset):
	"""
	计算数据集的基尼值
	:param dataset - 样例数据集
	:return gini - 信息熵
	"""
	m = len(dataset)
	label2count = {}
	for attr_vec in dataset:
		label = attr_vec[-1]
		if label not in label2count.keys():
			label2count[label] = 0
		label2count[label] += 1
	gini = 1.0
	for label in label2count:
		p = float(label2count[label]) / m
		gini -= p**2
	return gini

def calc_gini_index(dataset, attr_list, attr):
	"""
	计算属性对应的数据集的基尼指数
	:param dataset - 样例数据集
	:param attr - 指定的属性
	:return gini_index - 基尼指数
	"""
	gini_index = 0.0
	attr_index = attr_list.index(attr)
	dataset_len = len(dataset)
	attr_val2data = split_data(dataset, attr_index)
	for attr in attr_val2data.keys():
		sub_dataset = attr_val2data[attr]
		gini_index += (len(sub_dataset)/dataset_len)*calc_gini(sub_dataset)
	return gini_index


def find_split_attr(dataset, attr_list):
	"""
	选择最优划分属性，采用CART算法
	:param dataset - 样例数据集
	:param attr_list - 属性列表
	:return - split_attr 最优划分属性
	"""
	gini_index_list = []
	for attr in attr_list:
		ginni_index = calc_gini_index(dataset, attr_list, attr)
		gini_index_list.append(ginni_index)
	split_attr = attr_list[gini_index_list.index(min(gini_index_list))]
	return split_attr


def generate_tree(data_matrix, attr_list, split_list=None):
	"""
	递归构建决策树
	:prarm data_matrix - 样例的数据集
	:pram attr_list - 特征列表
	:return tree - 返回决策(子)树
	"""
	label_list = [example[-1] for example in data_matrix]
	if split_list == None:
		split_list = attr_list
	# 数据集属于同一类
	if (label_list.count(label_list[0]) == len(label_list)):	
		return label_list[0]
	# 特征纬度不够，取目前数量最多的类别
	if len(split_list) == 0:		
		label2count = {}
		for label in label_list:
			if label not in label2count.keys():
				label2count[label] = 0
			label2count[label] += 1
		max_count = max(list(label2count.values()))
		for label in label2count.keys():
			if label2count[label] == max_count:
				return label
	# 选择最优划分属性，划分数据集
	split_attr = find_split_attr(data_matrix, attr_list)
	split_index = attr_list.index(split_attr)
	attr_val_set = set(data_matrix[:,split_index].tolist())
	attr_val2data = split_data(data_matrix, split_index)
	tree = [
		# [	
		# 	# attribute
		# 	"xxx", 
		# 	# value
		# 	{
		# 		'yes': [], # element		
		# 		'no': [
		# 			"xxx",
		# 			{
		# 				...
		# 			}
		# 		]	
		# 	}
		# ]
		split_attr,	# attribute name
		{}			# attribute value
	]
	split_list.remove(split_attr)
	for val in list(attr_val_set):
		tree[1][val] = generate_tree(np.array(attr_val2data[val]), attr_list)
	return tree