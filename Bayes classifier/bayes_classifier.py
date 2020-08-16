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