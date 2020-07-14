import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties

"""
Description: 装载样本数据
Parameters:
	dir - 数据文本的路径
Returns:
	data_matrix - 数据矩阵 m*n
	labels - 分类的标签向量
"""
def load_data(dir):
	fr = open(dir)
	line_list = fr.readlines()
	data_matrix = np.zeros((len(line_list),3))
	labels = []
	index = 0
	for line in line_list:
		line = line.strip()
		attr_list = line.split('\t')
		data_matrix[index,:] = attr_list[0:3]
		label = attr_list[-1]
		if label == 'didntLike':
			labels.append(1)
		elif label == 'smallDoses':
			labels.append(2)
		elif label == 'largeDoses':
			labels.append(3)
		index += 1
	return data_matrix, labels


"""
Description: 数据可视化
Parameters:
	matrix - 数据文本的路径 m*n
	labels - 分类标注向量
Returns:
	None
"""
def draw_data(matrix, labels):
	font = FontProperties(fname=r"/System/Library/Fonts/STHeiti Medium.ttc", size=16)
	fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))
	labels_len = len(labels)
	labels_colour = []
	for i in labels:
		if i == 1:
			labels_colour.append('black')
		elif i == 2:
			labels_colour.append('blue')
		elif i == 3:
			labels_colour.append('red')

	# 左上区域，散点图，图的x和y轴
	axs[0][0].scatter(x=matrix[:,0], y=matrix[:,1], color=labels_colour, s=16, alpha=0.5)
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
	plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
	plt.setp(axs0_xlabel_text, size=9, weight='bold', color='black') 
	plt.setp(axs0_ylabel_text, size=9, weight='bold', color='black')

	# 设置、添加图例
	didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='blue', marker='.', markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	plt.show()


"""
Description: 数据归一化
Parameters:
	matrix - 数据矩阵 样本个数*特征值个数
Returns:
	norm_matrix - 归一化矩阵
	diff_arr - 各特征对应的差值向量
	min_arr - 各特征对应的最小值向量
"""
def normalization(matrix):
	# 矩阵行对应的最小值
	min_arr = matrix.min(0)
	max_arr = matrix.max(0)
	diff_arr = max_arr - min_arr
	m,n = np.shape(matrix)
	norm_matrix = (matrix - np.tile(min_arr, (m,1))) / (np.tile(diff_arr, (m,1)))
	return norm_matrix, diff_arr, min_arr


"""
Description: 分类器
Parameters:
	instance_matrix	- 输入矩阵
	trainning_matrix - 训练矩阵 样本个数*特征值个数
	labels - 分类，标记向量
	k - 选出距离目标点最近的k个样本点，k<=20
Returns:
	group - 数据集
	labels - 分类，标记向量
"""
def classfier(instance_vector, trainning_matrix, labels, k):
	label2class = {1: 'didntLike', 2: 'smallDoses', 3: 'largeDoses'}
	count_vector = [0, 0, 0]
	m,n = np.shape(trainning_matrix)
	d_vector = [0, 0, 0]
	for i in range(m):
		d_vector.append(distance(instance_vector, trainning_matrix[i]))
	index_list = np.argsort(d_vector)
	index_list = index_list[:k]
	for index in index_list:
		label = labels[index]
		if label == 1:
			count_vector[0] += 1
		elif label == 2:
			count_vector[1] += 1
		elif label == 3:
			count_vector[2] += 1
	max_count = max(count_vector)
	label = label2class[count_vector.index(max_count)+1]
	return label


def distance(vector, vector0):
	n = len(vector0)
	sum = 0
	for i in range(n):
		sum += (vector[i] - vector0[i])**2
	d = sum**0.5
	return d