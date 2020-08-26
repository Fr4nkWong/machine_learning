# -*- coding: UTF-8 -*-
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


def get_vocab_list(dataset):
	"""
	生成词汇表
	:param dataset - 数据集
	:param labels - 分类标注向量 m*1
	:return vocab_list - 词汇表 1*n
	"""
	vocab_set = set()
	for doc in dataset:
		vocab_set = vocab_set | set(doc)
	vocab_list = list(vocab_set)
	return vocab_list


def get_vocab_vec(word_list, vocab_list):
	"""
	生成词汇向量
	:param word_list - 词条
	:param vocab_list - 词汇表 1*n
	:return vocab_vec - 词汇向量，用词汇表表示词条中词汇出现情况 1*n
	"""
	vocab_vec = [0]*len(vocab_list)
	for word in word_list:
		if word in vocab_list:
			vocab_vec[vocab_list.index(word)] = 1
	return vocab_vec


def load_data(dir):
	"""
	加载&预处理数据集
	:param dir - 数据集的存放路径
	:return dataset - 数据集
	:return vocab_list - 词汇表 1*n
	"""
	# 加载&处理数据 (根据需求动态修改)
	fr = open(dir)
	line_list = fr.readlines()
	dataset = []
	for line in line_list:
		line = line.strip()
		attr_vector = line.split(' ')
		dataset.append(attr_vector)
	labels = [0,1,0,1,0,1]
	return dataset, labels



def handle_data(dataset):
	"""
	处理数据集
	:param dir - 数据集的存放路径
	:return vocab_list - 词汇表 1*n
	:return vocab_vec_list - 词汇向量组 m*n
	"""
	# 生成词汇表
	vocab_list = get_vocab_list(dataset)
	# 生成词汇向量组
	vocab_vec_list = []
	for line in dataset:
		vocab_vec_list.append(get_vocab_vec(line, vocab_list))
	return vocab_list, vocab_vec_list


def train_classifier(vec_list, labels):
	"""
	训练分类器，采用朴素贝叶斯算法
	:param vec_list - 向量组，即矩阵 m*n
	:param labels - 分类标注向量 m*1
	:return p0_vocab_vec - 非侮辱类的条件概率向量
	:return p1_vocab_vec - 侮辱类的条件概率向量
	:return p1 - 训练集属于侮辱类的概率
	"""
	num_vec = len(vec_list)	# 词汇向量的数量
	num_vocab = len(vec_list[0])	# 词汇表的长度
	# 平滑处理
	pa = (sum(labels)+1)/(float(num_vec)+2)	# P(c) = ｜Dc｜/|D|
	p0_vocab_num_vec = np.ones(num_vocab)
	p1_vocab_num_vec = np.ones(num_vocab)
	for i in range(num_vec):
		if labels[i] == 1:
			p1_vocab_num_vec += vec_list[i]
		else:
			p0_vocab_num_vec += vec_list[i]
	# P(xi|c) 采用log防止下溢出
	p1_label_num = sum(labels)
	p0_label_num = len(labels)-p1_label_num
	p0_vocab_vec = np.log(p0_vocab_num_vec/float(p0_label_num+2))
	p1_vocab_vec = np.log(p1_vocab_num_vec/float(p1_label_num+2))
	print(f'p0_vocab_vec: \n{p0_vocab_vec}')
	print(f'p1_vocab_vec: \n{p1_vocab_vec}')
	return p0_vocab_vec, p1_vocab_vec, pa


def classify(vocab_vec_list, p0_vec, p1_vec, pa):
	"""
	分类函数
	:param vocab_vec_list - 单词向量组
	:param p0_vec - 各单词对应的概率向量
	:param p1_vec - 各单词对应的概率向量
	:param pa - 分类标注向量
	:return label - 标记
	"""
	print(vocab_vec_list)
	# P(c|x) ～ P(c)*P(x|c)
	p0 = sum(p0_vec*vocab_vec_list) + np.log((1-pa))
	p1 = sum(p1_vec*vocab_vec_list) + np.log(pa)
	print(f'p0: {p0}')
	print(f'p1: {p1}')
	if p0 < p1:
		return 1
	else:
		return 0