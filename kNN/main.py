# -*- coding: UTF-8 -*-
import numpy as np
import kNN as knn

# 配置项
k = 20
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'

def test_model():
	pass


def main():
	# train model
	data_matrix, labels = knn.load_data(trainning_dir)
	# plot dataset
	# knn.draw_data(data_matrix, labels)
	training_norm_matrix, diff_arr, min_arr = knn.normalization(data_matrix)
	if is_batch:
		# for batch
		pass	
	else:
		fly_miles = float(input("每年获得的飞行常客里程数:"))
		game_percent = float(input("玩视频游戏所耗时间百分比:"))
		icecream_kilo = float(input("每周消费的冰激淋公升数:"))
		instance_vector = np.array([fly_miles, game_percent, icecream_kilo])
	instance_norm_vector = (instance_vector - min_arr) / diff_arr
	result = knn.classfier(instance_norm_vector, training_norm_matrix, labels, k)
	print('result:\t'+result)
	

if __name__ == '__main__':
	main()