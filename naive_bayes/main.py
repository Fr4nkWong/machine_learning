# -*- coding: UTF-8 -*-
import numpy as np
import naive_bayes as nb

# 配置项
is_batch = 0
trainning_dir = './training.txt'
testing_dir = './testing.txt'


def main():
	# handle data
	dataset, labels = nb.load_data(trainning_dir)
	vocab_list, vocab_vec_list = nb.handle_data(dataset)
	# train model
	p0_vec, p1_vec, pa = nb.train_classifier(vocab_vec_list, labels)
	# test
	test_dataset, test_labels = nb.load_data(testing_dir)
	test_vocab_vec_list = []
	for line in test_dataset:
		test_vocab_vec_list.append(nb.get_vocab_vec(line, vocab_list))
	res_vec = nb.classify(np.array(test_vocab_vec_list[0]), np.array(p0_vec), np.array(p1_vec), pa)
	if res_vec:
		print("侮辱类")
	else:
		print("非侮辱类")
	

if __name__ == '__main__':
	main()
