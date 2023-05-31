import random
import numpy as np
# from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pandas as pd 
import scipy.sparse as sp
import torch.utils.data as data

import math
import datasets_config


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

# 这个函数的意义是什么
# class DataLoaderHelper(object):
#     def __init__(self, dataloader):
#         self.loader = dataloader
#         self.dataiter = iter(self.loader)

#     def __next__(self):
#         try:
#             data, target = next(self.dataiter)
#         except StopIteration:
#             self.dataiter = iter(self.loader)
#             data, target = next(self.dataiter)
        
#         return data, target

# 为 DataLoader 定义next函数
class DataLoaderHelper(object):
    def __init__(self, dataloader):
        self.loader = dataloader
        self.dataiter = iter(self.loader)
        
        if self.loader.dataset.is_training:
            self.loader.dataset.ng_sample()

    def __next__(self):
        try:
            user, item, label = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            user, item, label = next(self.dataiter)
            
            if self.loader.dataset.is_training:
                self.loader.dataset.ng_sample()
        
        return user, item, label
    

def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=True, num_workers=4):
    if selected_idxs == None:
        dataloader = data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    else:
        partition = Partition(dataset, selected_idxs)
        dataloader = data.DataLoader(partition, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    
    return DataLoaderHelper(dataloader)


def load_all(test_num=1000):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(
		datasets_config.train_rating, 
		# nrows = test_num,
		sep='\t', header=None, names=['user', 'item'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1

	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	test_data = []
	with open(datasets_config.test_negative, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split('\t')
			u = eval(arr[0])[0]
			test_data.append([u, eval(arr[0])[1]])
			for i in arr[1:]:
				test_data.append([u, int(i)])
			line = fd.readline()
	# return train_data, test_data[:user_num * 100], user_num, item_num, train_mat
	return train_data, test_data, user_num, item_num, train_mat

# 每个客户端分配 相同数目用户 的数据
def simple_partition(train_data, test_data, user_num, item_num, worker_num):
    train_data_list = [[] for _ in range(worker_num)]
    test_data_list = [[] for _ in range(worker_num)]
    train_mat_list = [sp.dok_matrix((user_num, item_num), dtype=np.float32) for _ in range(worker_num)]
    
    user_step = math.ceil(user_num / worker_num)
    user_threshold = user_step
    list_index = 0
    for user, item in train_data:
        if user >= user_threshold:
            user_threshold += user_step
            list_index += 1
        train_data_list[list_index].append([user, item])
        train_mat_list[list_index][user, item] = 1.0

    user_threshold = user_step
    list_index = 0
    for user, item in test_data:
        if user >= user_threshold:
            user_threshold += user_step
            list_index += 1
        test_data_list[list_index].append([user, item])
        
    return train_data_list, test_data_list, train_mat_list

def load_datasets(worker_num, worker_rank, num_ng=0):
    train_data, test_data, user_num, item_num, train_mat = load_all()
    train_data_list, test_data_list, train_mat_list = simple_partition(train_data, test_data, user_num, item_num, worker_num)
    index = worker_rank - 1 
    train_dataset = NCFData(train_data_list[index], item_num, train_mat_list[index], num_ng, is_training=True)
    test_dataset = NCFData(test_data_list[index], item_num, None, 0, is_training=False)
    return train_dataset, test_dataset, user_num, item_num

def create_dataloaders(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    # return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return DataLoaderHelper(data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers))

'''
这段代码定义了一个继承自torch.utils.data.Dataset的NCFData类，用于加载NCF模型训练所需的数据集。

在__init__函数中，该类初始化了训练集/测试集的特征features、物品总数num_item、训练集中物品的交互记录train_mat、负样本采样数目num_ng和是否为训练集is_training。

ng_sample函数用于进行负采样。在训练集上，对于每个正例（u, i），根据交互记录train_mat在物品总数num_item中随机采样num_ng个负例（u, j），满足（u, j）不在train_mat中，以构建负样本数据集features_ng和对应的标签labels_ng。最终，将正例数据集features_ps和负例数据集features_ng和对应的标签labels_ps和labels_ng进行合并，构成最终的特征集features_fill和标签集labels_fill。

__len__函数返回数据集的总大小，即样本数目。

__getitem__函数用于查询某个样本。对于训练集，返回对应索引idx在特征集features_fill和标签集labels_fill中对应的用户、物品和标签。对于测试集，返回对应索引idx在特征集features_ps和标签集labels中对应的用户、物品和标签。'''
class NCFData(data.Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]	# test_data不需要 labels
		# self.labels = [1] + [0 for _ in range(1, len(features))]	# 如果要用 test_loss
		# self.features_fill = None
		# self.labels_fill = None

	# 对每个正例进行负采样
	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'
        
		print("begin ng_sampling")
		self.features_ng = []
        
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.features_fill = self.features_ps + self.features_ng
		self.labels_fill = labels_ps + labels_ng

	# 标签数据集总大小
	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	# 查询某样本，包括用户、物品和标签
	def __getitem__(self, idx):
		features = self.features_fill if self.is_training \
					else self.features_ps
		labels = self.labels_fill if self.is_training \
					else self.labels

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		return user, item, label