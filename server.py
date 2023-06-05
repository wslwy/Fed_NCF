from fileinput import filename
import imp
import os
import sys
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import time
from tkinter.tix import Tree
import numpy as np
import threading
import torch
import copy
import math
from config import *
import torch.nn.functional as F
import datasets, models
from training_utils import test

from mpi4py import MPI

import logging


# #init parameters
# parser = argparse.ArgumentParser(description='Distributed Client')
# parser.add_argument('--dataset_type', type=str, default='CIFAR10')
# # parser.add_argument('--model_type', type=str, default='AlexNet') # 修改模型参数
# parser.add_argument('--model_type', type=str, default='GMF')
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--data_pattern', type=int, default=0)
# parser.add_argument('--lr', type=float, default=0.1)
# parser.add_argument('--decay_rate', type=float, default=0.99)
# parser.add_argument('--min_lr', type=float, default=0.001)
# parser.add_argument('--epoch', type=int, default=500)   # 运行epoch数，原定为500
# parser.add_argument('--momentum', type=float, default=-1)
# parser.add_argument('--weight_decay', type=float, default=0.0)
# parser.add_argument('--data_path', type=str, default='/data/docker/data')
# parser.add_argument('--use_cuda', action="store_false", default=True)

#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
# parser.add_argument('--model_type', type=str, default='AlexNet') # 修改模型参数
parser.add_argument('--model_type', type=str, default='GMF')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=1000)   # 运行epoch数，原定为500
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='/data/docker/data')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

RESULT_PATH = os.getcwd() + '/server/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

# init logger (获取当前文件名称不包含后缀，创建日志记录器)
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

def main():
    logger.info("csize:{}".format(int(csize)))
    logger.info("server start (rank):{}".format(int(rank)))
    # init config (按需增删)
    common_config = CommonConfig()
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.data_pattern=args.data_pattern
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.min_lr=args.min_lr
    common_config.epoch = args.epoch
    common_config.momentum = args.momentum
    common_config.data_path = args.data_path
    common_config.weight_decay = args.weight_decay

    worker_num = int(csize) - 1

    # 全局模型参数数量
    # global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    # init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    # common_config.para_nums=init_para.nelement()
    # model_size = init_para.nelement() * 4 / 1024 / 1024
    # logger.info("para num: {}".format(common_config.para_nums))
    # logger.info("Model Size: {} MB".format(model_size))

    # 全局模型参数数量改(完整模型)
    user_num, item_num = 10, 3706  # 3706行
    # item_num =  # 所有行
    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type, user_num=user_num, item_num=item_num, factor_num=8)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    common_config.para_nums = init_para.nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    logger.info("para num: {}".format(common_config.para_nums))
    logger.info("Model Size: {} MB".format(model_size))

    # 提取需要分享的参数
    if common_config.model_type == 'GMF':
        params_to_share = {}
        params_to_share['item_embeddings.weight'] = global_model.embed_item_GMF.weight.data
        params_to_share['predict_layer.weight'] = global_model.predict_layer.weight.data
        params_to_share['predict_layer.bias'] = global_model.predict_layer.bias.data
    
    # 将分享的参数设为该字典
    init_para = params_to_share

    # create workers （设置客户端参数，但尚未发送到客户端）
    worker_list: List[Worker] = list()
    for worker_idx in range(worker_num):
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config), rank=worker_idx+1)
        )
    #到了这里，worker已经启动了

    # Create model instance
    # partition放到本地，也许可以在服务器决定策略
    # train_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern)

    # 第一次发送全局模型，item嵌入及网络部分
    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
    #     # worker.config.train_data_idxes = train_data_partition.use(worker_idx)

    # # connect socket and send init config (初始配置信息中拥有模型参数)
    communication_parallel(worker_list, 1, comm, action="init")

    # # recoder: SummaryWriter = SummaryWriter()
    # # 全局不作测试
    global_model.to(device)
    # # _, test_dataset = datasets.load_datasets(common_config.dataset_type,common_config.data_path)
    # # test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    # 没有验证集
    for epoch_idx in range(1, 1+common_config.epoch):
        logger.info("get begin")
        communication_parallel(worker_list, epoch_idx, comm, action="get_model")
        logger.info("get end")
        global_para = aggregate_model_para(global_model, worker_list)
        # logger.info(global_para['item_embeddings.weight'][:10])
        logger.info("send begin")
        communication_parallel(worker_list, epoch_idx, comm, action="send_model", data=global_para)
        logger.info("send end")

        # 全局没有测试集，收集worker的平均一下？
        # test_loss, acc = test(global_model, test_loader, device, model_type=args.model_type)
        # logger.info("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))
     
    # close socket
    print("server end running")

# def aggregate_model_para(global_model, worker_list):
#     global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
#     with torch.no_grad():
#         para_delta = torch.zeros_like(global_para)
#         for worker in worker_list:
#             model_delta = (worker.config.neighbor_paras - global_para)
#             #gradient
#             # model_delta = worker.config.neighbor_paras
#             para_delta += worker.config.average_weight * model_delta
#         global_para += para_delta
#     torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
#     return global_para

# 重写全局聚合
def aggregate_model_para(global_model, worker_list, aggregate_type='Simple_Fed_Avg'):
    # global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    # with torch.no_grad():
    #     para_delta = torch.zeros_like(global_para)
    #     for worker in worker_list:
    #         model_delta = (worker.config.neighbor_paras - global_para)
    #         #gradient
    #         # model_delta = worker.config.neighbor_paras
    #         para_delta += worker.config.average_weight * model_delta
    #     global_para += para_delta
    # torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
    # return global_para

    # 参数直接加权平均（权重尚未设置）
    if aggregate_type=='Simple_Fed_Avg':
        with torch.no_grad():
            if worker_list[0].config.common_config.model_type == 'GMF':
                global_paras = {}
                global_paras['item_embeddings.weight'] = torch.zeros_like(global_model.embed_item_GMF.weight.data)
                global_paras['predict_layer.weight'] = torch.zeros_like(global_model.predict_layer.weight.data)
                global_paras['predict_layer.bias'] = torch.zeros_like(global_model.predict_layer.bias.data)
            for worker in worker_list:
                for key, value in worker.config.neighbor_paras.items():
                    global_paras[key] += value / len(worker_list)
        global_model.init_weight_by_para(global_paras)

    elif aggregate_type=='Fed_Avg':
        with torch.no_grad():
            if worker_list[0].config.common_config.model_type == 'GMF':
                global_paras = {}
                global_paras['item_embeddings.weight'] = torch.zeros_like(global_model.embed_item_GMF.weight.data)
                global_paras['predict_layer.weight'] = torch.zeros_like(global_model.predict_layer.weight.data)
                global_paras['predict_layer.bias'] = torch.zeros_like(global_model.predict_layer.bias.data)
            for worker in worker_list:
                for key, value in worker.config.neighbor_paras.items():
                    global_paras[key] += value / len(worker_list)
        global_model.init_weight_by_para(global_paras)
    
    # elif aggregate_type=='MF_Fed_Avg':
    #     with torch.no_grad():
    #         if worker_list[0].config.common_config.model_type == 'GMF':
    #             global_paras = {}
    #             global_paras['item_embeddings.weight'] = torch.zeros_like(global_model.embed_item_GMF.weight.data)
    #             delta_item_emb = torch.zeros_like(global_model.embed_item_GMF.weight.data)
    #             global_paras['predict_layer.weight'] = torch.zeros_like(global_model.predict_layer.weight.data)
    #             global_paras['predict_layer.bias'] = torch.zeros_like(global_model.predict_layer.bias.data)
    #         for worker in worker_list:
    #             for key, value in worker.config.neighbor_paras.items():
    #                 if key == 'item_embeddings.weight':
    #                     cnt = 0
    #                     for i in range(len(value)):
    #                         if 
    #                 else:
    #                     global_paras[key] += value / len(worker_list)
    #     global_model.init_weight_by_para(global_paras)

    return global_paras


# 同步通信
def communication_parallel(worker_list, epoch_idx, comm, action, data=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for worker in worker_list:
        if action == "init":
            task = asyncio.ensure_future(worker.send_init_config(comm, epoch_idx))
        elif action == "get_model":
            task = asyncio.ensure_future(worker.get_model(comm, epoch_idx))
        elif action == "send_model":
            task = asyncio.ensure_future(worker.send_data(data, comm, epoch_idx))
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

    return partition_sizes

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, _ = datasets.load_datasets(dataset_type=dataset_type,data_path=args.data_path)

    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num=10
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    elif dataset_type == "EMNIST":
        train_class_num=62
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    if dataset_type == "CIFAR100" or dataset_type == "image100":
        train_class_num=100
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    return train_data_partition

if __name__ == "__main__":
    main()
