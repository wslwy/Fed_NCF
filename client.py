import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from pulp import *
import random
from config import ClientConfig, CommonConfig
from comm_utils import *
from training_utils import train, test
import datasets, models
from mpi4py import MPI
import logging

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(rank)% 4 + 0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


# init logger
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/clients/' + now + '/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)


filename = RESULT_PATH + now + "_" +os.path.basename(__file__).split('.')[0] + '_'+ str(int(rank)) +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
# end logger

MASTER_RANK=0

# 得到配置信息并且初始化本地配置信息
async def get_init_config(comm, MASTER_RANK, config):
    logger.info("before init")
    config_received = await get_data(comm, MASTER_RANK, 1)
    logger.info("after init")
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)

def main():
    logger.info("client_rank:{}".format(rank))
    # 创建客户端空的配置信息
    client_config = ClientConfig(
        common_config=CommonConfig()
    )
    # 从服务器接受配置信息并初始化
    logger.info("start")
    # 感觉下面是一个固定通信结构
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    task = asyncio.ensure_future(get_init_config(comm, MASTER_RANK, client_config))
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    # 将受到的配置文件从client_config复制到一个新的common_config里，添加了tag，意义不明   
    common_config = CommonConfig()
    common_config.model_type = client_config.common_config.model_type   
    common_config.dataset_type = client_config.common_config.dataset_type
    common_config.batch_size = client_config.common_config.batch_size
    common_config.data_pattern=client_config.common_config.data_pattern
    common_config.lr = client_config.common_config.lr
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr = client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.momentum = client_config.common_config.momentum
    common_config.weight_decay = client_config.common_config.weight_decay
    common_config.data_path = client_config.common_config.data_path
    common_config.para=client_config.para
    
    common_config.tag = 1
    # init config
    logger.info(str(common_config.__dict__))

    # # logger.info(str(len(client_config.train_data_idxes)))
    # # train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type, common_config.data_path)
    # # train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, selected_idxs=client_config.train_data_idxes)
    # # test_loader = datasets.create_dataloaders(test_dataset, batch_size=16, shuffle=False)

    # 以下为修改后的本地数据集获取，加载所有数据，再简单根据相同用户数划分
    print("worker {} begin loading data".format(rank))
    train_dataset, test_dataset, user_num, item_num = datasets.load_datasets(worker_num=int(csize)-1, worker_rank=rank, num_ng=4)
    train_loader = datasets.create_dataloaders(train_dataset, 
            batch_size=common_config.batch_size, shuffle=True, num_workers=0)
    test_loader = datasets.create_dataloaders(test_dataset,
            batch_size=client_config.test_num_ng+1, shuffle=False, num_workers=0)
    # print(len(train_loader.loader.dataset))
    print("worker {} end loading data".format(rank))

    

    # print("begin test ng_sample")
    # i = 0
    # print(len(test_dataset.features_ps))
    # train_loader.dataset.ng_sample()
    # # print(len(test_dataset[0:10]))
    # print(train_loader.dataset[-10:])
    # # print(len(test_loader))
    # print("end test ng_sample")

    # 本地模型实例化涉及config的改变, user_num应该使用总的还是本地的？
    item_num = 3706
    # 或许可以使用user_ID与数字索引处理，这里使用总人数
    local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type, user_num=user_num, item_num=item_num, factor_num=8)
    # print(local_model)


    # 本地训练，怎么改
    while True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training(comm, common_config, local_model, train_loader, test_loader)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        loop.close()
        # common_config.tag记录了训练轮数
        if common_config.tag==common_config.epoch+1:
            break

    print("worker {} end running".format(rank))

# 没有验证集
# async def local_training(comm, common_config, train_loader, test_loader):
#     # 用服务器传来的模型参数初始化模型
#     local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
#     torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
#     local_model.to(device)
#     epoch_lr = common_config.lr
    
#     local_steps = 20
#     # 根据周期设定学习率
#     if common_config.tag > 1 and common_config.tag % 1 == 0:
#         epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
#         common_config.lr = epoch_lr
#     logger.info("epoch-{} lr: {}".format(common_config.tag, epoch_lr))
    
#     # 设定优化器
#     if common_config.momentum<0:
#         optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
#     else:
#         optimizer = optim.SGD(local_model.parameters(),momentum=common_config.momentum, lr=epoch_lr, weight_decay=common_config.weight_decay)
    
#     # 本地训练并且测试
#     train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=common_config.model_type)
#     test_loss, acc = test(local_model, test_loader, device, model_type=common_config.model_type)
#     logger.info("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(common_config.tag, train_loss, test_loss, acc))
    
#     # 本地模型发回服务器
#     logger.info("send para")
#     local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
#     # send_data_await(comm, local_paras, MASTER_RANK, common_config.tag)
#     await send_data(comm, local_paras, MASTER_RANK, common_config.tag)
#     logger.info("after send")

#     # 从服务器收到更新后的模型，保存收到的参数，跟新周期
#     local_para = await get_data(comm, MASTER_RANK, common_config.tag)
#     common_config.para=local_para
#     common_config.tag = common_config.tag+1
#     logger.info("get end")

async def local_training(comm, common_config, local_model, train_loader, test_loader):
    
    # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    # torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    # 根据得到的参数更新模型
    local_model.init_weight_by_para(paras_dict=common_config.para)
    
    local_model.to(device)
    epoch_lr = common_config.lr
    
    local_steps = 2
    # # 自适应学习率
    # if common_config.tag > 1 and common_config.tag % 1 == 0:
    #     epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
    #     common_config.lr = epoch_lr
    logger.info("epoch-{} lr: {}".format(common_config.tag, epoch_lr))  # 日志记录周期和学习率
    # logger.info(local_model.embed_item_GMF.weight.data[:10])  # 日志记录嵌入表部分变化

    # # 优化器选择
    # if common_config.momentum<0:
    #     optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
    # else:
    #     optimizer = optim.SGD(local_model.parameters(),momentum=common_config.momentum, lr=epoch_lr, weight_decay=common_config.weight_decay)
    
    # 优化器，是否设置 weight_decay
    # if common_config.model == 'NeuMF-pre':
    #     optimizer = optim.SGD(local_model.parameters(), lr=args.lr)
    # else:
    #     optimizer = optim.Adam(local_model.parameters(), lr=args.lr)
    
    # optimizer_type = 'SGD'
    optimizer_type = 'Adam'
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr)
    else:
        optimizer = optim.Adam(local_model.parameters(), lr=epoch_lr)

    # 训练函数和测试函数
    train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=common_config.model_type)
    HR, NDCG = test(local_model, test_loader, top_k=10, device=device, model_type=common_config.model_type)

    # logger.info("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(common_config.tag, train_loss, test_loss, acc))
    logger.info("after aggregation, epoch: {}, train loss: {}, HR: {}, NDCG: {}".format(common_config.tag, train_loss, HR, NDCG))
    logger.info("send para")

    # local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    # 提取需要分享的参数
    if common_config.model_type == 'GMF':
        local_paras = {}
        local_paras['item_embeddings.weight'] = local_model.embed_item_GMF.weight.data
        local_paras['predict_layer.weight'] = local_model.predict_layer.weight.data
        local_paras['predict_layer.bias'] = local_model.predict_layer.bias.data
    # send_data_await(comm, local_paras, MASTER_RANK, common_config.tag)
    await send_data(comm, local_paras, MASTER_RANK, common_config.tag)
    logger.info("after send")

    # 从服务器收到更新后的模型
    local_para = await get_data(comm, MASTER_RANK, common_config.tag)
    common_config.para = local_para
    common_config.tag = common_config.tag+1
    logger.info("get end")


if __name__ == '__main__':
    main()
