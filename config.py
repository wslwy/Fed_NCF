import os
from typing import List
import paramiko
from scp import SCPClient
from torch.utils.tensorboard import SummaryWriter
from comm_utils import *


class ClientAction:
    LOCAL_TRAINING = "local_training"

'''
这是一个名为 Worker 的类，代表每个工作节点。其构造函数需要传入一个名为 config 的参数和一个 rank 参数。
类中定义了四个异步函数：
send_data 函数：用于向通信的对端节点发送数据，参数包括要发送的数据、通信对象 comm 以及当前的轮次数 epoch_idx。
send_init_config 函数：用于在每个轮次开始时将当前节点的配置信息发送给其他节点，参数包括通信对象 comm 以及当前的轮次数 epoch_idx。
get_model 函数：用于从通信的对端节点获取模型，参数包括通信对象 comm 以及当前的轮次数 epoch_idx。
get_para 函数：用于从通信的对端节点获取一些数据，包括训练时间和发送时间，参数包括通信对象 comm 以及当前的轮次数 epoch_idx。
在 get_model 和 get_para 函数中，获取到的数据会被保存在 config 对象中的相应属性中。
'''
class Worker:
    def __init__(self, config, rank):
        #这个config就是后面的client_config
        self.config = config
        self.rank = rank

    async def send_data(self, data, comm, epoch_idx):
        await send_data(comm, data, self.rank, epoch_idx)    

    async def send_init_config(self, comm, epoch_idx):
        print("before send", self.rank, "tag:", epoch_idx)
        await send_data(comm, self.config, self.rank, epoch_idx)    

    async def get_model(self, comm, epoch_idx):
        self.config.neighbor_paras = await get_data(comm, self.rank, epoch_idx)
    
    async def get_para(self, comm, epoch_idx):
        train_time, send_time = await get_data(comm, self.rank, epoch_idx)
        self.config.train_time=train_time
        self.config.send_time=send_time

class CommonConfig:
    def __init__(self):
        self.model_type = None
        self.dataset_type = None
        self.batch_size = None
        self.data_pattern = None
        self.lr = None
        self.decay_rate = None
        self.min_lr = None
        self.epoch = None
        self.momentum=None
        self.weight_decay=None
        self.para = None
        self.data_path = None
        #这里用来存worker的


class ClientConfig:
    def __init__(self,
                common_config,
                custom: dict = dict()
                ):
        self.para = None
        self.train_data_idxes = None
        self.common_config=common_config

        self.average_weight=0.1
        self.local_steps=20
        self.compre_ratio=1
        self.train_time=0
        self.send_time=0
        # neighbor 保存进程的参数
        self.neighbor_paras=None
        self.neighbor_indices=None

        # 添加
        self.test_num_ng = 99
        # 是否设置不同权重等
        # self.neighbor_weight/train_data_num
