# 该文件用来测试各文件性能

import models
from datasets import *

import torch
import torch.nn as nn
import torch.optim as optim

from training_utils import train, test

import torch.multiprocessing as mp

from config import *
import argparse

import copy
# from server import aggregate_model_para

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
# parser.add_argument('--model_type', type=str, default='AlexNet') # 修改模型参数
parser.add_argument('--model_type', type=str, default='GMF')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=1)   # 运行epoch数，原定为500
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='/data/docker/data')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

def main():
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

    batch_size = 48
    test_num_ng = 99
    train_data, test_data, user_num, item_num, train_mat = load_all()
    item_num = 3706
    train_dataset = NCFData(train_data, item_num, train_mat, num_ng=4, is_training=True)
    test_dataset = NCFData(test_data, item_num, None, 0, is_training=False)

    # 增加 num_worker 会报错
    train_loader = create_dataloaders(train_dataset, 
                batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = create_dataloaders(test_dataset,
                batch_size=test_num_ng+1, shuffle=False, num_workers=0)
    
    # print(user_num, item_num)
    # print(train_loader, test_loader)

    # print(len(train_loader.loader.dataset.features_ps))
    # train_loader.loader.dataset.ng_sample()
    # print(len(train_loader.loader.dataset.features_fill))

    model = models.create_model_instance(None, 'GMF', user_num, item_num, 8)

    

    worker_list: List[Worker] = list()
    for worker_idx in range(2):
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config), rank=worker_idx+1)
        )

    # 测试提取部分参数
    with torch.no_grad():
        params_to_share = {}
        params_to_share['item_embeddings.weight'] = torch.zeros_like(model.embed_item_GMF.weight.data)
        params_to_share['predict_layer.weight'] = torch.zeros_like(model.predict_layer.weight.data)
        params_to_share['predict_layer.bias'] = torch.zeros_like(model.predict_layer.bias.data)
    # print(params_to_share)
    worker_list[0].config.neighbor_paras = copy.deepcopy(params_to_share)
    # print(worker_list[0].config.neighbor_paras)
    with torch.no_grad():
        params_to_share['item_embeddings.weight'] = torch.randn_like(model.embed_item_GMF.weight.data)
        params_to_share['predict_layer.weight'] = torch.ones_like(model.predict_layer.weight.data)
        params_to_share['predict_layer.bias'] = torch.ones_like(model.predict_layer.bias.data)

    worker_list[1].config.neighbor_paras = copy.deepcopy(params_to_share)
    
    print(1, worker_list[0].config.neighbor_paras)
    print(2, worker_list[1].config.neighbor_paras)

    aggregate_model_para(model, worker_list, aggregate_type='Simple_Fed_Avg')

    print(model.embed_item_GMF.weight.data)

    # global_model = models.create_model_instance(None, 'GMF', user_num, item_num, 8)
    # print("before aggregate:")
    # state_dict = global_model.state_dict()
    # print(state_dict['embed_item_GMF.weight'].data)

    # global_para = aggregate_model_para(global_model=global_model, worker_list=worker_list)
    
    # print("after aggregate:")
    # state_dict = global_model.state_dict()
    # print(state_dict['embed_item_GMF.weight'].data)






    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # print(model.embed_user_GMF.weight.data)
    # epoch=2
    # for i in range(epoch):
    #     print("epoch: {}".format(i))
    #     train_loss = train(model, train_loader, optimizer, local_iters=20, device=torch.device("cpu"), model_type=None)
    #     print("train_loss: {}".format(train_loss))
    #     hr, ndcg = test(model, test_loader, top_k=10)
    #     print("hr: {}, ndcg: {}".format(hr, ndcg))



    # print(model.embed_user_GMF.weight.data)
    # train_loss = train(model, train_loader, optimizer, local_iters=20, device=torch.device("cpu"), model_type=None)
    # print("train_loss: {}".format(train_loss))
    # hr, ndcg = test(model, test_loader, top_k=10)
    # print("hr: {}, ndcg: {}".format(hr, ndcg))
    # print(model.embed_user_GMF.weight.data)
    # print(model)

    
# print("gt:{}, pred:{}".format(gt_item, pred_items))



# 测试数据集加载与划分
# train_data, test_data, user_num, item_num, train_mat = load_all()
# print(user_num, item_num)
# train_data_list, test_data_list, train_mat_list = simple_partition(train_data, test_data, user_num, item_num, worker_num=5)


# print(train_data_list[2])
# print(test_data_list[1])
# print(train_mat_list[3])

# 测试提取部分参数
# params_to_share = {}
# params_to_share['item_embeddings.weight'] = model.embed_item_GMF.weight.data
# params_to_share['predict_layer.weight'] = model.predict_layer.weight.data
# params_to_share['predict_layer.bias'] = model.predict_layer.bias.data
# # print(params_to_share)

# item_emb = torch.rand_like(params_to_share['item_embeddings.weight'])
# pred_w = torch.rand_like(params_to_share['predict_layer.weight'])
# pred_b = torch.rand_like(params_to_share['predict_layer.bias'])

# print(item_emb, pred_w, pred_b)

# # # 更新模型的物品嵌入和预测层的参数
# # model.item_embedding.weight.data = item_embedding_weights
# # model.predict_layer.weight.data = predict_layer_weights

# model.embed_item_GMF.weight.data.copy_(item_emb)
# model.predict_layer.weight.data.copy_(pred_w)
# model.predict_layer.bias.data.copy_(pred_b)

# # 创建一个预训练的ResNet模型
# model = models.resnet18(pretrained=True)

# 获取模型的参数字典
# state_dict = model.state_dict()

# # 查看参数字典的键
# parameter_keys = state_dict.keys()

# # 打印所有参数的键和对应的形状
# print("dict")
# for key in parameter_keys:
#     weight = state_dict[key]
#     print(key, weight.data)

# print(torch.equal(state_dict['embed_item_GMF.weight'].data, item_emb))
# print(torch.equal(state_dict['predict_layer.bias'].data, pred_b))

# print(model)

# test_emb = nn.Embedding(10, 5)

# print(test_emb)

# print(test_emb(torch.tensor(3)))
# async def local_training(comm, common_config, local_model, train_loader, test_loader):
    
#     # local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
#     # torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
#     # 根据得到的参数更新模型
#     local_model.init_weight_by_para(paras_dict=common_config.para)
    
#     local_model.to(device)
#     epoch_lr = common_config.lr
    
#     local_steps = 20
#     # # 自适应学习率
#     # if common_config.tag > 1 and common_config.tag % 1 == 0:
#     #     epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
#     #     common_config.lr = epoch_lr
#     logger.info("epoch-{} lr: {}".format(common_config.tag, epoch_lr))  # 日志记录周期和学习率

#     # # 优化器选择
#     # if common_config.momentum<0:
#     #     optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
#     # else:
#     #     optimizer = optim.SGD(local_model.parameters(),momentum=common_config.momentum, lr=epoch_lr, weight_decay=common_config.weight_decay)
    
#     # 优化器，是否设置 weight_decay
#     # if common_config.model == 'NeuMF-pre':
#     #     optimizer = optim.SGD(local_model.parameters(), lr=args.lr)
#     # else:
#     #     optimizer = optim.Adam(local_model.parameters(), lr=args.lr)
    
#     # optimizer_type = 'SGD'
#     optimizer_type = 'Adam'
#     if optimizer_type == 'SGD':
#         optimizer = optim.SGD(local_model.parameters(), lr=args.lr)
#     else:
#         optimizer = optim.Adam(local_model.parameters(), lr=args.lr)

#     # 训练函数和测试函数
#     train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=common_config.model_type)
#     HR, NDCG = test(local_model, test_loader, device, model_type=common_config.model_type)

#     # local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
#     # 提取需要分享的参数
#     if common_config.model_type == 'GMF':
#         local_paras = {}
#         local_paras['item_embeddings.weight'] = local_model.embed_item_GMF.weight.data
#         local_paras['predict_layer.weight'] = local_model.predict_layer.weight.data
#         local_paras['predict_layer.bias'] = local_model.predict_layer.bias.data
#     # send_data_await(comm, local_paras, MASTER_RANK, common_config.tag)
#     await send_data(comm, local_paras, MASTER_RANK, common_config.tag)
#     logger.info("after send")

#     # 从服务器收到更新后的模型
#     local_para = await get_data(comm, MASTER_RANK, common_config.tag)
#     common_config.para = local_para
#     common_config.tag = common_config.tag+1
#     logger.info("get end")

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

    return global_paras

if __name__ == '__main__':
    mp.freeze_support()
    main()