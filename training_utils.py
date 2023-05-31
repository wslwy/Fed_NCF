import sys
import time
import math
import re
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu"), model_type=None):
#     t_start = time.time()
#     model.train()
#     if local_iters is None:
#         local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)
#     #print("local_iters: ", local_iters)

#     train_loss = 0.0
#     samples_num = 0
#     for iter_idx in range(local_iters):
#         data, target = next(data_loader)

#         if model_type == 'LR':
#             data = data.squeeze(1).view(-1, 28 * 28)
            
#         data, target = data.to(device), target.to(device)
        
#         output = model(data)

#         optimizer.zero_grad()
        
#         loss_func = nn.CrossEntropyLoss() 
#         loss = loss_func(output, target)
#         #loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()

#         train_loss += (loss.item() * data.size(0))
#         samples_num += data.size(0)

#     if samples_num != 0:
#         train_loss /= samples_num
    
#     return train_loss

# 似乎能用了
def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu"), model_type=None):
    t_start = time.time()
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)
    #print("local_iters: ", local_iters)

    # 负采样放到 DataloaderHelper 里了

    train_loss = 0.0
    samples_num = 0
    for iter_idx in range(local_iters):
        user, item, label = next(data_loader)
        # if iter_idx == 1:
        #     print(user[0:10], item[0:10], label[0:10])
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        model.zero_grad()
        prediction = model(user, item)

        # loss_func = nn.CrossEntropyLoss() 
        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(prediction, label)
        loss.backward()
        optimizer.step()

        # print(loss)
        train_loss += (loss.item() * label.size(0))
        samples_num += label.size(0)

    if samples_num != 0:
        train_loss /= samples_num
        # print(train_loss)
    
    return train_loss




# def test(model, data_loader, device=torch.device("cpu"), model_type=None):
#     model.eval()
#     data_loader = data_loader.loader
    
#     test_loss = 0.0
#     test_accuracy = 0.0

#     correct = 0

#     with torch.no_grad():
#         for data, target in data_loader:

#             data, target = data.to(device), target.to(device)

#             if model_type == 'LR':
#                 data = data.squeeze(1).view(-1, 28 * 28)
#             output = model(data)

#             # sum up batch loss
#             loss_func = nn.CrossEntropyLoss(reduction='sum') 
#             test_loss += loss_func(output, target).item()
#             # test_loss += F.nll_loss(output, target, reduction='sum').item()
#             # get the index of the max log-probability
#             pred = output.argmax(1, keepdim=True)
#             batch_correct = pred.eq(target.view_as(pred)).sum().item()

#             correct += batch_correct
            

#     test_loss /= len(data_loader.dataset)
#     test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

#     # TODO: Record

#     return test_loss, test_accuracy

# test 暂定返回 HR@10 和 NDCG@10
def test(model, data_loader, top_k=10, device=torch.device("cpu"), model_type=None):
    model.eval()
    data_loader = data_loader.loader
    
    # test_loss = 0.0
    HR, NDCG = [], [] 

    for user, item, label in data_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist() # 如果 device 是 CPU ？

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    # print(HR, NDCG)
    return np.mean(HR), np.mean(NDCG)

    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            if model_type == 'LR':
                data = data.squeeze(1).view(-1, 28 * 28)
            output = model(data)

            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum') 
            test_loss += loss_func(output, target).item()
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    # TODO: Record

    return test_loss, test_accuracy


def hit(gt_item, pred_items):
    # print("gt:{}, pred:{}".format(gt_item, pred_items)) #观察预测结果
    if gt_item in pred_items:
        # print("gt:{}, pred:{}".format(gt_item, pred_items)) #观察命中预测结果
        return 1
    return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)