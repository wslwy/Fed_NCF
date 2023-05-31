import torch
import torch.nn as nn
import torch.nn.functional as F 


# class GMF(nn.Module):
# 	def __init__(self, user_num, item_num, factor_num, pre_model=None):
# 		super(GMF, self).__init__()
# 		"""
# 		user_num: number of users;
# 		item_num: number of items;
# 		factor_num: number of predictive factors;
# 		GMF_model: pre-trained GMF weights;
# 		"""		
# 		self.pre_model = pre_model

# 		# 嵌入层
# 		self.embed_user = nn.Embedding(user_num, factor_num)
# 		self.embed_item = nn.Embedding(item_num, factor_num)

# 		# 最后的预测层
# 		self.predict_layer = nn.Linear(factor_num, 1)

		
# 	def _init_weight_(self):
# 		""" We leave the weights initialization here. """
# 		if self.pre_model is None:
# 			nn.init.normal_(self.embed_user.weight, std=0.01)
# 			nn.init.normal_(self.embed_item.weight, std=0.01)

# 			for m in self.modules():
# 				if isinstance(m, nn.Linear) and m.bias is not None:
# 					m.bias.data.zero_()
# 		else:
# 			# embedding layers
# 			self.embed_user.weight.data.copy_(
# 							self.GMF_model.embed_user.weight)
# 			self.embed_item.weight.data.copy_(
# 							self.GMF_model.embed_item.weight)

# 			# predict layers
# 			self.predict_layer.weight.data.copy_(self.pre_model.predict_layer.weight)
# 			self.predict_layer.bias.data.copy_(self.pre_model.predict_layer.weight)

# 	def forward(self, user, item):
# 		embed_user = self.embed_user(user)
# 		embed_item = self.embed_item(item)
# 		output = embed_user * embed_item

# 		prediction = self.predict_layer(output)
# 		return prediction.view(-1)	# 将标量转为一维张量
	

class GMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(GMF, self).__init__()
        '''
		user_num:用户数量
		item_num:项目数量
		factor_映射维度
		'''
		
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.predict_layer = nn.Linear(factor_num, 1)
        
        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)


    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        #GMF部分就是求两个embedding的内积
        output_GMF = embed_user_GMF * embed_item_GMF
        prediction = self.predict_layer(output_GMF)
        return prediction.view(-1)	# 将标量转为一维张量