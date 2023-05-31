import torch
import torch.nn as nn
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        super(MLP, self).__init__()
        # 嵌入层
        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))
        
        # MLP
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        '''这里解释一下，这里是把每一层的定义放在列表里，然后
        nn.Sequential(*MLP_modules)函数直接取列表顺序定义网络，每一层的激活函		
        数都是relu'''
        self.MLP_layers = nn.Sequential(*MLP_modules)
        
        # 预测层
        self.predict_layer = nn.Linear(factor_num, 1)
        
        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_uniform_(self.predict_layer.weight,a=1, nonlinearity='sigmoid')
        
    def forward(self, user, item):
        # 嵌入层
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)

        # MLP
        output_MLP = self.MLP_layers(interaction)

        # 预测层
        prediction = self.predict_layer(output_MLP)

        return prediction.view(-1)