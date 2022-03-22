import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from layers import SageGCN

# 模型的定义
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, sample_num_lists, aggr_neighbor_method="mean",
                 aggr_method="sum"):
        """
        GraphSAGE模型的定义
        Args:
            input_dim:源节点的维度
            hidden_dim:每一层的隐藏（输出）维度列表(k,)
            sample_num_lists: 节点0阶，1阶，2阶...采样的邻居数量(k个每层采样到的邻居数)
        """
        super(GraphSage, self).__init__()
        self.name = "graphsage"
        # 初始化参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sample_num_lists = sample_num_lists
        # 网络的层数就是列表的长度
        self.num_layers = len(sample_num_lists)

        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_method = aggr_method

        # 定义layers
        self.layers = nn.ModuleList()  # 需要自己定义forward函数
        # 第0层是源节点的特征维度-->hidden[0]
        
        layer_in = SageGCN(input_dim, hidden_dim[0], aggr_neighbor_method=aggr_neighbor_method,
                                aggr_method=aggr_method)
        self.layers.append(layer_in)
        
        for layer in range(0, len(hidden_dim) - 2):
            if self.aggr_method == "concat":
                hidden_dim[layer] *= 2
            self.layers.append(SageGCN(hidden_dim[layer], hidden_dim[layer + 1], aggr_neighbor_method=aggr_neighbor_method,
                                    aggr_method=aggr_method))
        if self.aggr_method == "concat":
            hidden_dim[-2] *= 2
        
        layer_out = SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None, aggr_neighbor_method=aggr_neighbor_method,
                    aggr_method=aggr_method)
        
        self.layers.append(layer_out)  # 最后一层不需要激活

    def forward(self, node_features_list):
        """
        前向传播
        Args:
            node_features_list:节点0阶，1阶，2阶...邻居的特征列表(k+1,num_node,input_dim)
            其中第一层节点数假如num_node为5 则第二层节点数为5*sample_num 第三层节点数为5*sample_num*sample_num
        Returns:
            返回更新后的节点特征:node_features(num_src, output_dim)
        """
        # 采样后的节点k阶邻居特征列表
        hidden = node_features_list  
        # 不同的层，这部分"倒推"求解
        # 虽然是正序，但是最后结果其实是hidden[0]
        for k in range(self.num_layers):
            next_hidden = []
            layers = self.layers[k]
            # 每一层模型都需要进行的操作
            for hop in range(self.num_layers - k):
                # (num_src,input_dim)
                src_node_feats = node_features_list[hop]
                src_node_num = len(src_node_feats)
                # view改变形状
                # (num_src * num_neigh,input_dim)-->(num_src,num_neigh,input_dim)
                neighbor_node_feats = node_features_list[hop + 1].view((src_node_num, self.sample_num_lists[hop], -1))
                # 使用源节点和邻居节点进行聚合+更新操作
                h = layers(src_node_feats, neighbor_node_feats)
                # 加入到列表中
                next_hidden.append(h)
            # 重新赋值
            node_features_list = next_hidden
        return node_features_list[0]  # 最终hidden列表只剩下1个元素(num_src, output_dim)

    def extra_repr(self):
        return 'in_features={}, hidden_dim={}, out_features={}, sample_num_lists={}'.format(
            self.input_dim, self.hidden_dim[0:-1], self.hidden_dim[-1], self.sample_num_lists)