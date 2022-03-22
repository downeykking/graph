import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class NeighborAggregator(nn.Module):  # 继承了nn.Module，具有可学习的参数
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_neighbor_method="mean"):
        """聚合节点邻居
        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_neighbor_method: 邻居聚合方式 (default: {mean})--3种可选择的方式，不包括paper中的LSTM方法
        """
        super(NeighborAggregator, self).__init__()  # 继承父类的方法、属性
        # 形参初始化
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_neighbor_method = aggr_neighbor_method
       

    def forward(self, neighbor_feature):
        """
        前向传播
        Args:
            neighbor_feature:需要聚合的邻居节点特征(num_src,num_neigh,input_dim)--(源节点数量,邻居数量,input_dim)
        Returns:
            aggr_neighbor:聚合后的消息，用于更新节点的嵌入表示(num_src,output_dim)
        """
        if self.aggr_neighbor_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)  # 对第1维num_neigh进行聚合
        elif self.aggr_neighbor_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_neighbor_method == "max":
            aggr_neighbor, _ = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}".format(self.aggr_neighbor_method))

        return aggr_neighbor # (num_src,input_dim)


    def extra_repr(self):
        """
        Returns:返回参数字符串
        """
        return "in_features={}, out_features={}, aggr_neighbor_method={}".format(
            self.input_dim, self.output_dim, self.aggr_neighbor_method)