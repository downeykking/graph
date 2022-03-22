import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from aggregator import NeighborAggregator

class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu, 
                 aggr_neighbor_method="mean", aggr_method="sum"):
        """SageGCN层定义
        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，（就是聚合器的output_dim）
                当aggr_method=sum, 输出维度为hidden_dim
                当aggr_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        # 初始化参数
        assert aggr_neighbor_method in ["mean", "sum", "max"]  # 3种聚合方法
        assert aggr_method in ["sum", "concat"]  # 2种更新方法

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_method = aggr_method
        self.activation = activation
        # 定义聚合器aggregator
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_neighbor_method=aggr_neighbor_method)
        # 初始化更新的权重参数（无bias）
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化权重矩阵
        """
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        """
        前向传播
        Args:
            src_node_features:源节点特征(num_src, input_dim)
            neighbor_node_features:需要聚合的邻居节点特征(num_src,num_neigh, input_dim)
        Returns:
            combined:源节点和邻居节点concat后的特征(num_src, input_dim) or (num_src, 2*input_dim)
        """
        # 聚合操作
        # (num_src,num_neigh, input_dim)-->(num_src, input_dim)
        neighbor_feats = self.aggregator(neighbor_node_features)
        self_feats = src_node_features
        
        # 更新操作
        if self.aggr_method == "sum":  # 加法
            # (num_src, input_dim)-->(num_src, input_dim)
            combined = self_feats + neighbor_feats

        elif self.aggr_method == "concat":  # 拼接
            # (num_src, input_dim)-->(num_src, 2 * input_dim)
            combined = torch.cat([self_feats, neighbor_feats], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_method))
        
        # L2规范化
        # combined = F.normalize(combined, p=2, dim=1)
        
        # 乘权重
        combined = torch.mm(combined, self.weight)
        
        # 激活
        if self.activation:
            return self.activation(combined)
        else:
            return combined

    def extra_repr(self):
        """
        Returns:返回参数信息
        """
        # 当aggr_method = concat, 输出维度为hidden_dim * 2
        output_dim = self.hidden_dim if self.aggr_method == "sum" else self.hidden_dim * 2
        # 返回
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, output_dim, self.aggr_method)