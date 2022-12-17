import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.utils import sort_edge_index
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

from torch_geometric.nn import MessagePassing


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, head=8, bias=True, concat=True):
        super(GATConv, self).__init__(aggr='add', node_dim=0)
        # a new feature 'aggr', see 'https://pytorch-geometric.readthedocs.io/en/2.1.0/modules/nn.html#aggregation-operators'
        # super(GATConv, self).__init__(aggr=aggr.SumAggregation(), node_dim=0)

        self.W_src = nn.Linear(in_channels, out_channels * head, bias=bias)
        self.W_dst = nn.Linear(in_channels, out_channels * head, bias=bias)
        self.a_src = nn.Parameter(torch.Tensor(1, head, out_channels))
        self.a_dst = nn.Parameter(torch.Tensor(1, head, out_channels))

        self.head = head
        self.out_channels = out_channels
        self.concat = concat

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, head * out_channels, bias=bias)
            self.att_edge = nn.Parameter(torch.Tensor(1, head, out_channels))

    def forward(self, x, edge_index, edge_features):

        H, C = self.head, self.out_channels
        x_src = x_dst = self.W_src(x).view(-1, H, C)
        x = (x_src, x_dst)

        out = self.propagate(edge_index, x=x, edge_features=edge_features)

        if self.concat:
            out = out.view(-1, self.head * self.out_channels)
        else:
            out = out.mean(dim=1)

        return out

    def message(self, x_j, index, edge_features=None):
        x_src = x_j
        alpha = (x_src * self.a_src).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index)
        return alpha.unsqueeze(-1) * x_j
