import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import sort_edge_index
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

from torch_geometric.nn import MessagePassing

import math

# Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification, see <https://arxiv.org/abs/2009.03509>


class TransformerConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, head=8, bias=True, beta=True, concat=True):
        super(TransformerConv, self).__init__(aggr='add', node_dim=0)

        self.Wq = nn.Linear(in_channels, out_channels * head, bias=bias)
        self.Wk = nn.Linear(in_channels, out_channels * head, bias=bias)
        self.Wv = nn.Linear(in_channels, out_channels * head, bias=bias)
        self.We = nn.Linear(edge_dim, out_channels * head, bias=bias)

        self.head = head
        self.out_channels = out_channels
        self.concat = concat
        self.scale = math.sqrt(self.out_channels)
        self.lin_skip = nn.Linear(in_channels, head * out_channels, bias=bias)
        self.lin_beta = nn.Linear(3 * head * out_channels, 1, bias=False)

    def forward(self, x, edge_index, edge_features):

        H, C = self.head, self.out_channels
        query = self.Wq(x).view(-1, H, C)  # [edges, head, hidden]
        key = self.Wk(x).view(-1, H, C)  # [edges, head, hidden]
        value = self.Wv(x).view(-1, H, C)   # [edges, head, hidden]

        out = self.propagate(edge_index, q=query, k=key, v=value, edge_features=edge_features)

        if self.concat:
            out = out.view(-1, self.head * self.out_channels)
        else:
            out = out.mean(dim=1)

        x_skip = self.lin_skip(x)
        beta = self.lin_beta(torch.cat([out, x_skip, out - x_skip], dim=-1))
        beta = beta.sigmoid()
        out = (1 - beta) * out + beta * x_skip

        return out

    def message(self, edge_index_i, q_i, k_j, v_j, edge_features):
        # q_i [edges, head, hidden]
        edge_features = self.We(edge_features).view(-1, self.head, self.out_channels)
        k_j += edge_features

        alpha = (q_i * k_j).sum(dim=-1) / self.scale

        alpha_max = scatter(alpha, edge_index_i, dim=0, reduce='max')
        alpha_max = alpha_max.index_select(dim=0, index=edge_index_i)

        # do softmax, see https://zhuanlan.zhihu.com/p/29376573
        alpha = (alpha - alpha_max).exp()
        alpha_sum = scatter(alpha, edge_index_i, dim=0, reduce='sum')
        alpha_sum = alpha_sum.index_select(dim=0, index=edge_index_i)
        alpha = alpha / (alpha_sum + 1e-16)

        # score = [edges, head, head] v_j = [edges, head, hidden]
        v_j += edge_features

        out = v_j * alpha.view(-1, self.head, 1)

        return out
