from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import sort_edge_index
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

import numpy as np

from torch_geometric.utils import softmax
from torch_sparse import SparseTensor, matmul
import math

from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.nn import MessagePassing

from torch_geometric.nn import aggr


# Inductive Representation Learning on Large Graphs, see <https://arxiv.org/abs/1706.02216>
class SAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int,
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__(aggr='mean', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        edge_dim = edge_dim if edge_dim else 0
        self.lin_l = nn.Linear(in_channels[0] + edge_dim, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels[1], out_channels, bias=bias)

    def reset_parameters(self):

        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_features=None):

        x_j = self.propagate(edge_index, x=x, edge_features=edge_features)
        x_j = self.lin_l(x_j)
        x_i = self.lin_r(x)
        out = x_i + x_j

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j, edge_features):
        out = torch.cat([x_j, edge_features], dim=1)
        return out


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


class APPNP(MessagePassing):
    def __init__(self, K, alpha, dropout):
        super(APPNP, self).__init__(aggr='add', node_dim=0)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

    def forward(self, x, edge_index):

        edge_index, edge_weight = gcn_norm(edge_index)

        h = x
        for _ in range(self.K):
            edge_weight = F.dropout(edge_weight, p=self.dropout)

            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)

            x = x * (1 - self.alpha) + h * self.alpha

        return x

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GINConv(MessagePassing):
    def __init__(self, nn, eps=0.0, train_eps=False):

        super(GINConv, self).__init__(aggr='sum', node_dim=0)
        self.nn = nn
        self.initial_eps = eps
        self.train_eps = train_eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_j = scatter(x[row], index=col, dim=0, dim_size=x.size(0), reduce='sum')
        out = (1 + self.eps) * x + x_j
        return self.nn(out)

    # def forward(self, x, edge_index):
    #     x_j = self.propagate(edge_index, x=x)
    #     out = (1 + self.eps) * x + x_j
    #     return self.nn(out)

    def message(self, x_j):
        return x_j

    # is same to the above code
    # def forward(self, x, edge_index):
    #     row, col = edge_index
    #     x_j = scatter(x[row], index=col, dim=0, dim_size=x.size(0), reduce='sum')
    #     out = (1 + self.eps) * x + x_j
    #     return self.nn(out)


x = torch.tensor([[0.5],
                 [2.5],
                  [3.5],
                  [14.5]])

edge_index = torch.tensor([[0, 0, 1, 1, 3],
                           [1, 2, 0, 2, 1]])


edge_features = torch.FloatTensor([0, 1, 2, 3, 4]).reshape(-1, 1)

edge_index, edge_features = sort_edge_index(edge_index, edge_attr=edge_features, num_nodes=edge_index[0].max().item() + 1)

seed_everything(22)
out = TransformerConv(in_channels=1, out_channels=4, head=2, edge_dim=1)(x, edge_index, edge_features)
print(out)
# out2 = SAGEConv(in_channels=1, out_channels=4, edge_dim=1)(x, edge_index, edge_features)
# out3 = GATConv(in_channels=1, out_channels=4, head=2, edge_dim=1)(x, edge_index, edge_features)
# out4 = APPNP(K=10, alpha=2, dropout=0.5)(x, edge_index)
# out5 = GINConv()


def ginconv(input_dim, out_dim):
    return GINConv(nn=nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


g = ginconv(1, 1)(x, edge_index)
print(g)
