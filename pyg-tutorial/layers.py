from torch_scatter import scatter
from torch.nn import Linear
from torch_geometric.utils import add_self_loops, degree
import torch.nn as nn
from torch_geometric.utils import sort_edge_index
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
import math

from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.nn import MessagePassing


class SAGEConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)

        self.lin_r = Linear(in_channels[1], out_channels, bias=bias)

    def reset_parameters(self):

        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        adj_t
    ) -> Tensor:

        x_j = matmul(adj_t, x, reduce="mean")
        x_j = self.lin_l(x_j)

        x_i = self.lin_r(x)

        out = x_j + x_i
        print(out)

        # x_j = x[col]
        # x_j = scatter.scatter(x_j, row, dim=0, dim_size=x.size(0), reduce="sum")
        # x_j = self.lin_l(x_j)
        # x_i = self.lin_r(x)
        # out = 0.5 * x_j + x_i

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out


class TransformerConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, edge_dim, head=8, bias=True, concat=True):
        super(TransformerConv, self).__init__(aggr='add', node_dim=0)

        self.Wq = nn.Linear(in_channels, hidden_channels * head, bias=bias)
        self.Wk = nn.Linear(in_channels, hidden_channels * head, bias=bias)
        self.Wv = nn.Linear(in_channels, hidden_channels * head, bias=bias)
        self.We = nn.Linear(edge_dim, hidden_channels * head, bias=bias)

        self.head = head
        self.hidden_channels = hidden_channels
        self.concat = concat
        self.scale = math.sqrt(self.hidden_channels)

    def forward(self, x, edge_index, edge_features):

        H, C = self.head, self.hidden_channels
        query = self.Wq(x).view(-1, H, C)  # [edges, head, hidden]
        key = self.Wk(x).view(-1, H, C)  # [edges, head, hidden]
        value = self.Wv(x).view(-1, H, C)   # [edges, head, hidden]

        out = self.propagate(edge_index, q=query, k=key, v=value, edge_features=edge_features)

        if self.concat:
            out = out.view(-1, self.head * self.hidden_channels)
        else:
            out = out.mean(dim=1)

        return out

    def message(self, edge_index_i, q_i, k_j, v_j, edge_features):
        # q_i [edges, head, hidden]
        edge_features = self.We(edge_features).view(-1, self.head, self.hidden_channels)
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


x = torch.tensor([[0.5],
                 [2.5],
                  [3.5],
                  [14.5]])

seed_everything(42)
edge_index = torch.tensor([[0, 0, 1, 1, 3],
                           [1, 2, 0, 2, 1]])


edge_features = torch.FloatTensor([0, 1, 2, 3, 4]).reshape(-1, 1)

edge_index, edge_features = sort_edge_index(edge_index, edge_attr=edge_features, num_nodes=edge_index[0].max().item() + 1)

out = TransformerConv(in_channels=1, hidden_channels=4, head=2, edge_dim=1)(x, edge_index, edge_features)
print(out)
