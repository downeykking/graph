import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, softmax
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

import numpy as np

from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.typing import Adj, OptPairTensor
from torch_geometric.nn.inits import glorot


class GATConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.5, alpha=0.2, concat=True):
        super(GATConv, self).__init__()

        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_channels, out_channels)))

        self.a = nn.Parameter(torch.empty(size=(2 * out_channels, 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj):

        adj = to_dense_adj(edge_index=adj).squeeze(0)
        Wh = torch.mm(x, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_channels:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_channels) + ' -> ' \
            + str(self.out_channels) + ')'


class SparseGATConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.5, alpha=0.2, concat=True):
        super(SparseGATConv, self).__init__()

        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_channels, out_channels)))

        self.a_src = nn.Parameter(torch.empty(size=(out_channels, 1)))
        self.a_dst = nn.Parameter(torch.empty(size=(out_channels, 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

    def forward(self, x, edge_index):

        row, col = edge_index
        Wh = torch.mm(x, self.W)

        edge_1 = torch.matmul(Wh, self.a_src)
        edge_2 = torch.matmul(Wh, self.a_dst)

        e = edge_1 + edge_2

        # (Wh1 @ a1 + Wh1 @ a2) equals to (torch.cat([Wh1, Wh2], dim=1) @ A)
        e_final = e[row] + e[col]

        e = softmax(src=e_final, index=col, dim=0)
        attention = F.dropout(e, self.dropout, training=self.training)

        h_prime = attention * Wh[row]
        h_prime = scatter(h_prime, col, dim_size=edge_index.max().item() + 1, dim=0)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_channels) + ' -> ' \
            + str(self.out_channels) + ')'


if __name__ == "__main__":
    x = torch.tensor([[0.5],
                      [2.5],
                      [3.5],
                      [14.5]])

    edge_index = torch.tensor([[0, 0, 1, 1, 3],
                               [1, 2, 0, 2, 1]])

    row, col = edge_index

    conv = GATConv(x.size(1), 4)
    ans = conv(x, edge_index)
    print(ans)

    conv = SparseGATConv(x.size(1), 4)
    ans = conv(x, edge_index)
    print(ans)

    a = torch.randint(low=1, high=5, size=(1, 1)).float()
    # a1 = a[:1, :]
    # a2 = a[1:, :]

    # c = torch.mm(x, a1) + torch.mm(x, a2)
    # d = torch.mm(torch.cat([x[row], x[col]], dim=1), a)
    # print(c, d)
