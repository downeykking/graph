import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.utils import sort_edge_index
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


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
