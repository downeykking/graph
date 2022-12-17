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


class GINConv(MessagePassing):
    def __init__(self, net, eps=0.0, train_eps=False):

        super(GINConv, self).__init__(aggr='sum', node_dim=0)
        self.net = net
        self.initial_eps = eps
        self.train_eps = train_eps

        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_j = scatter(x[row], index=col, dim=0, dim_size=x.size(0), reduce='sum')
        out = (1 + self.eps) * x + x_j
        return self.net(out)

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


def ginconv(input_dim, out_dim):
    return GINConv(nn=nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
