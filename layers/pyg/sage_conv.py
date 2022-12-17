import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import sort_edge_index
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

from torch_geometric.nn import MessagePassing


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
