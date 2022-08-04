from typing import Optional, Dict, Tuple, Union
from torch_geometric.typing import OptTensor, OptPairTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


import torch_geometric

from torch_geometric.nn.conv import MessagePassing
from torch_geometric import seed_everything
from torch_sparse import SparseTensor, matmul

seed_everything(42)


class SAGEConv(MessagePassing):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 aggr: str = 'add',
                 normalize: bool = False,
                 bias: bool = True):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels[1], out_channels, bias=bias)

    def reset_parameters(self):

        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor],
                edge_index: Tensor,
                edge_attr: Tensor = None,
                edge_t: Tensor = None):

        x_j = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        print(x_j)
        # x_j = self.lin_l(x_j)
        # x_i = self.lin_r(x)
        x_j = x_j
        x_i = torch.cat([x, edge_attr], dim=-1)

        out = x_i + x_j

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j: Tensor, edge_attr) -> Tensor:
        # print(x_j)
        # has beed reduced by scatter and is_sorted
        return torch.cat([x_j, edge_attr], dim=1)
        # x_j = torch_scatter.scatter(x[j], row, dim=0, dim_size=x.size(0), reduce='mean')


model = SAGEConv(2, 2)

x = torch.randint(low=1, high=6, size=(3, 2)).float()
row = torch.LongTensor([1, 0, 2])
col = torch.LongTensor([0, 1, 2])
edge_index = torch.stack([row, col], dim=0)
edge_attr = torch.randint(low=1, high=6, size=(3, 2)).float()
# print(x[col])

print(x)
print(edge_attr)
a = model(x, edge_index, edge_attr)
print(a)
