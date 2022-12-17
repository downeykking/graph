import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

import numpy as np

from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.typing import Adj, OptPairTensor


class SAGEConv(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.aggr = aggr
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

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj
    ) -> Tensor:

        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            x_j = x[row]
            x_j = scatter(x_j, col, dim=0, dim_size=x.size(0), reduce=self.aggr)
            x_j = self.lin_l(x_j)
            x_i = self.lin_r(x)
            out = x_j + x_i

        elif isinstance(edge_index, SparseTensor):
            adj = edge_index
            x_j = matmul(adj.t(), x, reduce=self.aggr)
            x_j = self.lin_l(x_j)
            x_i = self.lin_r(x)
            out = x_j + x_i

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out


if __name__ == "__main__":
    x = torch.tensor([[0.5],
                      [2.5],
                      [3.5],
                      [14.5]])

    edge_index = torch.tensor([[0, 0, 1, 1, 3],
                               [1, 2, 0, 2, 1]])

    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(edge_index.max().item() + 1, edge_index.max().item() + 1))

    x_j = scatter(x[row], col, dim=0, dim_size=x.size(0), reduce="sum")
    x_j_sp = matmul(adj.t(), x)

    assert torch.allclose(x_j, x_j_sp)

    conv = SAGEConv(x.size(1), 4)

    ans = conv(x, edge_index)
    ans2 = conv(x, adj)

    assert torch.allclose(ans, ans2)
