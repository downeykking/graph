from torch import nn
from torch_geometric.datasets import Planetoid
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        return self.propagate(x, edge_index)
    
    def propagate(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        out = self.message(x, edge_index)
        out = self.aggregate(out, edge_index)
        out = self.update(out)
        
        return out
    
    def message(self, x, edge_index):
        x = self.lin(x)
        
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        x_j = x[col]
        x_j = norm.view(-1, 1) * x_j
        
        return x_j
    
    def aggregate(self, x_j, edge_index):
        # 从x_j 聚合到 x_i
        row, _ = edge_index
        aggr_out = scatter(x_j, row, dim=-2, reduce='sum')
        return aggr_out
    
    def update(self, aggr_out):
        return aggr_out

dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]

gcn = GCN(in_channels=data.num_features, out_channels=7)

out = gcn(data.x, data.edge_index)
print(out.shape) # torch.Size([2708, 7])