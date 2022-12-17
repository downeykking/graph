import torch
import torch.nn as nn
from torch_geometric import seed_everything
from torch_geometric.utils import sort_edge_index


x = torch.tensor([[0.5],
                 [2.5],
                  [3.5],
                  [14.5]])

edge_index = torch.tensor([[0, 0, 1, 1, 3],
                           [1, 2, 0, 2, 1]])


edge_features = torch.FloatTensor([0, 1, 2, 3, 4]).reshape(-1, 1)

edge_index, edge_features = sort_edge_index(edge_index, edge_attr=edge_features, num_nodes=edge_index[0].max().item() + 1)

seed_everything(22)
# out = TransformerConv(in_channels=1, out_channels=4, head=2, edge_dim=1)(x, edge_index, edge_features)
print(out)
# out2 = SAGEConv(in_channels=1, out_channels=4, edge_dim=1)(x, edge_index, edge_features)
# out3 = GATConv(in_channels=1, out_channels=4, head=2, edge_dim=1)(x, edge_index, edge_features)
# out4 = APPNP(K=10, alpha=2, dropout=0.5)(x, edge_index)
# out5 = GINConv()


def ginconv(input_dim, out_dim):
    return GINConv(nn=nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


g = ginconv(1, 1)(x, edge_index)
print(g)
