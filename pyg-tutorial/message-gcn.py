import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import networkx as nx
import matplotlib.pyplot as plt

# device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#
num_epochs = 200

# dataset
dataset = Planetoid(root='./data/Cora', name='Cora')

# dataloader 小图 所以直接使用一整个batch
data = dataset[0]

# model
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # "Add" aggregation (Step 5).
        # flow='source_to_target' 表示消息从源节点传播到目标节点
        # 对特征进行线性变换
        self.fc = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # (2, N+E)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: linearly transform node feature matrix.
        x = self.fc(x)

        # Step 3: Compute normalization.
        # row (E, ) col (E, )
        row, col = edge_index

        # deg (N, )
        deg = degree(col, x.size(0), dtype=x.dtype)

        deg_inv_sqrt = deg.pow(-0.5)

        # norm (E, )
        # deg_inv_sqrt[row] 表示 row中对应节点在deg中node的位置
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        # out (E, output_dim)
        return norm.view(-1, 1) * x_j

data = data.to(device)
model = GCNConv(data.num_features, 64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

model.train()
train_total = data.train_mask.sum().item()
val_total = data.val_mask.sum().item()

for epoch in range(num_epochs):

    # Forward pass
    outputs = model(data.x, data.edge_index)
    loss = criterion(outputs[data.train_mask], data.y[data.train_mask])

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, pred = torch.max(outputs.data, 1)
    train_correct = (pred[data.train_mask] ==
                     data.y[data.train_mask]).sum().item()
    val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()

    print(
        'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}, Val-Acc: {:.2f}'.format(
            epoch + 1, num_epochs, loss.item(),
            100 * train_correct / train_total, 100 * val_correct / val_total))
    
    from utils import *
    if epoch == num_epochs - 1:
        visualize(outputs, color=data.y, model_name="message-gcn")

model.eval()
outputs = model(data.x, data.edge_index)
_, pred = torch.max(outputs.data, dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('test acc: {:.4f}'.format(acc))


