from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

import networkx as nx
import matplotlib.pyplot as plt

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
num_epochs = 3
seed = 2022

# dataset
dataset = Planetoid(root='./data/Cora', name='Cora')

# dataloader 小图 所以直接使用一整个batch
# Data(x=[2708, 1433], edge_index=[2, 10556],
# y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
data = dataset[0]

fix_seed(seed)

# model


class GAT(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, dropout=0.5):
        super(GAT, self).__init__()
        # if heads > 1, then out_channels should be output_dim // heads
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden)
        self.conv2 = GATConv(in_channels=hidden, out_channels=output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, edge_index)
        return out


data = data.to(device)

model = GAT(input_dim=dataset.num_node_features, hidden=16,
            output_dim=dataset.num_classes).to(device)

# loss and optimizer
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

    # from utils import *
    # if epoch == num_epochs - 1:
    #     visualize(outputs, color=data.y, model_name="gcn")

model.eval()
outputs = model(data.x, data.edge_index)
_, pred = torch.max(outputs.data, dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('test acc: {:.4f}'.format(acc))
