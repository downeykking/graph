import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphSageLayer

class SupervisedGraphSage(nn.Module):
    def __init__(self, node_size, embedding_dim, hidden, nclass,
                 aggregator):
        super(SupervisedGraphSage, self).__init__()

        self.layer1 = GraphSageLayer(node_size, embedding_dim, hidden, aggregator)
        self.layer2 = GraphSageLayer(node_size, hidden, nclass, aggregator)
        # self.weight = nn.Parameter(torch.empty())
        # nn.init.xavier_uniform(self.weight.data)

    def forward(self, nodes, adj_lists, h):
        enc1 = self.layer1(nodes, adj_lists, h)
        out = F.relu(enc1)
        enc2 = self.layer2(nodes, adj_lists, out)
        return F.log_softmax(enc2, dim=1)
    
    
