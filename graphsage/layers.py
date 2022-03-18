import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class GraphSageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    node_size : int; 表示batch节点数目
    feature: (N, dim); 表示这一层嵌入层的特征
    adj_lists: list of set; 是一个邻接节点set列表，里面包括源节点的邻接节点
    embedding_dim: 这一层特征的维度
    hidden: 这一层转换后特征的维度
    aggregator: 何种方式聚合成邻居
    """
    def __init__(self, node_size, embedding_dim, hidden,
                 aggregator, num_sample=10, base_model=None, gcn=False,
                feature_transform=False): 
        super(GraphSageLayer, self).__init__()

        self.embedding = nn.Embedding(node_size, embedding_dim)
        
        # self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.hidden = hidden
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.weight = nn.Parameter(
                torch.empty(embedding_dim if gcn else 2 * embedding_dim, hidden))
        init.xavier_uniform(self.weight.data)

    def forward(self, nodes, adj_list, feature):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        return embed feature [nodes, hidden]
        """
        self.embedding.weight = nn.Parameter(torch.FloatTensor(feature), requires_grad=False)
        neigh_feats = self.aggregator.forward(nodes, [adj_list[int(node)] for node in nodes], 
                self.embedding, self.num_sample)
        
        if not self.gcn:
            self_feats = self.embedding(nodes)
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        
        else:
            combined = neigh_feats
        combined = F.relu(torch.mm(combined, self.weight))
        return combined