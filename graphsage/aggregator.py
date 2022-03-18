import torch
import torch.nn as nn

import numpy as np

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, gcn=False): 
        """
        Initializes the aggregator for a specific graph.
        embedding -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.gcn = gcn
        
    def forward(self, nodes, adj_list, embedding, num_sample=10):
        """
        nodes --- list of nodes in a batch
        adj_list --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        samp_neighs = []
        if not num_sample is None:
            for nid in nodes:
                # 如果邻居>num_sample无放回 邻居<num_sample 放回
                if len(adj_list[nid])>=num_sample:
                    res = np.random.choice(adj_list[nid], size=(num_sample, ), replace=False)
                else:
                    res = np.random.choice(adj_list[nid], size=(num_sample, ), replace=True)
                samp_neighs.append(set(res))
        else:
            samp_neighs = adj_list

        # 添加自环
        if self.gcn:
            samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        # 生成所有邻居节点，方便后面查embed
        unique_nodes_list = list(set.union(*samp_neighs))
        
        unique_nodes_map = {n:i for i,n in enumerate(unique_nodes_list)}
        
        # mask用来最后对应nodes含有哪些邻接节点
        mask =torch.zeros(len(nodes), len(unique_nodes_map),requires_grad=True)
        
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        column_indices = [unique_nodes_map[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        
        mask[row_indices, column_indices] = 1

        num_neigh = mask.sum(1, keepdim=True)
        # mean操作
        mask = torch.div(mask, num_neigh)
        # (neibors, )(2708, 1443) (neibors, 1443)
        embed_matrix = embedding(unique_nodes_list)

        # (nodes, neibors) (neibors, 1443) ->  (nodes, 1443)
        to_feats = torch.mm(mask, embed_matrix)
        return to_feats