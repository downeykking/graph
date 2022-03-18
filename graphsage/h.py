import numpy as np
from collections import defaultdict
import random
import torch


# samp_neighs = [set(np.random.sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
    
node_map = {}
with open("../dataset/cora/cora.content") as fp:
    for i,line in enumerate(fp):
        info = line.strip().split()
        node_map[info[0]] = i

adj_lists = defaultdict(set)
with open("../dataset/cora/cora.cites") as fp:
    for i,line in enumerate(fp):
        info = line.strip().split()
        paper1 = node_map[info[0]]
        paper2 = node_map[info[1]]
        adj_lists[paper1].add(paper2)
        adj_lists[paper2].add(paper1)

num_sample = 10
np.random.seed(2022)
random.seed(2022)
nodes = list(np.random.choice(list(adj_lists.keys()), 8))
# nodes [N] 
# [1855, 158, 451, 2422, 613, 1361, 1401, 2686]
# print(nodes)

to_neighs = [adj_lists[i] for i in nodes]
# to_neighs [N个set邻居]

samp_neighs = [set(random.sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
# samp_neighs [N个set邻居] 采样补全
#  [{1864, 2075, 2122, 75, 2482, 2299, 1854}, 
#  {184, 433, 707, 598}, 
#  {552, 2537, 2666}, 
#  {2376, 2499, 2500, 359}, 
#  {2496, 232, 616, 2571, 268, 14, 2107}, 
#  {1463}, 
#  {2605, 1554, 562, 565, 951}, 
#  {916, 2596, 101}


# GCN 加个自环，即把自身加进来
# samp_neighs = [samp_neigh | (set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]

unique_nodes_list = list(set.union(*samp_neighs))
# 这个batch中所有的邻居全部加进来
# print(unique_nodes_list)
# [2496, 707, 2499, 2500, 1864, 2376, 2122, 75, 2571, 268, 14, 1554, 916, 
# 598, 2075, 2596, 101, 359, 552, 2537, 2666, 
# 232, 616, 2107, 2605, 433, 2482, 562, 951, 565, 1463, 184, 2299, 1854]

unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
print(unique_nodes)
# 34个

# (N, 34)   (8, 34)
mask = torch.zeros(len(samp_neighs), len(unique_nodes))

column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh] 
print(column_indices)  
row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
print(row_indices)

mask[row_indices, column_indices] = 1
print(mask)

num_neigh = mask.sum(1, keepdim=True)

