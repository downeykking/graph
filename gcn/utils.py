import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_data(path="../dataset/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = pd.read_csv("{}{}.content".format(path, dataset), sep='\t', header=None)
    
    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    features = sp.csr_matrix(idx_features_labels.iloc[:, 1:-1], dtype=np.float32)
    
    # one-hot编码标签
    labels = pd.get_dummies(idx_features_labels.iloc[:, -1]).values
    
    # 节点idx
    idx = idx_features_labels.iloc[:,0].values
    # 由样本id到样本索引的映射字典
    idx_map = {j: i for i, j in enumerate(idx)}
    # 被引用论文 引用论文 有向图
    edges_data = pd.read_csv("{}{}.cites".format(path, dataset), sep='\t', header=None)
    # dict映射idx
    edges = edges_data.iloc[:,0:2].applymap(lambda x:idx_map.get(x))
    # 将edges转为[2, edge]的形式
    edges = edges.iloc[:,0:2].values.flatten('F').reshape(2, -1)
    
    # 稀疏格式邻接表表示邻接矩阵 单向
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # 构建对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    features = normalize_features(features)
    
    # 自环 + norm A* = A^(-1)D 
    adj = normalize(adj + sp.eye(adj.shape[0]))
    
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    # 返回one-hot中为1的坐标
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# 先求度，再转成度矩阵即可 其中 A* = D^(-1)A
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    mx = d_mat_inv.dot(mx)
    return mx


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(outputs, labels):
    _, pred = torch.max(outputs.data, dim=1)
    correct = (pred == labels).sum().item()
    return correct / len(labels)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
def visualize(h, color, model_name):
    z = TSNE(learning_rate='auto', init='random').fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.cpu().numpy(), cmap="Set2")
    plt.show()
    plt.savefig("{}.jpg".format(model_name))