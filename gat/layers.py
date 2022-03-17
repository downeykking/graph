import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    #layer层初始化中，除去基本的输入输出层，还需要指定alpha,concat
    #alpha 用于指定激活函数LeakyRelu中的参数
    #concat用于指定该层输出是否要拼接，因为用到了多头注意力机制
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        #W表示该层的特征变化矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        #一种初始化的方法
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #a表示用于计算注意力系数的单层前馈神经网络。
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.mm(h, self.W)

        # 用于得到计算注意力系数的矩阵。
        # 这里采用的是矩阵形式，一次计算便可得到网络中所有结点对之间的注意力系数
        # a_input = self._prepare_attentional_mechanism_input(Wh)
        # a_input (n, n, 2*out_features) 
        # a (2*out_features, 1) 
        # 二者相乘则为(n, n, 1)，需要通过 squeeze操作去掉第3个维度
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        e = self._prepare_attentional_mechanism_input(Wh)

        # 注意力系数可能为0，这里需要进行筛选操作，便于后续乘法
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 输出结合注意力系数的特征矩阵
        # (N, N) * (N, out_dim)
        h_prime = torch.matmul(attention, Wh)

        # 如果是多头拼接，则进行激活，反之不必
        if self.concat:
            return F.elu(h_prime)
        
        else:
            return h_prime

    # out of memory
    # def _prepare_attentional_mechanism_input(self, Wh):
    #     # number of nodes
    #     N = Wh.size()[0]
    #     Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
    #     Wh_repeated_alternating = Wh.repeat(N, 1)
    #     all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
    #     #将维度改为(N, N, 2 * self.out_features)
    #     return all_combinations_matrix.view(N, N, 2 * self.out_features)
    
    
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
    
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'