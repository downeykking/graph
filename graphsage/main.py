import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import GraphSage
from utils import load_data, accuracy, fix_seed, visualize
from sample import multihop_sampling

import time
import numpy as np

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
batch_size = 16
num_epochs = 20
seed = 2022
dropout = 0.5
hidden = 128
sample_num_lists = [10, 10]   # 每阶采样邻居的节点数
lr = 0.01
weight_decay = 5e-4
num_batch_per_epochs = 20    # 每个epoch循环的批次数
fix_seed(seed)


# data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

nclass = labels.max().item()+1
input_dim = features.size()[1]
hidden_dim = [hidden, nclass]   # 隐藏单元节点数

# Note: 采样的邻居阶数需要与GCN的层数保持一致
assert len(hidden_dim) == len(sample_num_lists)

model = GraphSage(input_dim=input_dim, hidden_dim=hidden_dim,
                  sample_num_lists=sample_num_lists).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(epoch):
    t = time.time()
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in range(num_batch_per_epochs):
        batch_src_index = np.random.choice(idx_train.cpu(), size=(batch_size,))
        batch_src_label = labels[batch_src_index]
        batch_sampling_result = multihop_sampling(batch_src_index, adj, sample_num_lists)
        batch_sampling_x = [features[idx].to(device) for idx in batch_sampling_result]
        
        outputs = model(batch_sampling_x)
        loss = criterion(outputs, batch_src_label)
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        total += batch_src_label.size(0)
        correct += (predicted == batch_src_label).sum().item()
        
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新

    model.eval() 
    with torch.no_grad():
        batch_sampling_result_val = multihop_sampling(idx_val.cpu().numpy(), adj, sample_num_lists)
        batch_sampling_x_val = [features[idx].to(device) for idx in batch_sampling_result_val]
        val_outputs = model(batch_sampling_x_val)
        acc_val = accuracy(val_outputs, labels[idx_val])

    print(
        'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}, Val-Acc: {:.2f}, time: {:.4f}s'.
        format(
            epoch + 1, num_epochs, total_loss/num_batch_per_epochs,
            correct * 100 / total, acc_val*100, time.time()-t)
        )
    
    
def test():
    model.eval() 
    with torch.no_grad():
        batch_sampling_result_test = multihop_sampling(idx_test.cpu().numpy(), adj, sample_num_lists)
        batch_sampling_x_test = [features[idx].to(device) for idx in batch_sampling_result_test]
        test_outputs = model(batch_sampling_x_test)
        acc_test = accuracy(test_outputs, labels[idx_test])

    print('test acc: {:.2f}'.format(acc_test*100))
    visualize(test_outputs, color=labels[idx_test], model_name=model.name)
    print('tSNE image is generated')
    
    
# Train model 
t_total = time.time()
for epoch in range(num_epochs):
    train(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()