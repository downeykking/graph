import numpy as np

def sampling(src_nodes, neighbor_table, sample_num=10):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    
    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表
    
    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for sid in src_nodes:
        # 如果邻居>num_sample无放回 邻居<num_sample 放回
        adj = neighbor_table[sid]
        if len(adj) >= sample_num:
            res = np.random.choice(list(adj), size=sample_num, replace=False)
        else:
            res = np.random.choice(list(adj), size=sample_num, replace=True)
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, neighbor_table, k_sample_num):
    """根据源节点进行多阶采样
    
    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        k_sample_num {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射
    
    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(k_sample_num):
        hopk_result = sampling(sampling_result[k], neighbor_table, hopk_num)
        # 注意，仍然保留了源节点列表，采样后的结果加在后面，这样保证了可以不断的向后采样
        # 最终sampling_result中存放有0，1，2，...，k阶的采样结果（邻居）
        sampling_result.append(hopk_result)
    return sampling_result