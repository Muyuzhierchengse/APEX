from dig.xgraph.dataset import *
import torch
from torch.utils.data import random_split
from tqdm import tqdm
from torch_geometric.data import Data
import scipy.sparse as ssp
import random
from torch_geometric.datasets import TUDataset
import os

def split_dataset(dataset, dataset_split=[0.8, 0.1, 0.1], seed=0):
    dataset_len = len(dataset)
    dataset_split = [int(dataset_len * dataset_split[0]),
                     int(dataset_len * dataset_split[1]),
                     0]
    dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, dataset_split, 
                                                 generator=generator)

    return {'train': train_set, 'val': val_set, 'test': test_set}


def load_dataset(data_path, dataset):
    if dataset in ['BA_shapes']:
        dataset = SynGraphDataset(data_path, dataset)
        data = dataset[0]
        dim_node = dataset.num_node_features
        dim_edge = dataset.num_edge_features
        num_classes = dataset.num_classes
        return data, 1, dim_node, num_classes

    if dataset in ['Graph-SST2', 'Graph-Twitter']:
        dataset = SentiGraphDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y
    if dataset in ['BBBP']:
        dataset = MoleculeDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y[:, 0]
    if dataset in ['BA_2Motifs']:
        dataset = SynGraphDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y
    # 新增 BACE 数据集支持（单任务二分类）
    if dataset in ['BACE']:
        dataset = MoleculeDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y[:, 0]  # BACE 只有1列标签
    if dataset in ['ClinTox']:
        dataset = MoleculeDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        raw_y = dataset.data.y[:, 1]           # ← CT_TOX

        # 先在完整数据集上把 y 替换为单列，再过滤
        # 必须先赋值再过滤，否则 dataset[valid_idx].data.y 仍是全量数据的切片
        dataset.data.y = raw_y.clamp(0, 1).long()

        valid_mask = ~torch.isnan(raw_y) & (raw_y != -1)
        valid_idx = valid_mask.nonzero(as_tuple=True)[0].tolist()
        dataset = dataset[valid_idx]
        # 新增 Tox21 数据集支持
    if dataset in ['Tox21']:
        dataset = MoleculeDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        raw_y = dataset.data.y[:, 0]

        # 先赋值再过滤（与 ClinTox 保持一致）
        dataset.data.y = raw_y.clamp(0, 1).long()

        valid_mask = ~torch.isnan(raw_y) & (raw_y != -1)
        valid_idx = valid_mask.nonzero(as_tuple=True)[0].tolist()
        dataset = dataset[valid_idx]

    if dataset in ['MUTAG']:
        raw = TUDataset(root=os.path.join(data_path, 'TUDataset'),
                        name='MUTAG',
                        use_node_attr=True)
        # 标签原始为 {-1, 1}，统一映射到 {0, 1}
        ys = raw.data.y.clone()
        ys = ((ys + 1) // 2).clamp(0, 1).long()   # -1→0, 1→1
        raw.data.x  = raw.data.x.to(torch.float32)
        raw.data.y  = ys
        dataset = raw

    # ---------- Mutagenicity ----------
    # 4337张图，二分类（诱变性），节点特征为14维原子类型 one-hot
    if dataset in ['Mutagenicity']:
        raw = TUDataset(root=os.path.join(data_path, 'TUDataset'),
                        name='Mutagenicity',
                        use_node_attr=True)
        raw.data.x = raw.data.x.to(torch.float32)
        raw.data.y = raw.data.y.long()
        # 过滤空图（少数样本边为空）
        valid_idx = [i for i, d in enumerate(raw)
                     if d.x is not None and d.x.size(0) > 0
                     and d.edge_index is not None and d.edge_index.size(1) > 0]
        dataset = raw[valid_idx]

    # ---------- Spurious-Motif ----------
    # DIG 内置合成数据集，偏置 b 可取 0.33 / 0.60 / 0.90
    # b 越大偏置越强；默认使用 0.60
    if dataset in ['Spurious-Motif']:
        dataset = SynGraphDataset(data_path, 'spurious_motif', b=0.60)
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y.long()

    # ---------- NCI1 ----------
    # ~4110张图，二分类（抗癌活性），无连续节点特征，仅离散标签
    # use_node_attr=False 时节点特征为 one-hot 编码的节点标签（37维）
    if dataset in ['NCI1']:
        raw = TUDataset(root=os.path.join(data_path, 'TUDataset'),
                        name='NCI1',
                        use_node_attr=False)
        # TUDataset 对 NCI1 自动生成 one-hot 节点特征
        raw.data.x = raw.data.x.to(torch.float32)
        raw.data.y = raw.data.y.long()
        dataset = raw
    # 新增 ToxCast 数据集支持
    if dataset in ['ToxCast']:
        dataset = MoleculeDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        raw_y = dataset.data.y[:, 0]

        # 先赋值再过滤，与其他数据集保持一致
        dataset.data.y = raw_y.clamp(0, 1).long()

        valid_mask = ~torch.isnan(raw_y) & (raw_y != -1)
        valid_idx = valid_mask.nonzero(as_tuple=True)[0].tolist()
        dataset = dataset[valid_idx]

        # 空图过滤保留
        dataset = dataset[[i for i, data in enumerate(dataset)
                        if data.x is not None and data.x.size(0) > 0
                        and data.edge_index is not None and data.edge_index.size(1) > 0]]

    dataset.data.y = dataset.data.y.long()
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes

    splitted_dataset = split_dataset(dataset)
    return splitted_dataset, 1, dim_node, num_classes


def construct_pyg_graph(node_ids, adj, node_features, y):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])

    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, node_id=node_ids, num_nodes=num_nodes)
    return data


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A,
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(u, num_hops, A, node_features, y):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [u]
    dists = [0, 0]
    visited = set([u])
    fringe = set([u])
    for dist in range(1, num_hops + 1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)

        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, node_features, y


def extract_enclosing_subgraphs(node_indices, A, x, label, num_hops):
    data_list = []
    for idx, u in enumerate(tqdm(node_indices)):
        tmp = k_hop_subgraph(u, num_hops, A, node_features=x, y=label[idx])
        data = construct_pyg_graph(*tmp)
        data_list.append(data)

    return data_list