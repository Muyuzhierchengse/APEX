"""
Description: The implement of PGExplainer model
<https://arxiv.org/abs/2011.04573>
"""

import tqdm
import time
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from math import sqrt
from torch import Tensor
from textwrap import wrap
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Tuple, List, Dict, Optional
from dig.xgraph.method.shapley import gnn_score, GnnNetsNC2valueFunc, GnnNetsGC2valueFunc, sparsity
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem

EPS = 1e-6


def k_hop_subgraph_with_default_whole_graph(
        edge_index, node_idx=None, num_hops=3, relabel_nodes=False,
        num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx


def calculate_selected_nodes(data, edge_mask, top_k):
    threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
    hard_mask = (edge_mask > threshold).cpu()
    edge_idx_list = torch.where(hard_mask == 1)[0]
    selected_nodes = []
    edge_index = data.edge_index.cpu().numpy()
    for edge_idx in edge_idx_list:
        selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
    selected_nodes = list(set(selected_nodes))
    return selected_nodes


class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot_subgraph(self, graph, nodelist, colors='#FFA500', labels=None, edge_color='gray',
                      edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)

        if self.is_show:
            plt.show()
        plt.close('all')

    def plot_subgraph_with_nodes(self, graph, nodelist, node_idx, colors='#FFA500', labels=None, edge_color='gray',
                                 edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph) # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()

    def plot_ba2motifs(self,
                       graph,
                       nodelist,
                       edgelist=None,
                       title_sentence=None,
                       figname=None):
        return self.plot_subgraph(graph, nodelist,
                                  edgelist=edgelist,
                                  title_sentence=title_sentence,
                                  figname=figname)

    def plot_molecule(self,
                      graph,
                      nodelist,
                      x,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        # collect the text information and node color
        if self.dataset_name == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in MoleculeNet.names.keys():
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist, colors=colors, labels=node_labels,
                           edgelist=edgelist, edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=title_sentence,
                           figname=figname)

    def plot_sentence(self,
                      graph,
                      nodelist,
                      words,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]
            nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=5,
                                   edge_color='yellow', arrows=False)

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='grey', arrows=False)
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()

    def plot_bashapes(self,
                      graph,
                      nodelist,
                      y,
                      node_idx,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph, nodelist, node_idx, colors,
                                      edgelist=edgelist,
                                      figname=figname,
                                      title_sentence=title_sentence,
                                      subgraph_edge_color='black')

    def get_topk_edges_subgraph(self,
                                edge_index,
                                edge_mask,
                                top_k,
                                un_directed=False):
        if un_directed:
            top_k = 2 * top_k
        edge_mask = edge_mask.reshape(-1)
        thres_index = max(edge_mask.shape[0] - top_k, 0)
        threshold = float(edge_mask.reshape(-1).sort().values[thres_index])
        hard_edge_mask = (edge_mask >= threshold)
        selected_edge_idx = np.where(hard_edge_mask == 1)[0].tolist()
        nodelist = []
        edgelist = []
        for edge_idx in selected_edge_idx:
            edges = edge_index[:, edge_idx].tolist()
            nodelist += [int(edges[0]), int(edges[1])]
            edgelist.append((edges[0], edges[1]))
        nodelist = list(set(nodelist))
        return nodelist, edgelist

    def plot_soft_edge_mask(self,
                            graph,
                            edge_mask,
                            top_k,
                            un_directed,
                            figname,
                            title_sentence=None,
                            **kwargs):
        edge_index = torch.tensor(list(graph.edges())).T
        edge_mask = torch.FloatTensor(edge_mask)
        if self.dataset_name.lower() in ['ba_2motifs', 'ba_lrp']:
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_ba2motifs(graph, nodelist, edgelist, title_sentence=title_sentence, figname=figname)

        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_molecule(graph, nodelist, x, edgelist, title_sentence=title_sentence, figname=figname)

        elif self.dataset_name.lower() in ['ba_shapes', 'ba_shapes', 'tree_grid', 'tree_cycle']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_bashapes(graph, nodelist, y, node_idx, edgelist, title_sentence=title_sentence, figname=figname)

        elif self.dataset_name.lower() in ['Graph_SST2'.lower()]:
            words = kwargs.get('words')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_sentence(graph, nodelist,
                               words=words,
                               edgelist=edgelist,
                               title_sentence=title_sentence,
                               figname=figname)

        else:
            raise NotImplementedError


class PGExplainer(nn.Module):
    r"""
    An implementation of PGExplainer in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.

    Args:
        model (:class:`torch.nn.Module`): The target model prepared to explain
        in_channels (:obj:`int`): Number of input channels for the explanation network
        explain_graph (:obj:`bool`): Whether to explain graph classification model (default: :obj:`True`)
        epochs (:obj:`int`): Number of epochs to train the explanation network
        lr (:obj:`float`): Learning rate to train the explanation network
        coff_size (:obj:`float`): Size regularization to constrain the explanation size
        coff_ent (:obj:`float`): Entropy regularization to constrain the connectivity of explanation
        t0 (:obj:`float`): The temperature at the first epoch
        t1(:obj:`float`): The temperature at the final epoch
        num_hops (:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
        (default: :obj:`None`)

    .. note: For node classification model, the :attr:`explain_graph` flag is False.
      If :attr:`num_hops` is set to :obj:`None`, it will be automatically calculated by calculating the
      :class:`torch_geometric.nn.MessagePassing` layers in the :attr:`model`.

    """
    def __init__(self, model, in_channels: int, device, explain_graph: bool = True, epochs: int = 5,
                 lr: float = 0.00001, coff_size: float = 0.001, coff_ent: float = 5e-6,
                 t0: float = 5.0, t1: float = 1.0, sample_bias: float = 0.0, num_hops: Optional[int] = None):
        super(PGExplainer, self).__init__()
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.in_channels = in_channels
        self.explain_graph = explain_graph

        # training parameters for PGExplainer
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.sample_bias = sample_bias

        self.num_hops = self.update_num_hops(num_hops)
        self.init_bias = 0.0

        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        r""" Set the edge weights before message passing

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            edge_mask (:obj:`torch.Tensor`): Edge weight matrix before message passing
              (default: :obj:`None`)

        The :attr:`edge_mask` will be randomly initialized when set to :obj:`None`.

        .. note:: When you use the :meth:`~PGExplainer.__set_masks__`,
          the explain flag for all the :class:`torch_geometric.nn.MessagePassing`
          modules in :attr:`model` will be assigned with :obj:`True`. In addition,
          the :attr:`edge_mask` will be assigned to all the modules.
          Please take :meth:`~PGExplainer.__clear_masks__` to reset.
        """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None, and set the explain flag to :obj:`False` """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def update_num_hops(self, num_hops: int):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __loss__(self, prob: Tensor, ori_pred: int):
        logit = prob[ori_pred]
        logit = torch.clamp(logit, min=EPS, max=1.0-EPS)
        pred_loss = -torch.log(logit + EPS)
        
        # 检查pred_loss是否有效
        if torch.isnan(pred_loss) or torch.isinf(pred_loss):
            print(f"Warning: pred_loss is NaN or Inf, prob[{ori_pred}]={prob[ori_pred]}")
            pred_loss = torch.tensor(0.0, device=prob.device)

        # size
        edge_mask = self.sparse_mask_values
        
        # 详细检查edge_mask
        if edge_mask is None:
            print("Error: edge_mask is None!")
            edge_mask = torch.ones(1, device=prob.device) * 0.5
            self.sparse_mask_values = edge_mask
        
        if torch.isnan(edge_mask).any() or torch.isinf(edge_mask).any():
            print(f"Warning: edge_mask contains NaN or Inf. Shape: {edge_mask.shape}, "
                f"NaN count: {torch.isnan(edge_mask).sum()}, Inf count: {torch.isinf(edge_mask).sum()}")
            edge_mask = torch.nan_to_num(edge_mask, nan=0.5, posinf=1.0-EPS, neginf=EPS)
            self.sparse_mask_values = edge_mask
        
        # 确保edge_mask在有效范围
        edge_mask = torch.clamp(edge_mask, min=EPS, max=1.0-EPS)
        
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy - 完全重写，使用更稳定的实现
        # 使用二值交叉熵的稳定版本
        # H = -p*log(p) - (1-p)*log(1-p)
        
        # 方法1：直接使用PyTorch的binary_cross_entropy
        # 但我们需要自己实现熵，因为BCE需要target
        
        # 方法2：使用数值稳定的log
        # 避免直接计算log(0)的情况
        p = edge_mask
        
        # 使用torch.where来避免log(0)
        # 当p接近0时，-p*log(p)趋向于0
        # 当p接近1时，-(1-p)*log(1-p)趋向于0
        term1 = torch.where(
            p > EPS,
            -p * torch.log(p),
            torch.zeros_like(p)
        )
        
        term2 = torch.where(
            (1.0 - p) > EPS,
            -(1.0 - p) * torch.log(1.0 - p),
            torch.zeros_like(p)
        )
        
        mask_ent = term1 + term2
        
        # 检查mask_ent的每个组成部分
        if torch.isnan(term1).any():
            print(f"Warning: term1 has NaN. p stats - min: {p.min()}, max: {p.max()}, mean: {p.mean()}")
            term1 = torch.nan_to_num(term1, nan=0.0)
        
        if torch.isnan(term2).any():
            print(f"Warning: term2 has NaN. (1-p) stats - min: {(1-p).min()}, max: {(1-p).max()}")
            term2 = torch.nan_to_num(term2, nan=0.0)
        
        if torch.isnan(mask_ent).any() or torch.isinf(mask_ent).any():
            print(f"Warning: mask_ent has NaN/Inf after computation. Setting invalid values to 0")
            mask_ent = torch.nan_to_num(mask_ent, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 使用sum而不是mean，避免除以0的情况
        if mask_ent.numel() > 0:
            mask_ent_loss = self.coff_ent * torch.sum(mask_ent) / max(mask_ent.numel(), 1)
        else:
            mask_ent_loss = torch.tensor(0.0, device=prob.device)
        
        # 最终检查mask_ent_loss
        if torch.isnan(mask_ent_loss) or torch.isinf(mask_ent_loss):
            print(f"Warning: mask_ent_loss is NaN or Inf, setting to 0")
            print(f"  mask_ent stats - min: {mask_ent.min()}, max: {mask_ent.max()}, mean: {mask_ent.mean()}")
            print(f"  edge_mask stats - min: {edge_mask.min()}, max: {edge_mask.max()}, mean: {edge_mask.mean()}")
            mask_ent_loss = torch.tensor(0.0, device=prob.device)

        loss = pred_loss + size_loss + mask_ent_loss
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Final loss is NaN or Inf. pred_loss={pred_loss}, size_loss={size_loss}, mask_ent_loss={mask_ent_loss}")
            loss = torch.tensor(0.0, device=prob.device, requires_grad=True)
        
        return loss

    def get_subgraph(self,
                     node_idx: int,
                     x: Tensor,
                     edge_index: Tensor,
                     y: Optional[Tensor] = None,
                     **kwargs)\
            -> Tuple[Tensor, Tensor, Tensor, List, Dict]:
        r""" extract the subgraph of target node

        Args:
            node_idx (:obj:`int`): The node index
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            y (:obj:`torch.Tensor`, :obj:`None`): Node label matrix with shape :obj:`[num_nodes]`
              (default :obj:`None`)
            kwargs(:obj:`Dict`, :obj:`None`): Additional parameters

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`,
          :obj:`List`, :class:`Dict`)

        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, self.num_hops, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        if y is not None:
            y = y[subset]
        return x, edge_index, y, subset, kwargs

    def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
        r""" Sample from the instantiation of concrete distribution when training """
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(log_alpha.shape, device=log_alpha.device)
            random_noise = torch.clamp(random_noise, min=1e-8, max=1.0 - 1e-8)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = torch.clamp(gate_inputs, min=-10, max=10)
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()
        
        # 添加检查
        if torch.isnan(gate_inputs).any() or torch.isinf(gate_inputs).any():
            print("Warning: gate_inputs has NaN/Inf in concrete_sample")
            gate_inputs = torch.nan_to_num(gate_inputs, nan=0.5, posinf=1.0-EPS, neginf=EPS)
        
        # 确保输出在有效范围内
        gate_inputs = torch.clamp(gate_inputs, min=EPS, max=1.0-EPS)
        
        return gate_inputs
    def explain(self,
                x: Tensor,
                edge_index: Tensor,
                embed: Tensor,
                tmp: float = 1.0,
                training: bool = False,
                **kwargs)\
            -> Tuple[float, Tensor]:
        r""" explain the GNN behavior for graph with explanation network """
        node_idx = kwargs.get('node_idx')
        nodesize = embed.shape[0]
        
        # 确保embed在正确的设备上并进行数值稳定化
        embed = embed.to(self.device)
        
        # 检查embed是否包含NaN或Inf
        if torch.isnan(embed).any() or torch.isinf(embed).any():
            print("Warning: embed contains NaN or Inf, replacing with zeros")
            embed = torch.nan_to_num(embed, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 归一化embed以防止数值溢出
        embed = torch.clamp(embed, min=-10, max=10)
        
        if self.explain_graph:
            col, row = edge_index
            f1 = embed[col]
            f2 = embed[row]
            f12self = torch.cat([f1, f2], dim=-1)
        else:
            col, row = edge_index
            f1 = embed[col]
            f2 = embed[row]
            self_embed = embed[node_idx].repeat(f1.shape[0], 1)
            f12self = torch.cat([f1, f2, self_embed], dim=-1)

        # using the node embedding to calculate the edge weight
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
            # 在每层后进行数值检查和稳定化
            if torch.isnan(h).any() or torch.isinf(h).any():
                h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
            h = torch.clamp(h, min=-10, max=10)
        
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        
        # 强制将values限制在(0, 1)范围内，避免边界值
        values = torch.clamp(values, min=EPS, max=1.0-EPS)
        
        # 检查values是否有效
        if torch.isnan(values).any() or torch.isinf(values).any():
            print("Warning: values contains NaN or Inf after concrete_sample")
            values = torch.nan_to_num(values, nan=0.5, posinf=1.0-EPS, neginf=EPS)
            values = torch.clamp(values, min=EPS, max=1.0-EPS)
        
        self.sparse_mask_values = values
        mask_sparse = torch.sparse_coo_tensor(
            edge_index, values, (nodesize, nodesize)
        ).to(self.device)
        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]
        
        # 再次确保edge_mask在有效范围内
        edge_mask = torch.clamp(edge_mask, min=EPS, max=1.0-EPS)

        # inverse the weights before sigmoid in MessagePassing Module
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # the model prediction with edge mask
        logits = self.model(x, edge_index)
        probs = F.softmax(logits, dim=-1)

        self.__clear_masks__()
        return probs, edge_mask

    def train_explanation_network(self, dataset):
        r""" training the explanation network by gradient descent(GD) using Adam optimizer """
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        if self.explain_graph:
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(self.device)
                    logits = self.model(data.x, data.edge_index)
                    emb = self.model.get_emb(data.x, data.edge_index)
                    emb_dict[gid] = emb.data.cpu()
                    ori_pred_dict[gid] = logits.argmax(-1).data.cpu()

            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                pred_list = []
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid]
                    data.to(self.device)
                    prob, edge_mask = self.explain(data.x, data.edge_index, embed=emb_dict[gid], tmp=tmp, training=True)
                    loss_tmp = self.__loss__(prob.squeeze(), ori_pred_dict[gid])
                    loss_tmp.backward()
                    loss += loss_tmp.item()
                    pred_label = prob.argmax(-1).item()
                    pred_list.append(pred_label)

                has_nan_grad = False
                for param in self.elayers.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    print(f"Warning: Skipping optimizer step due to NaN/Inf gradients at epoch {epoch}")
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(self.elayers.parameters(), max_norm=1.0)
                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss}')

        # ── 节点分类分支 ──────────────────────────────────────────────────────
        else:
            # dataset 可能是单张 Data 对象（节点分类场景，如 BA_shapes）
            # 也可能是支持下标的数据集对象，统一处理
            if hasattr(dataset, 'x'):
                # 单张 Data 对象，直接使用
                data = dataset.to(self.device)
            else:
                # 列表/数据集对象，取第一张图
                data = dataset[0].to(self.device)

            with torch.no_grad():
                self.model.eval()
                explain_node_index_list = torch.where(data.train_mask)[0].tolist()
                pred_dict = {}
                logits = self.model(data.x, data.edge_index)
                for node_idx in tqdm.tqdm(explain_node_index_list):
                    pred_dict[node_idx] = logits[node_idx].argmax(-1).item()

            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                optimizer.zero_grad()
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                self.elayers.train()
                tic = time.perf_counter()
                for iter_idx, node_idx in tqdm.tqdm(enumerate(explain_node_index_list)):
                    with torch.no_grad():
                        x, edge_index, y, subset, _ = \
                            self.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
                        emb = self.model.get_emb(x, edge_index)
                        new_node_index = int(torch.where(subset == node_idx)[0])
                    pred, edge_mask = self.explain(x, edge_index, emb, tmp, training=True, node_idx=new_node_index)
                    loss_tmp = self.__loss__(pred[new_node_index], pred_dict[node_idx])
                    loss_tmp.backward()
                    loss += loss_tmp.item()

                has_nan_grad = False
                for param in self.elayers.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    print(f"Warning: Skipping optimizer step due to NaN/Inf gradients at epoch {epoch}")
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(self.elayers.parameters(), max_norm=1.0)
                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}')
            print(f"training time is {duration:.5}s")

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs)\
            -> Tuple[None, List, List[Dict]]:
        num_classes = kwargs.get('num_classes')
        top_k = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        self.__clear_masks__()
        logits = self.model(x, edge_index)
        probs = F.softmax(logits, dim=-1)
        pred_labels = probs.argmax(dim=-1)
        embed = self.model.get_emb(x, edge_index)

        # ── 图分类分支（完全不动）────────────────────────────────────────────
        if self.explain_graph:
            probs = probs.squeeze()
            label = pred_labels
            _, edge_mask = self.explain(x, edge_index, embed=embed, tmp=1.0, training=False)
            return [edge_mask for _ in range(num_classes)]

            data = Data(x=x, edge_index=edge_index)
            selected_nodes = calculate_selected_nodes(data, edge_mask, top_k)
            masked_node_list = [node for node in range(data.x.shape[0]) if node in selected_nodes]
            maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            value_func = GnnNetsGC2valueFunc(self.model, target_class=label)

            masked_pred = gnn_score(masked_node_list, data,
                                    value_func=value_func,
                                    subgraph_building_method='zero_filling')

            maskout_pred = gnn_score(maskout_nodes_list, data, value_func,
                                     subgraph_building_method='zero_filling')

            sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]

            pred_mask = [edge_mask]
            related_preds = [{
                'masked': masked_pred,
                'maskout': maskout_pred,
                'origin': probs[label],
                'sparsity': sparsity_score}]
            return None, pred_mask, related_preds

        # ── 节点分类分支 ──────────────────────────────────────────────────────
        else:
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None, "please input the node_idx"

            num_nodes = x.size(0)
            num_edges = edge_index.size(1)

            max_nodes = kwargs.get('max_nodes', None)
            sparsity_val = kwargs.get('sparsity', 0.5)
            if max_nodes is not None:
                num_keep = max(1, int(max_nodes))
            else:
                num_keep = max(1, int(num_nodes * (1 - sparsity_val)))

            # ── 1. 提取 k-hop 子图（用于 explain 网络推理）────────────────
            x_sub, edge_index_sub, _, subset, _ = self.get_subgraph(
                node_idx, x, edge_index
            )
            new_node_idx = torch.where(subset == node_idx)[0]
            embed_sub = self.model.get_emb(x_sub, edge_index_sub)

            # ── 2. 在子图上运行 explain，得到子图边掩码 ───────────────────
            _, edge_mask_sub = self.explain(
                x_sub, edge_index_sub, embed_sub,
                tmp=1.0, training=False, node_idx=new_node_idx
            )
            # edge_mask_sub shape: [E_sub]，对应 edge_index_sub 的边

            # ── 3. 将子图边掩码映射回全图 ─────────────────────────────────
            # subset 记录子图节点在全图中的原始编号（relabel_nodes=True 时已重标签）
            # edge_index_sub 是重标签后的子图边，需要用 subset 还原全图节点编号
            # 先找全图 edge_index 中哪些边属于子图内部
            subset_set = set(subset.cpu().tolist())

            # 全图节点编号 → 子图节点编号的映射
            global_to_sub = {int(subset[i]): i for i in range(len(subset))}

            # 构建子图边（全图编号）→ 掩码值 的查找表
            # edge_index_sub 是子图内重标签编号，转回全图编号
            sub_src_global = subset[edge_index_sub[0]].cpu().tolist()
            sub_dst_global = subset[edge_index_sub[1]].cpu().tolist()
            sub_edge_dict = {}
            for i, (u, v) in enumerate(zip(sub_src_global, sub_dst_global)):
                sub_edge_dict[(u, v)] = edge_mask_sub[i].item()

            # 全图 edge_mask：子图内的边取对应掩码值，子图外的边取 0
            edge_mask_full = torch.zeros(num_edges, device=self.device)
            ei_cpu = edge_index.cpu()
            for i in range(num_edges):
                u = int(ei_cpu[0, i])
                v = int(ei_cpu[1, i])
                if (u, v) in sub_edge_dict:
                    edge_mask_full[i] = sub_edge_dict[(u, v)]
                elif (v, u) in sub_edge_dict:
                    # 无向图：若只存了一个方向，也匹配
                    edge_mask_full[i] = sub_edge_dict[(v, u)]

            # ── 4. 从全图 edge_mask 聚合出全图节点分数 ───────────────────
            raw_node_scores = []
            node_masks_list = []

            # 目标节点的直接邻居（含自身），用于限定 node_mask 范围
            s_cpu = edge_index[0].cpu()
            d_cpu = edge_index[1].cpu()
            node_idx_scalar = int(node_idx) if isinstance(node_idx, int) \
                else node_idx.item()
            neighbors = set()
            neighbors.add(node_idx_scalar)
            for i in range(num_edges):
                u, v = s_cpu[i].item(), d_cpu[i].item()
                if u == node_idx_scalar:
                    neighbors.add(v)
                if v == node_idx_scalar:
                    neighbors.add(u)
            neighbors       = sorted(neighbors)
            neighbor_tensor = torch.tensor(
                neighbors, dtype=torch.long, device=self.device
            )

            for cls in range(num_classes):
                # 所有类别共享同一套子图掩码（PGExplainer 只训练一个掩码网络）
                # node_scores：通过边掩码聚合
                ns  = torch.zeros(num_nodes, device=self.device)
                deg = torch.zeros(num_nodes, device=self.device)
                src_g = edge_index[0]
                dst_g = edge_index[1]
                ones  = torch.ones(num_edges, device=self.device)
                ns.scatter_add_(0, src_g, edge_mask_full)
                ns.scatter_add_(0, dst_g, edge_mask_full)
                deg.scatter_add_(0, src_g, ones)
                deg.scatter_add_(0, dst_g, ones)
                node_scores = ns / (deg + EPS)   # [N]
                raw_node_scores.append(node_scores)

                # node_mask：在直接邻居范围内选 top-k
                scores_sub_nb = node_scores[neighbor_tensor]
                k_sub         = min(num_keep, len(neighbors))
                topk_local    = scores_sub_nb.topk(k_sub).indices
                topk_global   = neighbor_tensor[topk_local]

                nm = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
                nm[topk_global] = 1.0
                node_masks_list.append(nm)

            # ── 5. 挂载供 main.py 使用的属性 ─────────────────────────────
            self.last_node_scores = raw_node_scores

            # 返回与图分类一致的格式：(edge_masks, node_masks)
            edge_masks = [edge_mask_full for _ in range(num_classes)]
            return edge_masks, node_masks_list

    def visualization(self, data: Data, edge_mask: Tensor, top_k: int, plot_utils: PlotUtils,
                      words: Optional[list] = None, node_idx: int = None, vis_name: Optional[str] = None):
        if vis_name is None:
            vis_name = f"filename.png"

        data = data.to('cpu')
        edge_mask = edge_mask.to('cpu')
        if self.explain_graph:
            graph = to_networkx(data)
            if words is None:
                plot_utils.plot_soft_edge_mask(graph,
                                               edge_mask,
                                               top_k=top_k,
                                               un_directed=True,
                                               words=words,
                                               figname=vis_name)
            else:
                plot_utils.plot_soft_edge_mask(graph,
                                               edge_mask,
                                               top_k=top_k,
                                               un_directed=True,
                                               x=data.x,
                                               figname=vis_name)
        else:
            assert node_idx is not None, "visualization method doesn't get the target node index"
            x, edge_index, y, subset, kwargs = \
                self.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
            new_node_idx = torch.where(subset == node_idx)[0]
            new_data = Data(x=x, edge_index=edge_index)
            graph = to_networkx(new_data)
            plot_utils.plot_soft_edge_mask(graph,
                                           edge_mask,
                                           top_k=top_k,
                                           un_directed=True,
                                           y=y,
                                           node_idx=new_node_idx,
                                           figname=vis_name)

    def __repr__(self):
        return f'{self.__class__.__name__}()'