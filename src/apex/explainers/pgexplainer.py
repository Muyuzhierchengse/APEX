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

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

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

    return subset, edge_index, inv, edge_mask


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

        pos = nx.kamada_kawai_layout(graph)
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
    def __init__(self, model, in_channels: int, device, explain_graph: bool = True, epochs: int = 5,
                 lr: float = 0.00001, coff_size: float = 0.001, coff_ent: float = 5e-6,
                 t0: float = 5.0, t1: float = 1.0, sample_bias: float = 0.0, num_hops: Optional[int] = None):
        super(PGExplainer, self).__init__()
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.in_channels = in_channels
        self.explain_graph = explain_graph

        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.sample_bias = sample_bias

        self.num_hops = self.update_num_hops(num_hops)
        self.init_bias = 0.0

        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
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

        if torch.isnan(pred_loss) or torch.isinf(pred_loss):
            print(f"Warning: pred_loss is NaN or Inf, prob[{ori_pred}]={prob[ori_pred]}")
            pred_loss = torch.tensor(0.0, device=prob.device)

        edge_mask = self.sparse_mask_values

        if edge_mask is None:
            print("Error: edge_mask is None!")
            edge_mask = torch.ones(1, device=prob.device) * 0.5
            self.sparse_mask_values = edge_mask
        
        if torch.isnan(edge_mask).any() or torch.isinf(edge_mask).any():
            print(f"Warning: edge_mask contains NaN or Inf. Shape: {edge_mask.shape}, "
                f"NaN count: {torch.isnan(edge_mask).sum()}, Inf count: {torch.isinf(edge_mask).sum()}")
            edge_mask = torch.nan_to_num(edge_mask, nan=0.5, posinf=1.0-EPS, neginf=EPS)
            self.sparse_mask_values = edge_mask

        edge_mask = torch.clamp(edge_mask, min=EPS, max=1.0-EPS)

        size_loss = self.coff_size * torch.sum(edge_mask)

        p = edge_mask

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

        if torch.isnan(term1).any():
            print(f"Warning: term1 has NaN. p stats - min: {p.min()}, max: {p.max()}, mean: {p.mean()}")
            term1 = torch.nan_to_num(term1, nan=0.0)
        
        if torch.isnan(term2).any():
            print(f"Warning: term2 has NaN. (1-p) stats - min: {(1-p).min()}, max: {(1-p).max()}")
            term2 = torch.nan_to_num(term2, nan=0.0)
        
        if torch.isnan(mask_ent).any() or torch.isinf(mask_ent).any():
            print(f"Warning: mask_ent has NaN/Inf after computation. Setting invalid values to 0")
            mask_ent = torch.nan_to_num(mask_ent, nan=0.0, posinf=0.0, neginf=0.0)

        if mask_ent.numel() > 0:
            mask_ent_loss = self.coff_ent * torch.sum(mask_ent) / max(mask_ent.numel(), 1)
        else:
            mask_ent_loss = torch.tensor(0.0, device=prob.device)

        if torch.isnan(mask_ent_loss) or torch.isinf(mask_ent_loss):
            print(f"Warning: mask_ent_loss is NaN or Inf, setting to 0")
            print(f"  mask_ent stats - min: {mask_ent.min()}, max: {mask_ent.max()}, mean: {mask_ent.mean()}")
            print(f"  edge_mask stats - min: {edge_mask.min()}, max: {edge_mask.max()}, mean: {edge_mask.mean()}")
            mask_ent_loss = torch.tensor(0.0, device=prob.device)

        loss = pred_loss + size_loss + mask_ent_loss

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

        if torch.isnan(gate_inputs).any() or torch.isinf(gate_inputs).any():
            print("Warning: gate_inputs has NaN/Inf in concrete_sample")
            gate_inputs = torch.nan_to_num(gate_inputs, nan=0.5, posinf=1.0-EPS, neginf=EPS)

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
        node_idx = kwargs.get('node_idx')
        nodesize = embed.shape[0]

        embed = embed.to(self.device)

        if torch.isnan(embed).any() or torch.isinf(embed).any():
            print("Warning: embed contains NaN or Inf, replacing with zeros")
            embed = torch.nan_to_num(embed, nan=0.0, posinf=1.0, neginf=-1.0)

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

        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
            if torch.isnan(h).any() or torch.isinf(h).any():
                h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
            h = torch.clamp(h, min=-10, max=10)

        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)

        values = torch.clamp(values, min=EPS, max=1.0-EPS)

        if torch.isnan(values).any() or torch.isinf(values).any():
            print("Warning: values contains NaN or Inf after concrete_sample")
            values = torch.nan_to_num(values, nan=0.5, posinf=1.0-EPS, neginf=EPS)
            values = torch.clamp(values, min=EPS, max=1.0-EPS)
        
        self.sparse_mask_values = values
        mask_sparse = torch.sparse_coo_tensor(
            edge_index, values, (nodesize, nodesize)
        ).to(self.device)
        mask_sigmoid = mask_sparse.to_dense()
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        edge_mask = torch.clamp(edge_mask, min=EPS, max=1.0-EPS)

        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        logits = self.model(x, edge_index)
        probs = F.softmax(logits, dim=-1)

        self.__clear_masks__()
        return probs, edge_mask

    def train_explanation_network(self, dataset):
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

        else:
            if hasattr(dataset, 'x'):
                data = dataset.to(self.device)
            else:
                data = dataset[0].to(self.device)

            with torch.no_grad():
                self.model.eval()
                explain_node_index_list = torch.where(data.train_mask)[0].tolist()
                pred_dict = {}
                logits = self.model(data.x, data.edge_index)
                for node_idx in tqdm.tqdm(explain_node_index_list):
                    pred_dict[node_idx] = logits[node_idx].argmax(-1).item()

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

            x_sub, edge_index_sub, _, subset, _ = self.get_subgraph(
                node_idx, x, edge_index
            )
            new_node_idx = torch.where(subset == node_idx)[0]
            embed_sub = self.model.get_emb(x_sub, edge_index_sub)

            _, edge_mask_sub = self.explain(
                x_sub, edge_index_sub, embed_sub,
                tmp=1.0, training=False, node_idx=new_node_idx
            )

            subset_set = set(subset.cpu().tolist())

            global_to_sub = {int(subset[i]): i for i in range(len(subset))}

            sub_src_global = subset[edge_index_sub[0]].cpu().tolist()
            sub_dst_global = subset[edge_index_sub[1]].cpu().tolist()
            sub_edge_dict = {}
            for i, (u, v) in enumerate(zip(sub_src_global, sub_dst_global)):
                sub_edge_dict[(u, v)] = edge_mask_sub[i].item()

            edge_mask_full = torch.zeros(num_edges, device=self.device)
            ei_cpu = edge_index.cpu()
            for i in range(num_edges):
                u = int(ei_cpu[0, i])
                v = int(ei_cpu[1, i])
                if (u, v) in sub_edge_dict:
                    edge_mask_full[i] = sub_edge_dict[(u, v)]
                elif (v, u) in sub_edge_dict:
                    edge_mask_full[i] = sub_edge_dict[(v, u)]

            raw_node_scores = []
            node_masks_list = []

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
                ns  = torch.zeros(num_nodes, device=self.device)
                deg = torch.zeros(num_nodes, device=self.device)
                src_g = edge_index[0]
                dst_g = edge_index[1]
                ones  = torch.ones(num_edges, device=self.device)
                ns.scatter_add_(0, src_g, edge_mask_full)
                ns.scatter_add_(0, dst_g, edge_mask_full)
                deg.scatter_add_(0, src_g, ones)
                deg.scatter_add_(0, dst_g, ones)
                node_scores = ns / (deg + EPS)
                raw_node_scores.append(node_scores)

                scores_sub_nb = node_scores[neighbor_tensor]
                k_sub         = min(num_keep, len(neighbors))
                topk_local    = scores_sub_nb.topk(k_sub).indices
                topk_global   = neighbor_tensor[topk_local]

                nm = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
                nm[topk_global] = 1.0
                node_masks_list.append(nm)

            self.last_node_scores = raw_node_scores

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