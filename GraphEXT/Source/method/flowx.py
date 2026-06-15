from rdkit import Chem
from torch_geometric.nn import MessagePassing
from itertools import combinations
from typing import List, Tuple, Union, Dict
import networkx as nx
import numpy as np
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from matplotlib.axes import Axes
from torch_geometric.utils.num_nodes import maybe_num_nodes
class ExplainerBase(nn.Module):

    def __init__(self, model: nn.Module, epochs: int = 0, lr: float = 0, explain_graph: bool = False,
                 molecule: bool = False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.explain_graph = explain_graph
        self.molecule = molecule
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

        self.ori_pred = None
        self.ex_labels = None
        self.edge_mask = None
        self.hard_edge_mask = None

        self.num_edges = None
        self.num_nodes = None
        self.device = None
        self.table = Chem.GetPeriodicTable().GetElementSymbol

    def __set_masks__(self, x: Tensor, edge_index: Tensor, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        self.node_feat_mask = torch.nn.Parameter(torch.randn(F, requires_grad=True, device=self.device) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E, requires_grad=True, device=self.device) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def __num_hops__(self):
        if self.explain_graph:
            return -1
        else:
            return self.num_layers

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx: int, x: Tensor, edge_index: Tensor, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = subgraph(
            node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device

    def control_sparsity(self, mask: Tensor, sparsity=None, **kwargs):
        if sparsity is None:
            sparsity = 0.7

        if not self.explain_graph:
            assert self.hard_edge_mask is not None
            mask_indices = torch.where(self.hard_edge_mask)[0]
            sub_mask = mask[self.hard_edge_mask]
            mask_len = sub_mask.shape[0]
            _, sub_indices = torch.sort(sub_mask, descending=True)
            split_point = int((1 - sparsity) * mask_len)
            important_sub_indices = sub_indices[: split_point]
            important_indices = mask_indices[important_sub_indices]
            unimportant_sub_indices = sub_indices[split_point:]
            unimportant_indices = mask_indices[unimportant_sub_indices]
            trans_mask = mask.clone()
            trans_mask[:] = - float('inf')
            trans_mask[important_indices] = float('inf')
        else:
            _, indices = torch.sort(mask, descending=True)
            mask_len = mask.shape[0]
            split_point = int((1 - sparsity) * mask_len)
            important_indices = indices[: split_point]
            unimportant_indices = indices[split_point:]
            trans_mask = mask.clone()
            trans_mask[important_indices] = float('inf')
            trans_mask[unimportant_indices] = - float('inf')

        return trans_mask

    def batch_input(self, x, edge_index, batch_size):
        data_list = []
        for _ in range(batch_size):
            data_list.append(Data(x=x.clone(), edge_index=edge_index.clone()))
    
        return Batch.from_data_list(data_list)

    def visualize_graph(self, node_idx: int, edge_index: Tensor, edge_mask: Tensor, y: Tensor = None,
                        threshold: float = None, nolabel: bool = True, **kwargs) -> Tuple[Axes, nx.DiGraph]:

        edge_index, _ = add_self_loops(edge_index, num_nodes=kwargs.get('num_nodes'))
        assert edge_mask.size(0) == edge_index.size(1)

        if self.molecule:
            atomic_num = torch.clone(y)

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = subgraph(
            node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        # --- temp ---
        edge_mask[edge_mask == float('inf')] = 1
        edge_mask[edge_mask == - float('inf')] = 0
        # ---

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if kwargs.get('dataset_name') == 'ba_lrp':
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset]

        if self.molecule:
            atom_colors = {6: '#8c69c5', 7: '#71bcf0', 8: '#aef5f1', 9: '#bdc499', 15: '#c22f72', 16: '#f3ea19',
                           17: '#bdc499', 35: '#cc7161'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]
        else:
            atom_colors = {0: '#8c69c5', 1: '#c56973', 2: '#a1c569', 3: '#69c5ba'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]


        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 250
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        # calculate Graph positions
        pos = nx.kamada_kawai_layout(G)
        ax = plt.gca()

        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    lw=max(data['att'], 0.5) * 2,
                    alpha=max(data['att'], 0.4),  # alpha control transparency
                    color='#e1442a',  # color control color
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.08",  # rad control angle
                ))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, **kwargs)
        # define node labels
        if self.molecule:
            if nolabel:
                node_labels = {n: f'{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
            else:
                node_labels = {n: f'{n}:{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
        else:
            if not nolabel:
                nx.draw_networkx_labels(G, pos, **kwargs)

        return ax, G

    def eval_related_pred(self, x: Tensor, edge_index: Tensor, edge_masks: List[Tensor], **kwargs):

        node_idx = kwargs.get('node_idx')
        node_idx = 0 if node_idx is None else node_idx  # graph level: 0, node level: node_idx
        related_preds = []

        # change the mask from -inf ~ +inf into 0 ~ 1
        for ex_label, edge_mask in enumerate(edge_masks):
            if self.hard_edge_mask is not None:
                sparsity = 1.0 - (edge_mask[self.hard_edge_mask] != 0).sum() / edge_mask[self.hard_edge_mask].size(0)
            else:
                sparsity = 1.0 - (edge_mask != 0).sum() / edge_mask.size(0)

            self.edge_mask.data = torch.ones(edge_mask.size(), device=self.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            self.edge_mask.data = edge_mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            self.edge_mask.data = 1.0 - edge_mask  # keep Parameter's id
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            self.edge_mask.data = torch.zeros(edge_mask.size(), device=self.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx],
                                  'sparsity': sparsity})

            # Adding proper activation function to the models' outputs.
            tmp_result_dict = {}
            for key, pred in related_preds[ex_label].items():
                if key in ['sparsity']:
                    tmp_result_dict[key] = pred.item()
                else:
                    tmp_result_dict[key] = pred.reshape(-1).softmax(0)[ex_label].item()
            related_preds[ex_label] = tmp_result_dict

        self.__clear_masks__()
        return related_preds


class WalkBase(ExplainerBase):

    def __init__(self, model: nn.Module, epochs: int = 0, lr: float = 0, explain_graph: bool = False, molecule: bool = False):
        super().__init__(model, epochs, lr, explain_graph, molecule)

    def extract_step(self, x: Tensor, edge_index: Tensor, detach: bool = True, split_fc: bool = False):

        layer_extractor = []
        hooks = []

        def register_hook(module: nn.Module):
            if not list(module.children()) or isinstance(module, MessagePassing):
                hooks.append(module.register_forward_hook(forward_hook))

        def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
            # input contains x and edge_index
            if detach:
                layer_extractor.append((module, input[0].clone().detach(), output.clone().detach()))
            else:
                layer_extractor.append((module, input[0], output))

        # --- register hooks ---
        self.model.apply(register_hook)

        pred = self.model(x, edge_index)

        for hook in hooks:
            hook.remove()

        # --- divide layer sets ---

        walk_steps = []
        fc_steps = []
        pool_flag = False
        step = {'input': None, 'module': [], 'output': None}
        for layer in layer_extractor:
            if isinstance(layer[0], MessagePassing) or isinstance(layer[0], GNNPool):
                if isinstance(layer[0], GNNPool):
                    pool_flag = True
                if step['module'] and step['input'] is not None:
                    walk_steps.append(step)
                step = {'input': layer[1], 'module': [], 'output': None}
            if pool_flag and split_fc and isinstance(layer[0], nn.Linear):
                if step['module']:
                    fc_steps.append(step)
                step = {'input': layer[1], 'module': [], 'output': None}
            step['module'].append(layer[0])
            step['output'] = layer[2]

        for walk_step in walk_steps:
            if hasattr(walk_step['module'][0], 'nn') and walk_step['module'][0].nn is not None:
                # We don't allow any outside nn during message flow process in GINs
                walk_step['module'] = [walk_step['module'][0]]

        if split_fc:
            if step['module']:
                fc_steps.append(step)
            return walk_steps, fc_steps
        else:
            fc_step = step

        return walk_steps, fc_step

    def walks_pick(self,
                   edge_index: Tensor,
                   pick_edge_indices: List,
                   walk_indices: List=[],
                   num_layers=0
                   ):
        walk_indices_list = []
        for edge_idx in pick_edge_indices:

            # Adding one edge
            walk_indices.append(edge_idx)
            _, new_src = src, tgt = edge_index[:, edge_idx]
            next_edge_indices = np.array((edge_index[0, :] == new_src).nonzero().view(-1))

            # Finding next edge
            if len(walk_indices) >= num_layers:
                # return one walk
                walk_indices_list.append(walk_indices.copy())
            else:
                walk_indices_list += self.walks_pick(edge_index, next_edge_indices, walk_indices, num_layers)

            # remove the last edge
            walk_indices.pop(-1)

        return walk_indices_list

    def eval_related_pred(self, x: Tensor, edge_index: Tensor, masks: List[Tensor], **kwargs):
        # place to add accuracy
        node_idx = kwargs.get('node_idx')
        pred_label = kwargs.get('pred_label')
        node_idx = 0 if node_idx is None else node_idx  # graph level: 0, node level: node_idx

        related_preds = []

        for label, edge_mask in enumerate(masks):
            if self.hard_edge_mask is not None:
                sparsity = 1.0 - (edge_mask[self.hard_edge_mask] != 0).sum() / edge_mask[self.hard_edge_mask].size(0)
            else:
                sparsity = 1.0 - (edge_mask != 0).sum() / edge_mask.size(0)

            # origin pred
            for mask in self.edge_mask:
                mask.data = torch.ones(edge_mask.size(), device=self.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            for mask in self.edge_mask:
                mask.data = edge_mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            for mask in self.edge_mask:
                mask.data = 1.0 - edge_mask
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            for mask in self.edge_mask:
                mask.data = torch.zeros(edge_mask.size(), device=self.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # Store related predictions for further evaluation.
            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx],
                                  'sparsity': sparsity})

            # Adding proper activation function to the models' outputs.
            if pred_label:
                label = pred_label
            tmp_result_dict = {}
            for key, pred in related_preds[label].items():
                if key in ['sparsity']:
                    tmp_result_dict[key] = pred.item()
                else:
                    tmp_result_dict[key] = pred.reshape(-1).softmax(0)[label].item()
            related_preds[label] = tmp_result_dict

        return related_preds

    def explain_edges_with_loop(self, x: Tensor, walks: Dict[Tensor, Tensor], ex_label):

        walks_ids = walks['ids']
        walks_score = walks['score'][:walks_ids.shape[0], ex_label].reshape(-1)
        if walks_ids.max() <= self.num_edges - 1:  # num_edges includes the self-loop
            idx_ensemble = torch.cat([(walks_ids == i).int().sum(dim=1).unsqueeze(0) for i in range(self.num_edges)], dim=0)
        else:
            idx_ensemble = torch.cat([(walks_ids == i).int().sum(dim=1).unsqueeze(0) for i in range(self.num_edges + self.num_nodes)], dim=0)
        hard_edge_attr_mask = (idx_ensemble.sum(1) > 0).long()
        hard_edge_attr_mask_value = torch.tensor([float('inf'), 0], dtype=torch.float, device=self.device)[hard_edge_attr_mask]
        edge_attr = (idx_ensemble * (walks_score.unsqueeze(0))).sum(1)
        # idx_ensemble1 = torch.cat(
        #     [(walks_ids == i).int().sum(dim=1).unsqueeze(1) for i in range(self.num_edges + self.num_nodes)], dim=1)
        # edge_attr1 = (idx_ensemble1 * (walks_score.unsqueeze(1))).sum(0)

        return edge_attr - hard_edge_attr_mask_value

    class connect_mask(object):

        def __init__(self, cls):
            self.cls = cls

        def __enter__(self):

            self.cls.edge_mask = [nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
                                 [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)]

            for idx, module in enumerate(self.cls.mp_layers):
                module._explain = True
                module.__edge_mask__ = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module._explain = False

    class temp_mask(object):

        def __init__(self, cls, temp_edge_mask):
            self.cls = cls
            self.temp_edge_mask = temp_edge_mask

        def __enter__(self):

            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain_flow__ = True
                module.layer_edge_mask = self.temp_edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain_flow__ = False

def gumbel_softmax(log_alpha: torch.Tensor, beta: float = 1.0, training: bool = True):
    if training:
        random_noise = torch.rand(log_alpha.shape)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
        gate_inputs = gate_inputs.sigmoid()
    else:
        gate_inputs = log_alpha.sigmoid()

    return gate_inputs
def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)
def subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
    else:
        node_idx = node_idx.to(row.device)

    inv = None

    if num_hops != -1:
        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]
    else:
        subsets = node_idx
        cur_subsets = node_idx
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

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

class FlowX(WalkBase):
    coeffs = {
        'edge_size': 5e-4,
        'edge_ent': 1e-1
    }

    def __init__(self, model, epochs=500, lr=3e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.score_structure = [(i % 2, term_idx)
                                for i in range(1, self.num_layers + 1)
                                for term_idx in combinations(range(self.num_layers), i)
                                ]

        self.ns_iter = 30 if explain_graph else 5
        self.ns_per_iter = None
        self.fidelity_plus = True
        self.score_lr = 0e-5

        self.no_mask = False
        if self.no_mask:
            self.epochs = 1
            self.lr = 0
            self.score_lr = 0

        if not explain_graph:
            self.epochs = 50

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:

        super().forward(x, edge_index, **kwargs)

        self.model.eval()

        # Initial original prediction
        _raw_pred = self.model(x, edge_index)
        if isinstance(_raw_pred, (tuple, list)):
            _raw_pred = _raw_pred[0]
        if _raw_pred.dim() == 1:
            _raw_pred = _raw_pred.unsqueeze(0)
        self.ori_logits_pred = _raw_pred.softmax(1)

        # Edge Index with self loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)

        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=self.num_nodes, flow=self.__flow__())

            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]

        import time
        start = time.time()
        self.time_list = []
        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        with self.connect_mask(self):
            iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count = \
                self.flow_shap(x, edge_index, edge_index_with_loop, walk_indices_list, **kwargs)

        walk_score_list = []
        for ex_label in ex_labels:
            self.train_mask(x,
                            edge_index,
                            ex_label,
                            walk_indices_list,
                            edge_index_with_loop,
                            iter_weighted_change_walks_list,
                            iter_changed_subsets_score_list,
                            walk_sample_count)

            walk_score_list.append(self.flow_mask.data)

        walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=1)}

        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
        start = time.time()
        masks = []
        for ex_label in ex_labels:
            edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
            mask = edge_attr
            masks.append(mask.detach())

        return masks

    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = cross_entropy_with_logit(raw_preds, x_label)
        else:
            loss = cross_entropy_with_logit(raw_preds[self.node_idx].unsqueeze(0), x_label)

        if self.fidelity_plus:
            loss = - loss

        return loss

    def train_mask(self,
                x: Tensor,
                edge_index: Tensor,
                ex_label: Tensor,
                walk_indices_list,
                edge_index_with_loop,
                iter_weighted_change_walks_list,
                iter_changed_subsets_score_list,
                walk_sample_count,
                t0=7.,
                t1=0.5,
                **kwargs
                ) -> None:

        self.to(x.device)

        self.nec_suf_mask = nn.Parameter(
            1e-1 * nn.init.uniform_(torch.empty((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device)))

        if self.no_mask:
            self.nec_suf_mask = nn.Parameter(
                100 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))
        self.iter_weighted_change_walks_list = nn.Parameter(iter_weighted_change_walks_list.clone().detach())

        walk_plain_indices_list = walk_indices_list + \
                                (edge_index_with_loop.shape[1]
                                * torch.arange(self.num_layers, device=self.device)).repeat(
                                    walk_indices_list.shape[0], 1)

        self.flow2layeredge_matrix = torch.stack([(walk_plain_indices_list == i).float().sum(dim=1)
                                                for i in
                                                range(self.num_layers * (self.num_edges + self.num_nodes))],
                                                dim=1).detach()

        optimizer = torch.optim.Adam([{'params': self.nec_suf_mask}], lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            masked_iter_weighted_change_walks_list = self.iter_weighted_change_walks_list * self.nec_suf_mask.sigmoid()

            walk_scores = (masked_iter_weighted_change_walks_list.unsqueeze(3).repeat(1, 1, 1,
                                                                                    iter_changed_subsets_score_list.shape[
                                                                                        2]) * iter_changed_subsets_score_list.unsqueeze(
                2)).sum(1).sum(0)
            EPS = 1e-18
            shap_flow_score = (walk_scores / (walk_sample_count.unsqueeze(1) + EPS))

            self.flow_mask = shap_flow_score[:, ex_label]

            self.layer_edge_mask = (self.flow_mask * self.flow2layeredge_matrix).view(self.flow_mask.shape[0],
                                                                                    self.num_layers,
                                                                                    -1).sum(0)
            mask = self.layer_edge_mask.sum(0)
            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)

            climb = True
            if climb:
                mask = mask ** 8
            else:
                end_epoch = 300
                temperature = float(t0 * ((t1 / t0) ** (epoch / end_epoch))) if epoch < end_epoch else t1
                mask = gumbel_softmax(mask, temperature, training=True)

            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)

            if self.fidelity_plus:
                mask = 1 - mask

            self.mask = mask

            temp_edge_mask = []
            for layer_idx in range(self.num_layers):
                temp_edge_mask.append(mask)

            with self.temp_mask(self, temp_edge_mask):
                _out = self.model(x, edge_index, **kwargs)
                raw_preds = _out[0] if isinstance(_out, (tuple, list)) else _out

            loss = self.__loss__(raw_preds, ex_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return

    def flow_shap(self,
                x,
                edge_index,
                edge_index_with_loop,
                walk_indices_list,
                **kwargs
                ):

        walk_sample_count = torch.zeros(walk_indices_list.shape[0], dtype=torch.float, device=self.device)

        iter_weighted_change_walks_list = []
        iter_changed_subsets_score_list = []

        num_nodes_total = x.size(0)
        nc = kwargs.get('num_classes')

        for iter_idx in range(self.ns_iter):

            unmask_pool = torch.cat([walk_indices_list[:, layer].unique() + layer * edge_index_with_loop.shape[1]
                                    for layer in range(self.num_layers)])
            self.ns_per_iter = unmask_pool.shape[0] if self.explain_graph or unmask_pool.shape[0] <= 100 else 100
            idx = torch.randperm(unmask_pool.nelement())
            unmask_pool = unmask_pool.view(-1)[idx].view(unmask_pool.size())

            mask_per_sub = unmask_pool.shape[0] // self.ns_per_iter
            weighted_change_walks_list = []
            last_eliminated_walks = torch.zeros(walk_indices_list.shape[0], dtype=torch.bool, device=self.device)
            layer_edge_mask_list = []

            for sub_idx in range(self.ns_per_iter):
                mask_pool = unmask_pool[: mask_per_sub * (sub_idx + 1)]

                eliminated_layer_edges = unmask_pool[mask_per_sub * sub_idx: mask_per_sub * (sub_idx + 1)]
                walk_plain_indices_list = walk_indices_list + \
                                        (edge_index_with_loop.shape[1] * torch.arange(self.num_layers,
                                                                                        device=self.device)).repeat(
                                            walk_indices_list.shape[0], 1)
                eliminated_walks = torch.stack([walk_plain_indices_list == edge for edge in eliminated_layer_edges],
                                            dim=0).long().sum(0).sum(1).bool().long()
                weighted_changed_walks = eliminated_walks.clone().float()
                weighted_changed_walks[eliminated_walks == last_eliminated_walks] = 0.
                weighted_changed_walks /= (weighted_changed_walks > 1e-20).sum() + 1e-30
                weighted_change_walks_list.append(weighted_changed_walks)
                last_eliminated_walks = eliminated_walks

                layer_edge_masks = torch.ones((self.num_layers, edge_index_with_loop.shape[1]),
                                            device=self.device)
                layer_edge_masks.view(-1)[mask_pool] -= 2
                layer_edge_mask_list.append(layer_edge_masks)

            weighted_change_walks_list = torch.stack(weighted_change_walks_list, dim=0)
            iter_weighted_change_walks_list.append(weighted_change_walks_list.detach())
            layer_edge_mask_list_stacked = torch.stack(layer_edge_mask_list, dim=0) * float('inf')

            if self.explain_graph:
                # 图分类：原始批处理路径
                for layer_idx in range(self.num_layers):
                    self.edge_mask[layer_idx].data = torch.cat(
                        [layer_edge_mask_list_stacked[:, layer_idx, :self.num_edges].reshape(-1),
                        layer_edge_mask_list_stacked[:, layer_idx, self.num_edges:].reshape(-1)]).sigmoid()

                batch = self.batch_input(x, edge_index, self.ns_per_iter)
                _raw = self.model(data=batch)
                if isinstance(_raw, (tuple, list)):
                    _raw = _raw[0]
                subsets_output = _raw.softmax(1).detach()
                last_subsets_output = torch.cat([self.ori_logits_pred, subsets_output.clone()[:-1]], dim=0)

            else:
                # 节点分类：逐条前向
                subsets_output = torch.zeros(self.ns_per_iter, nc, device=self.device)
                for sub_idx in range(self.ns_per_iter):
                    cur_layer_masks = layer_edge_mask_list[sub_idx] * float('inf')  # [num_layers, E_with_loop]
                    for layer_idx, module in enumerate(self.mp_layers):
                        full = torch.cat([
                            cur_layer_masks[layer_idx, :self.num_edges],
                            cur_layer_masks[layer_idx, self.num_edges:]
                        ])
                        module.__edge_mask__ = nn.Parameter(full.sigmoid().detach())
                    _batch = torch.zeros(num_nodes_total, dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        _raw = self.model(x, edge_index, batch=_batch)
                        if isinstance(_raw, (tuple, list)):
                            _raw = _raw[0]
                    subsets_output[sub_idx] = _raw[self.node_idx].softmax(0)

                last_subsets_output = torch.cat(
                    [self.ori_logits_pred[self.node_idx].unsqueeze(0),
                    subsets_output.clone()[:-1]], dim=0)

            changed_subsets_score_list = (last_subsets_output - subsets_output).detach()
            iter_changed_subsets_score_list.append(changed_subsets_score_list)

            walk_sample_count += (weighted_change_walks_list > 1e-30).float().sum(0)

        iter_weighted_change_walks_list = torch.stack(iter_weighted_change_walks_list, dim=0)
        iter_changed_subsets_score_list = torch.stack(iter_changed_subsets_score_list, dim=0)

        return iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count