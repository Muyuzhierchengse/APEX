
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


def _gauss_legendre_nodes_weights(m: int, device: torch.device):
    mu, omega = np.polynomial.legendre.leggauss(m)
    t = torch.tensor((mu + 1) / 2, dtype=torch.float32, device=device)
    w = torch.tensor(omega / 2,    dtype=torch.float32, device=device)
    return t, w


def _node_scores_to_binary_mask(node_scores: Tensor, num_nodes: int,
                                 sparsity: float) -> Tensor:
    num_keep = max(1, int(num_nodes * (1 - sparsity)))
    topk_idx = torch.topk(node_scores, k=num_keep, largest=True).indices
    mask = torch.zeros(num_nodes, dtype=torch.float32, device=node_scores.device)
    mask[topk_idx] = 1.0
    return mask


def _node_attribution_to_edge_mask(node_scores: Tensor, edge_index: Tensor) -> Tensor:
    src, dst = edge_index[0], edge_index[1]
    edge_scores = (node_scores[src] + node_scores[dst]) / 2.0
    mn, mx = edge_scores.min(), edge_scores.max()
    if (mx - mn).abs() < 1e-8:
        return torch.ones_like(edge_scores)
    return (edge_scores - mn) / (mx - mn + 1e-10)


class PolyGINExplainer:
    def __init__(self, model: nn.Module,
                 explain_graph: bool = True,
                 L: int = 4,         # L is fixed by the PolyGIN architecture.
                 num_hops: int = 3):
        self.model = model
        self.explain_graph = explain_graph
        self.L = L
        self.m = 2**(self.L-1)
        self.num_hops = num_hops
        self.last_node_scores: Optional[List[Tensor]] = None

    def __call__(self, x, edge_index, sparsity=0.5, num_classes=2,
                 node_idx=0, max_nodes=None, eval_sparsity=None, **kwargs):
        if self.explain_graph:
            return self._explain_graph(x, edge_index, sparsity, num_classes,
                                        node_idx, max_nodes, eval_sparsity)
        return self._explain_node(x, edge_index, sparsity, num_classes,
                                   node_idx, max_nodes, eval_sparsity)

    def _explain_graph(self, x, edge_index, sparsity, num_classes,
                        node_idx, max_nodes, eval_sparsity):
        device = x.device
        num_nodes = x.shape[0]

        if eval_sparsity is not None:
            cut_sparsity = eval_sparsity
        elif sparsity == 0 and max_nodes is not None:
            cut_sparsity = 1.0 - max_nodes / num_nodes
        else:
            cut_sparsity = sparsity

        t_nodes, w_nodes = _gauss_legendre_nodes_weights(self.m, device)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        self._set_reweight_false()

        edge_masks, node_masks, node_scores_list = [], [], []
        for cls in range(num_classes):
            IG_accum = torch.zeros_like(x)
            for k in range(self.m):
                t_k, w_k = t_nodes[k], w_nodes[k]
                with torch.enable_grad():
                    X_k = (t_k * x.detach()).requires_grad_(True)
                    logits = self.model(X_k, edge_index, batch)
                    score = logits.view(-1)[cls]
                    self.model.zero_grad()
                    score.backward()
                grad = X_k.grad if X_k.grad is not None else torch.zeros_like(x)
                IG_accum = IG_accum + w_k * grad.detach()

            Sh_exact = x.detach() * IG_accum
            node_scores = Sh_exact.sum(dim=1)
            node_scores_list.append(node_scores)
            node_masks.append(_node_scores_to_binary_mask(node_scores, num_nodes, cut_sparsity))
            edge_masks.append(_node_attribution_to_edge_mask(node_scores, edge_index))

        self._restore_reweight()
        self.last_node_scores = node_scores_list
        return edge_masks, node_masks

    def _explain_node(self, x, edge_index, sparsity, num_classes,
                       node_idx, max_nodes, eval_sparsity):
        device = x.device
        num_nodes = x.shape[0]

        if eval_sparsity is not None:
            cut_sparsity = eval_sparsity
        elif sparsity == 0 and max_nodes is not None:
            cut_sparsity = 1.0 - max_nodes / num_nodes
        else:
            cut_sparsity = sparsity

        t_nodes, w_nodes = _gauss_legendre_nodes_weights(self.m, device)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        self._set_reweight_false()

        x = x.detach()
        x_baseline = torch.zeros_like(x)

        with torch.no_grad():
            logits_ori_all = self.model(x, edge_index, batch)
            logits_ori_all = logits_ori_all[0] if isinstance(logits_ori_all, (tuple, list)) else logits_ori_all
            logits_base_all = self.model(x_baseline, edge_index, batch)
            logits_base_all = logits_base_all[0] if isinstance(logits_base_all, (tuple, list)) else logits_base_all

        edge_masks, node_masks, node_scores_list = [], [], []
        for cls in range(num_classes):
            IG_accum = torch.zeros_like(x)
            for k in range(self.m):
                t_k, w_k = t_nodes[k], w_nodes[k]
                with torch.enable_grad():
                    X_k = (t_k * x).requires_grad_(True)
                    logits = self.model(X_k, edge_index, batch)
                    logits = logits[0] if isinstance(logits, (tuple, list)) else logits
                    score = logits[node_idx, cls]
                    self.model.zero_grad()
                    score.backward()
                grad = X_k.grad if X_k.grad is not None else torch.zeros_like(x)
                IG_accum = IG_accum + w_k * grad.detach()

            phi = x * IG_accum
            node_scores = phi.sum(dim=1)

            completeness_gap = (
                node_scores.sum()
                - (logits_ori_all[node_idx, cls] - logits_base_all[node_idx, cls])
            ).abs().item()
            if completeness_gap > 1e-2:
                warnings.warn(
                    f'PolyGINExplainer node completeness check failed for '
                    f'node={node_idx}, cls={cls}: gap={completeness_gap:.6f}'
                )

            node_scores_list.append(node_scores)
            node_masks.append(_node_scores_to_binary_mask(node_scores, num_nodes, cut_sparsity))
            edge_masks.append(_node_attribution_to_edge_mask(node_scores, edge_index))

        self._restore_reweight()
        self.last_node_scores = node_scores_list
        return edge_masks, node_masks

    def _set_reweight_false(self):
        self._saved_edge_weights = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'reweight') and hasattr(module, 'edge_weight'):
                self._saved_edge_weights[name] = (module.reweight, module.edge_weight)
                module.reweight = False

    def _restore_reweight(self):
        for name, module in self.model.named_modules():
            if name in self._saved_edge_weights:
                module.reweight, module.edge_weight = self._saved_edge_weights[name]