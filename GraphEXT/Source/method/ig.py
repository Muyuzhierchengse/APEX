"""
method/ig.py
============
Integrated Gradients (IG) for GNN explanation.
支持图分类（explain_graph=True）和节点分类（explain_graph=False）。
"""

import torch
import numpy as np


class IntegratedGradients:
    def __init__(self, model, explain_graph=True, m_steps=50, baseline='zero'):
        self.model         = model
        self.explain_graph = explain_graph
        self.m_steps       = m_steps
        self.baseline_type = baseline
        self.last_node_scores = None

    def _get_baseline(self, x):
        if self.baseline_type == 'zero' or not isinstance(self.baseline_type, torch.Tensor):
            return torch.zeros_like(x)
        return self.baseline_type.to(x.device)

    def _interpolate(self, x, baseline, alpha):
        return baseline + alpha * (x - baseline)

    def _gradients_at(self, x_interp, edge_index, pred_cls, batch, node_idx=None):
        x_interp = x_interp.detach().requires_grad_(True)

        out = self.model(x_interp, edge_index, batch=batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        if node_idx is None:
            # 图分类：logits 形状 [C]
            scalar = logits[pred_cls]
        else:
            # 节点分类：logits 形状 [N, C]
            scalar = logits[node_idx, pred_cls]

        self.model.zero_grad()
        scalar.backward()

        grad = x_interp.grad.detach()
        return grad

    def _integrated_gradients_single_class(self, x, edge_index, baseline,
                                            pred_cls, batch, node_idx=None):
        m      = self.m_steps
        alphas = [k / m for k in range(1, m + 1)]

        grad_sum = torch.zeros_like(x)

        for alpha in alphas:
            x_interp = self._interpolate(x, baseline, alpha)
            grad     = self._gradients_at(
                x_interp, edge_index, pred_cls, batch,
                node_idx=node_idx,
            )
            grad_sum = grad_sum + grad

        avg_grad    = grad_sum / m
        attribution = (x - baseline) * avg_grad   # [N, D]

        return attribution

    def __call__(
        self,
        x,
        edge_index,
        sparsity=0.5,
        num_classes=2,
        node_idx=0,
        max_nodes=None,
    ):
        self.model.eval()

        device    = x.device
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        batch    = torch.zeros(num_nodes, dtype=torch.long, device=device)
        baseline = self._get_baseline(x)

        if max_nodes is not None:
            num_keep = max(1, int(max_nodes))
        else:
            num_keep = max(1, int(num_nodes * (1 - sparsity)))

        ig_node_idx = None if self.explain_graph else node_idx

        edge_masks         = []
        node_masks         = []
        signed_scores_list = []
        abs_scores_list    = []

        for cls in range(num_classes):
            attribution = self._integrated_gradients_single_class(
                x, edge_index, baseline, cls, batch,
                node_idx=ig_node_idx,
            )
            # attribution: [N, D]

            signed_scores = attribution.sum(dim=-1)        # [N]
            abs_scores    = attribution.abs().sum(dim=-1)  # [N]

            signed_scores_list.append(signed_scores)
            abs_scores_list.append(abs_scores)

            edge_masks.append(torch.zeros(num_edges, device=device))

            # ── 节点分类：只在目标节点的直接邻居+自身中选 top-k ──────────
            if not self.explain_graph:
                # 找 node_idx 的直接邻居（含自身）
                s_cpu = edge_index[0].cpu()
                d_cpu = edge_index[1].cpu()
                neighbors = set()
                neighbors.add(node_idx)
                for i in range(edge_index.size(1)):
                    u, v = s_cpu[i].item(), d_cpu[i].item()
                    if u == node_idx:
                        neighbors.add(v)
                    if v == node_idx:
                        neighbors.add(u)
                neighbors = sorted(neighbors)
                neighbor_tensor = torch.tensor(neighbors, dtype=torch.long, device=device)

                # 在邻居子集内按 abs_scores 选 top-k
                scores_sub = abs_scores[neighbor_tensor]
                k_sub      = min(num_keep, len(neighbors))
                topk_local = scores_sub.topk(k_sub).indices
                topk_global = neighbor_tensor[topk_local]

                nm = torch.zeros(num_nodes, dtype=torch.float32, device=device)
                nm[topk_global] = 1.0
            # ── 图分类：全图 top-k（原逻辑不变）────────────────────────────
            else:
                k    = min(num_keep, num_nodes)
                topk = abs_scores.topk(k).indices
                nm   = torch.zeros(num_nodes, dtype=torch.float32, device=device)
                nm[topk] = 1.0

            node_masks.append(nm)

        self.last_node_scores = signed_scores_list

        return edge_masks, node_masks