"""
PolyGINExplainer
================
基于 Aumann-Shapley 精确积分的 GNN 解释方法。

核心修复：
  1. 梯度计算路径改用 model.forward() 原生路径（task='pass' 绕过了reweight
     逻辑，导致 message() 里 edge_weight 不走 requires_grad 分支，梯度断裂）
  2. 节点级聚合恢复有符号 sum，但在 topk 选择时改为按绝对值排序
  3. 增加 baseline 可选（默认零基线，符合 Aumann-Shapley）
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
def _gauss_legendre_nodes_weights(m: int, device: torch.device):
    mu, omega = np.polynomial.legendre.leggauss(m)
    t = torch.tensor((mu + 1) / 2, dtype=torch.float32, device=device)
    w = torch.tensor(omega / 2,    dtype=torch.float32, device=device)
    return t, w


def _node_scores_to_binary_mask(node_scores: Tensor,
                                 num_nodes: int,
                                 sparsity: float) -> Tensor:
    """
    按节点 Shapley 分数的【有符号值】做 topk 选择：
    保留贡献最大（最正）的节点作为"重要节点"。
    这与 Fidelity+ 定义一致：移除正贡献节点，预测概率下降最多。
    """
    num_keep = max(1, int(num_nodes * (1 - sparsity)))
    topk_idx = torch.topk(node_scores, k=num_keep, largest=True).indices
    mask = torch.zeros(num_nodes, dtype=torch.float32,
                       device=node_scores.device)
    mask[topk_idx] = 1.0
    return mask


def _node_attribution_to_edge_mask(node_scores: Tensor,
                                    edge_index: Tensor) -> Tensor:
    src, dst = edge_index[0], edge_index[1]
    edge_scores = (node_scores[src] + node_scores[dst]) / 2.0
    mn, mx = edge_scores.min(), edge_scores.max()
    if (mx - mn).abs() < 1e-8:
        return torch.ones_like(edge_scores)
    return (edge_scores - mn) / (mx - mn + 1e-10)


# ──────────────────────────────────────────────────────────────────────────────
class PolyGINExplainer:
    def __init__(self, model: nn.Module,
                 explain_graph: bool = True,
                 L: int = 4):
        self.model = model
        self.explain_graph = explain_graph
        self.L = L
        self.m = 2 ** (L - 1)   # L=4 → m=8

        # 最近一次调用的原始归因结果，供外部读取 Efficiency Gap
        self.last_node_scores: Optional[List[Tensor]] = None

    def __call__(self,
                 x: Tensor,
                 edge_index: Tensor,
                 sparsity: float = 0.5,
                 num_classes: int = 2,
                 node_idx: int = 0,
                 max_nodes: int = None,
                 eval_sparsity: float = None,
                 **kwargs) -> Tuple[List[Tensor], List[Tensor]]:

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

        # 【修复】在整个归因过程中禁用 reweight，保证 message() 使用
        # 固定的全1 edge_weight，不走 requires_grad 分支干扰梯度
        self._set_reweight_false()

        edge_masks, node_masks = [], []
        node_scores_list = []          # ← 新增：保存每个类别的原始归因向量

        for cls in range(num_classes):
            IG_accum = torch.zeros_like(x)

            for k in range(self.m):
                t_k, w_k = t_nodes[k], w_nodes[k]

                with torch.enable_grad():
                    X_k = (t_k * x.detach()).requires_grad_(True)

                    # 【核心修复】直接调用 model.forward() 原生路径
                    # 而非 _forward_no_hook，确保梯度路径与训练时完全一致
                    logits = self.model(X_k, edge_index, batch)

                    score = logits.view(-1)[cls]
                    self.model.zero_grad()
                    score.backward()

                grad = X_k.grad if X_k.grad is not None else torch.zeros_like(x)
                IG_accum = IG_accum + w_k * grad.detach()

            # Hadamard 积 → 特征级 Shapley [num_nodes, feat_dim]
            Sh_exact = x.detach() * IG_accum

            # 节点级聚合：有符号 sum（符合理论，选正贡献节点）
            node_scores = Sh_exact.sum(dim=1)          # [num_nodes]
            node_scores_list.append(node_scores)       # ← 保存

            node_mask = _node_scores_to_binary_mask(node_scores, num_nodes, cut_sparsity)
            node_masks.append(node_mask)

            edge_mask = _node_attribution_to_edge_mask(node_scores, edge_index)
            edge_masks.append(edge_mask)

        self._restore_reweight()
        self.last_node_scores = node_scores_list       # ← 挂载到实例，供外部读取
        return edge_masks, node_masks

    # ──────────────────────────────────────────────────────────────────
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