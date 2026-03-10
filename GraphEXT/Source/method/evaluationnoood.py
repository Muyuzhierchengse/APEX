import torch
import numpy as np
from torch_geometric.utils import add_self_loops


def get_node_mask_from_edge_mask(edge_masks, num_nodes, edge_index, num_classes, sparsity):
    """
    将 edge-level importance mask 转换为 node-level binary mask。
    保留重要性最高的 top-(1-sparsity) 比例的节点。

    返回：list[Tensor]，每个元素 shape=[num_nodes]，1=重要节点，0=不重要节点。
    """
    if edge_masks[0].shape[0] != edge_index.shape[1]:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    num_keep = max(1, int(num_nodes * (1 - sparsity)))
    node_masks = []

    for pred in range(num_classes):
        mask = edge_masks[pred].detach().cpu().numpy()
        sorted_edge_idx = np.argsort(-mask)

        selected = set()
        for eidx in sorted_edge_idx:
            selected.add(edge_index[0][eidx].item())
            selected.add(edge_index[1][eidx].item())
            if len(selected) >= num_keep:
                break

        node_mask = torch.zeros(num_nodes, dtype=torch.float32)
        node_mask[list(selected)] = 1.0
        node_masks.append(node_mask)

    return node_masks


def eval_related_pred(model, x, edge_index, node_masks, pred_cls, device,
                      num_samples=10):
    """
    基于边际分布替换的软掩码评估 (Marginal-distribution soft-mask evaluation)。

    与物理切割或零掩码不同，本方法同时消除结构层面和特征层面的 OOD：
      1. 完整保留图拓扑（所有节点、边不变）          → 无结构 OOD
      2. 被"移除"的节点特征替换为边际分布采样        → 无特征 OOD
      3. 通过 Monte Carlo 多次采样取平均             → 鲁棒估计

    具体做法——对于每次 MC 采样：
      x_ref  = x[randperm(N)]                          （图内随机置换，保持精确边际分布）
      x_in   = M ⊙ x + (1 − M) ⊙ x_ref               （保留重要特征，其余→边际采样）
      x_out  = (1 − M) ⊙ x + M ⊙ x_ref               （移除重要特征→边际采样，保留其余）

    为什么用随机置换（permutation）而非高斯采样？
      - 分子图节点特征通常是 one-hot 编码，高斯采样会产生无意义的连续值
      - 随机置换在有限样本下精确保持经验边际分布 P(X_i)
      - 这是 Shapley value 框架下理论最优的"缺失"模拟方式

    参考文献：
      [1] Amara et al. "GraphFramEx: Towards Systematic Evaluation of
          Explainability Methods for Graph Neural Networks." NeurIPS 2022.
      [2] Mathis et al. "GInX-Eval: Towards In-Distribution Evaluation of
          Graph Neural Network Explanations." NeurIPS 2023.
      [3] Li et al. "DEGREE: Decomposition Based Explanation for Graph
          Neural Networks." NeurIPS 2023.

    参数和返回值与原版完全一致，main.py 无需修改。
    """
    model.eval()
    node_mask = node_masks[pred_cls].to(device)       # [N], binary 0/1
    mask_col = node_mask.unsqueeze(-1)                 # [N, 1]
    inv_mask_col = 1.0 - mask_col                      # [N, 1]

    N = x.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=device)

    with torch.no_grad():
        # ── 原始完整图 ─────────────────────────────────────────────
        logits_ori = model(x, edge_index, batch=batch)[0]
        ori_prob = torch.softmax(logits_ori, dim=0)[pred_cls].item()

        # ── Monte Carlo 边际分布软掩码 ────────────────────────────
        in_probs = []
        out_probs = []

        for _ in range(num_samples):
            # 随机置换：精确保持图内节点特征的经验边际分布
            perm = torch.randperm(N, device=device)
            x_ref = x[perm]                             # [N, D]

            # masked_in：重要节点保留原始特征，
            #            非重要节点 → 边际分布采样（模拟"未知/缺失"）
            x_in = mask_col * x + inv_mask_col * x_ref
            logits_in = model(x_in, edge_index, batch=batch)[0]
            in_probs.append(
                torch.softmax(logits_in, dim=0)[pred_cls].item()
            )

            # masked_out：重要节点 → 边际分布采样（模拟"移除"），
            #             非重要节点保留原始特征
            x_out = inv_mask_col * x + mask_col * x_ref
            logits_out = model(x_out, edge_index, batch=batch)[0]
            out_probs.append(
                torch.softmax(logits_out, dim=0)[pred_cls].item()
            )

    return {
        'ori':        ori_prob,
        'masked_in':  float(np.mean(in_probs)),
        'masked_out': float(np.mean(out_probs)),
    }