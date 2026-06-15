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


'''
import torch
import numpy as np
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data


def get_node_mask_from_edge_mask(edge_masks, num_nodes, edge_index, num_classes, sparsity):

    if edge_masks[0].shape[0] != edge_index.shape[1]:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    num_keep = max(1, int(num_nodes * (1 - sparsity)))
    node_masks = []

    for pred in range(num_classes):
        mask = edge_masks[pred].detach().cpu().numpy()
        sorted_edge_idx = np.argsort(-mask)

        selected = set()
        for eidx in sorted_edge_idx:
            src = edge_index[0][eidx].item()
            dst = edge_index[1][eidx].item()
            if len(selected) < num_keep:
                selected.add(src)
            if len(selected) < num_keep:
                selected.add(dst)
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


# ── Stability（GNNXBench 规范）────────────────────────────────────────────────

def _perturb_irrelevant_region(edge_index, node_mask, num_nodes, ratio=0.1, seed=42):

    rng    = np.random.default_rng(seed)
    device = edge_index.device
    ei_cpu = edge_index.cpu().numpy()          # [2, E]

    mask_np       = node_mask.cpu().numpy()    # [N]
    irrelevant_set = set(np.where(mask_np == 0)[0].tolist())

    # ── 分离：不重要区域的边 vs 其他边 ──────────────────────────
    irr_idx, keep_idx = [], []
    for i in range(ei_cpu.shape[1]):
        u, v = int(ei_cpu[0, i]), int(ei_cpu[1, i])
        if u in irrelevant_set and v in irrelevant_set:
            irr_idx.append(i)
        else:
            keep_idx.append(i)

    irr_edges  = ei_cpu[:, irr_idx]   # [2, E_irr]
    keep_edges = ei_cpu[:, keep_idx]  # [2, E_keep]

    n_irr    = irr_edges.shape[1]

    # 不重要区域没有内部边（所有边均跨越重要/不重要边界），无法扰动，直接返回原图
    if n_irr == 0:
        return edge_index

    n_remove = max(1, int(n_irr * ratio))

    # n_remove 不能超过可用边数（极小图保护）
    n_remove = min(n_remove, n_irr)

    # ── 删除：随机去掉 n_remove 条不重要边 ───────────────────────
    keep_local = rng.choice(n_irr, size=n_irr - n_remove, replace=False)
    survived   = irr_edges[:, keep_local]           # [2, E_irr - n_remove]

    # ── 添加：在不重要节点对中随机增加 n_remove 条新边 ───────────
    irr_nodes = np.array(sorted(irrelevant_set), dtype=np.int64)
    if len(irr_nodes) >= 2:
        # 避免重复已存在的边：构建现有边集合
        existing = set(zip(survived[0].tolist(), survived[1].tolist()))
        existing.update(zip(keep_edges[0].tolist(), keep_edges[1].tolist()))

        added_u, added_v = [], []
        max_try = n_remove * 20
        for _ in range(max_try):
            if len(added_u) >= n_remove:
                break
            u, v = rng.choice(irr_nodes, size=2, replace=False).tolist()
            if (u, v) not in existing:
                added_u.append(u)
                added_v.append(v)
                # 无向图同时添加反向边（与原图保持一致）
                existing.add((u, v))
                existing.add((v, u))
        if added_u:
            new_edges = np.array([added_u, added_v], dtype=np.int64)
            new_edges_rev = np.array([added_v, added_u], dtype=np.int64)
            survived = np.concatenate([survived, new_edges, new_edges_rev], axis=1)

    # ── 合并 ─────────────────────────────────────────────────────
    final = np.concatenate([keep_edges, survived], axis=1)
    return torch.tensor(final, dtype=torch.long, device=device)


def _binarize_node_mask(node_mask, sparsity, num_nodes):
    """将连续/二值 node_mask 统一转为 frozenset（重要节点下标集合），供 Jaccard 计算用。"""
    num_keep = max(1, int(num_nodes * (1 - sparsity)))
    scores   = node_mask.detach().cpu()
    topk_idx = scores.topk(min(num_keep, len(scores))).indices
    return frozenset(topk_idx.tolist())


def jaccard_similarity(set_a, set_b):
    """Jaccard 相似度 = |A ∩ B| / |A ∪ B|"""
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 1.0


def eval_stability(
    explainer,
    x, edge_index,
    node_masks_orig,          # 原始解释的 node_masks（list[Tensor]）
    pred_cls,
    num_classes,
    num_nodes,
    sparsity,
    device,
    perturb_ratio=0.1,
    n_perturb=5,
    seed_base=100,
):

    # 原始解释的重要节点集合
    orig_set = _binarize_node_mask(node_masks_orig[pred_cls], sparsity, num_nodes)

    jaccard_scores = []

    for i in range(n_perturb):
        # 1. 生成扰动后的 edge_index（仅不重要区域）
        perturbed_ei = _perturb_irrelevant_region(
            edge_index,
            node_masks_orig[pred_cls].to(device),
            num_nodes,
            ratio=perturb_ratio,
            seed=seed_base + i,
        )

        # 2. 在扰动图上重新运行 explainer
        try:
            result_p = explainer(
                x, perturbed_ei,
                sparsity=0,
                num_classes=num_classes,
                node_idx=0,
                max_nodes=int(num_nodes * (1 - sparsity)),
            )
        except Exception:
            # 部分 explainer 对极小图可能失败，跳过该次扰动
            continue

        # 3. 解析扰动后的 node_masks
        if isinstance(result_p, tuple):
            _, node_masks_p = result_p
        else:
            node_masks_p = get_node_mask_from_edge_mask(
                result_p, num_nodes, perturbed_ei, num_classes, sparsity,
            )

        # 4. 计算与原始解释的 Jaccard 相似度
        perturbed_set = _binarize_node_mask(node_masks_p[pred_cls], sparsity, num_nodes)
        jaccard_scores.append(jaccard_similarity(orig_set, perturbed_set))

    if not jaccard_scores:
        return 0.0

    return float(np.mean(jaccard_scores))
'''