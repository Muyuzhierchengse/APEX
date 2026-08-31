import torch
import numpy as np
from apex.evaluation.fidelity import get_node_mask_from_edge_mask


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
