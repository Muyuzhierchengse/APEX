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


from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph


def _get_khop_subset(edge_index, node_idx, num_nodes, num_hops, device):
    subset, _, _, _ = pyg_k_hop_subgraph(
        node_idx, num_hops, edge_index,
        relabel_nodes=False, num_nodes=num_nodes,
    )
    return subset.to(device)


def _remove_nodes_from_edge_index(edge_index, nodes_to_remove):

    remove_set = set(nodes_to_remove.cpu().tolist())
    src, dst = edge_index[0], edge_index[1]
    keep = torch.tensor(
        [i for i in range(edge_index.size(1))
         if src[i].item() not in remove_set
         and dst[i].item() not in remove_set],
        dtype=torch.long, device=edge_index.device,
    )
    if keep.numel() == 0:
        # 极端情况：所有边都被移除，返回空边
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    return edge_index[:, keep]


def eval_related_pred_node(model, x, edge_index,
                            raw_node_scores,
                            pred_cls, node_idx, device,
                            sparsity=0.5, num_hops=3):
    model.eval()
    num_nodes = x.size(0)

    from torch_geometric.utils import k_hop_subgraph
    subset, _, _, _ = k_hop_subgraph(
        node_idx, num_hops, edge_index,
        relabel_nodes=False, num_nodes=num_nodes,
    )
    
    khop_nodes = subset[subset != node_idx].tolist()
    n_khop = len(khop_nodes)

    if n_khop == 0:
        return {'ori': 0., 'masked_in': 0., 'masked_out': 0.,
                'fid_plus': 0., 'fid_minus': 0.}

    num_keep = max(1, int(n_khop * (1 - sparsity)))
    khop_tensor = torch.tensor(khop_nodes, dtype=torch.long, device=device)
    scores_khop = raw_node_scores[pred_cls].to(device)[khop_tensor]

    with torch.no_grad():
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

        def get_logit(feat):
            out = model(feat, edge_index, batch=batch)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            return logits[node_idx, pred_cls].item()

        logit_ori = get_logit(x)

        x_base = x.clone()
        x_base[khop_tensor] = 0.0
        logit_base = get_logit(x_base)

    align_dir      = 1.0 if logit_ori >= logit_base else -1.0
    aligned_scores = align_dir * scores_khop

    if aligned_scores.max().item() > 0:
        topk_idx = aligned_scores.topk(num_keep, largest=True).indices
    else:
        topk_idx = aligned_scores.abs().topk(num_keep, largest=True).indices

    important_set   = set(khop_tensor[topk_idx].cpu().tolist())
    unimportant_set = set(khop_nodes) - important_set

    important_tensor = torch.tensor(
        sorted(important_set), dtype=torch.long, device=device
    )
    unimportant_tensor = torch.tensor(
        sorted(unimportant_set), dtype=torch.long, device=device
    )

    x_out = x.clone()
    if important_tensor.numel() > 0:
        x_out[important_tensor] = 0.0

    x_in = x.clone()
    if unimportant_tensor.numel() > 0:
        x_in[unimportant_tensor] = 0.0

    with torch.no_grad():
        logit_out = get_logit(x_out)  # Fid+
        logit_in  = get_logit(x_in)   # Fid-

    dynamic_range = logit_ori - logit_base
    if abs(dynamic_range) < 1e-6:
        dynamic_range = 1e-6

    fid_plus_raw  = (logit_ori - logit_out) / dynamic_range
    fid_minus_raw = (logit_ori - logit_in)  / dynamic_range

    fid_plus  = float(max(0.0, min(1.0, fid_plus_raw)))
    fid_minus = float(max(0.0, min(1.0, fid_minus_raw)))
    '''
    print(f"    [DEBUG] node_idx={node_idx}, pred_cls={pred_cls}, "
          f"k-hop节点数={n_khop}, num_keep={num_keep}")
    print(f"    [DEBUG] 归因分数(k-hop): "
          f"min={scores_khop.min().item():.2f}, "
          f"max={scores_khop.max().item():.2f}, "
          f"mean={scores_khop.mean().item():.2f}")
    print(f"    [DEBUG] align_dir={align_dir:+.0f}, "
          f"logit_ori={logit_ori:.2f}, logit_base={logit_base:.2f}")
    print(f"    [DEBUG] important_set={len(important_set)}, "
          f"unimportant_set={len(unimportant_set)}")
    print(f"    [DEBUG] logit_ori={logit_ori:.2f}, logit_base={logit_base:.2f}, "
          f"logit_in={logit_in:.2f}, logit_out={logit_out:.2f}")
    print(f"    [DEBUG] dynamic_range={dynamic_range:.4f}")
    print(f"    [DEBUG] Fidelity+(raw)={fid_plus_raw:.4f} → {fid_plus:.4f}, "
          f"Fidelity-(raw)={fid_minus_raw:.4f} → {fid_minus:.4f}")
    '''
    return {
        'ori':        logit_ori,
        'masked_in':  logit_in,
        'masked_out': logit_out,
        'fid_plus':   fid_plus,
        'fid_minus':  fid_minus,
    }


def eval_stability_node(
    explainer, x, edge_index,
    node_masks_orig, pred_cls, node_idx,
    num_classes, num_nodes, sparsity, device,
    perturb_ratio=0.1, n_perturb=5, seed_base=100,
):
    orig_set = _binarize_node_mask(node_masks_orig[pred_cls], sparsity, num_nodes)
    jaccard_scores = []

    for i in range(n_perturb):
        perturbed_ei = _perturb_irrelevant_region(
            edge_index,
            node_masks_orig[pred_cls].to(device),
            num_nodes, ratio=perturb_ratio, seed=seed_base + i,
        )
        try:
            result_p = explainer(
                x, perturbed_ei, sparsity=0,
                num_classes=num_classes, node_idx=node_idx,
                max_nodes=int(num_nodes * (1 - sparsity)),
            )
        except Exception:
            continue

        node_masks_p = result_p[1] if isinstance(result_p, tuple) else \
            get_node_mask_from_edge_mask(result_p, num_nodes, perturbed_ei,
                                         num_classes, sparsity)
        perturbed_set = _binarize_node_mask(node_masks_p[pred_cls], sparsity, num_nodes)
        jaccard_scores.append(jaccard_similarity(orig_set, perturbed_set))

    return float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
