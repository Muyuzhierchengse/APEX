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


def eval_related_pred(model, x, edge_index, node_masks, pred_cls, device):
    """
    软掩码评估（Soft-mask evaluation）——与物理切割子图不同，
    本方法保持完整图结构（所有节点、所有边）不变，
    仅对节点特征向量乘以掩码值来模拟"保留 / 去除"子图。

    原理（参考 GraphFramEx, NeurIPS 2022; GStarX, NeurIPS 2022）：
      - masked_in  : x_in  = x ⊙ M        （保留重要节点特征，非重要节点特征置零）
      - masked_out : x_out = x ⊙ (1 − M)  （保留非重要节点特征，重要节点特征置零）

    由于图的拓扑结构始终完整，GNN 的消息传递范式不会遇到
    因节点 / 边被物理删除而产生的分布偏移（OOD）问题。

    参数与返回值的语义和签名与原版完全一致，main.py 无需修改。
    """
    model.eval()
    node_mask = node_masks[pred_cls].to(device)          # [N]，二值 0/1
    mask_col  = node_mask.unsqueeze(-1)                   # [N, 1]，用于广播到特征维度

    with torch.no_grad():
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # ── 原始完整图 ─────────────────────────────────────────────
        logits_ori = model(x, edge_index, batch=batch)[0]
        ori_prob   = torch.softmax(logits_ori, dim=0)[pred_cls].item()

        # ── 软掩码 masked_in：保留重要节点特征，其余置零 ──────────
        x_in       = x * mask_col                         # 非重要节点特征 → 0
        logits_in  = model(x_in, edge_index, batch=batch)[0]
        masked_in  = torch.softmax(logits_in, dim=0)[pred_cls].item()

        # ── 软掩码 masked_out：保留非重要节点特征，重要部分置零 ───
        x_out      = x * (1.0 - mask_col)                 # 重要节点特征 → 0
        logits_out = model(x_out, edge_index, batch=batch)[0]
        masked_out = torch.softmax(logits_out, dim=0)[pred_cls].item()

    return {
        'ori':        ori_prob,
        'masked_in':  masked_in,
        'masked_out': masked_out,
    }