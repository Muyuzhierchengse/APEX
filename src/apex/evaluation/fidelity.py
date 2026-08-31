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

    model.eval()
    node_mask = node_masks[pred_cls].to(device)
    mask_col = node_mask.unsqueeze(-1)
    inv_mask_col = 1.0 - mask_col

    N = x.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_ori = model(x, edge_index, batch=batch)[0]
        ori_prob = torch.softmax(logits_ori, dim=0)[pred_cls].item()

        in_probs = []
        out_probs = []

        for _ in range(num_samples):
            perm = torch.randperm(N, device=device)
            x_ref = x[perm]

            x_in = mask_col * x + inv_mask_col * x_ref
            logits_in = model(x_in, edge_index, batch=batch)[0]
            in_probs.append(
                torch.softmax(logits_in, dim=0)[pred_cls].item()
            )

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


