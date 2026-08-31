import argparse
import os
import os.path as osp
import time                            # ← 新增
import functools
import torch
from torch_geometric.loader import DataLoader
from torch_geometric import __version__

try:
    from torch_geometric.data.data import DataEdgeAttr
    torch.serialization.add_safe_globals([DataEdgeAttr])
except ImportError:
    pass

torch.load = functools.partial(torch.load, weights_only=False)

from apex.models.gnn import *
from apex.data.loaders import *
from apex.explainers.flowx import FlowX
from apex.explainers.gnnexplainer import GNNExplainer
from apex.explainers.gradcam import GradCAM
from apex.explainers.pgexplainer import PGExplainer
from apex.evaluation.fidelity import (
    get_node_mask_from_edge_mask,
    eval_related_pred,
)
from apex.evaluation.stability import eval_stability
from apex.explainers.poly_gin import PolyGINExplainer
from apex.explainers.integrated_gradients import IntegratedGradients
from apex.explainers.gauss_legendre_ig import GaussLegendreIG
from apex.explainers.adaptive_riemann_ig import RiemannOptIG
from apex.utils.checkpoints import compatible_state_dict
from apex.utils.paths import CHECKPOINT_DIR, DATA_DIR, LOG_DIR
from apex.utils.reproducibility import set_seed


def compute_efficiency_gap(model, x, edge_index, node_scores, pred_cls, device):
    num_nodes = x.size(0)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_x    = model(x,                   edge_index, batch=batch)[0]
        logits_zero = model(torch.zeros_like(x), edge_index, batch=batch)[0]

    fx      = logits_x[pred_cls].item()
    f0      = logits_zero[pred_cls].item()
    delta_f = fx - f0

    sum_phi = node_scores[pred_cls].sum().item()
    gap     = abs(sum_phi - delta_f)
    return gap, sum_phi, delta_f


def main():
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BBBP',
                        choices=['BBBP', 'Graph-SST2', 'BACE', 'Mutagenicity'])
    parser.add_argument('--model_used', type=str, default='GIN',
                        choices=['GCN_2l', 'GCN', 'GIN_2l', 'GIN', 'PolyGIN'])
    parser.add_argument('--explainer', type=str, default='GradCAM',
                        choices=['FlowX', 'GNNExplainer',
                                 'PGExplainer', 'GradCAM', 'PolyGINExplainer',
                                 'IntegratedGradients', 'GaussLegendreIG', 'RiemannOptIG'])
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--n_perturb',     type=int,   default=5)
    parser.add_argument('--perturb_ratio', type=float, default=0.1)
    args = parser.parse_args()

    data_path       = str(DATA_DIR)
    checkpoint_path = str(CHECKPOINT_DIR)
    model_save_path = osp.join(checkpoint_path, args.dataset,
                               args.model_used + f'_seed0.pkl')
    log_path = osp.join(str(LOG_DIR), args.dataset, args.explainer, args.model_used)
    os.makedirs(log_path, exist_ok=True)
    log_file = osp.join(log_path, f'Sparsity={args.sparsity}.log')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(log_file, 'a') as f:
        f.write(f'model used: {args.model_used}\n')
        f.write(f'method used: {args.explainer}\n')

    data, num_nodes, dim_node, num_classes = load_dataset(data_path, args.dataset)

    model = eval(args.model_used)(
        model_level='graph',
        dim_node=dim_node,
        dim_hidden=args.dim_hidden,
        num_classes=num_classes,
    ).to(device)

    raw_state = torch.load(model_save_path, map_location=device)
    model.load_state_dict(compatible_state_dict(raw_state))
    model.eval()

    if args.explainer == 'PGExplainer':
        explainer = PGExplainer(model, in_channels=600,
                                device=device, explain_graph=True)
        tmp_file = osp.join(checkpoint_path, args.dataset,
                            args.model_used + '_PGExplainer.pt')
        if not osp.exists(tmp_file):
            explainer.train_explanation_network(data['train'])
            torch.save(explainer.state_dict(), tmp_file)
        explainer.load_state_dict(torch.load(tmp_file, map_location=device))
    elif args.explainer == 'IntegratedGradients':
        explainer = IntegratedGradients(model, explain_graph=True, m_steps=50)
    elif args.explainer == 'GaussLegendreIG':
        explainer = GaussLegendreIG(model, explain_graph=True, n_steps=50)
    elif args.explainer in ('RiemannOptIG',):
        explainer = eval(args.explainer)(model, explain_graph=True, m_steps=50)
    else:
        explainer = eval(args.explainer)(model, explain_graph=True)

    data_loader = DataLoader(data['test'], batch_size=1, shuffle=False)
    fid_sum, fid_inv_sum, gap_sum, stab_sum, time_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    valid_count = 0

    for index, graph in enumerate(data_loader):
        if graph.num_nodes <= 1:
            continue

        graph = graph.to(device)
        print(f'Explaining graph #{index + 1}  (nodes={graph.num_nodes})')

        with torch.no_grad():
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            pred_cls = model(graph.x, graph.edge_index, batch=batch)[0].argmax(-1).item()
            if pred_cls != graph.y.item():
                print(f'  Skipping: prediction incorrect '
                      f'(pred={pred_cls}, true={graph.y.item()})\n')
                continue

        # ── 计时开始 ───────────────────────────────────────────────
        t_start = time.perf_counter()

        result = explainer(
            graph.x, graph.edge_index,
            sparsity=0,
            num_classes=num_classes,
            node_idx=0,
            max_nodes=int(graph.num_nodes * (1 - args.sparsity)),
        )

        if isinstance(result, tuple):
            edge_masks, node_masks = result
            raw_node_scores = explainer.last_node_scores
        else:
            edge_masks = result
            node_masks = get_node_mask_from_edge_mask(
                edge_masks, graph.num_nodes, graph.edge_index,
                num_classes, args.sparsity,
            )
            raw_node_scores = []
            for cls in range(num_classes):
                em  = edge_masks[cls].detach()
                ns  = torch.zeros(graph.num_nodes, device=device)
                src, dst = graph.edge_index[0], graph.edge_index[1]
                ns.scatter_add_(0, src, em)
                ns.scatter_add_(0, dst, em)
                deg  = torch.zeros(graph.num_nodes, device=device)
                ones = torch.ones(graph.edge_index.size(1), device=device)
                deg.scatter_add_(0, src, ones)
                deg.scatter_add_(0, dst, ones)
                ns = ns / (deg + 1e-8)
                raw_node_scores.append(ns)

        gap, sum_phi, delta_f = compute_efficiency_gap(
            model, graph.x, graph.edge_index,
            raw_node_scores, pred_cls, device,
        )

        r = eval_related_pred(
            model, graph.x, graph.edge_index,
            node_masks, pred_cls, device,
        )

        fid     = r['ori'] - r['masked_out']
        fid_inv = r['ori'] - r['masked_in']

        stability = eval_stability(
            explainer,
            graph.x, graph.edge_index,
            node_masks,
            pred_cls,
            num_classes,
            graph.num_nodes,
            args.sparsity,
            device,
            perturb_ratio=args.perturb_ratio,
            n_perturb=args.n_perturb,
            seed_base=100 + index * args.n_perturb,
        )

        # ── 计时结束 ───────────────────────────────────────────────
        t_elapsed = time.perf_counter() - t_start

        print(f"  Fidelity+        = {fid:.4f}")
        print(f"  Fidelity-        = {fid_inv:.4f}")
        print(f"  Efficiency Gap   = {gap:.6e}  "
              f"(Σφ={sum_phi:.4f}, Δf={delta_f:.4f})")
        print(f"  Stability        = {stability:.4f}  "
              f"(Jaccard@{args.n_perturb}×perturb={args.perturb_ratio})")
        print(f"  Sparsity         = {args.sparsity:.4f}")
        print(f"  time             = {t_elapsed:.4f} s\n")

        with open(log_file, 'a') as f:
            f.write(f'graph #{index + 1:d}  '
                    f'(fid+={fid:.4f}, fid-={fid_inv:.4f}, '
                    f'gap={gap:.6e}, sum_phi={sum_phi:.4f}, delta_f={delta_f:.4f}, '
                    f'stability={stability:.4f}, time={t_elapsed:.4f}s)\n')

        fid_sum     += fid
        fid_inv_sum += fid_inv
        gap_sum     += gap
        stab_sum    += stability
        time_sum    += t_elapsed
        valid_count += 1

    if valid_count == 0:
        print('No valid graphs to evaluate.')
        return

    avg_fid     = fid_sum     / valid_count
    avg_fid_inv = fid_inv_sum / valid_count
    avg_gap     = gap_sum     / valid_count
    avg_stab    = stab_sum    / valid_count
    avg_time    = time_sum    / valid_count

    summary = (
        f'\n=== Final Results ({valid_count} graphs) ===\n'
        f'  Fidelity+        = {avg_fid:.4f}  (higher is better)\n'
        f'  Fidelity-        = {avg_fid_inv:.4f}  (lower is better)\n'
        f'  Efficiency Gap   = {avg_gap:.6e}  (lower is better)\n'
        f'  Stability        = {avg_stab:.4f}  (higher is better)\n'
        f'  Sparsity         = {args.sparsity:.4f}\n'
        f'  Avg time/graph   = {avg_time:.4f} s\n'
    )
    print(summary)
    with open(log_file, 'a') as f:
        f.write(summary)


if __name__ == '__main__':
    main()
