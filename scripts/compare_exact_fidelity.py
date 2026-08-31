"""
main_fidelity.py
================
理论完备性实验：绝对保真度 (Absolute Fidelity)
新增 --tau 参数，对应 Step 1 死区截断阈值。
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import time
import functools
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.loader import DataLoader

try:
    from torch_geometric.data.data import DataEdgeAttr
    torch.serialization.add_safe_globals([DataEdgeAttr])
except ImportError:
    pass

torch.load = functools.partial(torch.load, weights_only=False)

from apex.models.gnn import *
from apex.data.loaders import *
from apex.explainers.algebraic_poly_gin import PolyGINExplainer
from apex.explainers.pgexplainer import PGExplainer
from apex.utils.checkpoints import compatible_state_dict
from apex.utils.reproducibility import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 模型安全前向
# ─────────────────────────────────────────────────────────────────────────────

def safe_forward(model, x, edge_index, device) -> Tensor:
    N = x.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, edge_index, batch)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    prob = F.softmax(logits.float(), dim=-1)
    return prob.clamp(1e-7, 1.0 - 1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# Fidelity 计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_fidelity(
    model, x, edge_index, E_exact_indices, pred_cls, device,
) -> Tuple[float, float]:
    num_edges = edge_index.size(1)
    p_orig = safe_forward(model, x, edge_index, device)
    p_orig_c = p_orig[pred_cls].item()

    # Fidelity+
    if E_exact_indices.numel() == 0:
        fid_plus = p_orig_c
    else:
        E_sub = edge_index[:, E_exact_indices]
        p_sub = safe_forward(model, x, E_sub, device)
        fid_plus = abs(p_orig_c - p_sub[pred_cls].item())

    # Fidelity-
    exact_set = set(E_exact_indices.tolist())
    compl_idx = torch.tensor(
        [i for i in range(num_edges) if i not in exact_set],
        dtype=torch.long, device=device,
    )
    if compl_idx.numel() == 0:
        fid_minus = p_orig_c
    else:
        E_compl = edge_index[:, compl_idx]
        p_compl = safe_forward(model, x, E_compl, device)
        fid_minus = abs(p_orig_c - p_compl[pred_cls].item())

    return float(fid_plus), float(fid_minus)


def edge_mask_to_indices(edge_mask, sparsity) -> Tensor:
    num_edges = edge_mask.size(0)
    keep_k = max(1, int(num_edges * (1.0 - sparsity)))
    return edge_mask.topk(keep_k).indices


# ─────────────────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────────────────

def run_fidelity_graph(args, model, data, num_classes, device, log_file, explainers):
    data_loader = DataLoader(data['test'], batch_size=1, shuffle=False)
    stats = {name: {'fid_plus': [], 'fid_minus': [], 'time': []}
             for name in explainers}

    for index, graph in enumerate(data_loader):
        if graph.num_nodes <= 1:
            continue

        graph = graph.to(device)
        N = graph.num_nodes
        batch = torch.zeros(N, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(graph.x, graph.edge_index, batch)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            pred_cls = logits.argmax(-1).item()

        if pred_cls != graph.y.item():
            continue

        print(f'\n── Graph #{index + 1}  (nodes={N}, edges={graph.edge_index.size(1)}) ──')

        for name, explainer in explainers.items():
            t0 = time.perf_counter()
            result = explainer(
                graph.x, graph.edge_index,
                sparsity=args.sparsity,
                num_classes=num_classes,
                node_idx=0,
                max_nodes=int(N * (1 - args.sparsity)),
            )
            t_elapsed = time.perf_counter() - t0

            if isinstance(result, tuple):
                edge_masks, _ = result
            else:
                edge_masks = result

            if name == 'PolyGINExplainer':
                E_exact_indices = explainer.last_E_exact_indices
                if E_exact_indices is None:
                    E_exact_indices = edge_mask_to_indices(
                        edge_masks[pred_cls], args.sparsity)
            else:
                E_exact_indices = edge_mask_to_indices(
                    edge_masks[pred_cls].detach(), args.sparsity)

            fid_plus, fid_minus = compute_fidelity(
                model, graph.x, graph.edge_index,
                E_exact_indices, pred_cls, device,
            )

            stats[name]['fid_plus'].append(fid_plus)
            stats[name]['fid_minus'].append(fid_minus)
            stats[name]['time'].append(t_elapsed)

            print(f'  [{name}]  F+={fid_plus:.6f}  F-={fid_minus:.6f}  '
                  f't={t_elapsed:.4f}s')

            with open(log_file, 'a') as f:
                f.write(f'graph #{index + 1}  [{name}]  '
                        f'F+={fid_plus:.6f}  F-={fid_minus:.6f}  '
                        f't={t_elapsed:.4f}s\n')

    _write_fidelity_summary(log_file, stats)


def _write_fidelity_summary(log_file, stats):
    lines = ['\n' + '=' * 60,
             '  Fidelity Summary',
             '=' * 60,
             f'  {"Method":<22} {"F+ (↓)":<14} {"F- (↑)":<14} {"Time/s":<10}',
             '-' * 60]
    for name, s in stats.items():
        n = len(s['fid_plus'])
        if n == 0:
            lines.append(f'  {name:<22}  No valid samples')
            continue
        lines.append(
            f'  {name:<22} {np.mean(s["fid_plus"]):<14.6f} '
            f'{np.mean(s["fid_minus"]):<14.6f} '
            f'{np.mean(s["time"]):<10.4f}  (n={n})'
        )
    lines.append('=' * 60)
    summary = '\n'.join(lines)
    print(summary)
    with open(log_file, 'a') as f:
        f.write(summary + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str, default='BBBP',
                        choices=['BBBP', 'Graph-SST2', 'BACE', 'Mutagenicity'])
    parser.add_argument('--model_used', type=str, default='PolyGIN',
                        choices=['PolyGIN'])
    parser.add_argument('--sparsity',   type=float, default=0.5)
    parser.add_argument('--dim_hidden', type=int,   default=300)
    parser.add_argument('--freeze_K',  type=int,   default=10 ** 6)
    parser.add_argument('--freeze_p',  type=int,   default=10 ** 9 + 7)
    # ── 新增 ──
    parser.add_argument('--tau',        type=int,   default=0,
                        help='Dead-zone threshold τ: |floor(W*K)| < τ → 0. '
                             '0 = disabled (backward compatible). '
                             'Try 100~10000 for BBBP/PolyGIN.')
    args = parser.parse_args()

    checkpoint_path = './model/checkpoint'
    log_path = osp.join('log', args.dataset, 'FidelityExperiment', args.model_used)
    os.makedirs(log_path, exist_ok=True)
    log_file = osp.join(log_path,
                        f'Sparsity={args.sparsity}_tau={args.tau}_fidelity.log')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}, Model: {args.model_used}, '
          f'Sparsity: {args.sparsity}, tau={args.tau}')

    with open(log_file, 'a') as f:
        f.write(f'dataset:  {args.dataset}\n'
                f'model:    {args.model_used}\n'
                f'sparsity: {args.sparsity}\n'
                f'K={args.freeze_K}, p={args.freeze_p}, tau={args.tau}\n\n')

    data, num_nodes, dim_node, num_classes = load_dataset('./dataset', args.dataset)

    model = eval(args.model_used)(
        model_level='graph', dim_node=dim_node,
        dim_hidden=args.dim_hidden, num_classes=num_classes,
    ).to(device)
    raw_state = torch.load(
        osp.join(checkpoint_path, args.dataset, args.model_used + '_seed0.pkl'),
        map_location=device,
    )
    model.load_state_dict(compatible_state_dict(raw_state))
    model.eval()

    # ── PolyGINExplainer（含 tau）──────────────────────────────────────────
    explainers = {}
    explainers['PolyGINExplainer'] = PolyGINExplainer(
        model,
        explain_graph=True,
        K=args.freeze_K,
        p=args.freeze_p,
        probe_prime=10007,
        tau=args.tau,           # ← 传入死区阈值
    )

    # ── PGExplainer ────────────────────────────────────────────────────────
    in_ch = args.dim_hidden * 2
    pg_explainer = PGExplainer(model, in_channels=in_ch,
                               device=device, explain_graph=True)
    pg_tmp = osp.join(checkpoint_path, args.dataset,
                      args.model_used + '_PGExplainer.pt')
    train_data = data['train'] if isinstance(data, dict) else data
    if not osp.exists(pg_tmp):
        pg_explainer.train_explanation_network(train_data)
        torch.save(pg_explainer.state_dict(), pg_tmp)
    pg_explainer.load_state_dict(torch.load(pg_tmp, map_location=device))
    explainers['PGExplainer'] = pg_explainer

    run_fidelity_graph(args, model, data, num_classes, device, log_file, explainers)


if __name__ == '__main__':
    main()
