import argparse
import os
import os.path as osp
import re
import time
import functools
from collections import OrderedDict
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric import __version__

try:
    from torch_geometric.data.data import DataEdgeAttr
    torch.serialization.add_safe_globals([DataEdgeAttr])
except ImportError:
    pass

torch.load = functools.partial(torch.load, weights_only=False)

from model.models import *
from dataset.data import *
from method.flowx import FlowX
from method.graphext import GraphEXT
from method.gnnexplainer import GNNExplainer
from method.gradcam import GradCAM
from method.pgexplainer import PGExplainer
from method.fsx import FSX
from method.evaluation import (
    get_node_mask_from_edge_mask,
    eval_related_pred,
    eval_related_pred_node,
)
from method.explainpoly import PolyGINExplainer
from method.ig import IntegratedGradients
from method.tig import TrapezoidalIG
from method.sig import SimpsonIG


def compatible_state_dict(state_dict):
    comp = OrderedDict()
    for key, value in state_dict.items():
        new_key = re.sub(r'conv(1|s\.[0-9]+)\.weight',
                         r'conv\1.lin.weight', key)
        comp[new_key] = value.T if new_key != key else value
    return comp


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── 图分类主逻辑 ─────────────────────────────────────────────────────────────

def run_graph_classification(args, model, data, num_classes, device,
                              log_file, explainer):
    data_loader = DataLoader(data['test'], batch_size=1, shuffle=False)
    fid_sum = fid_inv_sum = time_sum = 0.0
    valid_count = 0

    for index, graph in enumerate(data_loader):
        if graph.num_nodes <= 1:
            continue

        graph = graph.to(device)
        print(f'Explaining graph #{index + 1}  (nodes={graph.num_nodes})')

        with torch.no_grad():
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            out = model(graph.x, graph.edge_index, batch=batch)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            pred_cls = logits.argmax(-1).item()
            if pred_cls != graph.y.item():
                print(f'  Skipping: prediction incorrect '
                      f'(pred={pred_cls}, true={graph.y.item()})\n')
                continue

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
        else:
            edge_masks = result
            node_masks = get_node_mask_from_edge_mask(
                edge_masks, graph.num_nodes, graph.edge_index,
                num_classes, args.sparsity,
            )

        r = eval_related_pred(
            model, graph.x, graph.edge_index,
            node_masks, pred_cls, device,
        )
        fid     = r['ori'] - r['masked_out']
        fid_inv = r['ori'] - r['masked_in']

        t_elapsed = time.perf_counter() - t_start

        print(f"  Fidelity+      = {fid:.4f}")
        print(f"  Fidelity-      = {fid_inv:.4f}")
        print(f"  Sparsity       = {args.sparsity:.4f}")
        print(f"  time           = {t_elapsed:.4f} s\n")

        with open(log_file, 'a') as f:
            f.write(f'graph #{index + 1:d}  '
                    f'(fid+={fid:.4f}, fid-={fid_inv:.4f}, '
                    f'time={t_elapsed:.4f}s)\n')

        fid_sum     += fid
        fid_inv_sum += fid_inv
        time_sum    += t_elapsed
        valid_count += 1

    _write_summary(log_file, valid_count, fid_sum, fid_inv_sum,
                   time_sum, args.sparsity)


# ── 节点分类主逻辑 ───────────────────────────────────────────────────────────

def run_node_classification(args, model, data, num_classes, device,
                             log_file, explainer):
    if isinstance(data, dict):
        from torch_geometric.data import Batch
        graph = Batch.from_data_list(list(data['test'])).to(device)
    else:
        graph = data.to(device)

    num_nodes = graph.num_nodes

    if hasattr(graph, 'test_mask') and graph.test_mask is not None:
        target_nodes = graph.test_mask.nonzero(as_tuple=True)[0].tolist()
    else:
        target_nodes = list(range(num_nodes))

    print(f'Node classification: {len(target_nodes)} test nodes, '
          f'total nodes={num_nodes}')

    with torch.no_grad():
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        out = model(graph.x, graph.edge_index, batch=batch)
        all_logits = out[0] if isinstance(out, (tuple, list)) else out
        all_pred   = all_logits.argmax(dim=-1)

    fid_sum = fid_inv_sum = time_sum = 0.0
    valid_count = 0

    for index, node_idx in enumerate(target_nodes):
        pred_cls  = all_pred[node_idx].item()
        true_cls  = graph.y[node_idx].item()

        if pred_cls != true_cls:
            print(f'  Node {node_idx}: skipping (pred={pred_cls}, true={true_cls})')
            continue

        print(f'Explaining node #{node_idx}  '
              f'[{index + 1}/{len(target_nodes)}]  cls={pred_cls}')

        t_start = time.perf_counter()

        result = explainer(
            graph.x, graph.edge_index,
            sparsity=0,
            num_classes=num_classes,
            node_idx=node_idx,
            max_nodes=int(num_nodes * (1 - args.sparsity)),
        )

        if isinstance(result, tuple):
            edge_masks, node_masks = result
            raw_node_scores = explainer.last_node_scores
        else:
            edge_masks = result
            node_masks = get_node_mask_from_edge_mask(
                edge_masks, num_nodes, graph.edge_index,
                num_classes, args.sparsity,
            )
            raw_node_scores = []
            for cls in range(num_classes):
                em  = edge_masks[cls].detach()
                ns  = torch.zeros(num_nodes, device=device)
                src, dst = graph.edge_index[0], graph.edge_index[1]
                ns.scatter_add_(0, src, em)
                ns.scatter_add_(0, dst, em)
                deg  = torch.zeros(num_nodes, device=device)
                ones = torch.ones(graph.edge_index.size(1), device=device)
                deg.scatter_add_(0, src, ones)
                deg.scatter_add_(0, dst, ones)
                ns = ns / (deg + 1e-8)
                raw_node_scores.append(ns)

        r = eval_related_pred_node(
            model, graph.x, graph.edge_index,
            raw_node_scores,
            pred_cls, node_idx, device,
            sparsity=args.sparsity,
            num_hops=3,
        )
        fid     = r['fid_plus']
        fid_inv = r['fid_minus']

        t_elapsed = time.perf_counter() - t_start

        print(f"  Fidelity+      = {fid:.4f}")
        print(f"  Fidelity-      = {fid_inv:.4f}")
        print(f"  Sparsity       = {args.sparsity:.4f}")
        print(f"  time           = {t_elapsed:.4f} s\n")

        with open(log_file, 'a') as f:
            f.write(f'node #{node_idx:d}  '
                    f'(fid+={fid:.4f}, fid-={fid_inv:.4f}, '
                    f'time={t_elapsed:.4f}s)\n')

        fid_sum     += fid
        fid_inv_sum += fid_inv
        time_sum    += t_elapsed
        valid_count += 1

    _write_summary(log_file, valid_count, fid_sum, fid_inv_sum,
                   time_sum, args.sparsity)


# ── 公共 summary 输出 ────────────────────────────────────────────────────────

def _write_summary(log_file, valid_count, fid_sum, fid_inv_sum,
                   time_sum, sparsity):
    if valid_count == 0:
        print('No valid samples to evaluate.')
        return

    avg_fid     = fid_sum     / valid_count
    avg_fid_inv = fid_inv_sum / valid_count
    avg_time    = time_sum    / valid_count

    summary = (
        f'\n=== Final Results ({valid_count} samples) ===\n'
        f'  Fidelity+      = {avg_fid:.4f}  (higher is better)\n'
        f'  Fidelity-      = {avg_fid_inv:.4f}  (lower is better)\n'
        f'  Sparsity       = {sparsity:.4f}\n'
        f'  Avg time/sample= {avg_time:.4f} s\n'
    )
    print(summary)
    with open(log_file, 'a') as f:
        f.write(summary)


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BBBP',
                        choices=['BBBP', 'ClinTox', 'Graph-SST2', 'Graph-Twitter',
                                 'BA_2Motifs', 'BACE', 'Tox21', 'ToxCast',
                                 'BA_shapes','ogbg-molhiv','ogbg-molpcba','ogbn-proteins'])
    parser.add_argument('--model_used', type=str, default='GIN_3l',
                        choices=['GCN_2l', 'GCN_3l', 'GIN_2l', 'GIN_3l', 'PolyGIN_3l'])
    parser.add_argument('--explainer', type=str, default='GradCAM',
                        choices=['FlowX', 'GNNExplainer', 'GraphEXT',
                                 'PGExplainer', 'GradCAM', 'FSX', 'PolyGINExplainer',
                                 'IntegratedGradients', 'TrapezoidalIG', 'SimpsonIG'])
    parser.add_argument('--sparsity',      type=float, default=0.5)
    parser.add_argument('--dim_hidden',    type=int,   default=300)
    args = parser.parse_args()

    NODE_CLS_DATASETS = {'BA_shapes'}
    is_node_cls = args.dataset in NODE_CLS_DATASETS

    data_path       = './dataset'
    checkpoint_path = './model/checkpoint'
    model_save_path = osp.join(checkpoint_path, args.dataset,
                               args.model_used + f'_seed0.pkl')
    log_path = osp.join('log', args.dataset, args.explainer, args.model_used)
    os.makedirs(log_path, exist_ok=True)
    log_file = osp.join(log_path, f'Sparsity={args.sparsity}.log')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(log_file, 'a') as f:
        f.write(f'model used:  {args.model_used}\n')
        f.write(f'method used: {args.explainer}\n')
        f.write(f'task:        {"node" if is_node_cls else "graph"} classification\n')

    data, num_nodes, dim_node, num_classes = load_dataset(data_path, args.dataset)

    model_level = 'node' if is_node_cls else 'graph'
    model = eval(args.model_used)(
        model_level=model_level,
        dim_node=dim_node,
        dim_hidden=args.dim_hidden,
        num_classes=num_classes,
    ).to(device)

    raw_state = torch.load(model_save_path, map_location=device)
    model.load_state_dict(compatible_state_dict(raw_state))
    model.eval()

    explain_graph = not is_node_cls

    if args.explainer == 'PGExplainer':
        in_ch = args.dim_hidden * 3 if is_node_cls else args.dim_hidden * 2
        explainer = PGExplainer(model, in_channels=in_ch,
                                device=device, explain_graph=explain_graph)
        tmp_file = osp.join(checkpoint_path, args.dataset,
                            args.model_used + '_PGExplainer.pt')
        train_data = data['train'] if isinstance(data, dict) else data
        if not osp.exists(tmp_file):
            explainer.train_explanation_network(train_data)
            torch.save(explainer.state_dict(), tmp_file)
        explainer.load_state_dict(torch.load(tmp_file, map_location=device))
    elif args.explainer == 'IntegratedGradients':
        explainer = IntegratedGradients(model, explain_graph=explain_graph, m_steps=50)
    elif args.explainer == 'TrapezoidalIG':
        explainer = TrapezoidalIG(model, explain_graph=explain_graph, m_steps=50)
    elif args.explainer == 'SimpsonIG':
        explainer = SimpsonIG(model, explain_graph=explain_graph, m_steps=50)
    else:
        explainer = eval(args.explainer)(model, explain_graph=explain_graph)

    if is_node_cls:
        run_node_classification(args, model, data, num_classes, device,
                                log_file, explainer)
    else:
        run_graph_classification(args, model, data, num_classes, device,
                                 log_file, explainer)


if __name__ == '__main__':
    main()