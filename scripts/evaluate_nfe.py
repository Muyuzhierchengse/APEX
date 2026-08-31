import argparse
import os
import os.path as osp
import time
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
from apex.evaluation.nfe import NFECounter
from apex.explainers.poly_gin import PolyGINExplainer
from apex.explainers.integrated_gradients import IntegratedGradients
from apex.explainers.trapezoidal_ig import TrapezoidalIG
from apex.explainers.simpson_ig import SimpsonIG
from apex.utils.checkpoints import compatible_state_dict
from apex.utils.reproducibility import set_seed

# Checks the completeness axiom: sum of per-node attribution scores should
# match the model output gap between the real input and an all-zero baseline.
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
                                 'IntegratedGradients','TrapezoidalIG', 'SimpsonIG'])
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--dim_hidden', type=int, default=300)
    args = parser.parse_args()

    data_path       = './dataset'
    checkpoint_path = './model/checkpoint'
    model_save_path = osp.join(checkpoint_path, args.dataset,
                               args.model_used + f'_seed0.pkl')
    log_path = osp.join('log', args.dataset, args.explainer, args.model_used)
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

    nfe_counter = NFECounter()
    nfe_counter.register(model)

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
        explainer = IntegratedGradients(model, explain_graph=True, m_steps=1024)
    elif args.explainer == 'TrapezoidalIG':
        explainer = TrapezoidalIG(model, explain_graph=True, m_steps=1024)
    elif args.explainer == 'SimpsonIG':
        explainer = SimpsonIG(model, explain_graph=True, m_steps=1024)
    else:
        explainer = eval(args.explainer)(model, explain_graph=True)

    data_loader = DataLoader(data['test'], batch_size=1, shuffle=False)
    gap_sum, time_sum = 0.0, 0.0
    nfe_sum = 0
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

        nfe_counter.reset()
        t_start = time.perf_counter()

        result = explainer(
            graph.x, graph.edge_index,
            sparsity=0,
            num_classes=num_classes,
            node_idx=0,
            max_nodes=int(graph.num_nodes * (1 - args.sparsity)),
        )

        t_elapsed = time.perf_counter() - t_start
        # Hook fires once per class the explainer evaluates, so divide out num_classes
        # to get the NFE count for a single explanation.
        nfe = nfe_counter.value // num_classes

        if isinstance(result, tuple):
            edge_masks, node_masks = result
            raw_node_scores = explainer.last_node_scores
        else:
            edge_masks = result
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

        print(f"  NFE              = {nfe}")
        print(f"  Efficiency Gap   = {gap:.6e}  "
              f"(Σφ={sum_phi:.4f}, Δf={delta_f:.4f})")
        print(f"  time             = {t_elapsed:.4f} s\n")

        with open(log_file, 'a') as f:
            f.write(f'graph #{index + 1:d}  '
                    f'(nfe={nfe}, '
                    f'gap={gap:.6e}, sum_phi={sum_phi:.4f}, delta_f={delta_f:.4f}, '
                    f'time={t_elapsed:.4f}s)\n')

        gap_sum     += gap
        time_sum    += t_elapsed
        nfe_sum     += nfe
        valid_count += 1

    if valid_count == 0:
        print('No valid graphs to evaluate.')
        nfe_counter.remove()
        return

    avg_gap  = gap_sum  / valid_count
    avg_time = time_sum / valid_count
    avg_nfe  = nfe_sum  / valid_count

    summary = (
        f'\n=== Final Results ({valid_count} graphs) ===\n'
        f'  NFE (avg/graph)  = {avg_nfe:.1f}\n'
        f'  Efficiency Gap   = {avg_gap:.6e}  (lower is better)\n'
        f'  Avg time/graph   = {avg_time:.4f} s\n'
    )
    print(summary)
    with open(log_file, 'a') as f:
        f.write(summary)

    nfe_counter.remove()


if __name__ == '__main__':
    main()
