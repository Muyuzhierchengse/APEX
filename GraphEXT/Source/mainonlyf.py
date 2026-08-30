import argparse
import os
import os.path as osp
import re
import time
import functools
from collections import OrderedDict
import numpy as np
import random
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
from method.gnnexplainer import GNNExplainer
from method.gradcam import GradCAM
from method.pgexplainer import PGExplainer
from method.evaluation import (
    get_node_mask_from_edge_mask,
    eval_related_pred,
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

# Explains every correctly-predicted test graph and logs fidelity/sparsity/time,
# then writes the averaged results via _write_summary.
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


def main():
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BBBP',
                        choices=[ 'BBBP', 'Graph-SST2','BACE', 'Mutagenicity', 'BA_shapes'])
    parser.add_argument('--model_used', type=str, default='GIN',
                        choices=['GCN_2l', 'GCN', 'GIN_2l', 'GIN', 'PolyGIN'])
    parser.add_argument('--explainer', type=str, default='GradCAM',
                        choices=['FlowX', 'GNNExplainer', 
                                 'PGExplainer', 'GradCAM', 'PolyGINExplainer',
                                 'IntegratedGradients','TrapezoidalIG', 'SimpsonIG'])
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

    run_graph_classification(args, model, data, num_classes, device,
                                log_file, explainer)


if __name__ == '__main__':
    main()
    