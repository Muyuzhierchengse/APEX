import argparse
import os
import os.path as osp
import re
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
from method.evaluation import get_node_mask_from_edge_mask, eval_related_pred
from method.explainpoly import PolyGINExplainer

def compatible_state_dict(state_dict):
    """修复新版 PyG 中 GCNConv weight 键名及形状变化。"""
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

def main():
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BBBP',
                        choices=['BBBP', 'ClinTox', 'Graph-SST2', 'Graph-Twitter','BA_2Motifs','BACE','Tox21','ToxCast'])
    parser.add_argument('--model_used', type=str, default='GIN_3l',
                        choices=['GCN_2l', 'GCN_3l', 'GIN_2l', 'GIN_3l','PolyGIN_3l'])
    parser.add_argument('--explainer', type=str, default='GradCAM',
                        choices=['FlowX', 'GNNExplainer', 'GraphEXT',
                                 'PGExplainer', 'GradCAM', 'FSX','PolyGINExplainer'])
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--dim_hidden', type=int, default=300)
    args = parser.parse_args()

    # ── 路径设置 ──────────────────────────────────────────────────
    data_path       = './dataset'
    checkpoint_path = './model/checkpoint'
    model_save_path = osp.join(checkpoint_path, args.dataset,
                               args.model_used +  f'_seed0.pkl')
    log_path = osp.join('log', args.dataset, args.explainer, args.model_used)
    os.makedirs(log_path, exist_ok=True)
    log_file = osp.join(log_path, f'Sparsity={args.sparsity}.log')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(log_file, 'a') as f:
        f.write(f'model used: {args.model_used}\n')
        f.write(f'method used: {args.explainer}\n')

    # ── 数据 & 模型 ────────────────────────────────────────────────
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

    # ── Explainer 初始化 ───────────────────────────────────────────
    if args.explainer == 'PGExplainer':
        explainer = PGExplainer(model, in_channels=600,
                                device=device, explain_graph=True)
        tmp_file = osp.join(checkpoint_path, args.dataset,
                            args.model_used + '_PGExplainer.pt')
        if not osp.exists(tmp_file):
            explainer.train_explanation_network(data['train'])
            torch.save(explainer.state_dict(), tmp_file)
        explainer.load_state_dict(torch.load(tmp_file, map_location=device))
    else:
        explainer = eval(args.explainer)(model, explain_graph=True)

    # ── 评估循环 ───────────────────────────────────────────────────
    data_loader = DataLoader(data['test'], batch_size=1, shuffle=False)
    fid_sum, fid_inv_sum, valid_count = 0.0, 0.0, 0

    for index, graph in enumerate(data_loader):
        if graph.num_nodes <= 1:
            continue

        graph = graph.to(device)
        print(f'Explaining graph #{index + 1}  (nodes={graph.num_nodes})')

        with torch.no_grad():
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            pred_cls = model(graph.x, graph.edge_index, batch=batch)[0].argmax(-1).item()
            if pred_cls != graph.y.item():
                print(f'  Skipping: prediction incorrect (pred={pred_cls}, true={graph.y.item()})\n')
                continue
        # explainer 返回 edge-level importance mask
        result = explainer(
            graph.x, graph.edge_index,
            sparsity=0,
            num_classes=num_classes,
            node_idx=0,
            max_nodes=int(graph.num_nodes * (1 - args.sparsity)),
        )

        if isinstance(result, tuple):
            # PolyGINExplainer：直接使用解析 node mask，跳过启发式边转换
            _, node_masks = result
            # node_masks 已经按 sparsity 二值化，无需再调用 get_node_mask_from_edge_mask
        else:
            # 其他 explainer：走原有 edge_mask → node_mask 转换
            edge_masks = result
            node_masks = get_node_mask_from_edge_mask(
                edge_masks, graph.num_nodes, graph.edge_index,
                num_classes, args.sparsity,
            )

        # 物理子图切割 + 推理（只针对 pred_cls，共 3 次前向传播）
        r = eval_related_pred(
            model, graph.x, graph.edge_index,
            node_masks, pred_cls, device,
        )

        fid     = r['ori'] - r['masked_out']   # Fidelity+（越大越好）
        fid_inv = r['ori'] - r['masked_in']    # Fidelity-（越小越好）

        print(f"  Fidelity+  = {fid:.4f}")
        print(f"  Fidelity-  = {fid_inv:.4f}")
        print(f"  Sparsity   = {args.sparsity:.4f}\n")

        with open(log_file, 'a') as f:
            f.write(f'graph #{index + 1:d}  '
                    f'(fid+={fid:.4f}, fid-={fid_inv:.4f})\n')

        fid_sum     += fid
        fid_inv_sum += fid_inv
        valid_count += 1

    # ── 汇总结果 ───────────────────────────────────────────────────
    if valid_count == 0:
        print('No valid graphs to evaluate.')
        return

    avg_fid     = fid_sum     / valid_count
    avg_fid_inv = fid_inv_sum / valid_count

    summary = (
        f'\n=== Final Results ({valid_count} graphs) ===\n'
        f'  Fidelity+  = {avg_fid:.4f}  (higher is better)\n'
        f'  Fidelity-  = {avg_fid_inv:.4f}  (lower is better)\n'
        f'  Sparsity   = {args.sparsity:.4f}\n'
    )
    print(summary)
    with open(log_file, 'a') as f:
        f.write(summary)


if __name__ == '__main__':
    main()