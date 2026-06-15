"""
顶会标准 GNN 可解释性可视化 v7

修复策略（基于 v4 已验证的归一化管线）:
  1. 保留 v4 的 _normalise_shapley 幂变换管线（power=3 对比度增强）
  2. 统一所有方法使用同一归一化，确保公平对比
  3. PolyGINExplainer: 调用 explainpoly.py 取 last_node_scores，转为绝对值后用 power 归一化
  4. 修复 Mutagenicity 图筛选: _has_isolated_H → _has_truly_isolated_H
  5. 固定布局 + 统一可视化风格
"""

import os, os.path as osp, functools, re, random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.data import Data

try:
    from torch_geometric.data.data import DataEdgeAttr
    torch.serialization.add_safe_globals([DataEdgeAttr])
except ImportError:
    pass
torch.load = functools.partial(torch.load, weights_only=False)

from model.models import GIN_3l, PolyGIN_3l
from dataset.data import load_dataset
from method.flowx        import FlowX
from method.gradcam      import GradCAM
from method.explainpoly  import PolyGINExplainer
from method.ig           import IntegratedGradients


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compatible_state_dict(sd):
    comp = OrderedDict()
    for k, v in sd.items():
        nk = re.sub(r'conv(1|s\.[0-9]+)\.weight', r'conv\1.lin.weight', k)
        comp[nk] = v.T if nk != k else v
    return comp

DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_PATH  = './dataset'
CKPT_PATH  = './model/checkpoint'
DIM_HIDDEN = 300
SPARSITY   = 0.5

EXPLAINER_NAMES = ['FlowX', 'GradCAM', 'PolyGINExplainer', 'IntegratedGradients']

# 白→红 colormap
CMAP_IMPORTANCE = LinearSegmentedColormap.from_list(
    'importance', ['#FFFFFF', '#FFD4D4', '#FF6B6B', '#CC0000'], N=256)

# 原子序数 → 符号
ATOMIC_NUM_TO_SYM = {
    1:'H',  5:'B',  6:'C',  7:'N',  8:'O',  9:'F',
    11:'Na',12:'Mg',13:'Al',14:'Si',15:'P', 16:'S',
    17:'Cl',19:'K', 20:'Ca',26:'Fe',35:'Br',53:'I',
}

# Mutagenicity 14 维 one-hot
MUTAGENICITY_14 = ['C','O','Cl','H','N','F','Br','S','P','I','Na','K','Li','Ca']


# ═══════════════════════════════════════════════════════════════════════════
#  特征编码 + 原子解码
# ═══════════════════════════════════════════════════════════════════════════

def _detect_encoding(graph):
    if graph.x is None:
        return 'onehot'
    col0 = graph.x[:, 0].cpu().float()
    if (col0 > 1).any() and (col0 - col0.round()).abs().max() < 0.01:
        return 'atomic_num'
    return 'onehot'

def _decode_atom_atomic_num(x_row):
    anum = int(round(float(x_row[0].item())))
    return ATOMIC_NUM_TO_SYM.get(anum, f'?{anum}')

def _decode_atom_onehot_14(x_row):
    vec = x_row.cpu().float().numpy()
    idx = int(vec[:14].argmax())
    return MUTAGENICITY_14[idx] if idx < len(MUTAGENICITY_14) else '?'

def get_node_labels(graph, ds_name):
    if graph.x is None:
        return ['?'] * graph.num_nodes
    enc = _detect_encoding(graph)
    result = []
    for i in range(graph.num_nodes):
        row = graph.x[i]
        if enc == 'atomic_num':
            result.append(_decode_atom_atomic_num(row))
        else:
            result.append(_decode_atom_onehot_14(row))
    return result

def sst2_node_labels(graph, vocab):
    labels = []
    for i in range(graph.num_nodes):
        idx = int(graph.x[i].cpu().argmax())
        labels.append(vocab.get(idx, str(idx)))
    return labels


# ═══════════════════════════════════════════════════════════════════════════
#  图结构辅助
# ═══════════════════════════════════════════════════════════════════════════

def _adj_dict(graph):
    ei  = graph.edge_index.cpu().numpy()
    adj = {i: [] for i in range(graph.num_nodes)}
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        adj[u].append(v)
    return adj

def _count_atoms(labels, targets):
    return sum(1 for s in labels if s in targets)

def _c_fraction(labels):
    n = len(labels)
    return sum(1 for s in labels if s == 'C') / n if n > 0 else 0.0

def _heavy_c_fraction(labels):
    """C 在非H重原子中的占比（Mutagenicity 显式H场景用）"""
    heavy = sum(1 for s in labels if s != 'H')
    if heavy == 0:
        return 0.0
    return sum(1 for s in labels if s == 'C') / heavy

def _is_connected(graph):
    G = to_networkx(graph, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return nx.is_connected(G)

def _has_ring(graph, min_ring_size=5):
    G = to_networkx(graph, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return any(len(c) >= min_ring_size for c in nx.cycle_basis(G))

def _cycle_count(graph):
    G = to_networkx(graph, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return len(nx.cycle_basis(G))

def _no2_count(labels, adj):
    return sum(
        1 for i, sym in enumerate(labels)
        if sym == 'N' and sum(1 for nb in adj[i] if labels[nb] == 'O') >= 2
    )

def _nh2_count(labels, adj):
    count = 0
    for i, sym in enumerate(labels):
        if sym != 'N': continue
        nbs = adj[i]
        if any(labels[nb] == 'O' for nb in nbs): continue
        h_nb = sum(1 for nb in nbs if labels[nb] == 'H')
        if h_nb >= 2 or len(nbs) == 1:
            count += 1
    return count

def _has_truly_isolated_H(graph, labels):
    """
    只标记 degree==0 的 H（真正无键连）。
    degree==1 的 H 是显式 H 表示的正常情况（C-H 键），不算孤立。
    """
    G = to_networkx(graph, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return any(sym == 'H' and G.degree(i) == 0
               for i, sym in enumerate(labels))

def _topo_stats(graph):
    G = to_networkx(graph, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    n_cycles = len(nx.cycle_basis(G))
    try:
        diam = nx.diameter(G) if nx.is_connected(G) else 0
    except Exception:
        diam = 0
    return n_cycles, diam

def debug_graph(ds_name, graph, tag=''):
    labels = get_node_labels(graph, ds_name)
    adj    = _adj_dict(graph)
    enc    = _detect_encoding(graph)
    print(f'  [{tag}] nodes={graph.num_nodes}, y={graph.y.item()}, '
          f'enc={enc}, atoms={sorted(set(labels))}, '
          f'NO2={_no2_count(labels,adj)}, NH2={_nh2_count(labels,adj)}, '
          f'hal={_count_atoms(labels,{"F","Cl","Br","I"})}, '
          f'C_frac={_c_fraction(labels):.2f} '
          f'(heavy={_heavy_c_fraction(labels):.2f}), '
          f'connected={_is_connected(graph)}, '
          f'isolated_H={_has_truly_isolated_H(graph,labels)}')


# ═══════════════════════════════════════════════════════════════════════════
#  图筛选 — Mutagenicity（核心修复）
# ═══════════════════════════════════════════════════════════════════════════

def select_mutagenicity_graphs(graphs, n=2):
    """
    修复: 移除 _has_isolated_H（显式 H 原子 degree=1 是正常的）。
    改用 _has_truly_isolated_H（只检查 degree==0）。
    改用 _heavy_c_fraction（非H重原子中 C 的占比）避免显式 H 干扰。
    """
    def prep(g):
        if not (hasattr(g, 'y') and g.y.item() == 1):
            return None
        if not (10 <= g.num_nodes <= 35):
            return None
        if not _is_connected(g):
            return None
        labels = get_node_labels(g, 'Mutagenicity')
        if _has_truly_isolated_H(g, labels):
            return None
        adj = _adj_dict(g)
        return labels, adj

    # ─ 图1: 硝基芳烃 (NO2) —
    no2_cands = []
    for g in graphs:
        r = prep(g)
        if r is None:
            continue
        labels, adj = r
        nc = _no2_count(labels, adj)
        if nc < 1:
            continue
        hcf = _heavy_c_fraction(labels)
        other_hetero = _count_atoms(labels, {'F', 'Cl', 'Br', 'I', 'S', 'P'})
        extra_n = sum(1 for i, s in enumerate(labels)
                       if s == 'N' and sum(1 for nb in adj[i] if labels[nb] == 'O') < 2)
        size_penalty = 0 if 12 <= g.num_nodes <= 25 else abs(g.num_nodes - 18) * 0.3
        score = hcf * 15 - other_hetero * 3 - extra_n * 3 - size_penalty + nc * 2
        no2_cands.append((g, score))
    no2_cands.sort(key=lambda x: x[1], reverse=True)

    # ─ 图2: 芳香胺 (NH2) —
    nh2_cands = []
    for g in graphs:
        r = prep(g)
        if r is None:
            continue
        labels, adj = r
        if _no2_count(labels, adj) > 0:
            continue
        if _nh2_count(labels, adj) < 1:
            continue
        if _count_atoms(labels, {'O'}) > 0:
            continue
        hcf = _heavy_c_fraction(labels)
        n_cnt = _count_atoms(labels, {'N'})
        size_penalty = 0 if 12 <= g.num_nodes <= 25 else abs(g.num_nodes - 18) * 0.3
        score = hcf * 15 - abs(n_cnt - 1) * 4 - size_penalty
        nh2_cands.append((g, score))
    nh2_cands.sort(key=lambda x: x[1], reverse=True)

    result = []
    if no2_cands:
        result.append(no2_cands[0][0])
        print(f'  [Mutagenicity] 图1(NO2)找到, score={no2_cands[0][1]:.2f}')
        debug_graph('Mutagenicity', no2_cands[0][0], '图1-NO2')
    else:
        print('  [Mutagenicity] !! 图1 NO2 未找到，fallback')
        fallback = [g for g in graphs
                    if hasattr(g,'y') and g.y.item()==1
                    and 8 <= g.num_nodes <= 40
                    and _is_connected(g)
                    and _no2_count(get_node_labels(g,'Mutagenicity'),
                                   _adj_dict(g)) >= 1]
        if fallback:
            fallback.sort(key=lambda g: _heavy_c_fraction(get_node_labels(g,'Mutagenicity')), reverse=True)
            result.append(fallback[0])
            debug_graph('Mutagenicity', fallback[0], '图1-NO2(fallback)')

    for g, sc in nh2_cands:
        if g not in result:
            result.append(g)
            print(f'  [Mutagenicity] 图2(NH2)找到, score={sc:.2f}')
            debug_graph('Mutagenicity', g, '图2-NH2')
            break
    if len(result) < 2:
        print('  [Mutagenicity] !! 图2 NH2 未找到，放宽: 允许 O≤1')
        for g in graphs:
            if g in result: continue
            if not (hasattr(g,'y') and g.y.item()==1): continue
            if not (8 <= g.num_nodes <= 40): continue
            if not _is_connected(g): continue
            labels = get_node_labels(g, 'Mutagenicity')
            adj = _adj_dict(g)
            if _no2_count(labels, adj) > 0: continue
            if _nh2_count(labels, adj) >= 1 and _count_atoms(labels, {'O'}) <= 1:
                result.append(g)
                debug_graph('Mutagenicity', g, '图2-NH2(放宽)')
                break
    return result[:n]


# ═══════════════════════════════════════════════════════════════════════════
#  图筛选 — BACE
# ═══════════════════════════════════════════════════════════════════════════

def select_bace_graphs(graphs, n=2):
    def prep(g):
        if not (hasattr(g,'y') and g.y.item()==1): return None
        if not (15 <= g.num_nodes <= 35): return None
        if not _is_connected(g): return None
        labels = get_node_labels(g, 'BACE')
        adj = _adj_dict(g)
        return labels, adj

    db_cands, cp_cands = [], []
    for g in graphs:
        r = prep(g)
        if r is None: continue
        labels, adj = r
        n_cnt  = _count_atoms(labels, {'N'})
        o_cnt  = _count_atoms(labels, {'O'})
        hal    = _count_atoms(labels, {'F','Cl','Br','I'})
        n_cyc, diam = _topo_stats(g)
        if (n_cnt + o_cnt) >= 1 and hal >= 1:
            s = (n_cnt+o_cnt)*2 + hal*2 + diam*0.5 - abs(g.num_nodes-25)*0.2
            db_cands.append((g, s))
        if n_cyc >= 2 and (n_cnt+o_cnt) >= 2:
            s = n_cyc*3 + (n_cnt+o_cnt)*2 - abs(g.num_nodes-22)*0.2
            cp_cands.append((g, s))

    db_cands.sort(key=lambda x: x[1], reverse=True)
    cp_cands.sort(key=lambda x: x[1], reverse=True)

    result = []
    if db_cands:
        result.append(db_cands[0][0])
        print(f'  [BACE] 图1(哑铃)找到, score={db_cands[0][1]:.2f}')
        debug_graph('BACE', db_cands[0][0], '图1-哑铃')
    for g, sc in cp_cands:
        if g not in result:
            result.append(g)
            print(f'  [BACE] 图2(紧凑)找到, score={sc:.2f}')
            debug_graph('BACE', g, '图2-紧凑')
            break

    if len(result) < n:
        print('  [BACE] !! 兜底补充')
        for g in graphs:
            if not (hasattr(g,'y') and g.y.item()==1): continue
            if not (10 <= g.num_nodes <= 45): continue
            if not _is_connected(g): continue
            if g in result: continue
            labels = get_node_labels(g, 'BACE')
            sc = _count_atoms(labels,{'N'})*2 + _count_atoms(labels,{'O'}) \
                 + _count_atoms(labels,{'F','Cl','Br','I'})
            result.append(g)
            debug_graph('BACE', g, '兜底')
            if len(result) >= n: break
    return result[:n]


# ═══════════════════════════════════════════════════════════════════════════
#  图筛选 — BBBP
# ═══════════════════════════════════════════════════════════════════════════

def select_bbbp_graphs(graphs, n=2):
    def prep(g):
        if not (hasattr(g,'y') and g.y.item()==1): return None
        if not (10 <= g.num_nodes <= 25): return None
        if not _is_connected(g): return None
        return get_node_labels(g, 'BBBP')

    lipo_cands, halo_cands = [], []
    for g in graphs:
        labels = prep(g)
        if labels is None: continue
        cf  = _c_fraction(labels)
        hal = _count_atoms(labels, {'F','Cl','Br','I'})
        n_c = _count_atoms(labels, {'N'})
        o_c = _count_atoms(labels, {'O'})
        cyc = _cycle_count(g)
        if cf >= 0.70 and n_c <= 1 and o_c <= 1:
            s = cf*10 + cyc*2 - abs(g.num_nodes-16)*0.2
            lipo_cands.append((g, s))
        if hal >= 2 and _has_ring(g) and n_c <= 1 and o_c <= 1:
            s = hal*4 + cf*3 - abs(g.num_nodes-16)*0.2
            halo_cands.append((g, s))

    lipo_cands.sort(key=lambda x: x[1], reverse=True)
    halo_cands.sort(key=lambda x: x[1], reverse=True)

    result = []
    if lipo_cands:
        result.append(lipo_cands[0][0])
        print(f'  [BBBP] 图1(亲脂)找到, score={lipo_cands[0][1]:.2f}')
        debug_graph('BBBP', lipo_cands[0][0], '图1-亲脂')
    else:
        print('  [BBBP] !! 图1 未找到, 放宽 C_frac≥0.65')
        for g in graphs:
            if not (hasattr(g,'y') and g.y.item()==1): continue
            if not (8 <= g.num_nodes <= 30): continue
            if not _is_connected(g): continue
            if g in result: continue
            labels = get_node_labels(g, 'BBBP')
            cf = _c_fraction(labels)
            if cf >= 0.65 and _count_atoms(labels,{'N'})<=2 and _count_atoms(labels,{'O'})<=2:
                result.append(g)
                debug_graph('BBBP', g, '图1-亲脂(放宽)')
                break

    for g, sc in halo_cands:
        if g not in result:
            result.append(g)
            print(f'  [BBBP] 图2(多卤素)找到, score={sc:.2f}')
            debug_graph('BBBP', g, '图2-卤素')
            break
    if len(result) < 2:
        print('  [BBBP] !! 图2 未找到, 放宽卤素≥1')
        for g in graphs:
            if not (hasattr(g,'y') and g.y.item()==1): continue
            if not (8 <= g.num_nodes <= 30): continue
            if not _is_connected(g): continue
            if g in result: continue
            labels = get_node_labels(g, 'BBBP')
            if _count_atoms(labels, {'F','Cl','Br','I'}) >= 1 and _has_ring(g):
                result.append(g)
                debug_graph('BBBP', g, '图2-卤素(放宽)')
                break
    return result[:n]


def smart_select(ds_name, graphs, n=2, min_nodes=8, max_nodes=50):
    graphs = [g for g in graphs if min_nodes <= g.num_nodes <= max_nodes]
    print(f'\n[{ds_name}] 候选图数: {len(graphs)}，开始筛选...')
    if ds_name == 'Mutagenicity':
        return select_mutagenicity_graphs(graphs, n)
    elif ds_name == 'BACE':
        return select_bace_graphs(graphs, n)
    elif ds_name == 'BBBP':
        return select_bbbp_graphs(graphs, n)
    else:
        rng = np.random.RandomState(42)
        idx = list(range(len(graphs)))
        rng.shuffle(idx)
        return [graphs[i] for i in idx[:n]]


# ═══════════════════════════════════════════════════════════════════════════
#  Model loader
# ═══════════════════════════════════════════════════════════════════════════

def load_model(dataset, model_name, dim_node, num_classes, model_level='graph'):
    ckpt      = osp.join(CKPT_PATH, dataset, model_name + '_seed0.pkl')
    model_cls = PolyGIN_3l if model_name == 'PolyGIN_3l' else GIN_3l
    model = model_cls(
        model_level=model_level,
        dim_node=dim_node,
        dim_hidden=DIM_HIDDEN,
        num_classes=num_classes,
    ).to(DEVICE)
    raw = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(compatible_state_dict(raw))
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Explainer factory
# ═══════════════════════════════════════════════════════════════════════════

def build_explainer(name, model, dataset, explain_graph, train_data=None):
    if name == 'IntegratedGradients':
        return IntegratedGradients(model, explain_graph=explain_graph, m_steps=50)
    elif name == 'FlowX':
        return FlowX(model, explain_graph=explain_graph)
    elif name == 'GradCAM':
        return GradCAM(model, explain_graph=explain_graph)
    elif name == 'PolyGINExplainer':
        return PolyGINExplainer(model, explain_graph=explain_graph)
    else:
        raise ValueError(f'Unknown explainer: {name}')


# ═══════════════════════════════════════════════════════════════════════════
#  ★ 统一对比度归一化（保留 v4 已验证的幂变换管线）
# ═══════════════════════════════════════════════════════════════════════════

def _normalise_contrast(arr, power=2.5):
    """
    v4 _normalise_shapley 的升级版:
      - arr: 原始重要性分数（可以有正有负）
      - 先取绝对值得到「贡献幅度」
      - 用 90th 百分位作天花板（vs 旧 95th，稍保守）
      - 幂变换增强对比度
      - 所有方法共用此归一化，确保公平
    """
    if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
        return np.full(max(1, len(arr)), 0.5)
    arr = np.asarray(arr, dtype=float)
    # 贡献幅度（不分正负）
    mag = np.abs(arr)
    nonzero = mag[mag > 1e-8]
    if nonzero.size > 0:
        hi = np.percentile(nonzero, 90)
        if hi < 1e-8:
            hi = nonzero.max()
    else:
        hi = mag.max()
    if hi < 1e-8:
        return np.zeros_like(arr, dtype=float)
    # 除以天花板 → [0, ≥1] → clip → [0, 1] → power 变换增强对比度
    normalized = np.clip(mag / hi, 0, 1)
    return normalized ** power


# ═══════════════════════════════════════════════════════════════════════════
#  统一节点分数提取
# ═══════════════════════════════════════════════════════════════════════════

def _pred_class(model_or_explainer, graph, node_idx=0):
    model = getattr(model_or_explainer, 'model', model_or_explainer)
    with torch.no_grad():
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)
        out   = model(graph.x, graph.edge_index, batch=batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        pred   = logits.argmax(-1)
        return (pred[node_idx].item() if pred.ndim > 0 else pred.item())


def get_unified_node_importance(method, explainer, graph, num_classes, node_idx=0):
    """
    统一提取节点重要性分数（值域 [0, 1]，越大越重要）。

    PolyGINExplainer: 调用 explainpoly 获得 last_node_scores（Sh_exact.sum(dim=1)，
    有符号），取绝对值 → 对比度归一化。

    其他方法: 调用 explainer → 边掩码 → 节点分数 → 对比度归一化。
    """
    max_nodes = max(1, int(graph.num_nodes * (1 - SPARSITY)))

    if method == 'PolyGINExplainer':
        # 调用 explainpoly.py（处理 reweight + 已验证的 Shapley 计算）
        _ = explainer(graph.x, graph.edge_index, sparsity=0,
                       num_classes=num_classes, node_idx=node_idx,
                       max_nodes=max_nodes)
        pred_cls = _pred_class(explainer, graph, node_idx)

        if (hasattr(explainer, 'last_node_scores')
                and explainer.last_node_scores is not None
                and pred_cls < len(explainer.last_node_scores)):
            nm = explainer.last_node_scores[pred_cls]
            if isinstance(nm, torch.Tensor):
                nm = nm.detach().cpu().numpy()
            if nm.ndim == 2:
                nm = nm[:, pred_cls] if nm.shape[1] > pred_cls else nm[:, -1]
        else:
            nm = np.zeros(graph.num_nodes)
        return _normalise_contrast(nm)

    elif method == 'IntegratedGradients':
        model = getattr(explainer, 'model', explainer)
        model.eval()
        x          = graph.x.float().to(DEVICE)
        edge_index = graph.edge_index.to(DEVICE)
        batch      = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)
        baseline   = torch.zeros_like(x)
        m_steps    = getattr(explainer, 'm_steps', 50)
        grads_accum = torch.zeros_like(x)
        for alpha in np.linspace(0, 1, m_steps + 1):
            x_interp = (baseline + alpha*(x-baseline)).detach().requires_grad_(True)
            out = model(x_interp, edge_index, batch=batch)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            with torch.no_grad():
                pcls = logits.argmax(-1)
                cls_idx = pcls[node_idx].item() if pcls.ndim > 0 else pcls.item()
            logits[:, cls_idx].sum().backward()
            if x_interp.grad is not None:
                grads_accum += x_interp.grad.detach()
        ig = (grads_accum / (m_steps+1)) * (x - baseline)
        scores = ig.norm(dim=1).cpu().numpy()
        return _normalise_contrast(scores)

    else:
        # FlowX / GradCAM
        result = explainer(graph.x, graph.edge_index, sparsity=0,
                            num_classes=num_classes, node_idx=node_idx,
                            max_nodes=max_nodes)
        pred_cls = _pred_class(explainer, graph, node_idx)

        masks = result[0] if isinstance(result, tuple) else result
        if isinstance(masks, (list, tuple)):
            em = masks[pred_cls]
        elif isinstance(masks, dict):
            em = masks[pred_cls]
        else:
            em = masks
        if isinstance(em, torch.Tensor):
            em = em.detach().cpu().numpy()

        # 边 → 节点（取邻接边最大分数）
        node_imp = np.zeros(graph.num_nodes, dtype=float)
        ei = graph.edge_index.cpu().numpy()
        for e_idx in range(ei.shape[1]):
            u, v = int(ei[0, e_idx]), int(ei[1, e_idx])
            s = float(em[e_idx]) if e_idx < len(em) else 0.0
            if s > node_imp[u]:
                node_imp[u] = s
            if s > node_imp[v]:
                node_imp[v] = s
        return _normalise_contrast(node_imp)


# ═══════════════════════════════════════════════════════════════════════════
#  统一可视化核心函数
# ═══════════════════════════════════════════════════════════════════════════

def compute_graph_layout(graph, seed=0):
    G = to_networkx(graph, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    try:
        pos = nx.kamada_kawai_layout(G, scale=2.0)
    except Exception:
        pos = nx.spring_layout(G, seed=seed, k=1.5, scale=2.0)
    return pos


def draw_importance_graph(ax, graph, pos, node_scores,
                           node_labels=None, target_node=None):
    G = to_networkx(graph, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    n = graph.num_nodes

    node_colors = []
    for i in range(n):
        s = float(node_scores[i]) if i < len(node_scores) else 0.5
        node_colors.append(CMAP_IMPORTANCE(0.08 + s * 0.92))

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=180,
                           linewidths=0.6,
                           edgecolors='#555555',
                           margins=0.15)

    if target_node is not None and target_node < n:
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               nodelist=[target_node],
                               node_color=[node_colors[target_node]],
                               node_size=180,
                               linewidths=3.0,
                               edgecolors='#CC0000')

    if node_labels:
        for i in range(n):
            if i not in G.nodes:
                continue
            lbl = node_labels[i] if i < len(node_labels) else '?'
            s   = float(node_scores[i]) if i < len(node_scores) else 0.5
            txt_color = 'white' if s > 0.55 else '#222222'
            xi, yi = pos[i]
            ax.text(xi, yi, lbl, ha='center', va='center',
                    fontsize=5.5, fontweight='bold', color=txt_color, zorder=5)

    edge_list = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, ax=ax,
                           width=0.6, edge_color='#AAAAAA',
                           alpha=0.5, arrows=False)
    ax.set_axis_off()


def draw_mol_importance(ax, graph, pos, node_scores, node_labels):
    G = to_networkx(graph, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    n = graph.num_nodes

    node_colors = []
    for i in range(n):
        s = float(node_scores[i]) if i < len(node_scores) else 0.5
        node_colors.append(CMAP_IMPORTANCE(0.08 + s * 0.92))

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=200,
                           linewidths=0.5,
                           edgecolors='#444444')

    for i in range(n):
        if i not in G.nodes:
            continue
        lbl = node_labels[i] if (node_labels and i < len(node_labels)) else '?'
        s   = float(node_scores[i]) if i < len(node_scores) else 0.5
        txt_color = 'white' if s > 0.55 else '#222222'
        xi, yi = pos[i]
        ax.text(xi, yi, lbl, ha='center', va='center',
                fontsize=5.0, fontweight='bold', color=txt_color, zorder=5)

    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), ax=ax,
                           width=0.7, edge_color='#888888', alpha=0.6, arrows=False)
    ax.set_axis_off()
    ax.margins(0.12)


# ═══════════════════════════════════════════════════════════════════════════
#  Helper pickers
# ═══════════════════════════════════════════════════════════════════════════

def pick_graphs(dataset_graphs, n=3, min_nodes=5, max_nodes=30, seed=0):
    rng = np.random.RandomState(seed)
    candidates = [g for g in dataset_graphs if min_nodes<=g.num_nodes<=max_nodes]
    rng.shuffle(candidates)
    return candidates[:n]

def pick_nodes(graph, model, num_classes, n=4, seed=0):
    rng = np.random.RandomState(seed)
    graph = graph.to(DEVICE)
    with torch.no_grad():
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)
        out   = model(graph.x, graph.edge_index, batch=batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        preds  = logits.argmax(dim=-1).cpu()
    cands = (graph.test_mask.nonzero(as_tuple=True)[0].tolist()
             if hasattr(graph,'test_mask') and graph.test_mask is not None
             else list(range(graph.num_nodes)))
    correct = [i for i in cands if preds[i].item()==graph.y[i].item()]
    rng.shuffle(correct)
    return correct[:n]


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 : 分子数据集 (BBBP / BACE / Mutagenicity)
# ═══════════════════════════════════════════════════════════════════════════

def make_fig_mol():
    DATASETS = ['BBBP', 'BACE', 'Mutagenicity']
    N_GRAPHS = 2
    METHODS  = EXPLAINER_NAMES

    dataset_info, models_gin, models_poly = {}, {}, {}
    for ds in DATASETS:
        data, num_nodes, dim_node, num_classes = load_dataset(DATA_PATH, ds)
        dataset_info[ds] = (data, num_nodes, dim_node, num_classes)
        models_gin[ds]   = load_model(ds, 'GIN_3l',     dim_node, num_classes)
        models_poly[ds]  = load_model(ds, 'PolyGIN_3l', dim_node, num_classes)

    selected = {}
    for ds in DATASETS:
        data, _, _, _ = dataset_info[ds]
        all_test = list(data['test'])
        g0 = all_test[0]
        enc = _detect_encoding(g0)
        lbl0 = get_node_labels(g0, ds)
        print(f'\n=== [{ds}] 特征编码={enc}, 前5原子={lbl0[:5]}, '
              f'x[0,0]={g0.x[0,0].item():.1f}, x.shape={g0.x.shape} ===')
        chosen = smart_select(ds, all_test, n=N_GRAPHS)
        selected[ds] = chosen

    n_rows = N_GRAPHS * len(DATASETS)
    n_cols = len(METHODS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*2.4, n_rows*2.4),
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.08})

    for col, name in enumerate(METHODS):
        axes[0, col].set_title(name, fontsize=9, fontweight='bold', pad=8)

    row_labels = [f'{ds}\n#{gi+1}' for ds in DATASETS for gi in range(N_GRAPHS)]
    for row in range(n_rows):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=8,
                                rotation=0, labelpad=45, va='center',
                                fontweight='bold')

    for di, ds in enumerate(DATASETS):
        data, _, _, num_classes = dataset_info[ds]
        train_data = data.get('train', data)
        for gi, graph in enumerate(selected[ds]):
            row = di * N_GRAPHS + gi
            graph = graph.to(DEVICE)
            node_labels = get_node_labels(graph, ds)

            pos = compute_graph_layout(graph)

            for col, method in enumerate(METHODS):
                ax = axes[row, col]
                model = models_poly[ds] if method == 'PolyGINExplainer' else models_gin[ds]
                try:
                    exp = build_explainer(method, model, ds,
                                          explain_graph=True, train_data=train_data)
                    node_scores = get_unified_node_importance(
                        method, exp, graph, num_classes, node_idx=0)
                    draw_mol_importance(ax, graph, pos, node_scores, node_labels)
                except Exception as e:
                    import traceback
                    ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=5, color='red')
                    ax.set_axis_off()
                    print(f'[{ds}][{method}] Error: {e}')
                    traceback.print_exc()

    for di in range(1, len(DATASETS)):
        sep = di * N_GRAPHS
        for c in range(n_cols):
            axes[sep-1, c].spines['bottom'].set_visible(True)
            axes[sep-1, c].spines['bottom'].set_linewidth(1.5)
            axes[sep-1, c].spines['bottom'].set_color('#666666')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap=CMAP_IMPORTANCE, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Node importance', fontsize=9)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['Low', '', 'High'])
    cb.ax.tick_params(labelsize=7)

    fig.savefig('fig_mol.pdf', bbox_inches='tight', dpi=300)
    print('\nSaved fig_mol.pdf')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 : Graph-SST2
# ═══════════════════════════════════════════════════════════════════════════

def make_fig_sst2():
    DS       = 'Graph-SST2'
    N_GRAPHS = 4
    METHODS  = EXPLAINER_NAMES

    data, _, dim_node, num_classes = load_dataset(DATA_PATH, DS)
    model_gin  = load_model(DS, 'GIN_3l',     dim_node, num_classes)
    model_poly = load_model(DS, 'PolyGIN_3l', dim_node, num_classes)
    graphs = pick_graphs(list(data['test']), n=N_GRAPHS, min_nodes=4, max_nodes=25)

    vocab = {}
    vocab_path = osp.join(DATA_PATH, DS, 'vocab.pt')
    if osp.exists(vocab_path):
        raw = torch.load(vocab_path, map_location='cpu')
        if isinstance(raw, dict):
            if raw and isinstance(next(iter(raw)), str):
                vocab = {v: k for k, v in raw.items()}
            else:
                vocab = raw

    n_rows, n_cols = len(METHODS), N_GRAPHS
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*2.8, n_rows*2.5),
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.06})

    for col in range(n_cols):
        axes[0, col].set_title(f'Sentence #{col+1}', fontsize=9, fontweight='bold', pad=8)
    for row, method in enumerate(METHODS):
        axes[row, 0].set_ylabel(method, fontsize=8,
                                rotation=0, labelpad=55, va='center',
                                fontweight='bold')

    for row, method in enumerate(METHODS):
        model = model_poly if method == 'PolyGINExplainer' else model_gin
        for col, graph in enumerate(graphs):
            ax = axes[row, col]
            graph = graph.to(DEVICE)
            node_labels = sst2_node_labels(graph, vocab)

            try:
                exp = build_explainer(method, model, DS, explain_graph=True,
                                      train_data=data.get('train', data))
                pos = compute_graph_layout(graph)
                node_scores = get_unified_node_importance(
                    method, exp, graph, num_classes, node_idx=0)
                draw_importance_graph(ax, graph, pos, node_scores,
                                      node_labels=node_labels)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=5, color='red')
                ax.set_axis_off()

    fig.savefig('fig_sst2.pdf', bbox_inches='tight', dpi=300)
    print('Saved fig_sst2.pdf')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 : BA-shapes
# ═══════════════════════════════════════════════════════════════════════════

def make_fig_bashapes():
    DS       = 'BA_shapes'
    N_NODES  = 4
    METHODS  = EXPLAINER_NAMES

    data, _, dim_node, num_classes = load_dataset(DATA_PATH, DS)
    model_gin  = load_model(DS, 'GIN_3l',     dim_node, num_classes, model_level='node')
    model_poly = load_model(DS, 'PolyGIN_3l', dim_node, num_classes, model_level='node')

    if isinstance(data, dict):
        from torch_geometric.data import Batch
        graph_full = Batch.from_data_list(list(data['test'])).to(DEVICE)
    else:
        graph_full = data.to(DEVICE)

    target_nodes = pick_nodes(graph_full, model_gin, num_classes, n=N_NODES)

    n_rows, n_cols = len(METHODS), len(target_nodes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*2.8, n_rows*2.5),
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.06})
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, nidx in enumerate(target_nodes):
        cls = graph_full.y[nidx].item()
        axes[0, col].set_title(f'Node {nidx} (cls={cls})',
                               fontsize=9, fontweight='bold', pad=8)
    for row, method in enumerate(METHODS):
        axes[row, 0].set_ylabel(method, fontsize=8,
                                rotation=0, labelpad=55, va='center',
                                fontweight='bold')

    for row, method in enumerate(METHODS):
        model = model_poly if method == 'PolyGINExplainer' else model_gin
        for col, node_idx in enumerate(target_nodes):
            ax = axes[row, col]
            try:
                subset, edge_index_sub, mapping, _ = k_hop_subgraph(
                    node_idx, num_hops=3, edge_index=graph_full.edge_index,
                    relabel_nodes=True, num_nodes=graph_full.num_nodes)
                sub_graph = Data(x=graph_full.x[subset],
                                 edge_index=edge_index_sub,
                                 y=graph_full.y[subset],
                                 num_nodes=len(subset)).to(DEVICE)
                local_idx = (mapping.item() if mapping.ndim == 0
                             else mapping[0].item())

                exp = build_explainer(method, model, DS, explain_graph=False,
                                      train_data=data)
                pos = compute_graph_layout(sub_graph)
                node_scores = get_unified_node_importance(
                    method, exp, sub_graph, num_classes, node_idx=local_idx)
                draw_importance_graph(ax, sub_graph, pos, node_scores,
                                      target_node=local_idx)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=5, color='red')
                ax.set_axis_off()

    legend_handles = [
        mpatches.Patch(color='#CC0000', label='Target node'),
        plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=CMAP_IMPORTANCE(0.5), markersize=8,
                    label='Important'),
        plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=CMAP_IMPORTANCE(0.08), markersize=8,
                    label='Less important'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.savefig('fig_bashapes.pdf', bbox_inches='tight', dpi=300)
    print('Saved fig_bashapes.pdf')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    set_seed(0)
    print('=== Figure 1: Molecular datasets ===')
    make_fig_mol()
    print('=== Figure 2: Graph-SST2 ===')
    make_fig_sst2()
    print('=== Figure 3: BA-shapes ===')
    make_fig_bashapes()
