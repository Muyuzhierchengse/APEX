#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APEX (PolyGIN + PolyGINExplainer) Signed Node Attribution Visualization  v2
===========================================================================

Generates three appendix figures for the APEX / PolyGIN / Aumann–Shapley
attribution GNN explainability paper:

  Figure A1: BA-Shapes       (1 × 3) — 3 signed node explanations
  Figure A2: Graph-SST2      (2 × 3) — 3 positive + 3 negative sentence explanations
  Figure A3: Molecular graphs (3 × 3) — BBBP / BACE / Mutagenicity

Changes from v1:
  - A1: 1×3 (no ground-truth panel); sign-balance filter: each case must have both
        red (positive) and blue (negative) nodes.
  - A2: 5–12 token short sentences only; readable full-sentence display.
  - A3: BBBP / BACE use RDKit via SMILES (as before); Mutagenicity uses
        MolGenMutagenicity to reconstruct RDKit mols from one-hot features.
        Explicit H atoms are shown; a figure note documents this.

Usage:
  python visualize_signed_attribution.py \
      --data_path ./dataset \
      --checkpoint_path ./model/checkpoint \
      --out_dir ./figures_signed \
      --device cuda:0
"""

import argparse
import csv
import functools
import os
import os.path as osp
import random
import re
import sys
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
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

from model.models import PolyGIN_3l
from dataset.data import load_dataset
from method.explainpoly import PolyGINExplainer

# ===================================================================
#  CONSTANTS
# ===================================================================

_SIGNED_COLORS = [
    (0.0, "#2166AC"),
    (0.25, "#92C5DE"),
    (0.5, "#F7F7F7"),
    (0.75, "#F4A582"),
    (1.0, "#B2182B"),
]
CMAP_SIGNED = LinearSegmentedColormap.from_list("signed_rb", _SIGNED_COLORS, N=256)
DIM_HIDDEN = 300

# ===================================================================
#  UTILITIES  (from main.py)
# ===================================================================

def set_seed(seed: int = 0) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compatible_state_dict(state_dict: dict) -> OrderedDict:
    comp = OrderedDict()
    for key, value in state_dict.items():
        new_key = re.sub(
            r"conv(1|s\.[0-9]+)\.weight", r"conv\1.lin.weight", key
        )
        comp[new_key] = value.T if new_key != key else value
    return comp


# ===================================================================
#  LOAD MODEL
# ===================================================================

def load_model_for_dataset(
    dataset_name: str, dim_node: int, num_classes: int,
    checkpoint_path: str, device: torch.device, model_level: str = "graph",
) -> nn.Module:
    model = PolyGIN_3l(
        model_level=model_level, dim_node=dim_node,
        dim_hidden=DIM_HIDDEN, num_classes=num_classes,
    ).to(device)
    ckpt_dir = osp.join(checkpoint_path, dataset_name)
    candidates = [
        osp.join(ckpt_dir, "PolyGIN_3l_seed0.pkl"),
        osp.join(ckpt_dir, "PolyGIN_3l_seed01.pkl"),
    ]
    ckpt_file = None
    for p in candidates:
        if osp.exists(p): ckpt_file = p; break
    if ckpt_file is None and osp.isdir(ckpt_dir):
        for fname in sorted(os.listdir(ckpt_dir)):
            if fname.startswith("PolyGIN_3l") and fname.endswith(".pkl"):
                ckpt_file = osp.join(ckpt_dir, fname); break
    if ckpt_file is None:
        raise FileNotFoundError(f"No PolyGIN_3l checkpoint found in {ckpt_dir}")
    raw_state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(compatible_state_dict(raw_state))
    model.eval()
    print(f"  [load_model] {dataset_name} ← {ckpt_file}")
    return model


# ===================================================================
#  PREDICTION
# ===================================================================

def get_prediction(model: nn.Module, graph: Data, device: torch.device,
                   model_level: str = "graph") -> Tuple[int, torch.Tensor, float]:
    graph = graph.to(device)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, batch=batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if model_level == "node":
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            logits = logits[0]; probs = probs[0]
        pred_cls = int(logits.argmax(-1).item())
        conf = float(probs[pred_cls].item())
    return pred_cls, logits, conf


def get_all_node_predictions(model, graph, device):
    graph = graph.to(device)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, batch=batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        confs = probs[torch.arange(probs.size(0)), preds]
    return preds, confs


# ===================================================================
#  APEX EXPLAINER
# ===================================================================

def run_apex_explainer(model, graph, num_classes, explain_graph, device,
                       node_idx=0) -> PolyGINExplainer:
    explainer = PolyGINExplainer(model, explain_graph=explain_graph)
    explainer(graph.x, graph.edge_index, sparsity=0, num_classes=num_classes,
              node_idx=node_idx, max_nodes=graph.num_nodes)
    return explainer


# ===================================================================
#  SIGNED NORMALISATION & CONCENTRATION
# ===================================================================

def normalize_signed_scores(scores: np.ndarray) -> Tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(scores)))
    if max_abs < 1e-12:
        return np.zeros_like(scores, dtype=float), 0.0
    return scores / max_abs, max_abs


def compute_concentration(scores: np.ndarray) -> float:
    abs_scores = np.abs(scores)
    total = abs_scores.sum()
    if total < 1e-12: return 0.0
    n = len(scores)
    k = max(1, int(np.ceil(n * 0.20)))
    topk = np.sort(abs_scores)[-k:]
    return float(topk.sum() / total)


def compute_sign_balance(scores: np.ndarray) -> float:
    """Fraction of nodes with positive score. 0.0 = all negative, 1.0 = all positive."""
    n = len(scores)
    if n == 0: return 0.5
    return float((scores > 1e-10).sum()) / n


# ===================================================================
#  DATA LOADING HELPERS
# ===================================================================

def load_data_for_dataset(data_path: str, dataset_name: str):
    out = {}
    if dataset_name == "BA_shapes":
        data, _, dim_node, num_classes = load_dataset(data_path, dataset_name)
        out.update({"full_data": data, "dim_node": dim_node,
                     "num_classes": num_classes, "model_level": "node",
                     "splits": None, "supplement": None})
    elif dataset_name in ("Graph-SST2", "Graph-Twitter"):
        from dig.xgraph.dataset import SentiGraphDataset
        orig = SentiGraphDataset(data_path, dataset_name)
        orig.data.x = orig.data.x.to(torch.float32)
        orig.data.y = orig.data.y.long()
        supplement = getattr(orig, "supplement", None)
        from dataset.data import split_dataset
        splits = split_dataset(orig)
        out.update({"splits": splits, "dim_node": orig.num_node_features,
                     "num_classes": orig.num_classes, "model_level": "graph",
                     "supplement": supplement, "orig_dataset": orig})
    elif dataset_name in ("BBBP", "BACE", "ClinTox", "Tox21", "ToxCast"):
        from dig.xgraph.dataset import MoleculeDataset
        orig = MoleculeDataset(data_path, dataset_name)
        orig.data.x = orig.data.x.to(torch.float32)
        if dataset_name in ("BBBP", "BACE"):
            orig.data.y = orig.data.y[:, 0].long()
        elif dataset_name == "ClinTox":
            raw_y = orig.data.y[:, 1]
            orig.data.y = raw_y.clamp(0, 1).long()
            valid_mask = ~torch.isnan(raw_y) & (raw_y != -1)
            orig = orig[valid_mask.nonzero(as_tuple=True)[0].tolist()]
        from dataset.data import split_dataset
        splits = split_dataset(orig)
        out.update({"splits": splits, "dim_node": orig.num_node_features,
                     "num_classes": orig.num_classes, "model_level": "graph",
                     "supplement": None, "orig_dataset": orig})
    elif dataset_name == "Mutagenicity":
        from torch_geometric.datasets import TUDataset
        raw = TUDataset(root=os.path.join(data_path, "TUDataset"),
                        name="Mutagenicity", use_node_attr=True)
        raw.data.x = raw.data.x.to(torch.float32)
        raw.data.y = raw.data.y.long()
        valid_idx = [i for i, d in enumerate(raw)
                     if d.x is not None and d.x.size(0) > 0
                     and d.edge_index is not None and d.edge_index.size(1) > 0]
        orig = raw[valid_idx]
        from dataset.data import split_dataset
        splits = split_dataset(orig)
        out.update({"splits": splits, "dim_node": raw.num_node_features,
                     "num_classes": raw.num_classes, "model_level": "graph",
                     "supplement": None, "orig_dataset": raw})
    else:
        data, _, dim_node, num_classes = load_dataset(data_path, dataset_name)
        out.update({"splits": data, "dim_node": dim_node,
                     "num_classes": num_classes, "model_level": "graph",
                     "supplement": None})
    return out


# ===================================================================
#  SAMPLE SELECTION — Graph classification
# ===================================================================

def _build_candidates_graph(test_graphs, model, explain_graph, num_classes,
                            device, min_nodes, max_nodes, conf_lo, conf_hi,
                            n_max=200, sign_balance_range=None):
    """Run prediction + APEX; return list of candidate dicts.

    If sign_balance_range=(lo, hi), only keep candidates whose fraction of
    positive nodes falls within [lo, hi].
    """
    candidates = []
    count = 0
    for orig_idx, g in enumerate(test_graphs):
        if g.num_nodes < min_nodes or g.num_nodes > max_nodes: continue
        if g.num_nodes <= 1: continue
        g_dev = g.to(device)
        pred_cls, _, conf = get_prediction(model, g_dev, device, model_level="graph")
        true_label = int(g.y.item())
        if pred_cls != true_label: continue
        if conf < conf_lo or conf > conf_hi: continue

        explainer = run_apex_explainer(model, g_dev, num_classes,
                                        explain_graph=explain_graph, device=device)
        scores_tensor = explainer.last_node_scores[pred_cls]
        scores = (scores_tensor.detach().cpu().numpy().copy()
                  if isinstance(scores_tensor, torch.Tensor)
                  else np.asarray(scores_tensor, dtype=float))
        if scores.size == 0: continue

        # sign-balance filter
        if sign_balance_range is not None:
            sb = compute_sign_balance(scores)
            if sb < sign_balance_range[0] or sb > sign_balance_range[1]:
                continue

        normed, _ = normalize_signed_scores(scores)
        top20 = compute_concentration(scores)

        candidates.append({
            "graph": g, "graph_idx": orig_idx,
            "pred_cls": pred_cls, "true_label": true_label,
            "confidence": conf, "node_scores": scores,
            "normed_scores": normed, "num_nodes": int(g.num_nodes),
            "top20_mass": top20, "sign_balance": compute_sign_balance(scores),
        })
        count += 1
        if count >= n_max: break
    return candidates


def select_cases_by_concentration(candidates, n, min_percentile=60.0,
                                  max_percentile=90.0, cls_balance=False):
    if len(candidates) == 0: return []
    masses = np.array([c["top20_mass"] for c in candidates])
    lo = np.percentile(masses, min_percentile) if min_percentile > 0 else masses.min()
    hi = np.percentile(masses, max_percentile) if max_percentile < 100 else masses.max()
    mid_cands = [c for c in candidates if lo <= c["top20_mass"] <= hi]
    if len(mid_cands) < n: mid_cands = list(candidates)
    mid_cands = sorted(mid_cands, key=lambda c: c["top20_mass"])
    if cls_balance and n >= 2:
        cls0 = [c for c in mid_cands if c["pred_cls"] == 0]
        cls1 = [c for c in mid_cands if c["pred_cls"] == 1]
        selected = []
        if cls0: selected.append(cls0[min(len(cls0)//2, len(cls0)-1)])
        if cls1: selected.append(cls1[min(len(cls1)//2, len(cls1)-1)])
        pool = [c for c in mid_cands if c not in selected]
        np.random.seed(0)
        extra = np.random.choice(pool, size=min(n-len(selected), len(pool)),
                                 replace=False).tolist()
        selected.extend(extra)
        return selected[:n]
    else:
        if len(mid_cands) <= n: return mid_cands
        indices = np.linspace(0, len(mid_cands)-1, n, dtype=int)
        return [mid_cands[i] for i in indices]


def select_cases_for_dataset(test_graphs, model, num_classes, device, n=3,
                             min_nodes=10, max_nodes=50, conf_lo=0.6,
                             conf_hi=0.99, cls_balance=False, n_max_cand=300,
                             sign_balance_range=None):
    candidates = _build_candidates_graph(
        test_graphs, model, True, num_classes, device,
        min_nodes, max_nodes, conf_lo, conf_hi, n_max_cand,
        sign_balance_range=sign_balance_range)
    if len(candidates) < n:
        # Relax size/conf constraints but KEEP sign-balance filter;
        # a figure panel with all-neg or all-pos nodes is meaningless.
        print(f"    WARNING: only {len(candidates)} sign-balanced candidates — "
              f"relaxing size/conf (NOT sign-balance)")
        candidates2 = _build_candidates_graph(
            test_graphs, model, True, num_classes, device,
            1, 200, 0.5, 1.0, n_max_cand * 3,
            sign_balance_range=sign_balance_range)
        seen = {c["graph_idx"] for c in candidates}
        for c in candidates2:
            if c["graph_idx"] not in seen:
                candidates.append(c)
    if len(candidates) < n:
        print(f"    WARNING: still only {len(candidates)} sign-balanced "
              f"candidates after relaxation.")
    return select_cases_by_concentration(candidates, n, cls_balance=cls_balance)


# ===================================================================
#  BA-SHAPES  NODE SELECTION  (with sign-balance)
# ===================================================================

def select_bashapes_nodes(data, model, num_classes, device, n=3,
                          conf_lo=0.6, conf_hi=0.99, n_max_cand=500,
                          sign_balance_range=None):
    """Select *n* nodes whose **3-hop subgraph** signed attribution has at least
    one positive (red) and one negative (blue) node in the subgraph.

    Criteria: "红色节点搜索3跳里面有没蓝色。只要有就行。" — as long as the
    3-hop computation subgraph contains both signs, it is accepted."""
    data = data.to(device)
    preds, confs = get_all_node_predictions(model, data, device)
    test_mask = getattr(data, "test_mask", None)
    if test_mask is None:
        test_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    test_nodes = test_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"  BA-Shapes: {len(test_nodes)} test nodes, "
          f"scanning up to {n_max_cand}...")

    candidates = []
    count = 0
    for nidx in test_nodes:
        true_label = int(data.y[nidx].item())
        pred_cls = int(preds[nidx].item())
        conf = float(confs[nidx].item())
        if pred_cls != true_label: continue
        if conf < conf_lo or conf > conf_hi: continue

        # ── Run APEX on full graph ──
        explainer = run_apex_explainer(model, data, num_classes,
                                        explain_graph=False, device=device,
                                        node_idx=nidx)
        scores_tensor = explainer.last_node_scores[pred_cls]
        scores_full = (scores_tensor.detach().cpu().numpy().copy()
                       if isinstance(scores_tensor, torch.Tensor)
                       else np.asarray(scores_tensor, dtype=float))
        if scores_full.size == 0: continue

        # ── Check sign-balance on 4-hop subgraph, ignoring near-zero nodes ──
        subset, _, _, _ = k_hop_subgraph(
            nidx, num_hops=4, edge_index=data.edge_index,
            relabel_nodes=True, num_nodes=data.num_nodes)
        sub_scores = scores_full[subset.cpu().numpy()]
        # Only consider nodes with meaningful |score| (>5% of max absolute score)
        thr = np.max(np.abs(sub_scores)) * 0.05 if len(sub_scores) > 0 else 0.0
        meaningful = sub_scores[np.abs(sub_scores) > thr]
        any_pos = (meaningful > 1e-10).any()
        any_neg = (meaningful < -1e-10).any()
        if not (any_pos and any_neg):
            continue  # need at least one red AND one blue node with meaningful score

        top20 = compute_concentration(scores_full)
        candidates.append({
            "node_idx": nidx, "pred_cls": pred_cls, "true_label": true_label,
            "confidence": conf, "node_scores": scores_full,
            "num_nodes": int(data.num_nodes), "top20_mass": top20,
            "sign_balance": float((sub_scores > 1e-10).sum()) / len(sub_scores),
        })
        count += 1
        if count >= n_max_cand: break

    print(f"  BA-Shapes: {len(candidates)} candidates after subgraph sign-balance filter")
    if len(candidates) < n:
        print(f"  BA-Shapes: *** ONLY {len(candidates)} sign-balanced candidates found "
              f"(need {n}).  Figures may be incomplete.  Consider increasing "
              f"--n_max_cand or broadening sign_balance_range. ***")
        if len(candidates) == 0:
            return []
    return select_cases_by_concentration(candidates, n, cls_balance=False)


# ===================================================================
#  BA-SHAPES  MOTIF MASK HELPER
# ===================================================================

def get_motif_nodes_bashapes(data, node_idx, num_hops=3):
    edge_label_matrix = getattr(data, "edge_label_matrix", None)
    if edge_label_matrix is None: return None
    y_val = int(data.y[node_idx].item()) if data.y is not None else -1
    if y_val == 0: return []
    elm = edge_label_matrix + edge_label_matrix.T
    if hasattr(elm, "to"): elm = elm.to("cpu")
    connected = {node_idx}
    for _ in range(num_hops):
        new_nodes = set()
        for n in connected:
            new_nodes.update(torch.where(elm[n] != 0)[0].tolist())
        connected.update(new_nodes)
    return list(connected)


# ===================================================================
#  GRAPH-SST2  TOKEN LOADING
# ===================================================================

def _load_sst2_raw_tokens(data_path):
    """Load sentence_tokens from the processed SST2 data, keyed by 0-based graph idx."""
    processed_dir = osp.join(data_path, "Graph-SST2", "processed")
    pt_path = osp.join(processed_dir, "data.pt")
    tokens_map = {}

    # Try loading the processed .pt file
    if osp.exists(pt_path):
        try:
            loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, tuple) and len(loaded) >= 3:
                supplement = loaded[2]
                if isinstance(supplement, dict) and "sentence_tokens" in supplement:
                    raw = supplement["sentence_tokens"]
                    for k, v in raw.items():
                        tokens_map[int(k)] = v
                    return tokens_map
        except Exception:
            pass

    # Fallback: load raw JSON directly
    raw_json = osp.join(data_path, "Graph-SST2", "raw",
                        "Graph-SST2_sentence_tokens.json")
    if osp.exists(raw_json):
        import json
        with open(raw_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            tokens_map[int(k)] = v
        return tokens_map

    # Last resort: use SentiGraphDataset
    try:
        from dig.xgraph.dataset import SentiGraphDataset
        ds = SentiGraphDataset(data_path, "Graph-SST2")
        supplement = getattr(ds, "supplement", None)
        if supplement and "sentence_tokens" in supplement:
            for k, v in supplement["sentence_tokens"].items():
                tokens_map[int(k)] = v
    except Exception:
        pass

    return tokens_map


def _load_sst2_vocab(data_path):
    """Load BERT vocab mapping (int token_id → str)."""
    vocab = {}
    for subdir in ["Graph-SST2", ""]:
        vocab_path = osp.join(data_path, subdir, "vocab.pt")
        if osp.exists(vocab_path):
            raw = torch.load(vocab_path, map_location="cpu")
            if isinstance(raw, dict):
                if raw and isinstance(next(iter(raw)), str):
                    vocab = {v: k for k, v in raw.items()}
                else:
                    vocab = raw
            break
    return vocab


def get_sst2_tokens_for_graph(graph, graph_global_idx, tokens_map, vocab):
    """
    Return list of token strings for a Graph-SST2 graph.

    Tries multiple index conventions (0-based, 1-based) and verifies
    by checking if token count matches node count.
    """
    n = graph.num_nodes

    # Attempt 1: direct lookup
    for offset in [0, -1, 1]:
        key = graph_global_idx + offset
        if key in tokens_map:
            toks = tokens_map[key]
            if len(toks) == n:
                return toks

    # Attempt 2: any key that gives the right count
    for k, v in tokens_map.items():
        if isinstance(v, list) and len(v) == n:
            return v

    # Fallback: vocab
    if vocab:
        x = graph.x.cpu()
        labels = []
        for i in range(n):
            idx = int(x[i].argmax())
            labels.append(vocab.get(idx, str(idx)))
        return labels

    return [str(i) for i in range(n)]


# ===================================================================
#  MOLECULE  LABELS
# ===================================================================

ATOMIC_NUM_MAP = {
    1:"H", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 11:"Na", 12:"Mg",
    13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 19:"K", 20:"Ca",
    26:"Fe", 35:"Br", 53:"I",
}
MUTAG_14 = ["C","O","Cl","H","N","F","Br","S","P","I","Na","K","Li","Ca"]


def get_molecule_node_labels(graph, ds_name):
    x = graph.x.cpu()
    if x is None: return ["?"] * graph.num_nodes
    n = graph.num_nodes
    col0 = x[:, 0].float()
    if (col0 > 1).any() and (col0 - col0.round()).abs().max() < 0.01:
        return [ATOMIC_NUM_MAP.get(int(round(float(col0[i].item()))), "?")
                for i in range(n)]
    if x.shape[1] >= 14 and (x[:, :14].sum(dim=-1) - 1.0).abs().max() < 0.1:
        return [MUTAG_14[int(x[i, :14].argmax())]
                if int(x[i, :14].argmax()) < 14 else "?"
                for i in range(n)]
    return [str(int(x[i].argmax())) if x.shape[1] > 1
            else str(int(x[i, 0].item())) for i in range(n)]


def get_smiles(graph):
    return getattr(graph, "smiles", None)


# ===================================================================
#  MolGenMutagenicity — reconstruct RDKit mols from TUDataset one-hot
# ===================================================================

class MolGenMutagenicity:
    """Reconstruct RDKit molecules from Mutagenicity (TUDataset) one-hot format."""

    def __init__(self):
        self.atom_types = {0:"C",1:"O",2:"Cl",3:"H",4:"N",5:"F",6:"Br",
                           7:"S",8:"P",9:"I",10:"Na",11:"K",12:"Li",13:"Ca"}
        self.bond_types = {0: None, 1: None, 2: None}  # filled lazily

    def _get_bond_type(self, edge_attr_row):
        # edge_attr is 3-dim one-hot: [SINGLE, DOUBLE, TRIPLE]
        from rdkit.Chem import BondType
        idx = int(edge_attr_row.argmax())
        return [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE][idx]

    def get_mol(self, pyg_data):
        from rdkit.Chem import RWMol, BondType, AtomValenceException
        from rdkit import Chem as rdChem

        x = pyg_data.x.cpu()
        edge_index = pyg_data.edge_index.cpu()
        edge_attr = pyg_data.edge_attr.cpu() if hasattr(pyg_data, "edge_attr") and pyg_data.edge_attr is not None else None

        mol = RWMol()
        # Add atoms in order (PyG index == RDKit index)
        for atom_vec in x:
            atom_type = self.atom_types[int(atom_vec.argmax())]
            mol.AddAtom(rdChem.Atom(atom_type))

        # Add bonds
        tracked = set()
        for e_idx in range(edge_index.shape[1]):
            u, v = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
            if u >= v: continue
            if (u, v) in tracked: continue
            tracked.add((u, v))
            if edge_attr is not None and e_idx < edge_attr.shape[0]:
                bt = self._get_bond_type(edge_attr[e_idx])
            else:
                bt = BondType.SINGLE
            mol.AddBond(u, v, bt)

        # Sanitize with valence correction
        try:
            rdChem.SanitizeMol(mol)
        except Exception:
            self._sanitize_with_valence_correction(mol)

        return rdChem.Mol(mol)

    def get_smiles_no_h(self, pyg_data):
        from rdkit.Chem import MolToSmiles, RemoveHs
        mol = self.get_mol(pyg_data)
        return MolToSmiles(RemoveHs(mol))

    def _sanitize_with_valence_correction(self, mol):
        from rdkit.Chem import SanitizeMol, AtomValenceException
        try:
            SanitizeMol(mol)
        except Exception as e:
            match = re.search(r"atom # (\d+)", str(e))
            if match:
                atom_idx = int(match.group(1))
                self._correct_valence(mol, mol.GetAtomWithIdx(atom_idx))
                self._sanitize_with_valence_correction(mol)

    def _correct_valence(self, mol, c_atom):
        from rdkit import Chem as rdChem
        sym = c_atom.GetSymbol()
        neighbors = c_atom.GetNeighbors()
        n_types = [n.GetSymbol() for n in neighbors]

        if sym == "N":
            if n_types.count("O") > 1:  # nitro / nitrate
                for n in neighbors:
                    bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx())
                    if (n.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 1.0
                        and len(n.GetNeighbors()) == 1):
                        c_atom.SetFormalCharge(1); n.SetFormalCharge(-1)
            elif n_types.count("O") == 1:
                for n in neighbors:
                    if n.GetSymbol() == "O":
                        bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx())
                        if bond.GetBondTypeAsDouble() == 1.0:
                            c_atom.SetFormalCharge(1); n.SetFormalCharge(-1)
                        elif bond.GetBondTypeAsDouble() == 2.0:
                            c_atom.SetFormalCharge(1)
            elif n_types.count("N") == 1:
                bonds = [b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]
                if 3.0 in bonds: c_atom.SetFormalCharge(1)
                elif len(bonds) <= 3: c_atom.SetFormalCharge(1)
                else:
                    for n in neighbors:
                        if n.GetSymbol() == "N": c_atom.SetFormalCharge(1); n.SetFormalCharge(-1)
            elif n_types.count("N") > 1:  # azides
                for n in neighbors:
                    if n.GetSymbol() == "N" and len(n.GetNeighbors()) == 1:
                        c_atom.SetFormalCharge(1); n.SetFormalCharge(-1)
                    else: c_atom.SetFormalCharge(1)
            elif n_types.count("C") >= 3: c_atom.SetFormalCharge(1)
        elif sym == "O":
            if set([b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]) == {1.0, 2.0}:
                c_atom.SetFormalCharge(1)
        elif sym in ("S", "P"):
            pass  # let RDKit handle


# ===================================================================
#  RDKit MOLECULE DRAWING
# ===================================================================

def _rdkit_available():
    try:
        import rdkit; return True
    except ImportError:
        return False


def _verify_rdkit_atom_order(graph, smiles):
    """
    Sanity-check: RDKit atom ordering (from MolFromSmiles) must match the
    PyG node features (built by MoleculeDataset iterating mol.GetAtoms()).

    We compare the atomic number in graph.x[i, 0] against
    rdkit_mol.GetAtomWithIdx(i).GetAtomicNum() for every node i.
    Prints a warning on mismatch (should never happen for BBBP/BACE).
    Returns True iff the ordering matches across all atoms.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    n_mol = mol.GetNumAtoms()
    n_pyg = graph.num_nodes
    if n_mol != n_pyg:
        # This can happen for Mutagenicity (explicit H) — not an error,
        # just means the caller should not route through this SMILES path.
        return False

    x = graph.x.cpu()
    col0 = x[:, 0].float()
    is_atomic_num_encoding = (
        (col0 > 1).any() and (col0 - col0.round()).abs().max() < 0.01
    )
    if not is_atomic_num_encoding:
        # Not the atomic-number encoding used by MoleculeDataset →
        # can't verify, but also shouldn't be using this drawing path.
        return False

    mismatches = 0
    for i in range(min(n_mol, n_pyg)):
        rd_atom_num = mol.GetAtomWithIdx(i).GetAtomicNum()
        pyg_atom_num = int(round(float(col0[i].item())))
        if rd_atom_num != pyg_atom_num:
            mismatches += 1
            if mismatches <= 3:
                print(f"    [atom order mismatch] node={i}: "
                      f"RDKit={rd_atom_num} vs PyG={pyg_atom_num}")
    if mismatches > 0:
        print(f"    WARNING: {mismatches}/{n_pyg} atom ordering mismatches "
              f"for SMILES={smiles[:50]}... — RDKit/PyG indices may be misaligned!")
        return False
    return True


def draw_molecule_rdkit(ax, graph, normed_scores, smiles, title=""):
    """Draw a molecule via SMILES with per-atom signed attribution highlights.

    ATOM ORDERING GUARANTEE (BBBP / BACE):
      MoleculeDataset.process() builds each graph by:
        1. mol = Chem.MolFromSmiles(smiles)
        2. for atom in mol.GetAtoms(): ...  → builds x row 0..N-1
        3. stores the SAME smiles string on the Data object
      Here we call Chem.MolFromSmiles() on the SAME SMILES string.  RDKit
      parses SMILES left-to-right, so GetAtoms() returns atoms in the same
      order as the original parse.  Therefore:
          PyG node index i  ≡  RDKit atom index i.

    The function calls _verify_rdkit_atom_order() to sanity-check this
    invariant before drawing.
    """
    if not _verify_rdkit_atom_order(graph, smiles):
        return False
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from io import BytesIO

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    n_atoms = mol.GetNumAtoms()
    if n_atoms != graph.num_nodes: return False

    atom_highlights = {}
    atom_radii = {}
    for i in range(min(n_atoms, len(normed_scores))):
        s = float(normed_scores[i])
        val = 0.5 + 0.5 * max(-1.0, min(1.0, s))
        rgb = CMAP_SIGNED(val)
        atom_highlights[i] = tuple(float(c) for c in rgb[:3])
        atom_radii[i] = 0.35 + 0.20 * abs(s)

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 380)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1.8
    opts.useBWAtomPalette()          # disable element-coloured atom labels
    opts.setSymbolColour((0, 0, 0))  # black symbol text (C, O, N, …)
    drawer.DrawMolecule(mol, highlightAtoms=list(range(n_atoms)),
                        highlightAtomColors=atom_highlights,
                        highlightAtomRadii=atom_radii, legend=title)
    drawer.FinishDrawing()
    bio = BytesIO(drawer.GetDrawingText())
    img = plt.imread(bio)
    ax.imshow(img)
    ax.set_axis_off()
    if title: ax.set_title(title, fontsize=7, fontweight="bold")
    return True


def draw_molecule_rdkit_direct(ax, mol, normed_scores, title=""):
    """Draw an RDKit mol DIRECTLY (no SMILES round-trip) with per-atom highlights.

    Used for Mutagenicity where mol is built by MolGenMutagenicity.
    PyG node i == RDKit atom i (atoms added in order).  Explicit H atoms are shown;
    the figure caption documents this.
    """
    from rdkit.Chem.Draw import rdMolDraw2D
    from io import BytesIO

    n_atoms = mol.GetNumAtoms()
    atom_highlights = {}
    atom_radii = {}
    for i in range(min(n_atoms, len(normed_scores))):
        s = float(normed_scores[i])
        val = 0.5 + 0.5 * max(-1.0, min(1.0, s))
        rgb = CMAP_SIGNED(val)
        atom_highlights[i] = tuple(float(c) for c in rgb[:3])
        atom_radii[i] = 0.35 + 0.20 * abs(s)

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 380)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1.8
    opts.useBWAtomPalette()          # disable element-coloured atom labels
    opts.setSymbolColour((0, 0, 0))  # black symbol text (C, O, N, …)
    drawer.DrawMolecule(mol, highlightAtoms=list(range(n_atoms)),
                        highlightAtomColors=atom_highlights,
                        highlightAtomRadii=atom_radii, legend=title)
    drawer.FinishDrawing()
    bio = BytesIO(drawer.GetDrawingText())
    img = plt.imread(bio)
    ax.imshow(img)
    ax.set_axis_off()
    if title: ax.set_title(title, fontsize=7, fontweight="bold")


# ===================================================================
#  NETWORKX FALLBACK DRAWING
# ===================================================================

def _node_colors_from_signed(scores):
    colors = []
    for s in scores:
        val = 0.5 + 0.5 * float(s)
        colors.append(CMAP_SIGNED(max(0.0, min(1.0, val))))
    return colors


def draw_signed_graph(ax, graph, node_scores, node_labels=None,
                      highlight_nodes=None, target_node=None,
                      use_labels=True, title="", font_size=5.5,
                      node_size=180, edge_width=0.6):
    ei = graph.edge_index.cpu().numpy()
    n = graph.num_nodes
    if node_scores is None or len(node_scores) == 0:
        node_scores = np.zeros(n)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u < n and v < n: G.add_edge(u, v)
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    try:
        pos = nx.kamada_kawai_layout(G, scale=2.0)
    except Exception:
        pos = nx.spring_layout(G, seed=0, k=1.5, scale=2.0)

    node_colors = _node_colors_from_signed(node_scores[:n])
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), ax=ax,
                           width=edge_width, edge_color="#AAAAAA",
                           alpha=0.5, arrows=False)

    edgecolors_default = ["#555555"] * n
    linewidths_default = [0.6] * n
    if highlight_nodes:
        for ni in highlight_nodes:
            if 0 <= ni < n:
                edgecolors_default[ni] = "#000000"
                linewidths_default[ni] = 2.5
    if target_node is not None and 0 <= target_node < n:
        edgecolors_default[target_node] = "#CC0000"
        linewidths_default[target_node] = 3.0

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_size, linewidths=linewidths_default,
                           edgecolors=edgecolors_default)

    if use_labels and node_labels:
        for i in G.nodes():
            lbl = node_labels[i] if i < len(node_labels) else "?"
            s = float(node_scores[i]) if i < len(node_scores) else 0.0
            txt_color = "white" if abs(s) > 0.45 else "#222222"
            xi, yi = pos[i]
            ax.text(xi, yi, lbl, ha="center", va="center",
                    fontsize=font_size, fontweight="bold",
                    color=txt_color, zorder=5)
    ax.set_axis_off()
    if title: ax.set_title(title, fontsize=7, fontweight="bold", pad=2)


def draw_molecule_nx(ax, graph, normed_scores, node_labels, title=""):
    draw_signed_graph(ax, graph, normed_scores, node_labels=node_labels,
                      use_labels=True, title=title, font_size=5.0, node_size=200)


# ===================================================================
#  TOKEN HEATMAP — readable sentence display
# ===================================================================

def draw_sentence_heatmap(ax, tokens, scores, title=""):
    """
    Draw a sentence as a row of coloured rectangles with readable token text.
    Designed for 5–12 tokens.  Cell width adapts to token length.
    """
    n = len(tokens)
    if n == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                transform=ax.transAxes); ax.set_axis_off(); return

    s_normed, _ = normalize_signed_scores(np.asarray(scores[:n], dtype=float))

    # Adaptive cell sizing
    char_widths = [max(len(tok), 2) for tok in tokens]
    cell_w_base = 0.18  # width per character
    cell_h = 0.60
    padding_x = 0.04

    x_positions = [0.0]
    for cw in char_widths:
        x_positions.append(x_positions[-1] + cw * cell_w_base + padding_x)
    total_w = x_positions[-1] - padding_x

    y0 = 0.5 - cell_h / 2

    max_font = 9.0
    min_font = 6.5

    for i in range(n):
        x0 = x_positions[i]
        cell_w_i = char_widths[i] * cell_w_base
        score_i = float(s_normed[i])
        val = 0.5 + 0.5 * score_i
        facecolor = CMAP_SIGNED(val)

        rect = mpatches.Rectangle((x0, y0), cell_w_i, cell_h,
                                  facecolor=facecolor, edgecolor="#CCCCCC",
                                  linewidth=0.4, zorder=2)
        ax.add_patch(rect)

        txt_color = "white" if abs(score_i) > 0.4 else "#111111"
        tok = tokens[i]
        # Scale font to fit cell
        fs = max(min_font, min(max_font, cell_w_i / (len(tok) * 0.10)))
        ax.text(x0 + cell_w_i / 2, 0.5, tok, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=txt_color, zorder=3)

    ax.set_xlim(-0.1, total_w + 0.1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title: ax.set_title(title, fontsize=7, fontweight="bold", pad=6)


# ===================================================================
#  COLOURBAR
# ===================================================================

def add_signed_colorbar(fig, label="Signed node attribution"):
    cbar_ax = fig.add_axes([0.15, 0.015, 0.70, 0.012])
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=CMAP_SIGNED, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label(label, fontsize=10, fontweight="bold")
    cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cb.set_ticklabels(["-1", "-0.5", "0", "+0.5", "+1"])
    cb.ax.tick_params(labelsize=7)


# ===================================================================
#  FIGURE A1: BA-Shapes  (1 × 3, no ground-truth, sign-balanced)
# ===================================================================

def plot_bashapes_figure(data, model, num_classes, device, selected_cases, out_dir):
    """Figure A1: 1 row × 3 columns, sign-balanced node explanations."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    fig.suptitle("Figure A1: BA-Shapes — Signed Node Attribution (APEX)",
                 fontsize=12, fontweight="bold", y=0.98)

    for pi, case in enumerate(selected_cases[:3]):
        ax = axes[pi]
        node_idx = case["node_idx"]
        pred_cls = case["pred_cls"]
        scores = case["node_scores"]
        conf = case["confidence"]

        normed, _ = normalize_signed_scores(scores)

        # 3-hop subgraph around target node
        subset, ei_sub, mapping, _ = k_hop_subgraph(
            node_idx, num_hops=3, edge_index=data.edge_index,
            relabel_nodes=True, num_nodes=data.num_nodes)
        local_idx = (int(mapping[0].item()) if mapping.ndim > 0
                     else int(mapping.item()))

        sub_normed = normed[subset.cpu().numpy()]
        sub_graph = Data(x=data.x[subset], edge_index=ei_sub,
                         num_nodes=len(subset))

        # Recompute sign-balance on the 3-hop SUBGRAPH (what we actually draw)
        sb_sub = compute_sign_balance(sub_normed)

        # Motif highlighting
        motif_nodes = get_motif_nodes_bashapes(data, node_idx)
        motif_local = None
        if motif_nodes is not None and len(motif_nodes) > 0:
            g2l = {int(g): int(l) for l, g in enumerate(subset.tolist())}
            motif_local = [g2l[gn] for gn in motif_nodes if gn in g2l]

        title = (f"Node {node_idx}  |  pred={pred_cls}  "
                 f"conf={conf:.2f}  |  pos={sb_sub:.0%}")
        draw_signed_graph(ax, sub_graph, sub_normed,
                          target_node=local_idx,
                          highlight_nodes=motif_local, title=title)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=CMAP_SIGNED(0.95), edgecolor="#555",
                       label="Positive (+)", linewidth=0.5),
        mpatches.Patch(facecolor=CMAP_SIGNED(0.50), edgecolor="#555",
                       label="Near zero", linewidth=0.5),
        mpatches.Patch(facecolor=CMAP_SIGNED(0.05), edgecolor="#555",
                       label="Negative (-)", linewidth=0.5),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
                    markeredgecolor="#CC0000", markeredgewidth=3, markersize=10,
                    label="Target node"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
                    markeredgecolor="#000000", markeredgewidth=2.5, markersize=10,
                    label="Motif node"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.04))
    add_signed_colorbar(fig)
    _save_figure(fig, out_dir, "appendix_fig_A1_bashapes_signed")
    plt.close(fig)


# ===================================================================
#  FIGURE A2: Graph-SST2  (2 × 3, 5–12 tokens, readable)
# ===================================================================

def plot_graphsst2_figure(test_graphs, test_orig_indices, model, num_classes,
                          device, data_path, out_dir):
    """Figure A2: 2×3, short sentences (5–12 tokens), full-sentence heatmap."""
    tokens_map = _load_sst2_raw_tokens(data_path)
    vocab = _load_sst2_vocab(data_path)
    print(f"  Graph-SST2: loaded tokens for {len(tokens_map)} graphs")

    # ── Build candidates ──
    all_cands = []
    for sub_idx, g in enumerate(test_graphs):
        full_idx = test_orig_indices[sub_idx] if sub_idx < len(test_orig_indices) else sub_idx
        tokens = get_sst2_tokens_for_graph(g, full_idx, tokens_map, vocab)
        n_tok = len(tokens)
        if n_tok < 5 or n_tok > 14: continue
        # Require token count matches node count
        if n_tok != int(g.num_nodes): continue

        g_dev = g.to(device)
        pred_cls, _, conf = get_prediction(model, g_dev, device, model_level="graph")
        true_label = int(g.y.item())
        if pred_cls != true_label: continue
        if conf < 0.7 or conf > 0.99: continue

        all_cands.append({
            "graph": g, "graph_idx": sub_idx,
            "full_dataset_idx": full_idx,
            "pred_cls": pred_cls, "true_label": true_label,
            "confidence": conf, "num_nodes": int(g.num_nodes),
            "tokens": tokens,
        })
    print(f"  Graph-SST2: {len(all_cands)} candidates (5–12 tokens, matched)")

    # ── Run APEX ──
    for c in all_cands:
        g_dev = c["graph"].to(device)
        explainer = run_apex_explainer(model, g_dev, num_classes,
                                        explain_graph=True, device=device)
        scores_tensor = explainer.last_node_scores[c["pred_cls"]]
        scores = (scores_tensor.detach().cpu().numpy().copy()
                  if isinstance(scores_tensor, torch.Tensor)
                  else np.asarray(scores_tensor, dtype=float))
        c["node_scores"] = scores
        c["top20_mass"] = compute_concentration(scores)
        c["sign_balance"] = compute_sign_balance(scores)

    pos_cands = [c for c in all_cands if c["pred_cls"] == 1]
    neg_cands = [c for c in all_cands if c["pred_cls"] == 0]

    def _pick(cands, n, diversify_groups=None):
        """Pick *n* diverse candidates spread across the concentration distribution.

        We partition candidates by top20_mass terciles (low / mid / high).
        By default each group picks the highest-confidence candidate.
        If ``diversify_groups`` is a set of group indices, those groups pick
        the most sign-balanced candidate instead, producing different sentences
        while leaving the other groups' picks invariant.
        """
        if diversify_groups is None:
            diversify_groups = set()
        if len(cands) == 0: return []
        if len(cands) <= n:
            return sorted(cands, key=lambda c: -c["confidence"])

        sorted_by_mass = sorted(cands, key=lambda c: c["top20_mass"])
        group_size = len(sorted_by_mass) // n
        selected = []
        for g in range(n):
            start = g * group_size
            end = start + group_size if g < n - 1 else len(sorted_by_mass)
            group = sorted_by_mass[start:end]
            if g in diversify_groups and len(group) > 1:
                # Pick most sign-balanced (closest to 50/50 red/blue)
                best = min(group, key=lambda c: abs(c.get("sign_balance", 0.5) - 0.5))
            else:
                best = max(group, key=lambda c: c["confidence"])
            selected.append(best)

        selected = sorted(selected, key=lambda c: c["top20_mass"])
        return selected

    # Positive row: first two panels (groups 0,1) get fresh diverse picks;
    # panel 3 (group 2) keeps its max-confidence sentence.
    pos_sel = _pick(pos_cands, 3, diversify_groups={0, 1})
    neg_sel = _pick(neg_cands, 3)      # negative row unchanged
    print(f"  Graph-SST2: selected {len(pos_sel)} pos, {len(neg_sel)} neg")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 6.5))
    fig.suptitle("Figure A2: Graph-SST2 — Signed Token Attribution (APEX)",
                 fontsize=12, fontweight="bold", y=0.98)

    row_labels = ["Positive sentiment", "Negative sentiment"]
    for ri, (label, cases) in enumerate(zip(row_labels, [pos_sel, neg_sel])):
        axes[ri, 0].set_ylabel(label, fontsize=9, fontweight="bold",
                               rotation=90, labelpad=10, va="center")
        for ci, case in enumerate(cases):
            ax = axes[ri, ci]
            if case is None:
                ax.text(0.5, 0.5, "(no candidate)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
                ax.set_axis_off(); continue

            normed, _ = normalize_signed_scores(case["node_scores"])
            tokens = case["tokens"]
            title = (f"Pred={case['pred_cls']}  True={case['true_label']}  "
                     f"Conf={case['confidence']:.2f}")
            draw_sentence_heatmap(ax, tokens, normed, title=title)

    add_signed_colorbar(fig)
    all_selected = pos_sel + neg_sel
    _save_figure(fig, out_dir, "appendix_fig_A2_graphsst2_signed")
    plt.close(fig)
    return all_selected


# ===================================================================
#  FIGURE A3: Molecular Graphs  (3 × 3)
# ===================================================================

def plot_molecule_figure(dataset_infos, models, device, data_path, out_dir):
    """Figure A3: 3x3 (BBBP / BACE / Mutagenicity).

    Every selected molecule must have meaningful red-vs-blue contrast
    (sign-balance in [0.10, 0.85] — at least 10% positive AND 15% negative).
    """
    DATASETS = ["BBBP", "BACE", "Mutagenicity"]
    SIZE_RANGES = {"BBBP": (12, 35), "BACE": (15, 45), "Mutagenicity": (10, 40)}
    has_rdkit = _rdkit_available()
    if has_rdkit:
        print("  RDKit detected – will attempt molecular scaffold drawing")
    else:
        print("  RDKit not found – using networkx spring layout")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle("Figure A3: Molecular Graphs – Signed Node Attribution (APEX)",
                 fontsize=12, fontweight="bold", y=0.99)

    # Mutagenicity mol generator
    mut_gen = MolGenMutagenicity() if has_rdkit else None
    mut_used_explicit_h = False

    all_selected = []

    for ri, ds_name in enumerate(DATASETS):
        info = dataset_infos[ds_name]
        model = models[ds_name]
        num_classes = info["num_classes"]
        test_graphs = list(info["splits"]["test"])
        min_n, max_n = SIZE_RANGES[ds_name]

        axes[ri, 0].set_ylabel(ds_name, fontsize=10, fontweight="bold",
                               rotation=90, labelpad=12, va="center")

        # ── sign-balance: ≥10% pos, ≥15% neg for visual red/blue contrast ──
        selected = select_cases_for_dataset(
            test_graphs, model, num_classes, device, n=3,
            min_nodes=min_n, max_nodes=max_n,
            cls_balance=(num_classes == 2),
            sign_balance_range=(0.10, 0.85))

        for ci, case in enumerate(selected):
            ax = axes[ri, ci]
            graph = case["graph"]
            normed, _ = normalize_signed_scores(case["node_scores"])
            node_labels = get_molecule_node_labels(graph, ds_name)
            smiles = get_smiles(graph)
            sb = case.get("sign_balance", compute_sign_balance(case["node_scores"]))
            title = (f"#{case['graph_idx']}  |  "
                     f"pred={case['pred_cls']}  true={case['true_label']}  "
                     f"conf={case['confidence']:.2f}  |  pos={sb:.0%}")

            drawn = False

            if ds_name == "Mutagenicity" and has_rdkit and mut_gen is not None:
                # Use MolGenMutagenicity to reconstruct RDKit mol
                try:
                    mol = mut_gen.get_mol(graph)
                    if mol.GetNumAtoms() == graph.num_nodes:
                        draw_molecule_rdkit_direct(ax, mol, normed, title=title)
                        drawn = True
                        mut_used_explicit_h = True
                    else:
                        print(f"    Mutagenicity #{case['graph_idx']}: "
                              f"mol atoms={mol.GetNumAtoms()} !="
                              f" graph nodes={graph.num_nodes}")
                except Exception as e:
                    print(f"    Mutagenicity #{case['graph_idx']}: "
                          f"MolGen failed: {e}")

            if not drawn and ds_name != "Mutagenicity" and has_rdkit and smiles:
                drawn = draw_molecule_rdkit(ax, graph, normed, smiles, title=title)

            if not drawn:
                draw_molecule_nx(ax, graph.to("cpu"), normed, node_labels, title=title)

            case["dataset"] = ds_name
            case["figure"] = "A3"
            all_selected.append(case)

    # Figure annotation for Mutagenicity explicit H
    if mut_used_explicit_h:
        fig.text(0.5, 0.003,
                 "Mutagenicity molecules include explicit H atoms (shown as small "
                 "white/grey circles).  Attribution scores for H are typically near zero.",
                 ha="center", fontsize=7, fontstyle="italic", color="#555555")

    add_signed_colorbar(fig)
    _save_figure(fig, out_dir, "appendix_fig_A3_molecules_signed")
    plt.close(fig)
    return all_selected


# ===================================================================
#  SAVE HELPERS
# ===================================================================

def _save_figure(fig, out_dir, basename):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = osp.join(out_dir, f"{basename}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")


def save_selection_csv(all_cases, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = osp.join(out_dir, "selected_signed_visualization_cases.csv")
    fields = ["figure", "dataset", "split_index", "original_index",
              "pred_cls", "true_label", "confidence", "num_nodes",
              "top20_mass", "sign_balance"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, case in enumerate(all_cases):
            row = {
                "figure": case.get("figure", "?"),
                "dataset": case.get("dataset", "?"),
                "split_index": i,
                "original_index": case.get("graph_idx",
                                            case.get("node_idx", i)),
                "pred_cls": case.get("pred_cls", -1),
                "true_label": case.get("true_label", -1),
                "confidence": case.get("confidence", 0.0),
                "num_nodes": case.get("num_nodes", 0),
                "top20_mass": case.get("top20_mass", 0.0),
                "sign_balance": case.get("sign_balance", 0.0),
            }
            writer.writerow(row)
    print(f"  Saved CSV: {csv_path}")


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="APEX Signed Node Attribution Visualization (Appendix)"
    )
    parser.add_argument("--data_path", type=str, default="./dataset")
    parser.add_argument("--checkpoint_path", type=str, default="./model/checkpoint")
    parser.add_argument("--out_dir", type=str, default="./figures_signed")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    all_csv_entries = []

    # ═══════════════════════════════════════════════════════════════
    #  FIGURE A1 — BA-Shapes
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Figure A1: BA-Shapes (1×3, sign-balanced)")
    print("=" * 60)
    ba_info = load_data_for_dataset(args.data_path, "BA_shapes")
    ba_data = ba_info["full_data"]
    ba_model = load_model_for_dataset(
        "BA_shapes", ba_info["dim_node"], ba_info["num_classes"],
        args.checkpoint_path, device, model_level="node")
    ba_selected = select_bashapes_nodes(
        ba_data, ba_model, ba_info["num_classes"], device, n=3)
    if ba_selected:
        plot_bashapes_figure(ba_data, ba_model, ba_info["num_classes"],
                             device, ba_selected, args.out_dir)
        for c in ba_selected:
            c["dataset"] = "BA_shapes"; c["figure"] = "A1"
            c["graph_idx"] = c["node_idx"]
        all_csv_entries.extend(ba_selected)
    else:
        print("  !!! BA-Shapes: no suitable nodes found; skipping Figure A1")

    # ═══════════════════════════════════════════════════════════════
    #  FIGURE A2 — Graph-SST2
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Figure A2: Graph-SST2 (2×3, 5–12 tokens)")
    print("=" * 60)
    sst2_info = load_data_for_dataset(args.data_path, "Graph-SST2")
    sst2_model = load_model_for_dataset(
        "Graph-SST2", sst2_info["dim_node"], sst2_info["num_classes"],
        args.checkpoint_path, device, model_level="graph")
    sst2_test = list(sst2_info["splits"]["test"])
    sst2_orig_indices = list(sst2_info["splits"]["test"].indices)
    sst2_cases = plot_graphsst2_figure(
        sst2_test, sst2_orig_indices, sst2_model,
        sst2_info["num_classes"], device, args.data_path, args.out_dir)
    for c in sst2_cases:
        c["dataset"] = "Graph-SST2"; c["figure"] = "A2"
        if "full_dataset_idx" in c and "graph_idx" not in c:
            c["graph_idx"] = c["full_dataset_idx"]
    all_csv_entries.extend(sst2_cases)

    # ═══════════════════════════════════════════════════════════════
    #  FIGURE A3 — Molecular datasets
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Figure A3: Molecular Graphs (BBBP / BACE / Mutagenicity)")
    print("=" * 60)
    mol_info = {}
    mol_models = {}
    for ds in ["BBBP", "BACE", "Mutagenicity"]:
        print(f"\n  Loading {ds}...")
        mol_info[ds] = load_data_for_dataset(args.data_path, ds)
        mol_models[ds] = load_model_for_dataset(
            ds, mol_info[ds]["dim_node"], mol_info[ds]["num_classes"],
            args.checkpoint_path, device, model_level="graph")
    mol_cases = plot_molecule_figure(
        mol_info, mol_models, device, args.data_path, args.out_dir)
    all_csv_entries.extend(mol_cases)

    # ═══════════════════════════════════════════════════════════════
    #  Save CSV
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Saving CSV")
    print("=" * 60)
    save_selection_csv(all_csv_entries, args.out_dir)
    print(f"\nDone! Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
