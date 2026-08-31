"""
subexplainer.py
================
核心框架：基于有限域 PIT 的零误差代数消元（CRT 双素数 + 死区截断版本）

[Step 1 升级]
  引入死区阈值 tau，将 |W_scaled| < tau 的权重强制归零，
  在 GF(p) 上人为构造稀疏多项式理想，使 Step 3 消元得以实际触发。

  tilde_W = 0                          if |floor(W * K)| < tau
           floor(W * K) mod p          otherwise
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

_PRIME1: int = 10 ** 9 + 7
_PRIME2: int = 10 ** 9 + 9
_DEFAULT_PRIME: int = _PRIME1
_DEFAULT_K: int = 10 ** 6
_PROBE_PRIME: int = 10007
_DEFAULT_TAU: int = 0   # 死区阈值默认值（0 = 不截断，向后兼容）

_B = 31623  # ceil(sqrt(10^9+9))


def _safe_mod(t: Tensor, p: int) -> Tensor:
    return t.remainder(p)


# ─────────────────────────────────────────────────────────────────────────────
# 精确 GF(p) 矩阵乘：双重拆位
# ─────────────────────────────────────────────────────────────────────────────

def _gfp_linear(x: Tensor, W: Tensor, bias: Tensor, p: int) -> Tensor:
    B = _B
    x_hi = x // B;  x_lo = x % B
    W_hi = W // B;  W_lo = W % B

    def mm(A, Bt):
        return (A.double() @ Bt.double()).round().to(torch.int64)

    hh = _safe_mod(mm(x_hi, W_hi.T), p)
    hl = _safe_mod(mm(x_hi, W_lo.T), p)
    lh = _safe_mod(mm(x_lo, W_hi.T), p)
    ll = _safe_mod(mm(x_lo, W_lo.T), p)

    B2 = (B * B) % p
    acc = _safe_mod(hh * B2, p)
    acc = _safe_mod(acc + _safe_mod((hl + lh) * B, p), p)
    acc = _safe_mod(acc + ll, p)
    acc = _safe_mod(acc + bias.unsqueeze(0), p)
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 死区量化辅助：将已缩放的整数张量做死区截断
# ─────────────────────────────────────────────────────────────────────────────

def _dead_zone(q: Tensor, tau: int) -> Tensor:
    """
    将绝对值 < tau 的整数元素归零（死区滤波）。
    tau=0 时为恒等映射（向后兼容）。
    """
    if tau <= 0:
        return q
    return torch.where(q.abs() < tau, torch.zeros_like(q), q)


# ─────────────────────────────────────────────────────────────────────────────
# 提取冻结权重和非线性参数（含死区截断）
# ─────────────────────────────────────────────────────────────────────────────

def _get_frozen_weights(model: nn.Module, K: int, p: int,
                        tau: int = _DEFAULT_TAU) -> dict:
    """
    提取所有 Linear 层量化整数权重，值域 [0, p-1]。
    死区阈值 tau：|floor(W*K)| < tau 的元素被强制归零。
    """
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = torch.floor(module.weight.data.double() * K).to(torch.int64)
            W = _dead_zone(W, tau)          # ← 死区截断
            W = _safe_mod(W, p)
            if module.bias is not None:
                b = torch.floor(module.bias.data.double() * K).to(torch.int64)
                b = _dead_zone(b, tau)      # ← 死区截断
                b = _safe_mod(b, p)
            else:
                b = torch.zeros(module.out_features, dtype=torch.int64,
                                device=module.weight.device)
            weights[name] = {'weight': W, 'bias': b}
    return weights


def _get_extra_params(model: nn.Module, K: int, p: int,
                      tau: int = _DEFAULT_TAU) -> dict:
    """
    提取 PolyScaleNorm.scale 和 PolyActivation.alpha 的量化整数值。
    同样施加死区截断。
    """
    params = {}
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls == 'PolyScaleNorm':
            s = torch.floor(module.scale.data.double() * K).to(torch.int64)
            s = _dead_zone(s, tau)          # ← 死区截断
            params[name] = {'type': cls, 'scale': _safe_mod(s, p)}
        elif cls == 'PolyActivation':
            a = module.alpha.data.double().clamp(-0.5, 0.5)
            a_int = torch.floor(a * K).to(torch.int64)
            a_int = _dead_zone(a_int, tau)  # ← 死区截断
            params[name] = {'type': cls, 'alpha': _safe_mod(a_int, p)}
    return params


# ─────────────────────────────────────────────────────────────────────────────
# 精确 GF(p) PolyMLP 前向
# ─────────────────────────────────────────────────────────────────────────────

def _gfp_poly_mlp(x, prefix, weights, params, K, p):
    h = _gfp_linear(x, weights[f'{prefix}.0']['weight'],
                    weights[f'{prefix}.0']['bias'], p)
    psn = params.get(f'{prefix}.1')
    if psn and psn['type'] == 'PolyScaleNorm':
        h = _safe_mod(h * psn['scale'].unsqueeze(0), p)
    pact = params.get(f'{prefix}.2')
    if pact and pact['type'] == 'PolyActivation':
        h_sq = _gfp_sq(h, p)
        h = _safe_mod(h + _safe_mod(pact['alpha'] * h_sq, p), p)
    h = _gfp_linear(h, weights[f'{prefix}.3']['weight'],
                    weights[f'{prefix}.3']['bias'], p)
    return h


def _gfp_sq(x: Tensor, p: int) -> Tensor:
    B = _B
    x_hi = x // B;  x_lo = x % B
    B2 = (B * B) % p

    def sq_term(a, b, coef):
        v = (a.double() * b.double()).round().to(torch.int64)
        return _safe_mod(_safe_mod(v, p) * coef, p)

    r = sq_term(x_hi, x_hi, B2)
    r = _safe_mod(r + sq_term(x_hi, x_lo, (2 * B) % p), p)
    r = _safe_mod(r + sq_term(x_lo, x_lo, 1), p)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# 精确 GF(p) GINConv 聚合
# ─────────────────────────────────────────────────────────────────────────────

def _gfp_gin_aggregate(h, edge_index, N, p, device):
    src, dst = edge_index[0], edge_index[1]
    agg = torch.zeros(N, h.shape[1], dtype=torch.int64, device=device)
    BLOCK = 512
    E = edge_index.shape[1]
    for s in range(0, E, BLOCK):
        e = min(s + BLOCK, E)
        agg.scatter_add_(0, dst[s:e].unsqueeze(1).expand(-1, h.shape[1]),
                         h[src[s:e]])
        agg = _safe_mod(agg, p)
    return _safe_mod(h + agg, p)


# ─────────────────────────────────────────────────────────────────────────────
# 完整精确 GF(p) 前向（单素数）
# ─────────────────────────────────────────────────────────────────────────────

def _forward_gfp_exact(
    weights, params, z, edge_index, K, p, device,
    batch=None, verbose=False, label='',
) -> Tensor:
    N = z.shape[0]
    h = z.to(torch.int64).to(device)

    for prefix in ['conv1.nn', 'convs.0.nn', 'convs.1.nn']:
        h = _gfp_gin_aggregate(h, edge_index, N, p, device)
        h = _gfp_poly_mlp(h, prefix, weights, params, K, p)

    if verbose:
        uniq = h.unique(dim=0).shape[0]
        print(f'  [{label}] after convs: unique_node_vecs={uniq}/{N}')

    h_pool = _safe_mod(h.sum(dim=0, keepdim=True), p)

    h_pool = _gfp_linear(h_pool, weights['ffn.0']['weight'],
                         weights['ffn.0']['bias'], p)
    pact_ffn = params.get('ffn.1')
    if pact_ffn and pact_ffn['type'] == 'PolyActivation':
        h_sq = _gfp_sq(h_pool, p)
        h_pool = _safe_mod(h_pool + _safe_mod(pact_ffn['alpha'] * h_sq, p), p)
    h_pool = _gfp_linear(h_pool, weights['ffn.3']['weight'],
                         weights['ffn.3']['bias'], p)

    if verbose:
        print(f'  [{label}] output (p={p}): {h_pool.tolist()}')

    return h_pool


# ─────────────────────────────────────────────────────────────────────────────
# CRT 双素数签名
# ─────────────────────────────────────────────────────────────────────────────

def _forward_crt(
    weights1, params1, weights2, params2,
    z, edge_index, K, device, batch=None, verbose=False, label='',
) -> Tensor:
    y1 = _forward_gfp_exact(weights1, params1, z, edge_index,
                             K, _PRIME1, device, batch, verbose, label + '_p1')
    y2 = _forward_gfp_exact(weights2, params2, z, edge_index,
                             K, _PRIME2, device, batch, verbose, label + '_p2')
    return torch.cat([y1, y2], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 代数冻结与稀疏化重定义（含死区截断）
# ─────────────────────────────────────────────────────────────────────────────

def algebraic_freeze(model: nn.Module, K: int = _DEFAULT_K,
                     p: int = _DEFAULT_PRIME,
                     tau: int = _DEFAULT_TAU) -> nn.Module:
    """
    将浮点模型映射到 GF(p)，同时施加死区截断 tau。
    |floor(param * K)| < tau 的参数被强制归零（稀疏理想）。
    """
    frozen = copy.deepcopy(model)
    frozen.eval()
    num_params = num_zeroed_tau = num_zeroed_mod = 0

    with torch.no_grad():
        for name, param in frozen.named_parameters():
            q = torch.floor(param.data.double() * K).to(torch.int64)
            # 死区截断（新增）
            zeroed_tau = (q.abs() < tau).sum().item() if tau > 0 else 0
            q = _dead_zone(q, tau)
            # 取模映射
            q = _safe_mod(q, p)
            zeroed_mod = (q == 0).sum().item()

            param.data = q.double()
            num_zeroed_tau += zeroed_tau
            num_zeroed_mod += zeroed_mod
            num_params += q.numel()

    print(f'[Step 1] Algebraic Freezing: K={K}, p={p}, tau={tau}')
    print(f'         Total params : {num_params}')
    print(f'         Zeroed by tau: {num_zeroed_tau} '
          f'({100.*num_zeroed_tau/max(num_params,1):.2f}%)')
    print(f'         Zeroed total : {num_zeroed_mod} '
          f'({100.*num_zeroed_mod/max(num_params,1):.2f}%)')
    return frozen


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 探针采样与代数锚定（CRT）
# ─────────────────────────────────────────────────────────────────────────────

def probe_anchoring(
    weights1, params1, weights2, params2,
    num_nodes, d0, edge_index, K, device,
    batch=None, probe_prime=_PROBE_PRIME, verbose=False,
):
    z = torch.randint(0, probe_prime, (num_nodes, d0),
                      dtype=torch.int64, device=device)
    if verbose:
        print(f'[Step 2] Probe: shape={z.shape}, '
              f'domain=[0,{probe_prime}), edges={edge_index.size(1)}')
    y_base = _forward_crt(weights1, params1, weights2, params2,
                          z, edge_index, K, device, batch, verbose, 'y_base')
    if verbose:
        print(f'         y_base: {y_base.tolist()}')
    return z, y_base


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: O(|E|) 代数重要性排序 + 贪心保留
# ─────────────────────────────────────────────────────────────────────────────

def absolute_rigid_elimination(
    weights1, params1, weights2, params2,
    z_probe, y_base, edge_index_full, K, device,
    batch=None, verbose=False,
    sparsity: float = 0.5,
):
    """
    对每条边计算"删除后 CRT 签名的 L1 变化量"作为重要性分数，
    按分数降序保留 top-(1-sparsity) 比例的边。

    判定标准从"严格恒等"改为"重要性排序"，解决 GIN 全边敏感问题。
    保留数量 keep_k = max(1, round(num_edges * (1 - sparsity)))。
    """
    num_edges = edge_index_full.size(1)
    scores = torch.zeros(num_edges, dtype=torch.float64, device=device)

    # 先统计严格冗余边（仍然保留原逻辑，作为诊断信息）
    strict_redundant = 0
    for i in range(num_edges):
        keep = torch.ones(num_edges, dtype=torch.bool, device=device)
        keep[i] = False
        E_test = edge_index_full[:, keep]
        y_test = _forward_crt(weights1, params1, weights2, params2,
                               z_probe, E_test, K, device, batch)
        if torch.equal(y_test, y_base):
            strict_redundant += 1
            scores[i] = 0.0
        else:
            # L1 距离作为重要性（两个素数拼接后的绝对差之和）
            scores[i] = (y_test.double() - y_base.double()).abs().sum().item()

    keep_k = max(1, round(num_edges * (1.0 - sparsity)))
    topk = scores.topk(keep_k)
    E_exact = topk.indices.sort().values

    print(f'[Step 3] Importance ranking: {num_edges} edges, '
          f'strict_redundant={strict_redundant}, '
          f'keep_k={keep_k} (sparsity={sparsity:.2f})')
    print(f'         Score range: min={scores.min():.3e}  '
          f'median={scores.median():.3e}  max={scores.max():.3e}')
    return E_exact


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 决定性图子式重构
# ─────────────────────────────────────────────────────────────────────────────

def exact_motif_reconstruction(x, edge_index_full, E_exact_indices):
    E_exact = edge_index_full[:, E_exact_indices]
    print(f'[Step 4] Motif: {E_exact.size(1)} edges, {x.size(0)} nodes')
    return x, E_exact


# ─────────────────────────────────────────────────────────────────────────────
# 主类：PolyGINExplainer
# ─────────────────────────────────────────────────────────────────────────────

class PolyGINExplainer:
    def __init__(
        self,
        model: nn.Module,
        explain_graph: bool = True,
        K: int = _DEFAULT_K,
        p: int = _DEFAULT_PRIME,
        probe_prime: int = _PROBE_PRIME,
        tau: int = _DEFAULT_TAU,          # ← 新增死区阈值
        verbose: bool = False,
    ):
        self.model = model
        self.explain_graph = explain_graph
        self.K = K
        self.probe_prime = probe_prime
        self.tau = tau
        self.verbose = verbose
        self.device = next(model.parameters()).device

        # Step 1：冻结（含死区截断）
        self.frozen_model = algebraic_freeze(model, K=K, p=_PRIME1, tau=tau)
        self.frozen_model.to(self.device).eval()

        # 预提取两个素数下的权重（含 tau 截断）
        self.weights1 = _get_frozen_weights(model, K, _PRIME1, tau)
        self.params1  = _get_extra_params(model, K, _PRIME1, tau)
        self.weights2 = _get_frozen_weights(model, K, _PRIME2, tau)
        self.params2  = _get_extra_params(model, K, _PRIME2, tau)

        self.last_node_scores = None
        self.last_E_exact: Optional[Tensor] = None
        self.last_E_exact_indices: Optional[Tensor] = None

    def _explain_single(self, x, edge_index, batch=None, sparsity=0.5):
        N, d0 = x.shape
        device = self.device
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        z_probe, y_base = probe_anchoring(
            self.weights1, self.params1, self.weights2, self.params2,
            N, d0, edge_index, self.K, device, batch,
            probe_prime=self.probe_prime, verbose=self.verbose,
        )

        E_exact_indices = absolute_rigid_elimination(
            self.weights1, self.params1, self.weights2, self.params2,
            z_probe, y_base, edge_index, self.K, device, batch,
            verbose=self.verbose, sparsity=sparsity,
        )

        x_out, E_exact = exact_motif_reconstruction(x, edge_index, E_exact_indices)
        self.last_E_exact = E_exact
        self.last_E_exact_indices = E_exact_indices
        return x_out, E_exact

    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        sparsity: float = 0.0,
        num_classes: int = 2,
        node_idx: int = 0,
        max_nodes: int = -1,
        **kwargs,
    ):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        N = x.size(0)
        batch = torch.zeros(N, dtype=torch.long, device=self.device)

        _, _ = self._explain_single(x, edge_index, batch, sparsity=sparsity)

        num_edges = edge_index.size(1)
        E_exact_indices = self.last_E_exact_indices

        mask = torch.zeros(num_edges, dtype=torch.float, device=self.device)
        if E_exact_indices is not None and E_exact_indices.numel() > 0:
            mask[E_exact_indices] = 1.0

        edge_masks = [mask for _ in range(num_classes)]

        src, dst = edge_index[0], edge_index[1]
        ones = torch.ones(num_edges, device=self.device)
        deg = torch.zeros(N, device=self.device)
        deg.scatter_add_(0, src, ones)
        deg.scatter_add_(0, dst, ones)

        node_scores_list = []
        for _ in range(num_classes):
            ns = torch.zeros(N, device=self.device)
            ns.scatter_add_(0, src, mask)
            ns.scatter_add_(0, dst, mask)
            ns = ns / (deg + 1e-8)
            node_scores_list.append(ns)

        self.last_node_scores = node_scores_list
        return edge_masks, node_scores_list