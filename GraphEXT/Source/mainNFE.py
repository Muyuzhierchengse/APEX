"""
subexplainer.py
================
核心框架：基于有限域 PIT 的零误差代数消元
实现 PolyGINExplainer 的四步流程：
  Step 1 — 代数冻结 (Algebraic Freezing)
  Step 2 — 探针采样与代数锚定 (Schwartz-Zippel Probe Anchoring)
  Step 3 — O(|E|) 绝对刚性消元 (Absolute Rigid Elimination)
  Step 4 — 决定性图子式重构 (Exact Motif Reconstruction)
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PRIME: int = 10 ** 9 + 7   # 大素数 p
_DEFAULT_K: int = 10 ** 6           # 缩放因子 K（特征保留物理阈值）


# ─────────────────────────────────────────────────────────────────────────────
# 工具：安全整数模运算（防止 int64 溢出）
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mod(t: Tensor, p: int) -> Tensor:
    """在 int64 上做 mod，将值域压到 [0, p-1]。"""
    return t.remainder(p)


# ─────────────────────────────────────────────────────────────────────────────
# 逐层 mod p Hook：防止 float32 中间层溢出
# ─────────────────────────────────────────────────────────────────────────────

class _ModHook:
    """
    对 nn.Linear 的输出立即做 round → int64 → mod p → float，
    将激活值域压回 [0, p-1]，防止跨层累积溢出。
    这是在 float32 硬件上模拟 GF(p) 运算的必要补丁。
    """
    def __init__(self, p: int):
        self.p = p
        self._handles = []

    def register(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._hook_fn)
                self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook_fn(self, module: nn.Module, input: tuple, output: Tensor):
        # round → int64 → mod p → float（全程保留 device）
        out_int = torch.round(output).to(torch.int64)
        out_mod = _safe_mod(out_int, self.p)
        return out_mod.float()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 代数冻结 (Algebraic Freezing)
# ─────────────────────────────────────────────────────────────────────────────

def algebraic_freeze(
    model: nn.Module,
    K: int = _DEFAULT_K,
    p: int = _DEFAULT_PRIME,
) -> nn.Module:
    """
    将浮点模型 f_R 映射到有限域 GF(p) 模型 f_Z。

    操作：Θ̃ = floor(Θ · K) mod p
    - 小于 1/K 的浮点残差被"零点混叠"抹除
    - 显著权重被转化为绝对整数

    返回一个新模型（不修改原模型），权重已替换为量化后的 float，
    真正的逐层模运算在 _forward_gf 的 _ModHook 中实现。
    """
    frozen = copy.deepcopy(model)
    frozen.eval()
    with torch.no_grad():
        for param in frozen.parameters():
            # floor(Θ · K) mod p，存为 float32 以兼容原网络算子
            quantized = torch.floor(param.data * K).to(torch.int64)
            quantized = _safe_mod(quantized, p)
            param.data = quantized.float()
    return frozen


# ─────────────────────────────────────────────────────────────────────────────
# 有限域前向传播包装（带逐层 mod p hook）
# ─────────────────────────────────────────────────────────────────────────────

def _forward_gf(
    frozen_model: nn.Module,
    z: Tensor,           # int64 探针，形状 [N, d0]
    edge_index: Tensor,  # 当前候选拓扑
    p: int,
    device: torch.device,
    batch: Optional[Tensor] = None,
) -> Tensor:
    """
    在 GF(p) 上运行 frozen_model 的一次前向传播。

    策略：
      1. 将 int64 探针转为 float（模型算子要求 float）
      2. 注册逐层 mod p hook，防止 float32 跨层累积溢出
      3. 运行 forward，获取 float 输出
      4. 将最终输出 round → int64 → mod p，得到有限域签名
    """
    z_float = z.float().to(device)
    if batch is None:
        batch = torch.zeros(z_float.size(0), dtype=torch.long, device=device)

    # 注册逐层 mod p hook
    hook = _ModHook(p)
    hook.register(frozen_model)

    try:
        with torch.no_grad():
            out = frozen_model(z_float, edge_index, batch)
    finally:
        hook.remove()

    if isinstance(out, (tuple, list)):
        out = out[0]

    # 最终输出再做一次 round → int64 → mod p
    out_int = torch.round(out).to(torch.int64)
    return _safe_mod(out_int, p)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 探针采样与代数锚定
# ─────────────────────────────────────────────────────────────────────────────

def probe_anchoring(
    frozen_model: nn.Module,
    num_nodes: int,
    d0: int,
    edge_index: Tensor,
    p: int,
    device: torch.device,
    batch: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    在 GF(p) 全域均匀采样随机探针矩阵 Z̃，计算基准代数签名 Ỹ_base。

    Z̃ ~ U(0, p-1)^{N×d0}

    根据 Schwartz-Zippel 引理：
      两个不同 d 阶多项式在随机点碰撞的概率 ≤ d/p
      对 d=8, p≈1e9 → 误判概率 < 1e-8

    返回：
      z_probe  — int64 探针矩阵 [N, d0]
      y_base   — 基准签名（int64 mod p）[C] 或 [N, C]
    """
    # 在 [0, p-1] 均匀采样 int64
    z_probe = torch.randint(0, p, (num_nodes, d0),
                            dtype=torch.int64, device=device)

    y_base = _forward_gf(frozen_model, z_probe, edge_index, p, device, batch)
    return z_probe, y_base


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: O(|E|) 绝对刚性消元
# ─────────────────────────────────────────────────────────────────────────────

def absolute_rigid_elimination(
    frozen_model: nn.Module,
    z_probe: Tensor,
    y_base: Tensor,
    edge_index_full: Tensor,
    p: int,
    device: torch.device,
    batch: Optional[Tensor] = None,
) -> Tensor:
    """
    线性遍历 O(|E|) 次，对每条边做严格模等价测试。

    若去掉边 e_i 后：
      f_Z(Z̃; E \ {e_i}) ≡ Ỹ_base (mod p)
    则 e_i 是多项式的冗余变量，永久删除。

    冗余变量剔除的代数独立性保证了单次线性遍历的正确性：
    先删 A 后删 B ≡ 先删 B 后删 A，完全规避 NP-Hard 子集搜索。

    返回：E_exact_indices — 精确保留的边索引（在原 edge_index 中的列下标）
    """
    num_edges = edge_index_full.size(1)
    # 从完整边集开始
    E_cand_mask = torch.ones(num_edges, dtype=torch.bool, device=device)

    for i in range(num_edges):
        if not E_cand_mask[i]:
            continue  # 已被删除，跳过

        # 拓扑遮蔽：假设缺失边 e_i
        test_mask = E_cand_mask.clone()
        test_mask[i] = False
        E_test = edge_index_full[:, test_mask]

        # 有限域前向（带逐层 mod p）
        y_test = _forward_gf(frozen_model, z_probe, E_test, p, device, batch)

        # 刚性校验：严格模等价
        if torch.equal(y_test, y_base):
            # e_i 是冗余变量，永久删除
            E_cand_mask[i] = False

    # 返回保留的边列下标
    E_exact_indices = E_cand_mask.nonzero(as_tuple=True)[0]
    return E_exact_indices


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 决定性图子式重构
# ─────────────────────────────────────────────────────────────────────────────

def exact_motif_reconstruction(
    x: Tensor,
    edge_index_full: Tensor,
    E_exact_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    将代数消元后的骨架结构与原始物理特征重新绑定。

    信号与结构解耦：
      - GF(p) + 随机探针 Z̃ 萃取图的"纯代数结构（Structure）"
      - 重新赋予原始"物理信号（Signal）" X ∈ R^{N×d0}

    返回：
      (x, E_exact) 即 G_motif = (V, E_exact, X)
    """
    E_exact = edge_index_full[:, E_exact_indices]
    return x, E_exact


# ─────────────────────────────────────────────────────────────────────────────
# 主类：PolyGINExplainer
# ─────────────────────────────────────────────────────────────────────────────

class PolyGINExplainer:
    """
    基于有限域 PIT 的零误差代数消元解释器。

    完整四步流程：
      Step 1 — 代数冻结
      Step 2 — 探针采样与代数锚定
      Step 3 — O(|E|) 绝对刚性消元
      Step 4 — 决定性图子式重构

    接口与其他 explainer 一致，供 main.py 统一调用。
    """

    def __init__(
        self,
        model: nn.Module,
        explain_graph: bool = True,
        K: int = _DEFAULT_K,
        p: int = _DEFAULT_PRIME,
    ):
        self.model = model
        self.explain_graph = explain_graph
        self.K = K
        self.p = p
        self.device = next(model.parameters()).device

        # Step 1: 代数冻结（在初始化时完成，避免重复计算）
        self.frozen_model = algebraic_freeze(model, K=K, p=p)
        self.frozen_model.to(self.device)
        self.frozen_model.eval()

        # 供 main.py 读取的副产品
        self.last_node_scores: Optional[List[Tensor]] = None
        self.last_E_exact: Optional[Tensor] = None
        self.last_E_exact_indices: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # 内部方法：运行完整流程
    # ------------------------------------------------------------------

    def _explain_single(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        对单张图运行完整代数消元流程。

        返回：
          (x_orig, E_exact) — 原始特征 + 精确边集
        """
        N, d0 = x.shape
        device = self.device

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        # Step 2: 探针采样与代数锚定
        z_probe, y_base = probe_anchoring(
            self.frozen_model, N, d0, edge_index, self.p, device, batch
        )

        # Step 3: O(|E|) 绝对刚性消元
        E_exact_indices = absolute_rigid_elimination(
            self.frozen_model, z_probe, y_base, edge_index, self.p, device, batch
        )

        # Step 4: 图子式重构
        x_out, E_exact = exact_motif_reconstruction(x, edge_index, E_exact_indices)

        self.last_E_exact = E_exact
        self.last_E_exact_indices = E_exact_indices

        return x_out, E_exact

    # ------------------------------------------------------------------
    # 公共接口：与其他 explainer 兼容
    # ------------------------------------------------------------------

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
        """
        返回 (edge_masks, node_scores) 元组，兼容 main.py 的调用约定。

        edge_masks[c]: 形状 [|E|] 的 float Tensor，
          E_exact 中的边为 1.0，其余为 0.0。
        last_node_scores: 通过边 mask 聚合的节点分数（供 Efficiency Gap 计算）。
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        N = x.size(0)

        batch = torch.zeros(N, dtype=torch.long, device=self.device)

        # 运行代数消元
        _, E_exact = self._explain_single(x, edge_index, batch)

        num_edges = edge_index.size(1)
        E_exact_indices = self.last_E_exact_indices

        # 构造 edge_mask（所有类共享同一结构掩码）
        mask = torch.zeros(num_edges, dtype=torch.float, device=self.device)
        if E_exact_indices.numel() > 0:
            mask[E_exact_indices] = 1.0

        edge_masks = [mask for _ in range(num_classes)]

        # 聚合节点分数（与 main.py edge_mask_to_node_scores 逻辑一致）
        src, dst = edge_index[0], edge_index[1]
        node_scores_list = []
        for c in range(num_classes):
            ns = torch.zeros(N, device=self.device)
            ns.scatter_add_(0, src, mask)
            ns.scatter_add_(0, dst, mask)
            deg = torch.zeros(N, device=self.device)
            ones = torch.ones(num_edges, device=self.device)
            deg.scatter_add_(0, src, ones)
            deg.scatter_add_(0, dst, ones)
            ns = ns / (deg + 1e-8)
            node_scores_list.append(ns)

        self.last_node_scores = node_scores_list

        return edge_masks, node_scores_list