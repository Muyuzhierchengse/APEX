import torch
import numpy as np


class RiemannOptIG:


    def __init__(self, model, explain_graph=True, m_steps=50,
                 n_probe=None, baseline='zero'):
        self.model         = model
        self.explain_graph = explain_graph
        self.m_steps       = m_steps
        self.n_probe       = n_probe if n_probe is not None else max(20, m_steps // 2)
        self.baseline_type = baseline

        self.last_node_scores = None

    # ------------------------------------------------------------------ #
    #  工具                                                                #
    # ------------------------------------------------------------------ #

    def _baseline(self, x):
        if self.baseline_type == 'zero':
            return torch.zeros_like(x)
        return self.baseline_type.to(x.device)

    def _grad_at_alpha(self, x, baseline, alpha, edge_index, batch, pred_cls):
        """计算单个 α 处的梯度 g(α) = ∂F_{pred_cls}/∂x_interp，返回 [N,D]。"""
        x_interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        logits   = self.model(x_interp, edge_index, batch=batch)[0]
        self.model.zero_grad()
        logits[pred_cls].backward()
        return x_interp.grad.detach()   # [N, D]

    # ------------------------------------------------------------------ #
    #  Step 1：探针采样，估计 |g'(α)|                                       #
    # ------------------------------------------------------------------ #

    def _probe_gradients(self, x, baseline, edge_index, batch, pred_cls):
        """
        在 n_probe 个等间隔 α 处计算梯度，返回：
          alphas_probe : ndarray [n_probe]
          grads_probe  : List[Tensor [N,D]]，长度 n_probe
        """
        n  = self.n_probe
        alphas = np.linspace(0.0, 1.0, n + 1)[1:]   # 避免 α=0（全零处梯度可能退化）
        grads  = []
        for a in alphas:
            g = self._grad_at_alpha(x, baseline, float(a), edge_index, batch, pred_cls)
            grads.append(g)
        return alphas, grads

    # ------------------------------------------------------------------ #
    #  Step 2：计算最优 α 调度                                              #
    # ------------------------------------------------------------------ #

    def _optimal_alphas(self, alphas_probe, grads_probe):
        """
        根据探针梯度估计 |g'(α)|，用 CDF 逆变换采样 m_steps 个最优 α。

        |g'(αk)| ≈ ‖grads[k] − grads[k-1]‖₁ / Δα（逐特征 L1 均值）
        """
        n  = len(grads_probe)
        da = alphas_probe[1] - alphas_probe[0] if n > 1 else 1.0

        # 相邻梯度之差的 L1 范数，近似 |g'|（每段一个标量）
        density = np.zeros(n)
        for k in range(n):
            if k == 0:
                diff = grads_probe[1] - grads_probe[0] if n > 1 else grads_probe[0]
            else:
                diff = grads_probe[k] - grads_probe[k - 1]
            density[k] = diff.abs().mean().item()

        # 防止全零（梯度完全平坦）退化为均匀采样
        density = density + 1e-10

        # CDF（累计密度函数）
        cdf    = np.cumsum(density)
        cdf    = cdf / cdf[-1]   # 归一化到 [0,1]

        # 用 CDF 逆变换（quantile）生成 m_steps 个最优 α 值
        # 目标分位数：等间隔分布在 [0,1]
        quantiles    = np.linspace(0.0, 1.0, self.m_steps + 2)[1:-1]  # 避开端点
        opt_alphas_q = np.interp(quantiles, cdf, alphas_probe)

        # 加入端点保证完整路径覆盖
        opt_alphas = np.concatenate([[0.0], opt_alphas_q, [1.0]])
        opt_alphas = np.clip(opt_alphas, 0.0, 1.0)
        opt_alphas = np.sort(opt_alphas)
        return opt_alphas   # [m_steps+2] 个点

    # ------------------------------------------------------------------ #
    #  Step 3：梯形法则积分                                                 #
    # ------------------------------------------------------------------ #

    def _integrate(self, x, baseline, opt_alphas, edge_index, batch, pred_cls):
        """
        用梯形法则在最优 α 点处积分：
          ∫ g(α) dα ≈ Σk  [g(αk)+g(αk+1)]/2 · (αk+1 − αk)
        """
        # 计算所有最优 α 处的梯度
        grads = []
        for a in opt_alphas:
            g = self._grad_at_alpha(x, baseline, float(a), edge_index, batch, pred_cls)
            grads.append(g)

        # 梯形积分
        integral = torch.zeros_like(x)   # [N, D]
        for k in range(len(opt_alphas) - 1):
            da         = float(opt_alphas[k + 1] - opt_alphas[k])
            trap       = (grads[k] + grads[k + 1]) * 0.5
            integral   = integral + trap * da

        # 标准 IG 乘以路径差
        attribution = (x - baseline) * integral   # [N, D]
        return attribution

    # ------------------------------------------------------------------ #
    #  公开接口                                                             #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        x,
        edge_index,
        sparsity=0.5,
        num_classes=2,
        node_idx=0,
        max_nodes=None,
    ):
        self.model.eval()

        device    = x.device
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        batch     = torch.zeros(num_nodes, dtype=torch.long, device=device)
        baseline  = self._baseline(x)

        num_keep  = max(1, int(max_nodes if max_nodes is not None
                               else num_nodes * (1 - sparsity)))

        edge_masks         = []
        node_masks         = []
        signed_scores_list = []

        for cls in range(num_classes):
            # Step 1：探针采样
            alphas_probe, grads_probe = self._probe_gradients(
                x, baseline, edge_index, batch, cls
            )

            # Step 2：最优 α 调度
            opt_alphas = self._optimal_alphas(alphas_probe, grads_probe)

            # Step 3：最优采样点上的梯形积分
            attribution = self._integrate(
                x, baseline, opt_alphas, edge_index, batch, cls
            )   # [N, D]

            # 有符号节点分数（供 Efficiency Gap）
            signed_scores = attribution.sum(dim=-1)
            # L1 节点分数（供 top-k）
            abs_scores    = attribution.abs().sum(dim=-1)

            signed_scores_list.append(signed_scores)
            edge_masks.append(torch.zeros(num_edges, device=device))

            k    = min(num_keep, num_nodes)
            topk = abs_scores.topk(k).indices
            nm   = torch.zeros(num_nodes, dtype=torch.float32, device=device)
            nm[topk] = 1.0
            node_masks.append(nm)

        self.last_node_scores = signed_scores_list

        return edge_masks, node_masks