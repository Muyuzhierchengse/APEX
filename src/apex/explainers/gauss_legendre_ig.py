import torch
import numpy as np


class GaussLegendreIG:
    """
    Gauss-Legendre Integrated Gradients explainer（图级分类）。

    Parameters
    ----------
    model        : 待解释的 GNN 模型，输出 (logits, ...)
    explain_graph : 保持接口一致，固定 True
    n_steps      : GL 积分点数，默认 20（等效精度远超 Riemann 50步）
    baseline     : 'zero'（全零，默认）
    """

    def __init__(self, model, explain_graph=True, n_steps=20, baseline='zero'):
        self.model         = model
        self.explain_graph = explain_graph
        self.n_steps       = n_steps
        self.baseline_type = baseline
        self.last_node_scores = None   # List[Tensor [N]]，有符号，供 Efficiency Gap

        # 预计算 GL 节点与权重，映射到 [0, 1]
        # leggauss 返回 [-1,1] 上的节点 t_k 和权重 w_k
        t, w = np.polynomial.legendre.leggauss(n_steps)
        self._alphas  = torch.tensor((t + 1) / 2, dtype=torch.float64)  # [n_steps]
        self._weights = torch.tensor(w / 2,        dtype=torch.float64)  # [n_steps]

    # ------------------------------------------------------------------ #
    #  内部工具                                                             #
    # ------------------------------------------------------------------ #

    def _get_baseline(self, x):
        if self.baseline_type == 'zero' or not isinstance(self.baseline_type, torch.Tensor):
            return torch.zeros_like(x)
        return self.baseline_type.to(x.device)

    def _gradient_at(self, x_interp, edge_index, pred_cls, batch):
        """在插值点 x_interp 处计算 ∂F_{pred_cls} / ∂x_interp，返回 [N, D]。"""
        x_interp = x_interp.detach().requires_grad_(True)
        logits   = self.model(x_interp, edge_index, batch=batch)[0]
        scalar   = logits[pred_cls]
        self.model.zero_grad()
        scalar.backward()
        return x_interp.grad.detach()   # [N, D]

    # ------------------------------------------------------------------ #
    #  核心：Gauss-Legendre 数值积分                                        #
    # ------------------------------------------------------------------ #

    def _gl_integrated_gradients(self, x, edge_index, baseline, pred_cls, batch):
        """
        用 n 点 GL 正交规则计算 IG 归因。

        Returns
        -------
        attribution : Tensor [N, D]
        """
        device  = x.device
        alphas  = self._alphas.to(dtype=x.dtype,  device=device)   # [n]
        weights = self._weights.to(dtype=x.dtype, device=device)   # [n]

        grad_accum = torch.zeros_like(x)   # [N, D]

        for alpha, weight in zip(alphas, weights):
            x_interp   = baseline + alpha * (x - baseline)   # 插值点
            grad       = self._gradient_at(x_interp, edge_index, pred_cls, batch)
            grad_accum = grad_accum + weight * grad           # 加权累加

        # IG = (x - x') ⊙ Σ w_k · grad_k
        attribution = (x - baseline) * grad_accum   # [N, D]
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
        baseline  = self._get_baseline(x)

        num_keep = max(1, int(max_nodes)) if max_nodes is not None \
                   else max(1, int(num_nodes * (1 - sparsity)))

        edge_masks         = []
        node_masks         = []
        signed_scores_list = []

        for cls in range(num_classes):
            attribution = self._gl_integrated_gradients(
                x, edge_index, baseline, cls, batch
            )   # [N, D]

            # 有符号：对特征维度求和（满足完备性公理 Σφ = f(x) − f(x')）
            signed_scores = attribution.sum(dim=-1)        # [N]
            # 无符号：L1 范数，用于 top-k 选节点
            abs_scores    = attribution.abs().sum(dim=-1)  # [N]

            signed_scores_list.append(signed_scores)

            # 伪 edge_mask（全零占位，GL-IG 无原生 edge 归因）
            edge_masks.append(torch.zeros(num_edges, device=device))

            # 二值 node_mask：L1 分数 top-k
            k    = min(num_keep, num_nodes)
            topk = abs_scores.topk(k).indices
            nm   = torch.zeros(num_nodes, dtype=torch.float32, device=device)
            nm[topk] = 1.0
            node_masks.append(nm)

        # 有符号归因供 Efficiency Gap 计算
        self.last_node_scores = signed_scores_list

        return edge_masks, node_masks