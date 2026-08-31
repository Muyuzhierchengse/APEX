import torch
import numpy as np
from torch_geometric.utils import add_self_loops


class SimpsonIG:
    def __init__(self, model, explain_graph=True, m_steps=51, baseline='zero'):
        if m_steps < 3:
            raise ValueError("m_steps must be >= 3 for Simpson's rule.")
        if m_steps % 2 == 0:
            import warnings
            m_steps += 1
            warnings.warn(
                f"Simpson's rule requires an odd number of points. "
                f"m_steps incremented to {m_steps}.",
                UserWarning,
            )
        self.model             = model
        self.explain_graph     = explain_graph
        self.m_steps           = m_steps
        self.baseline_mode     = baseline
        self.last_node_scores  = None

    @staticmethod
    def _simpson_weights(m, device):
        w = torch.ones(m, device=device)
        w[1:-1:2] = 4.0
        w[2:-2:2] = 2.0
        w = w / (3.0 * (m - 1))
        return w

    def __call__(self, x, edge_index,
                 sparsity=0, num_classes=2, node_idx=0, max_nodes=None):
        device = x.device
        N      = x.size(0)
        batch  = torch.zeros(N, dtype=torch.long, device=device)

        if self.baseline_mode == 'mean':
            x_base = x.mean(dim=0, keepdim=True).expand_as(x)
        else:
            x_base = torch.zeros_like(x)

        delta = x - x_base

        m       = self.m_steps
        alphas  = torch.linspace(0.0, 1.0, m, device=device)
        weights = self._simpson_weights(m, device)

        ig_node_idx = None if self.explain_graph else node_idx

        node_scores = [torch.zeros(N, device=device) for _ in range(num_classes)]

        self.model.eval()

        for k, (alpha, w) in enumerate(zip(alphas, weights)):
            for cls in range(num_classes):
                x_interp = (x_base + alpha * delta).detach().requires_grad_(True)

                out    = self.model(x_interp, edge_index, batch=batch)
                logits = out[0] if isinstance(out, (tuple, list)) else out

                if ig_node_idx is None:
                    scalar = logits[cls]
                else:
                    scalar = logits[ig_node_idx, cls]

                grad = torch.autograd.grad(
                    scalar, x_interp,
                    create_graph=False,
                )[0]
                node_scores[cls] = node_scores[cls] + w * (delta * grad).sum(dim=-1)

        self.last_node_scores = node_scores

        node_scores_abs = [s.abs() for s in node_scores]

        ei_sl, _ = add_self_loops(edge_index, num_nodes=N)
        src, dst  = ei_sl[0], ei_sl[1]

        edge_masks = []
        for cls in range(num_classes):
            ns = node_scores_abs[cls]
            em = (ns[src] + ns[dst]) / 2.0
            edge_masks.append(em)

        num_keep   = max(1, int(N * (1 - sparsity)))
        node_masks = []
        for cls in range(num_classes):
            nm   = torch.zeros(N, dtype=torch.float32, device=device)
            topk = node_scores_abs[cls].topk(min(num_keep, N)).indices
            nm[topk] = 1.0
            node_masks.append(nm)

        return edge_masks, node_masks