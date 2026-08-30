import torch
import numpy as np


class IntegratedGradients:
    def __init__(self, model, explain_graph=True, m_steps=50, baseline='zero', isabs=False):
        self.model         = model
        self.explain_graph = explain_graph
        self.m_steps       = m_steps
        self.baseline_type = baseline
        self.isabs         = isabs
        self.last_node_scores = None

    def _get_baseline(self, x):
        if self.baseline_type == 'zero' or not isinstance(self.baseline_type, torch.Tensor):
            return torch.zeros_like(x)
        return self.baseline_type.to(x.device)

    def _interpolate(self, x, baseline, alpha):
        return baseline + alpha * (x - baseline)

    def _gradients_at(self, x_interp, edge_index, pred_cls, batch, node_idx=None):
        x_interp = x_interp.detach().requires_grad_(True)

        out = self.model(x_interp, edge_index, batch=batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        if node_idx is None:
            scalar = logits[0, pred_cls]
        else:
            scalar = logits[node_idx, pred_cls]

        self.model.zero_grad()
        scalar.backward()

        grad = x_interp.grad.detach()
        return grad

    def _integrated_gradients_single_class(self, x, edge_index, baseline,
                                            pred_cls, batch, node_idx=None):
        m      = self.m_steps
        alphas = [k / m for k in range(1, m + 1)]

        grad_sum = torch.zeros_like(x)

        for alpha in alphas:
            x_interp = self._interpolate(x, baseline, alpha)
            grad     = self._gradients_at(
                x_interp, edge_index, pred_cls, batch,
                node_idx=node_idx,
            )
            grad_sum = grad_sum + grad

        avg_grad    = grad_sum / m
        attribution = (x - baseline) * avg_grad

        return attribution

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

        batch    = torch.zeros(num_nodes, dtype=torch.long, device=device)
        baseline = self._get_baseline(x)

        if max_nodes is not None:
            num_keep = max(1, int(max_nodes))
        else:
            num_keep = max(1, int(num_nodes * (1 - sparsity)))

        ig_node_idx = None if self.explain_graph else node_idx

        edge_masks         = []
        node_masks         = []
        signed_scores_list = []
        abs_scores_list    = []

        for cls in range(num_classes):
            attribution = self._integrated_gradients_single_class(
                x, edge_index, baseline, cls, batch,
                node_idx=ig_node_idx,
            )

            signed_scores = attribution.sum(dim=-1)
            abs_scores    = attribution.abs().sum(dim=-1)
            select_scores = abs_scores if self.isabs else signed_scores

            signed_scores_list.append(signed_scores)
            abs_scores_list.append(abs_scores)

            edge_masks.append(torch.zeros(num_edges, device=device))

            if not self.explain_graph:
                s_cpu = edge_index[0].cpu()
                d_cpu = edge_index[1].cpu()
                neighbors = set()
                neighbors.add(node_idx)
                for i in range(edge_index.size(1)):
                    u, v = s_cpu[i].item(), d_cpu[i].item()
                    if u == node_idx:
                        neighbors.add(v)
                    if v == node_idx:
                        neighbors.add(u)
                neighbors = sorted(neighbors)
                neighbor_tensor = torch.tensor(neighbors, dtype=torch.long, device=device)

                scores_sub = select_scores[neighbor_tensor]
                k_sub      = min(num_keep, len(neighbors))
                topk_local = scores_sub.topk(k_sub).indices
                topk_global = neighbor_tensor[topk_local]

                nm = torch.zeros(num_nodes, dtype=torch.float32, device=device)
                nm[topk_global] = 1.0
            else:
                k    = min(num_keep, num_nodes)
                topk = select_scores.topk(k).indices
                nm   = torch.zeros(num_nodes, dtype=torch.float32, device=device)
                nm[topk] = 1.0

            node_masks.append(nm)

        self.last_node_scores = signed_scores_list

        return edge_masks, node_masks