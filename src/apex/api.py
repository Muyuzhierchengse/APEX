"""Small public API for explaining a single graph with APEX."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from apex.explainers.poly_gin import PolyGINExplainer


@dataclass(frozen=True)
class GraphExplanation:
    """Named, CPU-backed outputs for one graph explanation."""

    target_class: int
    predicted_class: int
    logits: Tensor
    baseline_logits: Tensor
    node_scores: Tensor
    node_mask: Tensor
    edge_scores: Tensor
    completeness_gap: float
    quadrature_points: int

    def top_nodes(self, k: int = 5, *, by_absolute_score: bool = True) -> Tensor:
        """Return the indices of the ``k`` most influential nodes."""
        if k < 1:
            raise ValueError("k must be at least 1")
        values = self.node_scores.abs() if by_absolute_score else self.node_scores
        return torch.topk(values, k=min(k, values.numel())).indices


class APEX:
    """High-level, single-graph interface to the APEX attribution engine.

    Exactness requires a compatible polynomial model and the correct number of
    polynomial transformation blocks in ``depth``. The current implementation
    uses the all-zero feature tensor as the attribution baseline.
    """

    def __init__(self, model: nn.Module, *, depth: int = 4,
                 device: Optional[torch.device] = None):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.model = model
        self.depth = depth
        self.device = torch.device(device) if device is not None else None
        self._engine = PolyGINExplainer(model, explain_graph=True, L=depth)

    def _resolve_device(self, x: Tensor) -> torch.device:
        if self.device is not None:
            return self.device
        parameter = next(self.model.parameters(), None)
        return parameter.device if parameter is not None else x.device

    def _forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        output = self.model(x, edge_index, batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 1:
            output = output.unsqueeze(0)
        if output.ndim != 2 or output.size(0) != 1:
            raise ValueError("APEX.explain expects a single graph and [1, C] logits")
        return output

    def explain(self, graph, *, target: Optional[int] = None,
                sparsity: float = 0.5) -> GraphExplanation:
        """Explain ``graph`` and return signed node and normalized edge scores.

        ``graph`` must expose ``x`` and ``edge_index`` tensors. When ``target``
        is omitted, the model's predicted class is explained.
        """
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must satisfy 0 <= sparsity < 1")
        if not hasattr(graph, "x") or not hasattr(graph, "edge_index"):
            raise TypeError("graph must expose x and edge_index tensors")

        device = self._resolve_device(graph.x)
        self.model.to(device)
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                logits = self._forward(x, edge_index)
                baseline_logits = self._forward(torch.zeros_like(x), edge_index)

            predicted_class = int(logits[0].argmax().item())
            target_class = predicted_class if target is None else int(target)
            num_classes = logits.size(1)
            if not 0 <= target_class < num_classes:
                raise ValueError(
                    f"target must be in [0, {num_classes - 1}], got {target_class}"
                )

            edge_masks, node_masks = self._engine(
                x,
                edge_index,
                sparsity=sparsity,
                num_classes=num_classes,
            )
            if self._engine.last_node_scores is None:
                raise RuntimeError("APEX attribution engine returned no node scores")

            node_scores = self._engine.last_node_scores[target_class]
            target_delta = logits[0, target_class] - baseline_logits[0, target_class]
            completeness_gap = float((node_scores.sum() - target_delta).abs().item())

            return GraphExplanation(
                target_class=target_class,
                predicted_class=predicted_class,
                logits=logits[0].detach().cpu().clone(),
                baseline_logits=baseline_logits[0].detach().cpu().clone(),
                node_scores=node_scores.detach().cpu().clone(),
                node_mask=node_masks[target_class].detach().cpu().clone(),
                edge_scores=edge_masks[target_class].detach().cpu().clone(),
                completeness_gap=completeness_gap,
                quadrature_points=self._engine.m,
            )
        finally:
            self.model.train(was_training)
