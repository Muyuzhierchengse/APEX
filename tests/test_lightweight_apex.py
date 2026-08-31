import random

import pytest


pytestmark = pytest.mark.lightweight_graph
SEED = 20260831


def _build_analytic_polynomial_model(torch):
    class AnalyticPolynomialModel(torch.nn.Module):
        """A two-class degree-two model with an exact zero-baseline solution."""

        def forward(self, x, edge_index, batch=None):
            class_zero = (x[:, 0].square() + 2.0 * x[:, 1]).sum()
            class_one = (-0.5 * x[:, 0].square() + x[:, 2]).sum()
            return torch.stack((class_zero, class_one)).unsqueeze(0)

    return AnalyticPolynomialModel().cpu().eval()


def test_apex_analytic_attribution_contract(source_on_path, tiny_graph):
    import numpy as np
    import torch
    from method.explainpoly import PolyGINExplainer

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    x, edge_index, _ = tiny_graph
    model = _build_analytic_polynomial_model(torch)
    explainer = PolyGINExplainer(model, explain_graph=True, L=2)

    edge_masks, node_masks = explainer(
        x,
        edge_index,
        sparsity=0.5,
        num_classes=2,
    )
    expected = [
        x[:, 0].square() + 2.0 * x[:, 1],
        -0.5 * x[:, 0].square() + x[:, 2],
    ]

    assert len(edge_masks) == len(node_masks) == len(explainer.last_node_scores) == 2
    for class_index in range(2):
        assert tuple(edge_masks[class_index].shape) == (edge_index.size(1),)
        assert tuple(node_masks[class_index].shape) == (x.size(0),)
        assert tuple(explainer.last_node_scores[class_index].shape) == (x.size(0),)
        assert torch.isfinite(edge_masks[class_index]).all()
        assert torch.isfinite(explainer.last_node_scores[class_index]).all()
        torch.testing.assert_close(
            explainer.last_node_scores[class_index],
            expected[class_index],
            atol=1e-5,
            rtol=1e-4,
        )

    with torch.no_grad():
        delta = model(x, edge_index)[0] - model(torch.zeros_like(x), edge_index)[0]
    attribution_sums = torch.stack([scores.sum() for scores in explainer.last_node_scores])
    torch.testing.assert_close(attribution_sums, delta, atol=1e-5, rtol=1e-4)
    assert (explainer.last_node_scores[0] > 0).any()
    assert (explainer.last_node_scores[0] < 0).any()
