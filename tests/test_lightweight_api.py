from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.lightweight_graph


def test_high_level_api_returns_named_complete_explanation(source_on_path, tiny_graph):
    import torch
    from apex import APEX, GraphExplanation

    class AnalyticPolynomialModel(torch.nn.Module):
        def forward(self, x, edge_index, batch=None):
            class_zero = (x[:, 0].square() + 2.0 * x[:, 1]).sum()
            class_one = (-0.5 * x[:, 0].square() + x[:, 2]).sum()
            return torch.stack((class_zero, class_one)).unsqueeze(0)

    x, edge_index, _ = tiny_graph
    graph = SimpleNamespace(x=x, edge_index=edge_index)
    model = AnalyticPolynomialModel().cpu().eval()

    explanation = APEX(model, depth=2).explain(graph, sparsity=0.5)

    assert isinstance(explanation, GraphExplanation)
    assert explanation.target_class == explanation.predicted_class
    assert explanation.quadrature_points == 2
    assert explanation.node_scores.shape == (x.size(0),)
    assert explanation.node_mask.shape == (x.size(0),)
    assert explanation.edge_scores.shape == (edge_index.size(1),)
    assert explanation.top_nodes(2).shape == (2,)
    assert explanation.completeness_gap < 1e-5
    assert torch.isfinite(explanation.node_scores).all()
    assert model.training is False


def test_high_level_api_runs_with_repository_polygin(source_on_path, tiny_graph):
    import torch
    from apex import APEX
    from apex.models.gnn import PolyGIN

    torch.manual_seed(20260903)
    x, edge_index, _ = tiny_graph
    graph = SimpleNamespace(x=x, edge_index=edge_index)
    model = PolyGIN(
        model_level="graph",
        dim_node=x.size(1),
        dim_hidden=4,
        num_classes=2,
    ).cpu().eval()

    explanation = APEX(model, depth=4).explain(graph)

    assert explanation.quadrature_points == 8
    assert explanation.node_scores.shape == (x.size(0),)
    assert explanation.edge_scores.shape == (edge_index.size(1),)
    assert explanation.completeness_gap < 1e-3
    assert torch.isfinite(explanation.node_scores).all()
