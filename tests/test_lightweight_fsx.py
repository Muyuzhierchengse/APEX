import random

import pytest


pytestmark = pytest.mark.lightweight_graph
SEED = 20260831


def _build_tiny_model(torch):
    class TinyMessageLayer(torch.nn.Module):
        def forward(self, x, edge_index):
            output = torch.zeros_like(x)
            output.index_add_(0, edge_index[1], x[edge_index[0]])
            return output

    class TinyGraphModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TinyMessageLayer()
            self.head = torch.nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(
                    torch.tensor(
                        [[0.5, -0.25, 0.75], [-0.5, 0.5, 0.25]],
                        dtype=torch.float32,
                    )
                )

        def forward(self, x, edge_index, batch=None):
            hidden = self.conv1(x, edge_index)
            return self.head(hidden.sum(dim=0, keepdim=True))

    return TinyGraphModel().cpu().eval()


def _set_seed(torch, np):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def test_fsx_tiny_graph_structure_and_repeatability(source_on_path, tiny_graph):
    import numpy as np
    import torch
    from method.fsx import FSX

    x, edge_index, _ = tiny_graph
    model = _build_tiny_model(torch)
    explainer = FSX(model, explain_graph=True)

    _set_seed(torch, np)
    first = explainer(
        x,
        edge_index,
        sparsity=0.5,
        num_classes=2,
        max_nodes=2,
    )
    _set_seed(torch, np)
    second = explainer(
        x,
        edge_index,
        sparsity=0.5,
        num_classes=2,
        max_nodes=2,
    )

    assert set(first) == {0, 1}
    assert set(second) == {0, 1}
    for class_index in range(2):
        assert tuple(first[class_index].shape) == (edge_index.size(1),)
        assert first[class_index].device.type == "cpu"
        assert torch.isfinite(first[class_index]).all()
        torch.testing.assert_close(
            first[class_index], second[class_index], atol=0.0, rtol=0.0
        )
