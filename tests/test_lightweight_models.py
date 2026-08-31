import copy
import random

import pytest


pytestmark = pytest.mark.lightweight_graph
SEED = 20260831


def _set_cpu_seed(torch):
    random.seed(SEED)
    torch.manual_seed(SEED)


@pytest.mark.parametrize(
    ("model_name", "model_level", "expected_shape"),
    [
        ("GCN", "graph", (1, 2)),
        ("GCN", "node", (5, 2)),
        ("GIN", "graph", (1, 2)),
        ("GIN", "node", (5, 2)),
        ("PolyGIN", "graph", (1, 2)),
        ("PolyGIN", "node", (5, 2)),
    ],
)
def test_small_model_forward_contract(
    source_on_path, tiny_graph, model_name, model_level, expected_shape
):
    import torch
    from model import models

    _set_cpu_seed(torch)
    x, edge_index, batch = tiny_graph
    model_cls = getattr(models, model_name)
    model = model_cls(
        model_level=model_level,
        dim_node=3,
        dim_hidden=4,
        num_classes=2,
    ).cpu().eval()
    with torch.no_grad():
        output = model(x, edge_index, batch)
    assert output.device.type == "cpu"
    assert tuple(output.shape) == expected_shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("model_name", ["GCN", "GIN", "PolyGIN"])
def test_state_dict_memory_round_trip(source_on_path, tiny_graph, model_name):
    import torch
    from model import models

    _set_cpu_seed(torch)
    x, edge_index, batch = tiny_graph
    model_cls = getattr(models, model_name)
    original = model_cls("graph", 3, 4, 2).cpu().eval()
    state = copy.deepcopy(original.state_dict())
    restored = model_cls("graph", 3, 4, 2).cpu().eval()
    restored.load_state_dict(state, strict=True)

    with torch.no_grad():
        expected = original(x, edge_index, batch)
        actual = restored(x, edge_index, batch)
    assert original.state_dict().keys() == restored.state_dict().keys()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
