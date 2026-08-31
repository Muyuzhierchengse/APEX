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
    from apex.models import gnn as models

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
    from apex.models import gnn as models

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


def test_shared_set_seed_reproduces_python_numpy_and_torch(source_on_path):
    import numpy as np
    import torch
    from apex.utils.reproducibility import set_seed

    set_seed(SEED)
    first = (random.random(), np.random.random(), torch.rand(3))
    set_seed(SEED)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], atol=0.0, rtol=0.0)


def test_compatible_state_dict_preserves_key_and_transpose_contract(source_on_path):
    from collections import OrderedDict

    import torch
    from apex.utils.checkpoints import compatible_state_dict

    state = OrderedDict(
        [
            ("conv1.weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
            ("convs.0.weight", torch.tensor([[5.0, 6.0], [7.0, 8.0]])),
            ("classifier.bias", torch.tensor([9.0, 10.0])),
        ]
    )
    converted = compatible_state_dict(state)
    assert list(converted) == [
        "conv1.lin.weight",
        "convs.0.lin.weight",
        "classifier.bias",
    ]
    torch.testing.assert_close(converted["conv1.lin.weight"], state["conv1.weight"].T)
    torch.testing.assert_close(
        converted["convs.0.lin.weight"], state["convs.0.weight"].T
    )
    torch.testing.assert_close(converted["classifier.bias"], state["classifier.bias"])
