import random

import pytest


pytestmark = pytest.mark.lightweight_graph
SEED = 20260831


def test_edge_mask_to_node_mask_on_hand_computed_graph(source_on_path):
    import torch
    from apex.evaluation.fidelity import get_node_mask_from_edge_mask

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    edge_masks = [
        torch.tensor([0.1, 0.2, 0.9, 0.8, 0.3, 0.4]),
        torch.tensor([0.2, 0.1, 0.8, 0.9, 0.4, 0.3]),
    ]
    node_masks = get_node_mask_from_edge_mask(
        edge_masks,
        num_nodes=4,
        edge_index=edge_index,
        num_classes=2,
        sparsity=0.5,
    )
    assert len(node_masks) == 2
    assert [set(mask.nonzero(as_tuple=True)[0].tolist()) for mask in node_masks] == [
        {1, 2},
        {1, 2},
    ]


@pytest.mark.parametrize("important", [True, False])
def test_masked_in_out_extreme_masks_are_reproducible(
    source_on_path, tiny_graph, important
):
    import numpy as np
    import torch
    from apex.evaluation.fidelity import eval_related_pred

    class TinyEvaluationModel(torch.nn.Module):
        def forward(self, x, edge_index, batch=None):
            score = x[0, 0] + 0.25 * x[1, 1]
            return torch.stack((score, -score)).unsqueeze(0)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    x, edge_index, _ = tiny_graph
    value = 1.0 if important else 0.0
    node_masks = [torch.full((x.size(0),), value) for _ in range(2)]
    first = eval_related_pred(
        TinyEvaluationModel().cpu().eval(),
        x,
        edge_index,
        node_masks,
        pred_cls=0,
        device=torch.device("cpu"),
        num_samples=3,
    )
    torch.manual_seed(SEED)
    second = eval_related_pred(
        TinyEvaluationModel().cpu().eval(),
        x,
        edge_index,
        node_masks,
        pred_cls=0,
        device=torch.device("cpu"),
        num_samples=3,
    )
    assert first == second
    assert set(first) == {"ori", "masked_in", "masked_out"}
    assert all(np.isfinite(value) for value in first.values())
    if important:
        assert first["masked_in"] == pytest.approx(first["ori"], abs=1e-7, rel=1e-7)
    else:
        assert first["masked_out"] == pytest.approx(first["ori"], abs=1e-7, rel=1e-7)


def test_nfe_counter_register_reset_remove(source_on_path):
    import torch
    from apex.evaluation.nfe import NFECounter

    model = torch.nn.Linear(2, 2).cpu().eval()
    counter = NFECounter()
    assert counter.value == 0
    counter.register(model)
    model(torch.ones(1, 2))
    model(torch.ones(1, 2))
    assert counter.value == 2
    counter.reset()
    model(torch.ones(1, 2))
    assert counter.value == 1
    counter.remove()
    assert counter._hook_handle is None
    model(torch.ones(1, 2))
    assert counter.value == 1


def test_nfe_counter_double_register_is_idempotent(source_on_path):
    import torch
    from apex.evaluation.nfe import NFECounter

    model = torch.nn.Linear(2, 2).cpu().eval()
    counter = NFECounter()
    counter.register(model)
    counter.register(model)
    model(torch.ones(1, 2))
    assert counter.value == 1
    counter.remove()


def test_nfe_counter_remove_is_idempotent(source_on_path):
    import torch
    from apex.evaluation.nfe import NFECounter

    model = torch.nn.Linear(2, 2).cpu().eval()
    counter = NFECounter()
    counter.register(model)
    counter.remove()
    counter.remove()
    assert counter._hook_handle is None
    model(torch.ones(1, 2))
    assert counter.value == 0


def test_fixed_explanation_has_exact_stability(source_on_path, tiny_graph):
    import torch
    from apex.evaluation.stability import eval_stability, jaccard_similarity

    x, edge_index, _ = tiny_graph
    node_masks = [
        torch.tensor([1.0, 0.9, 0.0, 0.0, 0.0]),
        torch.tensor([0.8, 1.0, 0.0, 0.0, 0.0]),
    ]

    class FixedMaskExplainer:
        def __call__(self, x, edge_index, **kwargs):
            edge_masks = [torch.ones(edge_index.size(1)) for _ in range(2)]
            return edge_masks, [mask.clone() for mask in node_masks]

    random.seed(SEED)
    torch.manual_seed(SEED)
    stability = eval_stability(
        FixedMaskExplainer(),
        x,
        edge_index,
        node_masks,
        pred_cls=0,
        num_classes=2,
        num_nodes=x.size(0),
        sparsity=0.6,
        device=torch.device("cpu"),
        perturb_ratio=0.1,
        n_perturb=3,
        seed_base=SEED,
    )
    assert jaccard_similarity(frozenset({0, 1}), frozenset({0, 1})) == 1.0
    assert stability == 1.0
