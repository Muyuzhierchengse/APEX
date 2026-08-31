import json
import random
import sys
from pathlib import Path

import pytest


SEED = 20260831


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "lightweight_graph: CPU-only tests requiring the graph-learning dependencies; "
        "not executed while the current environment is incompatible.",
    )
    config.addinivalue_line(
        "markers",
        "pre_split: migration-stage check against the current worktree; skipped once "
        "HEAD moves away from the frozen baseline commit.",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def manifest(repo_root):
    return json.loads((repo_root / "tests" / "baseline_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def source_on_path(repo_root, monkeypatch):
    source = repo_root / "src"
    monkeypatch.syspath_prepend(str(source))
    return source


@pytest.fixture
def tiny_graph():
    """Return a five-node CPU graph without reading or downloading data."""
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    x = torch.tensor(
        [
            [1.0, -0.5, 0.25],
            [0.5, 0.75, -0.25],
            [-1.0, 0.25, 0.50],
            [0.25, -0.75, 1.00],
            [0.75, 0.50, -0.50],
        ],
        dtype=torch.float32,
        device="cpu",
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
        dtype=torch.long,
        device="cpu",
    )
    batch = torch.zeros(x.size(0), dtype=torch.long, device="cpu")
    return x, edge_index, batch
