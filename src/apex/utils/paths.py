"""Explicit project-root paths used by APEX research scripts."""

import os
from pathlib import Path


_PROJECT_ROOT_ENV = "APEX_PROJECT_ROOT"


def _is_source_checkout(candidate):
    return (candidate / "pyproject.toml").is_file() and (
        candidate / "src" / "apex"
    ).is_dir()


def _resolve_project_root():
    configured_root = os.environ.get(_PROJECT_ROOT_ENV)
    if configured_root is not None:
        if not configured_root.strip():
            raise RuntimeError(
                "APEX_PROJECT_ROOT is empty; set it to the APEX project root."
            )
        return Path(configured_root).expanduser().resolve()

    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if _is_source_checkout(candidate):
            return candidate

    raise RuntimeError(
        "Cannot locate an APEX source checkout from the installed package. "
        "Set APEX_PROJECT_ROOT to the project root containing pyproject.toml "
        "and src/apex before using APEX research paths."
    )


PROJECT_ROOT = _resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
