import json
import re


def test_quickstart_notebook_is_clean_and_uses_public_api(repo_root):
    path = repo_root / "examples/apex_quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 5
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)

    executable = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "from apex import APEX" in executable
    assert ".explain(graph" in executable
    for forbidden in [
        r"\bcuda\b",
        r"torch\.load\s*\(",
        r"torch\.save\s*\(",
        r"\boptimizer\b",
        r"\.train\s*\(",
        r"https?://",
    ]:
        assert not re.search(forbidden, executable, flags=re.IGNORECASE)
