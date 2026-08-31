import ast
import re


NO_EXTENSION_SOURCES = {
    "GraphEXT/Source/method/explainpolyfuben",
    "GraphEXT/Source/method/explainpolynoood",
}


def _entry(manifest, path):
    return next(item for item in manifest["files"] if item["path"] == path)


def _top_level_classes(path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def test_all_source_files_compile_without_importing(repo_root, manifest):
    source_entries = [
        item for item in manifest["files"]
        if item["type"] in {"python", "python-no-extension"}
    ]
    assert {item["path"] for item in source_entries if item["type"] == "python-no-extension"} == NO_EXTENSION_SOURCES
    for item in source_entries:
        path = repo_root / item["path"]
        compile(path.read_bytes(), str(path), "exec")


def test_required_top_level_classes_exist(repo_root):
    models = _top_level_classes(repo_root / "GraphEXT/Source/model/models.py")
    assert {"GCN", "GIN", "PolyGIN"} <= models
    assert "FSX" in _top_level_classes(repo_root / "GraphEXT/Source/method/fsx.py")
    assert "PolyGINExplainer" in _top_level_classes(
        repo_root / "GraphEXT/Source/method/explainpoly.py"
    )
    assert "NFECounter" in _top_level_classes(repo_root / "GraphEXT/Source/mainNFE.py")


def test_apex_files_are_not_fsx_only(manifest):
    apex_paths = {
        "GraphEXT/Source/method/explainpoly.py",
        "GraphEXT/Source/method/explainpolyfuben",
        "GraphEXT/Source/method/explainpolynoood",
        "GraphEXT/Source/method/pexplain.py",
        "GraphEXT/Source/method/subexplainer.py",
    }
    for path in apex_paths:
        entry = _entry(manifest, path)
        assert entry["target_repository"] != "FSX"


def test_fsx_core_is_not_apex_only(manifest):
    entry = _entry(manifest, "GraphEXT/Source/method/fsx.py")
    assert entry["ownership"] == "FSX"
    assert entry["target_repository"] == "FSX"


def test_known_legacy_model_name_break_is_recorded(repo_root, manifest):
    ids = {item["id"] for item in manifest["known_baseline_issues"]}
    text = "\n".join(
        (repo_root / path).read_text(encoding="utf-8")
        for path in [
            "GraphEXT/Source/main.py",
            "GraphEXT/Source/main_fidelity.py",
            "GraphEXT/Source/visualize_signed_attribution.py",
        ]
    )
    assert re.search(r"\b(?:GCN|GIN|PolyGIN)_3l\b", text)
    assert "legacy_3l_model_names" in ids


def test_missing_eval_stability_is_recorded(repo_root, manifest):
    ids = {item["id"] for item in manifest["known_baseline_issues"]}
    main_tree = ast.parse((repo_root / "GraphEXT/Source/main.py").read_bytes())
    evaluation_tree = ast.parse((repo_root / "GraphEXT/Source/method/evaluation.py").read_bytes())
    imported = {
        alias.name
        for node in main_tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "method.evaluation"
        for alias in node.names
    }
    defined = {
        node.name for node in evaluation_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "eval_stability" in imported
    assert "eval_stability" not in defined
    assert "missing_eval_stability" in ids


def test_required_known_issue_set_is_present(manifest):
    ids = {item["id"] for item in manifest["known_baseline_issues"]}
    assert {
        "missing_eval_stability",
        "legacy_3l_model_names",
        "dataset_choice_loader_mismatch",
        "flowx_missing_sqrt_import",
        "mainonlyf_bashapes_path",
        "orphan_graph_twitter_checkpoint",
    } <= ids


def test_lightweight_graph_tests_are_marked_and_avoid_forbidden_operations(repo_root):
    paths = [
        repo_root / "tests/test_lightweight_models.py",
        repo_root / "tests/test_lightweight_fsx.py",
        repo_root / "tests/test_lightweight_apex.py",
        repo_root / "tests/test_lightweight_evaluation.py",
    ]
    forbidden = [
        r"\bcuda\b",
        r"torch\.load\s*\(",
        r"torch\.save\s*\(",
        r"\.backward\s*\(",
        r"\boptimizer\b",
        r"\.train\s*\(",
        r"\.pkl\b",
        r"\.pt\b",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "pytestmark = pytest.mark.lightweight_graph" in text
        for pattern in forbidden:
            assert not re.search(pattern, text, flags=re.IGNORECASE), (
                f"forbidden lightweight-test operation {pattern!r} in {path.name}"
            )
