import ast
import re
import subprocess


NO_EXTENSION_SOURCES = {
    "GraphEXT/Source/method/explainpolyfuben",
    "GraphEXT/Source/method/explainpolynoood",
}
APEX_LAYOUT_BASE_COMMIT = "ec82c36d50327af3738d663d36b2589dedd666c4"
MOVE_MAP = {
    "GraphEXT/Source/model/models.py": "src/apex/models/gnn.py",
    "GraphEXT/Source/dataset/data.py": "src/apex/data/loaders.py",
    "GraphEXT/Source/dataset/datafuben.py": "experiments/legacy/data_loaders_extended.py",
    "GraphEXT/Source/method/evaluation.py": "src/apex/evaluation/fidelity.py",
    "GraphEXT/Source/method/evaluationfuben.py": "experiments/legacy/evaluation_stability_candidate.py",
    "GraphEXT/Source/method/evaluationnoood.py": "experiments/variants/evaluation_marginal_ood.py",
    "GraphEXT/Source/method/explainpoly.py": "src/apex/explainers/poly_gin.py",
    "GraphEXT/Source/method/subexplainer.py": "src/apex/explainers/algebraic_poly_gin.py",
    "GraphEXT/Source/method/ig.py": "src/apex/explainers/integrated_gradients.py",
    "GraphEXT/Source/method/glig.py": "src/apex/explainers/gauss_legendre_ig.py",
    "GraphEXT/Source/method/tig.py": "src/apex/explainers/trapezoidal_ig.py",
    "GraphEXT/Source/method/sig.py": "src/apex/explainers/simpson_ig.py",
    "GraphEXT/Source/method/roig.py": "src/apex/explainers/adaptive_riemann_ig.py",
    "GraphEXT/Source/method/flowx.py": "src/apex/explainers/flowx.py",
    "GraphEXT/Source/method/gnnexplainer.py": "src/apex/explainers/gnnexplainer.py",
    "GraphEXT/Source/method/gradcam.py": "src/apex/explainers/gradcam.py",
    "GraphEXT/Source/method/pgexplainer.py": "src/apex/explainers/pgexplainer.py",
    "GraphEXT/Source/method/explainpolyfuben": "experiments/variants/explainpoly_aumann_shapley.py",
    "GraphEXT/Source/method/explainpolynoood": "experiments/variants/explainpoly_expected_ig.py",
    "GraphEXT/Source/method/pexplain.py": "experiments/variants/polygnn_shapley.py",
    "GraphEXT/Source/main.py": "scripts/evaluate.py",
    "GraphEXT/Source/mainNFE.py": "scripts/evaluate_nfe.py",
    "GraphEXT/Source/mainonlyf.py": "scripts/evaluate_fidelity_only.py",
    "GraphEXT/Source/main_fidelity.py": "scripts/compare_exact_fidelity.py",
    "GraphEXT/Source/mainkeshihua.py": "scripts/visualize_method_comparison.py",
    "GraphEXT/Source/visualize_signed_attribution.py": "scripts/visualize_signed_attribution.py",
    "GraphEXT/Source/model_train.py": "scripts/train.py",
    "GraphEXT/Source/mainfuben.py": "experiments/legacy/evaluate_copy.py",
    "GraphEXT/Source/model_trainfuben.py": "experiments/legacy/train_imbalanced.py",
}


class _RemoveImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def _ast_without_imports(content, filename):
    tree = ast.parse(content, filename=filename)
    tree = _RemoveImports().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def _entry(manifest, path):
    return next(item for item in manifest["files"] if item["path"] == path)


def _top_level_classes(path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _baseline_blob(repo_root, baseline_commit, path):
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            "show",
            f"{baseline_commit}:{path}",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _current_python_sources(repo_root):
    roots = [repo_root / "src/apex", repo_root / "scripts", repo_root / "experiments"]
    return sorted(path for root in roots for path in root.rglob("*.py"))


def test_frozen_baseline_source_files_compile_without_importing(repo_root, manifest):
    source_entries = [
        item for item in manifest["files"]
        if item["type"] in {"python", "python-no-extension"}
    ]
    assert {item["path"] for item in source_entries if item["type"] == "python-no-extension"} == NO_EXTENSION_SOURCES
    for item in source_entries:
        content = _baseline_blob(repo_root, manifest["baseline_commit"], item["path"])
        compile(content, item["path"], "exec")


def test_current_apex_source_files_compile_without_importing(repo_root):
    sources = _current_python_sources(repo_root)
    assert sources
    for path in sources:
        compile(path.read_bytes(), str(path), "exec")


def test_required_top_level_classes_exist(repo_root):
    models = _top_level_classes(repo_root / "src/apex/models/gnn.py")
    assert {"GCN", "GIN", "PolyGIN"} <= models
    assert "PolyGINExplainer" in _top_level_classes(
        repo_root / "src/apex/explainers/poly_gin.py"
    )
    assert "PolyGINExplainer" in _top_level_classes(
        repo_root / "src/apex/explainers/algebraic_poly_gin.py"
    )
    assert "NFECounter" in _top_level_classes(repo_root / "scripts/evaluate_nfe.py")


def test_all_layout_moves_preserve_non_import_ast(repo_root):
    assert len(MOVE_MAP) == 29
    for old_path, new_path in MOVE_MAP.items():
        old_content = _baseline_blob(repo_root, APEX_LAYOUT_BASE_COMMIT, old_path)
        current_path = repo_root / new_path
        assert current_path.is_file()
        assert _ast_without_imports(old_content, old_path) == _ast_without_imports(
            current_path.read_bytes(), new_path
        ), f"non-import AST changed for {old_path} -> {new_path}"
    assert not (repo_root / "GraphEXT/Source").exists()


def test_current_apex_entries_have_no_fsx_or_graphext_activity(repo_root):
    forbidden_modules = {"method.fsx", "method.graphext"}
    for relative_path in ["scripts/evaluate.py", "experiments/legacy/evaluate_copy.py"]:
        path = repo_root / relative_path
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        imported_modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }
        assert imported_modules.isdisjoint(forbidden_modules)
        assert not re.search(r"['\"](?:FSX|GraphEXT)['\"]", text)


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
            "scripts/evaluate.py",
            "scripts/compare_exact_fidelity.py",
            "scripts/visualize_signed_attribution.py",
        ]
    )
    assert re.search(r"\b(?:GCN|GIN|PolyGIN)_3l\b", text)
    assert "legacy_3l_model_names" in ids


def test_missing_eval_stability_is_recorded(repo_root, manifest):
    ids = {item["id"] for item in manifest["known_baseline_issues"]}
    main_tree = ast.parse((repo_root / "scripts/evaluate.py").read_bytes())
    evaluation_tree = ast.parse((repo_root / "src/apex/evaluation/fidelity.py").read_bytes())
    imported = {
        alias.name
        for node in main_tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "apex.evaluation.fidelity"
        for alias in node.names
    }
    defined = {
        node.name for node in evaluation_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "eval_stability" in imported
    assert "eval_stability" not in defined
    assert "missing_eval_stability" in ids


def test_current_sources_have_no_legacy_internal_imports(repo_root):
    forbidden_exact = {"model.models", "dataset.data"}
    for path in _current_python_sources(repo_root):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
        assert not forbidden_exact.intersection(imported_modules), path
        assert not any(
            module == "method" or module.startswith("method.")
            for module in imported_modules
        ), path


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
