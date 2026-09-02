import ast
import importlib
import importlib.util
from pathlib import Path

import pytest
import tomli
from setuptools import find_packages


ACTIVE_SCRIPTS = [
    Path("scripts/evaluate.py"),
    Path("scripts/evaluate_nfe.py"),
    Path("scripts/evaluate_fidelity_only.py"),
    Path("scripts/compare_exact_fidelity.py"),
    Path("scripts/train.py"),
    Path("scripts/visualize_method_comparison.py"),
    Path("scripts/visualize_signed_attribution.py"),
]
DOCUMENTED_SCRIPTS = ACTIVE_SCRIPTS
IMPORT_TO_REQUIREMENT = {
    "captum": "captum",
    "dig": "dive-into-graphs",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "numpy": "numpy",
    "ogb": "ogb",
    "rdkit": "rdkit",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "torch_sparse": "torch-sparse",
    "tqdm": "tqdm",
}
STANDARD_LIBRARY = {
    "__future__",
    "argparse",
    "collections",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "functools",
    "io",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "random",
    "re",
    "sys",
    "textwrap",
    "time",
    "traceback",
    "typing",
    "warnings",
}


def _production_import_roots(repo_root):
    roots = [repo_root / "src/apex", repo_root / "scripts", repo_root / "experiments"]
    imported = set()
    for path in (path for root in roots for path in root.rglob("*.py")):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
    return imported


def _requirement_names(path):
    names = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(separator, 1)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _path_values(paths):
    return {
        name: getattr(paths, name)
        for name in [
            "PROJECT_ROOT",
            "DATA_DIR",
            "CHECKPOINT_DIR",
            "LOG_DIR",
            "OUTPUT_DIR",
            "FIGURE_DIR",
        ]
    }


def _expected_paths(root):
    return {
        "PROJECT_ROOT": root,
        "DATA_DIR": root / "data",
        "CHECKPOINT_DIR": root / "artifacts/checkpoints",
        "LOG_DIR": root / "outputs/logs",
        "OUTPUT_DIR": root / "outputs/results",
        "FIGURE_DIR": root / "outputs/figures",
    }


def test_source_checkout_is_detected_without_environment_override(
    repo_root, source_on_path, monkeypatch
):
    monkeypatch.delenv("APEX_PROJECT_ROOT", raising=False)
    import apex.utils.paths as paths

    importlib.reload(paths)
    assert _path_values(paths) == _expected_paths(repo_root)


def test_paths_are_independent_of_current_working_directory(
    repo_root, source_on_path, monkeypatch, tmp_path
):
    monkeypatch.delenv("APEX_PROJECT_ROOT", raising=False)
    import apex.utils.paths as paths

    importlib.reload(paths)
    monkeypatch.chdir(tmp_path)
    importlib.reload(paths)
    assert _path_values(paths) == _expected_paths(repo_root)


def test_environment_override_has_priority_and_controls_all_paths(
    repo_root, source_on_path, monkeypatch, tmp_path
):
    import apex.utils.paths as paths

    configured_root = tmp_path / "configured-root"
    monkeypatch.setenv("APEX_PROJECT_ROOT", str(configured_root))
    importlib.reload(paths)
    assert _path_values(paths) == _expected_paths(configured_root.resolve())
    assert paths.PROJECT_ROOT != repo_root

    monkeypatch.delenv("APEX_PROJECT_ROOT")
    importlib.reload(paths)


def test_empty_environment_override_is_rejected(source_on_path, monkeypatch):
    import apex.utils.paths as paths

    monkeypatch.setenv("APEX_PROJECT_ROOT", "   ")
    with pytest.raises(RuntimeError, match="APEX_PROJECT_ROOT"):
        importlib.reload(paths)


def test_unconfigured_noneditable_install_location_fails_without_fallback(
    repo_root, monkeypatch, tmp_path
):
    monkeypatch.delenv("APEX_PROJECT_ROOT", raising=False)
    installed_path = (
        tmp_path / "conda_env" / "Lib" / "site-packages" / "apex" / "utils" / "paths.py"
    )
    installed_path.parent.mkdir(parents=True)
    installed_path.write_bytes((repo_root / "src/apex/utils/paths.py").read_bytes())
    monkeypatch.chdir(repo_root)

    spec = importlib.util.spec_from_file_location("isolated_installed_paths", installed_path)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="APEX_PROJECT_ROOT"):
        spec.loader.exec_module(module)


def test_active_scripts_use_shared_paths_and_no_legacy_roots(repo_root):
    forbidden = ["./dataset", "./model/checkpoint", "./figures_signed", "'./output'"]
    for relative_path in ACTIVE_SCRIPTS:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "apex.utils.paths" in text
        assert all(value not in text for value in forbidden)


def test_requirements_cover_production_third_party_imports(repo_root):
    imported = _production_import_roots(repo_root)
    unknown = imported - STANDARD_LIBRARY - {"apex"} - set(IMPORT_TO_REQUIREMENT)
    assert not unknown, f"unclassified production imports: {sorted(unknown)}"
    requirements = _requirement_names(repo_root / "requirements.txt")
    required_for_imports = {
        requirement
        for imported_name, requirement in IMPORT_TO_REQUIREMENT.items()
        if imported_name in imported
    }
    assert required_for_imports <= requirements
    assert {"torch-scatter", "torch-sparse"} <= requirements


def test_pyproject_metadata_and_package_discovery(repo_root):
    with (repo_root / "pyproject.toml").open("rb") as stream:
        config = tomli.load(stream)
    assert config["project"]["name"] == "apex-gnn-explanations"
    assert config["project"]["version"] == "0.1.0"
    assert config["project"]["requires-python"] == ">=3.9"
    assert config["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "scripts" not in config["project"]
    discovery = config["tool"]["setuptools"]["packages"]["find"]
    assert config["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert discovery["where"] == ["src"]
    assert discovery["include"] == ["apex", "apex.*"]
    packages = find_packages(
        where=str(repo_root / "src"), include=tuple(discovery["include"])
    )
    assert packages
    assert all(package == "apex" or package.startswith("apex.") for package in packages)


def test_readme_documents_paths_contract_and_existing_scripts(repo_root):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "--no-build-isolation -e ." in readme
    assert '$env:APEX_PROJECT_ROOT = "E:\\workplace\\APEX"' in readme
    assert "pyproject.toml" in readme and "src/apex" in readme
    for relative_path in DOCUMENTED_SCRIPTS:
        assert relative_path.as_posix() in readme
        assert (repo_root / relative_path).is_file()
    for value in [
        "BA_shapes", "BBBP", "Graph-SST2", "BACE", "Mutagenicity",
        "GCN", "GCN_2l", "GIN", "GIN_2l", "PolyGIN",
    ]:
        assert value in readme
