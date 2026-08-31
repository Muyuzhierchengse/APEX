import hashlib
import subprocess

import pytest


OWNERSHIPS = {"FSX", "APEX", "SHARED", "EXPERIMENT", "REFERENCE"}
TARGETS = {"FSX", "APEX", "BOTH", "待定"}
SOURCE_TYPES = {"python", "python-no-extension"}
MIXED_FILES = {
    "GraphEXT/Source/model/models.py",
    "GraphEXT/Source/main.py",
    "GraphEXT/Source/model_train.py",
    "GraphEXT/Source/dataset/data.py",
}


def _git(repo_root, *args):
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            *args,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _baseline_files(repo_root, baseline_commit):
    output = _git(repo_root, "ls-tree", "-r", "--name-only", baseline_commit)
    return {line.strip() for line in output.splitlines() if line.strip()}


def _head_commit(repo_root):
    return _git(repo_root, "rev-parse", "HEAD").strip()


def test_manifest_exactly_covers_baseline(repo_root, manifest):
    baseline_files = _baseline_files(repo_root, manifest["baseline_commit"])
    recorded = {entry["path"] for entry in manifest["files"]}
    assert manifest["baseline_commit"] == "c4c5277574c04b91fe62a413df3f2dd23ad94b5c"
    assert manifest["tracked_file_count"] == 44
    assert len(manifest["files"]) == 44
    assert len(recorded) == 44
    assert len(baseline_files) == 44
    assert recorded == baseline_files
    assert not any(path.startswith("tests/") for path in baseline_files)


def test_manifest_paths_exist_in_baseline_commit(repo_root, manifest):
    baseline_files = _baseline_files(repo_root, manifest["baseline_commit"])
    missing = [entry["path"] for entry in manifest["files"] if entry["path"] not in baseline_files]
    assert missing == []


def test_all_python_sources_are_classified(manifest):
    sources = [entry for entry in manifest["files"] if entry["type"] in SOURCE_TYPES]
    assert sum(entry["type"] == "python" for entry in sources) == 29
    assert sum(entry["type"] == "python-no-extension" for entry in sources) == 2
    assert all(entry["ownership"] in OWNERSHIPS for entry in sources)


def test_bytecode_is_not_source(manifest):
    bytecode = [entry for entry in manifest["files"] if entry["type"] == "python-bytecode"]
    assert len(bytecode) == 8
    assert all(entry["disposition"] == "record_derived_cache" for entry in bytecode)
    assert not any(entry["type"] in SOURCE_TYPES for entry in bytecode)


def test_controlled_enums(manifest):
    assert {entry["ownership"] for entry in manifest["files"]} <= OWNERSHIPS
    assert {entry["target_repository"] for entry in manifest["files"]} <= TARGETS
    assert {entry["ownership"] for entry in manifest["files"]} == OWNERSHIPS


def test_mixed_files_have_symbol_maps(manifest):
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    for path in MIXED_FILES:
        symbols = by_path[path].get("symbol_mappings")
        assert symbols, f"missing symbol map for {path}"
        assert all(item["ownership"] in OWNERSHIPS for item in symbols)
        assert all(item["target_repository"] in TARGETS for item in symbols)


def test_manifest_records_immutable_file_identity(manifest):
    for entry in manifest["files"]:
        assert len(entry["git_blob_id"]) == 40
        assert len(entry["sha256"]) == 64
        assert entry["size"] >= 0


@pytest.mark.pre_split
def test_pre_split_worktree_hashes_match_manifest(repo_root, manifest):
    if _head_commit(repo_root) != manifest["baseline_commit"]:
        pytest.skip("pre-split worktree check only applies while HEAD is the baseline commit")

    mismatches = []
    for entry in manifest["files"]:
        path = repo_root / entry["path"]
        if not path.is_file():
            mismatches.append(entry["path"])
            continue
        content = path.read_bytes()
        if (
            len(content) != entry["size"]
            or hashlib.sha256(content).hexdigest() != entry["sha256"]
        ):
            mismatches.append(entry["path"])
    assert mismatches == []


def test_recorded_blob_ids_match_baseline_commit(repo_root, manifest):
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            "ls-tree",
            "-r",
            "--full-tree",
            manifest["baseline_commit"],
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    blobs = {}
    for line in result.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        _, object_type, object_id = metadata.split()
        assert object_type == "blob"
        blobs[path] = object_id
    assert {
        entry["path"]: entry["git_blob_id"] for entry in manifest["files"]
    } == blobs
