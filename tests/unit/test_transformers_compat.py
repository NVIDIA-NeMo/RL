"""Tests for Transformers runtime compatibility patches."""

from pathlib import Path

import pytest
import transformers
import transformers.dynamic_module_utils as dynamic_module_utils

from nemo_rl.transformers_compat import (
    _SYMLINK_CACHE_PATCH_MARKER,
    _compute_local_source_files_hash_with_symlink_fix,
    patch_transformers_dynamic_module_symlink_cache,
)

_SOURCE_FILES = {
    "configuration_test.py": "from .dependency import VALUE\n",
    "dependency.py": "from .leaf import LEAF\nVALUE = LEAF\n",
    "leaf.py": "LEAF = 1\n",
}


def _create_source_tree(root: Path, *, symlinked: bool) -> Path:
    if not symlinked:
        root.mkdir()
        for filename, content in _SOURCE_FILES.items():
            (root / filename).write_text(content)
        return root

    blobs = root / "blobs"
    snapshot = root / "snapshots" / "revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for index, (filename, content) in enumerate(_SOURCE_FILES.items()):
        blob = blobs / f"hash-{index}"
        blob.write_text(content)
        (snapshot / filename).symlink_to(Path("../../blobs") / blob.name)
    return snapshot


def test_source_hash_preserves_filenames_in_symlinked_snapshot(tmp_path):
    plain_model = _create_source_tree(tmp_path / "plain", symlinked=False)
    cached_model = _create_source_tree(tmp_path / "cached", symlinked=True)

    plain_hash = _compute_local_source_files_hash_with_symlink_fix(
        plain_model, plain_model / "configuration_test.py"
    )
    cached_hash = _compute_local_source_files_hash_with_symlink_fix(
        cached_model, cached_model / "configuration_test.py"
    )

    assert cached_hash == plain_hash
    assert len(cached_hash) == 16


def test_patch_only_applies_to_transformers_5_12_1(monkeypatch):
    original_function = object()
    monkeypatch.setattr(transformers, "__version__", "5.13.0")
    monkeypatch.setattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        original_function,
    )

    assert patch_transformers_dynamic_module_symlink_cache() is False
    assert dynamic_module_utils._compute_local_source_files_hash is original_function


def test_patch_is_idempotent_on_transformers_5_12_1(monkeypatch):
    monkeypatch.setattr(transformers, "__version__", "5.12.1")
    monkeypatch.setattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        object(),
    )

    assert patch_transformers_dynamic_module_symlink_cache() is True
    patched_function = dynamic_module_utils._compute_local_source_files_hash
    assert getattr(patched_function, _SYMLINK_CACHE_PATCH_MARKER) is True

    assert patch_transformers_dynamic_module_symlink_cache() is False
    assert dynamic_module_utils._compute_local_source_files_hash is patched_function


def test_patch_fails_loudly_if_transformers_target_is_missing(monkeypatch):
    monkeypatch.setattr(transformers, "__version__", "5.12.1")
    monkeypatch.delattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="cannot apply"):
        patch_transformers_dynamic_module_symlink_cache()
