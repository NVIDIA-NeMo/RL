# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Transformers runtime compatibility patches."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
import transformers.dynamic_module_utils as dynamic_module_utils

import nemo_rl.transformers_compat as transformers_compat
from nemo_rl.transformers_compat import (
    _compute_local_source_files_hash_with_symlink_fix,
    _patch_transformers_dynamic_module_symlink_cache,
)

_SOURCE_FILES = {
    "configuration_test.py": "from .dependency import VALUE\n",
    "dependency.py": "from .leaf import LEAF\nVALUE = LEAF\n",
    "leaf.py": "LEAF = 1\n",
}


def _create_symlinked_source_tree(root: Path) -> Path:
    blobs = root / "blobs"
    snapshot = root / "snapshots" / "revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for index, (filename, content) in enumerate(_SOURCE_FILES.items()):
        blob = blobs / f"hash-{index}"
        blob.write_text(content, encoding="utf-8")
        (snapshot / filename).symlink_to(Path("../../blobs") / blob.name)
    return snapshot


def test_get_cached_module_file_handles_symlinked_hub_cache(monkeypatch, tmp_path):
    modules_cache = tmp_path / "hf_modules_cache"
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(modules_cache))
    snapshot = _create_symlinked_source_tree(tmp_path / "models--org--repo")

    cached_module = dynamic_module_utils.get_cached_module_file(
        str(snapshot), "configuration_test.py"
    )
    cached_source_dir = modules_cache / Path(cached_module).parent

    assert dynamic_module_utils._compute_local_source_files_hash is (
        _compute_local_source_files_hash_with_symlink_fix
    ), (
        "The symlink-cache patch is not installed. If Transformers is now "
        "5.13.0 or newer the upstream fix is already in place: delete "
        "nemo_rl/transformers_compat.py, its bootstrap at the bottom of "
        "nemo_rl/__init__.py, and this test file. Do not widen the version gate."
    )
    for filename, content in _SOURCE_FILES.items():
        assert (cached_source_dir / filename).read_text(encoding="utf-8") == content


@pytest.mark.parametrize(
    "version",
    ("5.11.0", "5.12.0", "5.12.1", "5.12.1+vendor.1"),
)
def test_patch_applies_to_affected_transformers_versions(monkeypatch, version):
    def original_function(pretrained_model_name_or_path, resolved_module_file):
        return None

    monkeypatch.setattr(transformers_compat, "distribution_version", lambda _: version)
    monkeypatch.setattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        original_function,
    )

    assert _patch_transformers_dynamic_module_symlink_cache() is True
    assert (
        dynamic_module_utils._compute_local_source_files_hash
        is _compute_local_source_files_hash_with_symlink_fix
    )


@pytest.mark.parametrize("version", ("5.10.0", "5.13.0", "6.0.0"))
def test_patch_skips_unaffected_transformers_versions(monkeypatch, version):
    original_function = object()
    monkeypatch.setattr(transformers_compat, "distribution_version", lambda _: version)
    monkeypatch.setattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        original_function,
    )

    assert _patch_transformers_dynamic_module_symlink_cache() is False
    assert dynamic_module_utils._compute_local_source_files_hash is original_function


def test_patch_is_idempotent_without_function_attribute(monkeypatch):
    monkeypatch.setattr(transformers_compat, "distribution_version", lambda _: "5.12.1")
    monkeypatch.setattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        _compute_local_source_files_hash_with_symlink_fix,
    )

    assert _patch_transformers_dynamic_module_symlink_cache() is False
    assert not vars(_compute_local_source_files_hash_with_symlink_fix)


def test_patch_skips_when_transformers_is_not_installed(monkeypatch):
    def missing_distribution(_):
        raise PackageNotFoundError

    monkeypatch.setattr(
        transformers_compat, "distribution_version", missing_distribution
    )

    assert _patch_transformers_dynamic_module_symlink_cache() is False


def test_patch_propagates_import_error_from_transformers(monkeypatch):
    monkeypatch.setattr(transformers_compat, "distribution_version", lambda _: "5.12.1")

    def broken_import(_):
        raise ImportError("broken Transformers installation")

    monkeypatch.setattr(transformers_compat.importlib, "import_module", broken_import)

    with pytest.raises(ImportError, match="broken Transformers installation"):
        _patch_transformers_dynamic_module_symlink_cache()


def test_patch_fails_loudly_if_transformers_target_is_missing(monkeypatch):
    monkeypatch.setattr(transformers_compat, "distribution_version", lambda _: "5.12.1")
    monkeypatch.delattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="cannot apply"):
        _patch_transformers_dynamic_module_symlink_cache()


def test_patch_fails_loudly_if_transformers_target_signature_changed(monkeypatch):
    def incompatible_function(unexpected_parameter):
        return None

    monkeypatch.setattr(transformers_compat, "distribution_version", lambda _: "5.12.1")
    monkeypatch.setattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        incompatible_function,
    )

    with pytest.raises(RuntimeError, match="unexpected.*signature"):
        _patch_transformers_dynamic_module_symlink_cache()
