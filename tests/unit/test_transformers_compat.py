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

import json
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest
import transformers.dynamic_module_utils as dynamic_module_utils
import transformers.utils.import_utils as transformers_import_utils
from packaging.version import Version
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.auto import tokenization_auto

import nemo_rl.transformers_compat as transformers_compat
from nemo_rl.models.policy import _patch_transformers_tokenizer_class_set
from nemo_rl.transformers_compat import (
    _compute_local_source_files_hash_with_symlink_fix,
    _patch_transformers_dynamic_module_symlink_cache,
    _patch_transformers_torch_fx_compat,
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

    # Backend extras pin different Transformers releases, so the patch must be
    # active exactly on affected versions and absent where upstream has the fix.
    # Once every extra is on 5.13.0+, delete the dynamic-module hashing patch,
    # its call in nemo_rl/__init__.py and its tests. Keep the torch-FX helper.
    installed_version = Version(distribution_version("transformers"))
    is_patched = (
        dynamic_module_utils._compute_local_source_files_hash
        is _compute_local_source_files_hash_with_symlink_fix
    )
    if (
        Version(transformers_compat._MIN_AFFECTED_TRANSFORMERS_VERSION)
        <= installed_version
        < Version(transformers_compat._FIXED_TRANSFORMERS_VERSION)
    ):
        assert is_patched, (
            f"The symlink-cache patch is not installed on affected Transformers "
            f"{installed_version}. Do not widen the version gate."
        )
    else:
        assert not is_patched, (
            f"The symlink-cache patch is installed on unaffected Transformers "
            f"{installed_version}."
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


def test_torch_fx_compat_patch_restores_removed_helper(monkeypatch):
    monkeypatch.setattr(transformers_compat, "distribution_version", lambda _: "5.12.1")
    monkeypatch.delattr(
        transformers_import_utils,
        "is_torch_fx_available",
        raising=False,
    )

    assert _patch_transformers_torch_fx_compat() is True
    assert (
        transformers_import_utils.is_torch_fx_available
        is transformers_compat._is_torch_fx_available_compat
    )
    assert _patch_transformers_torch_fx_compat() is False


def test_torch_fx_compat_patch_preserves_existing_helper(monkeypatch):
    existing_helper = object()
    monkeypatch.setattr(transformers_compat, "distribution_version", lambda _: "4.57.1")
    monkeypatch.setattr(
        transformers_import_utils,
        "is_torch_fx_available",
        existing_helper,
        raising=False,
    )

    assert _patch_transformers_torch_fx_compat() is False
    assert transformers_import_utils.is_torch_fx_available is existing_helper


def test_deepseek_v3_local_fast_tokenizer_keeps_native_backend(tmp_path: Path) -> None:
    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    ).save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "LlamaTokenizerFast"
    config_path.write_text(json.dumps(config))

    tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)

    assert type(tokenizer).__name__ == "TokenizersBackend"
    assert tokenizer("hello", add_special_tokens=True)["input_ids"] == [2]
    assert "deepseek_v3" in tokenization_auto.MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS
    assert (
        tokenization_auto.TOKENIZER_MAPPING_NAMES["deepseek_v3"] == "TokenizersBackend"
    )


@pytest.mark.parametrize("fail_first", (False, True))
def test_deepseek_v3_tokenizer_registry_retry(
    monkeypatch: pytest.MonkeyPatch, fail_first: bool
) -> None:
    blocked = {"deepseek_v3", "unrelated"}
    mapping = {"deepseek_v3": "TokenizersBackend", "unrelated": "TokenizersBackend"}
    monkeypatch.setattr(
        tokenization_auto, "MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS", blocked
    )
    monkeypatch.setattr(tokenization_auto, "TOKENIZER_MAPPING_NAMES", mapping)
    calls = []
    result = object()

    def load_tokenizer(name: str, **kwargs: object) -> object:
        assert name == "local-fixture"
        assert kwargs == {"trust_remote_code": True}
        calls.append(("deepseek_v3" in blocked, "deepseek_v3" in mapping))
        if fail_first and len(calls) == 1:
            raise OSError("missing tokenizer.json")
        return result

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    _patch_transformers_tokenizer_class_set()

    assert (
        AutoTokenizer.from_pretrained("local-fixture", trust_remote_code=True) is result
    )
    assert calls == ([(True, True), (False, False)] if fail_first else [(True, True)])
    assert "unrelated" in blocked
    assert mapping["unrelated"] == "TokenizersBackend"


@pytest.mark.parametrize("version", ("5.21.0", "6.0.0"))
def test_deepseek_v3_tokenizer_patch_rejects_unverified_versions(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    # Patch the guard's live import target, not a collection-time module alias.
    live_transformers = import_module("transformers")
    monkeypatch.setattr(live_transformers, "__version__", version)
    assert import_module("transformers") is live_transformers
    assert import_module("transformers").__version__ == version
    with pytest.raises(AssertionError, match="verified on 5.12.1 and 5.20.1"):
        _patch_transformers_tokenizer_class_set()
