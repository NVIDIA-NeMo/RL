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

"""Compatibility patches must target mutations present in the pinned Megatron."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.sglang


@pytest.fixture
def patches() -> ModuleType:
    # These stdlib-only helpers do not need optional SGLang or Torch imports.
    path = (
        Path(__file__).resolve().parents[5]
        / "nemo_rl/models/generation/sglang/utils/patches.py"
    )
    spec = importlib.util.spec_from_file_location("sglang_compat_patches", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_only_patches_dynamic_context(
    patches: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = tmp_path / "dynamic_context.py"
    context.write_text(
        "if HAVE_TMS:\n"
        "    import torch_memory_saver\n"
        '    torch_memory_saver.hook_mode = "torch"\n'
    )
    locate = Mock(return_value=str(context))
    monkeypatch.setattr(patches, "_get_megatron_file", locate)
    for helper in (
        "_patch_sglang_transformers_compat_bootstrap",
        "_patch_sglang_safe_unpickler",
        "_override_sglang_imbalance_check_env",
    ):
        monkeypatch.setattr(patches, helper, Mock())

    patches._apply_sglang_compat_patches()

    locate.assert_called_once_with("core", "inference/contexts/dynamic_context.py")
    updated = context.read_text()
    assert '    # torch_memory_saver.hook_mode = "torch"' in updated
    assert '\n    torch_memory_saver.hook_mode = "torch"' not in updated
    compile(updated, str(context), "exec")
    patches._apply_sglang_compat_patches()
    assert context.read_text() == updated


def test_missing_dynamic_context_anchor_fails(
    patches: ModuleType, tmp_path: Path
) -> None:
    context = tmp_path / "dynamic_context.py"
    context.write_text("# Unexpected upstream change.\n")
    with pytest.raises(RuntimeError, match="Megatron hook_mode anchor"):
        patches._patch_megatron_hook_mode_in(str(context))
    assert context.read_text() == "# Unexpected upstream change.\n"


def test_missing_megatron_needs_no_patch(
    patches: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(patches, "_get_megatron_file", Mock(return_value=None))
    patch_file = Mock()
    monkeypatch.setattr(patches, "_patch_megatron_hook_mode_in", patch_file)

    patches._patch_megatron_dynamic_context_hook_mode()

    patch_file.assert_not_called()
