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

"""CPU coverage for the pinned BEAM SGLang allowlist."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def patches():
    path = (
        Path(__file__).resolve().parents[5]
        / "nemo_rl/models/generation/sglang/utils/patches.py"
    )
    spec = importlib.util.spec_from_file_location("beam_sglang_patches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_allowlist_preserves_restrictions_and_is_idempotent(
    tmp_path, monkeypatch, patches
):
    path = tmp_path / "common.py"
    path.write_text(
        'class SafeUnpickler:\n    ALLOWED_GLOBALS = {\n        ("builtins", "bool"),\n    }\n'
    )
    monkeypatch.setattr(patches, "_get_sglang_file", lambda _: str(path))
    patches._patch_sglang_safe_unpickler()
    namespace = {}
    exec(path.read_text(), namespace)
    allowed = namespace["SafeUnpickler"].ALLOWED_GLOBALS
    module = "nemo_rl.models.generation.sglang.utils.train_utils"
    assert allowed == {("builtins", "bool")} | {
        (module, name)
        for name in (
            "_rebuild_cuda_tensor_modified",
            "FlattenedTensorMetadata",
            "FlattenedTensorBucket",
        )
    }
    before = path.read_text()
    patches._patch_sglang_safe_unpickler()
    assert path.read_text() == before


def test_unknown_allowlist_fails_without_writing(tmp_path, monkeypatch, patches):
    path = tmp_path / "common.py"
    path.write_text("# unsupported source layout\n")
    monkeypatch.setattr(patches, "_get_sglang_file", lambda _: str(path))
    with pytest.raises(RuntimeError, match="anchor"):
        patches._patch_sglang_safe_unpickler()
    assert path.read_text() == "# unsupported source layout\n"
