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

from pathlib import Path

import pytest

from nemo_rl.models.generation.vllm.cache_utils import (
    worker_vllm_cache_root,
    writeback_vllm_cache,
)


def test_worker_vllm_cache_root_uses_node_local_cache_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NRL_VLLM_CACHE_ROOT_BASE", str(tmp_path / "vllm"))

    assert worker_vllm_cache_root(2050) == str(tmp_path / "vllm" / "vllm_2050")


def test_worker_vllm_cache_root_rejects_relative_cache_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NRL_VLLM_CACHE_ROOT_BASE", "relative/cache")

    with pytest.raises(ValueError, match="must be an absolute path"):
        worker_vllm_cache_root(0)


def test_writeback_vllm_cache_merges_worker_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "local" / "vllm_3"
    destination_base = tmp_path / "persistent"
    (source / "torch_compile_cache").mkdir(parents=True)
    (source / "torch_compile_cache" / "artifact.bin").write_bytes(b"compiled")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(source))
    monkeypatch.setenv("NRL_VLLM_CACHE_WRITEBACK_DIR", str(destination_base))

    assert writeback_vllm_cache()
    assert (
        destination_base / "vllm_3" / "torch_compile_cache" / "artifact.bin"
    ).read_bytes() == b"compiled"


def test_writeback_vllm_cache_is_optional(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "missing"))
    monkeypatch.delenv("NRL_VLLM_CACHE_WRITEBACK_DIR", raising=False)

    assert not writeback_vllm_cache()
