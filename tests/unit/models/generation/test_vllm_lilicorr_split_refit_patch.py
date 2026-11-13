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
"""A LiLiCorr drafter must accept a refit split across load_weights calls."""

import logging
from pathlib import Path

import pytest

from nemo_rl.models.generation.vllm import patches


def _source_with(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "lilicorr.py"
    path.write_text(
        "class LiLiCorrHead:\n    def __init__(self):\n"
        + patches._LILICORR_INIT_OLD
        + patches._LILICORR_MATERIALIZE_OLD
        + "        pass\n\n    def load_weights(self):\n"
        + body
    )
    return path


def test_patch_gates_coverage_check_and_is_idempotent(tmp_path, monkeypatch):
    path = _source_with(tmp_path, patches._LILICORR_COVERAGE_OLD)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _rel: str(path))
    logger = logging.getLogger("test")
    patches._patch_vllm_lilicorr_split_refit(logger)
    once = path.read_text()
    assert patches._LILICORR_COVERAGE_NEW in once
    assert "if missing and _nrl_coverage_check_enabled():" in once
    assert "_nrl_build_inference_buffers" in once  # in-place refill wrapper
    assert "self.materialize_inference_buffers(" in once  # built at construction
    patches._patch_vllm_lilicorr_split_refit(logger)
    assert path.read_text() == once


def test_patch_refuses_an_unknown_layout(tmp_path, monkeypatch):
    path = _source_with(tmp_path, "        pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _rel: str(path))
    with pytest.raises(RuntimeError, match="LiLiCorr refit patch"):
        patches._patch_vllm_lilicorr_split_refit(logging.getLogger("test"))


def test_patch_skips_vllm_without_lilicorr(monkeypatch):
    def missing(_rel):
        raise RuntimeError("not found")

    monkeypatch.setattr(patches, "_get_vllm_file", missing)
    patches._patch_vllm_lilicorr_split_refit(logging.getLogger("test"))


def test_patch_applies_to_installed_vllm_lilicorr(tmp_path, monkeypatch):
    try:
        source = Path(patches._get_vllm_file("model_executor/models/lilicorr.py"))
    except RuntimeError:
        pytest.skip("installed vLLM has no LiLiCorr drafter (< 0.31)")
    copy = tmp_path / "lilicorr.py"
    copy.write_text(source.read_text())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _rel: str(copy))
    patches._patch_vllm_lilicorr_split_refit(logging.getLogger("test"))
    compile(copy.read_text(), str(copy), "exec")
    assert patches._LILICORR_COVERAGE_NEW in copy.read_text()
