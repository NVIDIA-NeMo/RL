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
"""vLLM source patches must not fail every model when an unrelated model file changed."""

import logging
import sys
import types

from nemo_rl.models.generation.vllm import patches


def test_final_layernorm_patch_skips_unrecognized_source(tmp_path, monkeypatch, caplog):
    src = tmp_path / "nano_nemotron_vl.py"
    original = "# a newer vLLM layout without the 0.29 anchors\n"
    src.write_text(original)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _rel: str(src))

    logger = logging.getLogger("test_vllm_patch")
    with caplog.at_level(logging.ERROR, logger="test_vllm_patch"):
        patches._patch_vllm_radio_final_layernorm(logger)

    assert src.read_text() == original
    assert "NOT applied" in caplog.text


def test_v1_init_workers_patch_tolerates_missing_v1_executor(monkeypatch):
    def missing(_rel):
        raise RuntimeError("Failed to locate expected vLLM file to patch.")

    monkeypatch.setattr(patches, "_get_vllm_file", missing)
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)

    assert patches._patch_vllm_init_workers_ray("/venv/bin/python", ["FOO"]) is False
    import os

    copied = set(os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"].split(","))
    assert {"FOO", "RAY_ENABLE_UV_RUN_RUNTIME_ENV"} <= copied


def test_ray_executor_v2_assumed_when_switch_removed(monkeypatch):
    envs = types.ModuleType("vllm.envs")
    vllm = types.ModuleType("vllm")
    vllm.envs = envs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    assert patches.vllm_uses_ray_executor_v2() is True

    envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND = False
    assert patches.vllm_uses_ray_executor_v2() is False
