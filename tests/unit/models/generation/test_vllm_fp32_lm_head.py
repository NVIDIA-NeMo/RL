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

"""vllm_cfg.fp32_lm_head maps to vLLM's native head_dtype="float32" override."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.vllm.config import (
    LEGACY_VLLM_FP32_LM_HEAD_ENV_VAR,
    vllm_fp32_lm_head_enabled,
)
from nemo_rl.models.generation.vllm.vllm_worker import _apply_fp32_lm_head


@pytest.mark.parametrize(
    ("vllm_cfg", "expected"),
    [
        ({}, False),
        ({"fp32_lm_head": False}, False),
        ({"fp32_lm_head": True}, True),
        ({"env_vars": {LEGACY_VLLM_FP32_LM_HEAD_ENV_VAR: "1"}}, False),
    ],
)
def test_vllm_fp32_lm_head_enabled(vllm_cfg, expected):
    assert vllm_fp32_lm_head_enabled(vllm_cfg) is expected


def test_fp32_lm_head_sets_head_dtype_and_keeps_other_overrides():
    vllm_kwargs = {"hf_overrides": {"max_position_embeddings": 8192}}

    _apply_fp32_lm_head({"fp32_lm_head": True}, vllm_kwargs)

    assert vllm_kwargs["hf_overrides"] == {
        "max_position_embeddings": 8192,
        "head_dtype": "float32",
    }


@pytest.mark.parametrize("vllm_cfg", [{}, {"fp32_lm_head": False}])
def test_fp32_lm_head_disabled_leaves_overrides_alone(vllm_cfg):
    vllm_kwargs = {"hf_overrides": {"max_position_embeddings": 8192}}

    _apply_fp32_lm_head(vllm_cfg, vllm_kwargs)

    assert vllm_kwargs["hf_overrides"] == {"max_position_embeddings": 8192}


def test_fp32_lm_head_accepts_a_matching_head_dtype_override():
    vllm_kwargs = {"hf_overrides": {"head_dtype": "float32"}}

    _apply_fp32_lm_head({"fp32_lm_head": True}, vllm_kwargs)

    assert vllm_kwargs["hf_overrides"] == {"head_dtype": "float32"}


@pytest.mark.parametrize("head_dtype", ["model", "bfloat16"])
def test_fp32_lm_head_rejects_a_conflicting_head_dtype_override(head_dtype):
    vllm_kwargs = {"hf_overrides": {"head_dtype": head_dtype}}

    with pytest.raises(ValueError, match="conflicts with hf_overrides.head_dtype"):
        _apply_fp32_lm_head({"fp32_lm_head": True}, vllm_kwargs)


@pytest.mark.vllm
def test_installed_vllm_resolves_an_fp32_generation_head():
    """Pin the vLLM contract the flag relies on (vllm-project/vllm#48390)."""
    from vllm.config.model import _get_head_dtype
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    config = SimpleNamespace(head_dtype="float32")
    assert _get_head_dtype(config, torch.bfloat16, "generate") is torch.float32
    assert _get_head_dtype(SimpleNamespace(), torch.bfloat16, "generate") is (
        torch.bfloat16
    )
    assert hasattr(LogitsProcessor, "_apply_head")
