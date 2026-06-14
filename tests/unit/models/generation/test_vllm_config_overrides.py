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

import pytest

from nemo_rl.models.huggingface.common import apply_hf_config_overrides


def test_apply_hf_config_overrides_initializes_hf_overrides():
    vllm_kwargs = {"hf_overrides": None}
    vllm_cfg = {"skip_tokenizer_init": True}

    apply_hf_config_overrides(vllm_kwargs, vllm_cfg, {}, "auto")

    assert vllm_kwargs["hf_overrides"] == {}
    assert vllm_cfg["skip_tokenizer_init"] is True


def test_apply_hf_config_overrides_disables_gpt_oss_quantization_for_dummy():
    vllm_kwargs = {}
    vllm_cfg = {"skip_tokenizer_init": True}
    hf_config = {
        "architectures": ["GptOssForCausalLM"],
        "quantization_config": {"quant_method": "mxfp4"},
    }

    apply_hf_config_overrides(vllm_kwargs, vllm_cfg, hf_config, "dummy")

    assert vllm_kwargs["hf_overrides"]["quantization_config"] == {}
    assert vllm_cfg["skip_tokenizer_init"] is True


def test_apply_hf_config_overrides_rejects_quantized_gpt_oss_non_dummy():
    hf_config = {
        "architectures": ["GptOssForCausalLM"],
        "quantization_config": {"quant_method": "mxfp4"},
    }

    with pytest.raises(AssertionError, match="load_format='dummy'"):
        apply_hf_config_overrides({}, {"skip_tokenizer_init": False}, hf_config, "auto")


def test_apply_hf_config_overrides_forces_tokenizer_for_vlm_architecture():
    vllm_kwargs = {}
    vllm_cfg = {"skip_tokenizer_init": True}

    apply_hf_config_overrides(
        vllm_kwargs,
        vllm_cfg,
        {"architectures": ["Qwen3_5ForConditionalGeneration"]},
        "auto",
    )

    assert vllm_kwargs["hf_overrides"] == {}
    assert vllm_cfg["skip_tokenizer_init"] is False
