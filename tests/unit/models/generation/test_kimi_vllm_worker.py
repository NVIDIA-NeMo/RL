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

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.vllm_worker import (
    _validate_kimi_mla_max_model_len,
)


def test_kimi_mla_max_model_len_accepts_128_multiple() -> None:
    hf_config = SimpleNamespace(architectures=["KimiK25ForConditionalGeneration"])

    _validate_kimi_mla_max_model_len(hf_config, 1280)


def test_kimi_mla_max_model_len_rejects_unaligned_length() -> None:
    hf_config = SimpleNamespace(architectures=["KimiK25ForConditionalGeneration"])

    with pytest.raises(ValueError, match="multiple of 128"):
        _validate_kimi_mla_max_model_len(hf_config, 1216)


def test_kimi_mla_max_model_len_ignores_other_architectures() -> None:
    hf_config = SimpleNamespace(architectures=["Qwen2ForCausalLM"])

    _validate_kimi_mla_max_model_len(hf_config, 1216)
