# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import patch

import pytest

from nemo_rl.models.huggingface.common import (
    ModelFlag,
    is_gemma_model,
    is_nano_nemotron_vl_model,
)


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "model_name",
    [
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "google/gemma-3-1b-pt",
        "google/gemma-3-4b-pt",
        "google/gemma-3-12b-pt",
        "google/gemma-3-27b-pt",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
    ],
)
def test_gemma_models(model_name):
    assert is_gemma_model(model_name)
    assert ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(model_name)


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ],
)
def test_non_gemma_models(model_name):
    assert not is_gemma_model(model_name)
    assert not ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(model_name)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("NemotronH_Nano_VL_V2", True),
        ("NemotronH_Nano_Omni_Reasoning_V3", True),
        # Nemotron 3.5 Super VL reports ``nemotron_h_omni``.
        ("nemotron_h_omni", True),
        ("nemotron_h", False),
        ("llama", False),
    ],
)
def test_is_nano_nemotron_vl_model_by_model_type(model_type, expected):
    with patch(
        "nemo_rl.models.huggingface.common.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type=model_type),
    ) as from_pretrained:
        assert is_nano_nemotron_vl_model("some/model") is expected
        assert ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches("some/model") is expected
    from_pretrained.assert_called_with("some/model", trust_remote_code=True)
