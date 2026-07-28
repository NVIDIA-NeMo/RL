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

import json

import pytest

from nemo_rl.models.huggingface.common import (
    ModelFlag,
    is_gemma_model,
    is_nano_nemotron_vl_model,
)


@pytest.mark.parametrize(
    "model_type",
    ["NemotronH_Nano_VL_V2", "NemotronH_Nano_Omni_Reasoning_V3"],
)
def test_nemotron_model_flag_does_not_import_remote_code(tmp_path, model_type):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "auto_map": {"AutoConfig": "configuration.ShouldNotBeImported"},
                "model_type": model_type,
            }
        )
    )
    (tmp_path / "configuration.py").write_text(
        "raise AssertionError('model flag imported checkpoint code')\n"
    )

    assert is_nano_nemotron_vl_model(str(tmp_path))
    assert ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(str(tmp_path))


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
