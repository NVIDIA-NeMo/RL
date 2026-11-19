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

"""Native model detection through the text model and its multimodal wrapper."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.vllm


@pytest.fixture
def native():
    from nemo_rl.models.generation.vllm.quantization import deepseek_v41_native

    return deepseek_v41_native


def test_native_detection_uses_text_owner_and_resolved_quant_config(native):
    quant = SimpleNamespace(
        get_name=lambda: "deepseek_v4_fp8",
        expert_dtype="fp4",
        weight_block_size=[32, 32],
    )
    # HF JSON may omit expert_dtype; vLLM resolves it from the quantization config.
    config = SimpleNamespace(model_type="deepseek_v41")
    text = SimpleNamespace(config=config, model=SimpleNamespace(quant_config=quant))
    wrapper = SimpleNamespace(config=config, language_model=text)
    assert native.is_native_model(text)
    assert native.is_native_model(wrapper)
    quant.expert_dtype = "fp8"
    assert not native.is_native_model(wrapper)
    text.model.quant_config = None
    assert not native.is_native_model(wrapper)
