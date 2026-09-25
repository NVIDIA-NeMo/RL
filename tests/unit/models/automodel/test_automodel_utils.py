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
import pytest

from nemo_rl.models.automodel.utils import resolve_model_class


@pytest.mark.automodel
@pytest.mark.parametrize(
    ("model_type", "class_name"),
    [
        ("qwen2_5_vl", "nemo_image_text"),
        ("qwen2_5_omni", "nemo_text_waveform"),
        ("unknown_model", "nemo_causal_lm"),
    ],
)
def test_resolve_model_class_maps_model_type_to_loader(
    monkeypatch: pytest.MonkeyPatch, model_type: str, class_name: str
) -> None:
    """An unmapped model type falls back to the causal-LM loader."""
    classes = {
        "nemo_image_text": object(),
        "nemo_text_waveform": object(),
        "nemo_causal_lm": object(),
    }
    monkeypatch.setattr(
        "nemo_rl.models.automodel.utils.AUTOMODEL_FACTORY",
        {
            "qwen2_5_vl": classes["nemo_image_text"],
            "qwen2_5_omni": classes["nemo_text_waveform"],
        },
    )
    monkeypatch.setattr(
        "nemo_rl.models.automodel.utils.NeMoAutoModelForCausalLM",
        classes["nemo_causal_lm"],
    )

    assert resolve_model_class(model_type) is classes[class_name]


@pytest.mark.automodel
def test_resolve_model_class_routes_gemma4_unified_to_image_text_model():
    assert "ImageTextToText" in resolve_model_class("gemma4_unified").__name__
