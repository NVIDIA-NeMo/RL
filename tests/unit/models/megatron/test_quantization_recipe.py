# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nemo_rl.models.megatron.quantization_recipe import (
    _find_bf16_config_key,
    first_last_bf16_local_layers,
)


@pytest.mark.parametrize(
    ("quantization_key", "quantization_recipe"),
    [
        ("fp8_quantization_recipe", "mxfp8"),
        ("fp4_quantization_recipe", "nvfp4"),
    ],
)
def test_find_bf16_config_key_rejects_quantized_evaluation(
    quantization_key: str, quantization_recipe: str
) -> None:
    recipe = SimpleNamespace(
        configs={
            "bf16": {
                "training_recipe": {},
                "evaluation_recipe": {quantization_key: quantization_recipe},
            },
            "full_precision": {
                "training_recipe": {},
                "evaluation_recipe": {},
            },
        }
    )

    assert _find_bf16_config_key(recipe) == "full_precision"


def test_find_bf16_config_key_inherits_training_when_evaluation_missing() -> None:
    recipe = SimpleNamespace(configs={"bf16": {"training_recipe": {}}})

    assert _find_bf16_config_key(recipe) == "bf16"


def test_first_last_bf16_layers_reject_overlapping_global_ranges() -> None:
    with pytest.raises(ValueError, match="overlap"):
        first_last_bf16_local_layers(
            total_layers=8,
            global_layer_offset=0,
            local_layer_count=8,
            num_layers_at_start_in_bf16=5,
            num_layers_at_end_in_bf16=4,
        )
