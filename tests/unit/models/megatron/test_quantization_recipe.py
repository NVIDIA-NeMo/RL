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

import pytest

from nemo_rl.models.megatron.quantization_recipe import (
    first_last_bf16_local_layers,
)


def test_first_last_bf16_layers_reject_overlapping_global_ranges() -> None:
    with pytest.raises(ValueError, match="overlap"):
        first_last_bf16_local_layers(
            total_layers=8,
            global_layer_offset=0,
            local_layer_count=8,
            num_layers_at_start_in_bf16=5,
            num_layers_at_end_in_bf16=4,
        )
