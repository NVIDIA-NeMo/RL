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

from typing import Any


def get_embedded_mtp_ignore_patterns(config: Any) -> list[str]:
    text_config = getattr(config, "text_config", None) or config
    num_mtp_layers = getattr(text_config, "num_nextn_predict_layers", None)
    if num_mtp_layers is None:
        num_mtp_layers = getattr(text_config, "mtp_num_hidden_layers", 0)

    if num_mtp_layers == 0:
        return []
    if (
        isinstance(num_mtp_layers, bool)
        or not isinstance(num_mtp_layers, int)
        or num_mtp_layers < 0
    ):
        raise ValueError(
            "MTP layer metadata must contain a non-negative integer layer count"
        )

    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    if (
        isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers < 0
    ):
        raise ValueError(
            "MTP layer metadata must contain a non-negative integer num_hidden_layers"
        )

    return [
        f"model.layers.{layer_idx}.*"
        for layer_idx in range(
            num_hidden_layers,
            num_hidden_layers + num_mtp_layers,
        )
    ]
