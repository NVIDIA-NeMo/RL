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

import difflib
from typing import Literal

from pydantic import BaseModel, model_validator


class Eagle3DraftConfig(BaseModel, extra="allow"):
    """Configuration for EAGLE-3 draft-model co-training with the policy."""

    speculator_type: Literal["eagle3"] = "eagle3"
    enabled: bool = False
    model_name: str | None = None
    loss_weight: float = 0.1
    num_layers: int | None = None
    aux_layer_indices: list[int] | None = None

    @model_validator(mode="after")
    def _reject_near_miss_extra_keys(self) -> "Eagle3DraftConfig":
        # extra="allow" preserves genuinely novel legacy keys, but a typo of a
        # declared field (e.g. "enalbed") would otherwise silently no-op the
        # real field's default. Reject extras that look like misspellings.
        declared = set(type(self).model_fields)
        for key in self.model_extra or {}:
            close = difflib.get_close_matches(key, declared, n=1, cutoff=0.8)
            if close:
                raise ValueError(
                    f"unknown draft config key {key!r}; did you mean {close[0]!r}?"
                )
        return self


def draft_refit_enabled(config: Eagle3DraftConfig | None) -> bool:
    """Return whether generation must accept refitted draft weights."""
    return config is not None and config.enabled
