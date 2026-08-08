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

"""Validated configuration for the turn-level credit experiment."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class TurnCreditConfig(BaseModel):
    """Configuration owned by the research entrypoint.

    Attributes:
        enabled: Whether to record native turn rewards and modify advantages.
        source: Credit source. The first research slice supports only native
            environment rewards.
        environment_mode: Whether each turn receives its immediate reward or
            its discounted return-to-go.
        discount: Return-to-go discount in the closed interval ``[0, 1]``.
        macro_weight: Weight applied to the existing trajectory-level GRPO
            advantage.
        turn_weight: Weight applied to the token-aligned turn credit.
        raw_reward_atol: Absolute tolerance used to validate that raw turn
            rewards sum to the trajectory reward before reward transforms.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    source: Literal["environment"] = "environment"
    environment_mode: Literal["immediate", "return_to_go"] = "immediate"
    discount: float = 1.0
    macro_weight: float = 1.0
    turn_weight: float = 0.0
    raw_reward_atol: float = 1.0e-6

    @model_validator(mode="after")
    def _validate_numeric_ranges(self) -> "TurnCreditConfig":
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("turn_credit.discount must be in [0, 1]")
        if self.macro_weight < 0.0:
            raise ValueError("turn_credit.macro_weight must be non-negative")
        if self.turn_weight < 0.0:
            raise ValueError("turn_credit.turn_weight must be non-negative")
        if self.raw_reward_atol < 0.0:
            raise ValueError("turn_credit.raw_reward_atol must be non-negative")
        return self
