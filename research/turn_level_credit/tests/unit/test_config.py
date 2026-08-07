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

"""Tests for turn-level credit configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from run_grpo_turn_credit import load_master_and_turn_credit_config
from turn_level_credit.config import TurnCreditConfig


def test_config_defaults_are_macro_only():
    config = TurnCreditConfig()

    assert not config.enabled
    assert config.source == "environment"
    assert config.environment_mode == "immediate"
    assert config.macro_weight == 1.0
    assert config.turn_weight == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discount", -0.1),
        ("discount", 1.1),
        ("macro_weight", -1.0),
        ("turn_weight", -1.0),
        ("raw_reward_atol", -1.0),
    ],
)
def test_config_rejects_invalid_numeric_ranges(field, value):
    with pytest.raises(ValidationError):
        TurnCreditConfig.model_validate({field: value})


def test_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        TurnCreditConfig.model_validate({"silent_fallback": True})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discount", float("nan")),
        ("discount", float("inf")),
        ("discount", float("-inf")),
        ("macro_weight", float("nan")),
        ("macro_weight", float("inf")),
        ("macro_weight", float("-inf")),
        ("turn_weight", float("nan")),
        ("turn_weight", float("inf")),
        ("turn_weight", float("-inf")),
        ("raw_reward_atol", float("nan")),
        ("raw_reward_atol", float("inf")),
        ("raw_reward_atol", float("-inf")),
    ],
)
def test_config_rejects_non_finite_numbers(field, value):
    with pytest.raises(ValidationError):
        TurnCreditConfig.model_validate({field: value})


def test_sliding_puzzle_pilot_is_multi_turn_and_macro_only_by_default():
    config_path = (
        Path(__file__).parents[2] / "configs" / "grpo_sliding_puzzle_turn_credit.yaml"
    )

    master_config, turn_credit_config = load_master_and_turn_credit_config(
        str(config_path),
        [],
    )

    assert master_config.grpo.max_rollout_turns == 6
    assert master_config.grpo.max_num_steps == 10
    assert master_config.policy["train_global_batch_size"] == 16
    assert turn_credit_config.enabled
    assert turn_credit_config.turn_weight == 0.0
