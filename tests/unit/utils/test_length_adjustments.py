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

import pytest
import torch

from nemo_rl.utils.length_adjustments import (
    _band_multiplier,
    _generated_assistant_token_lengths,
    apply_group_length_adjustments,
)


def _result(
    *,
    reward: float = 1.0,
    agent_name: str = "video_agent",
    prompt_history: list[dict] | None = None,
    generated_token_ids: list[int] | None = None,
    profile_band: dict | None = None,
) -> dict:
    prompt_history = prompt_history or [
        {"role": "user", "token_ids": torch.tensor([90])}
    ]
    generated_token_ids = generated_token_ids or [1, 2, 13, 3, 4]
    return {
        "agent_ref": {"name": agent_name},
        "input_message_log": prompt_history,
        "message_log": [
            *prompt_history,
            {
                "role": "assistant",
                "token_ids": torch.tensor(generated_token_ids),
            },
        ],
        "profile_band": profile_band,
        "full_result": {"reward": reward, "response": {"output": []}},
    }


def _config(
    default: dict,
    *,
    global_profile_band: dict | None = None,
    agent_overrides: dict | None = None,
) -> dict:
    length_bonus = {"default": {"enabled": True, **default}}
    if global_profile_band is not None:
        length_bonus["profile_band"] = {
            "enabled": True,
            "defaults": global_profile_band,
        }
    if agent_overrides is not None:
        length_bonus["agent_overrides"] = agent_overrides
    return {
        "grpo": {
            "num_generations_per_prompt": 1,
            "length_bonus": length_bonus,
        }
    }


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (1024, 1.0),
        (2560, 0.975),
        (4096, 0.95),
        (8192, 0.95),
    ],
)
def test_profile_band_multiplier_stays_at_floor_after_b(length, expected):
    band = {"a": 1024, "b": 4096, "f": 0.95}

    assert _band_multiplier(length, band) == pytest.approx(expected)


def test_generated_reasoning_excludes_prompt_history_and_splits_omni_end_token():
    prompt_history = [
        {"role": "user", "token_ids": torch.tensor([90])},
        {"role": "assistant", "token_ids": torch.tensor([7, 8, 9, 13])},
        {"role": "user", "token_ids": torch.tensor([91])},
    ]
    result = _result(
        prompt_history=prompt_history,
        generated_token_ids=[1, 2, 13, 3, 4, 5],
    )

    assert _generated_assistant_token_lengths(result, 13) == (2, 3)


def test_generated_reasoning_rejects_non_vector_token_ids():
    result = _result()
    result["message_log"][-1]["token_ids"] = torch.tensor([[1, 2, 13]])

    with pytest.raises(ValueError, match="one-dimensional"):
        _generated_assistant_token_lengths(result, 13)


@pytest.mark.parametrize("reasoning_end_token_id", [True, -1, "13"])
def test_reasoning_end_token_id_must_be_non_negative_integer(
    reasoning_end_token_id,
):
    config = _config(
        {
            "profile_band_reasoning": True,
            "reasoning_end_token_id": reasoning_end_token_id,
        },
        global_profile_band={"reasoning": {"a": 1, "b": 2, "f": 0.5}},
    )

    with pytest.raises(ValueError, match="reasoning_end_token_id"):
        apply_group_length_adjustments([], config)


def test_global_reasoning_profile_band_uses_generated_tokens():
    result = _result(generated_token_ids=[1, 2, 13, 3, 4, 5])
    config = _config(
        {
            "profile_band_reasoning": True,
            "reasoning_end_token_id": 13,
        },
        global_profile_band={"reasoning": {"a": 1, "b": 2, "f": 0.5}},
    )

    apply_group_length_adjustments([result], config)

    assert result["full_result"]["reward"] == pytest.approx(0.5)
    feature = result["full_result"]["gdpo_reward_features"][
        "profile_band_reasoning"
    ]
    assert feature["multiplier"] == pytest.approx(0.5)


def test_row_profile_band_takes_precedence_over_global_config():
    result = _result(
        generated_token_ids=[1, 2, 13, 3, 4, 5],
        profile_band={"reasoning": {"a": 10, "b": 20, "f": 0.1}},
    )
    config = _config(
        {
            "profile_band_reasoning": True,
            "reasoning_end_token_id": 13,
        },
        global_profile_band={"reasoning": {"a": 1, "b": 2, "f": 0.5}},
    )

    apply_group_length_adjustments([result], config)

    assert result["full_result"]["reward"] == pytest.approx(1.0)


def test_agent_override_profile_band_takes_precedence_over_global_config():
    result = _result(generated_token_ids=[1, 2, 13])
    config = _config(
        {
            "profile_band_reasoning": True,
            "reasoning_end_token_id": 13,
        },
        global_profile_band={"reasoning": {"a": 1, "b": 2, "f": 0.5}},
        agent_overrides={
            "video_agent": {
                "profile_band": {
                    "reasoning": {"a": 1, "b": 2, "f": 0.8},
                }
            }
        },
    )

    apply_group_length_adjustments([result], config)

    assert result["full_result"]["reward"] == pytest.approx(0.8)


def test_profile_band_does_not_change_incorrect_reward():
    result = _result(reward=0.0, generated_token_ids=[1, 2, 13])
    config = _config(
        {
            "profile_band_reasoning": True,
            "reasoning_end_token_id": 13,
        },
        global_profile_band={"reasoning": {"a": 1, "b": 2, "f": 0.5}},
    )

    apply_group_length_adjustments([result], config)

    assert result["full_result"]["reward"] == pytest.approx(0.0)
