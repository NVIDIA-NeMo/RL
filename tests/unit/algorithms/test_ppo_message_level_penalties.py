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

from types import SimpleNamespace

import pytest
import torch

import nemo_rl.algorithms.ppo as ppo_module
from nemo_rl.algorithms.ppo import (
    _apply_ppo_message_level_advantage_penalties,
    _resolve_ppo_message_level_advantage_penalties,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture(autouse=True)
def _gym_flag_from_env(monkeypatch):
    monkeypatch.setattr(
        ppo_module, "_should_use_nemo_gym", lambda cfg: cfg.env["should_use_nemo_gym"]
    )


def _config(invalid=None, malformed=None, gym=True, cross=False):
    return SimpleNamespace(
        ppo={
            "invalid_tool_call_advantage": invalid,
            "malformed_thinking_advantage": malformed,
            "adv_estimator": {"cross_trajectory": cross},
        },
        env={"should_use_nemo_gym": gym},
    )


def _msg(role, n, **flags):
    m = {"role": role, "token_ids": torch.arange(n)}
    if role == "assistant":
        m["generation_logprobs"] = torch.zeros(n)
    m.update(flags)
    return m


def test_overrides_only_flagged_assistant_spans():
    message_logs = [
        [
            _msg("user", 3),
            _msg("assistant", 2, is_invalid_tool_call=True),
            _msg("user", 1),
            _msg("assistant", 2, has_malformed_thinking=True),
            _msg("assistant", 1),
        ]
    ]
    adv = torch.full((1, 9), 0.5)
    data = BatchedDataDict({"advantages": adv})
    metrics = _apply_ppo_message_level_advantage_penalties(
        data, message_logs, _config(invalid=-5.0, malformed=-3.0)
    )
    expected = torch.tensor([[0.5, 0.5, 0.5, -5.0, -5.0, 0.5, -3.0, -3.0, 0.5]])
    torch.testing.assert_close(data["advantages"], expected)
    assert metrics["advantage_penalty/num_invalid_tool_calls"] == 1
    assert metrics["advantage_penalty/num_malformed_thinking"] == 1


def test_noop_when_unset():
    data = BatchedDataDict({"advantages": torch.ones(1, 4)})
    msgs = [[_msg("assistant", 4, is_invalid_tool_call=True)]]
    assert _apply_ppo_message_level_advantage_penalties(data, msgs, _config()) == {}
    torch.testing.assert_close(data["advantages"], torch.ones(1, 4))


def test_validation():
    with pytest.raises(ValueError, match="NeMo-Gym"):
        _resolve_ppo_message_level_advantage_penalties(_config(invalid=-5.0, gym=False))
    with pytest.raises(ValueError, match="cross_trajectory"):
        _resolve_ppo_message_level_advantage_penalties(_config(invalid=-5.0, cross=True))


def _config_empty(empty=-5.0, invalid=None, exempt=("finish",)):
    cfg = _config(invalid=invalid)
    cfg.ppo["empty_tool_call_advantage"] = empty
    cfg.ppo["empty_tool_call_exempt_tools"] = list(exempt)
    return cfg


def test_empty_tool_call_override_skips_exempt_and_invalid():
    message_logs = [
        [
            _msg("user", 2),
            _msg("assistant", 2, is_empty_tool_call=True, tool_call_name="think"),
            _msg("assistant", 2, is_empty_tool_call=True, tool_call_name="finish"),
            _msg(
                "assistant",
                2,
                is_empty_tool_call=True,
                is_invalid_tool_call=True,
                tool_call_name="str_replace_editor",
            ),
            _msg("assistant", 1, tool_call_name="execute_bash"),
        ]
    ]
    data = BatchedDataDict({"advantages": torch.full((1, 9), 0.5)})
    metrics = _apply_ppo_message_level_advantage_penalties(
        data, message_logs, _config_empty(empty=-4.0, invalid=-5.0)
    )
    expected = torch.tensor([[0.5, 0.5, -4.0, -4.0, 0.5, 0.5, -5.0, -5.0, 0.5]])
    torch.testing.assert_close(data["advantages"], expected)
    assert metrics["advantage_penalty/num_empty_tool_calls"] == 1
    assert metrics["advantage_penalty/num_invalid_tool_calls"] == 1


def test_empty_only_config_is_validated():
    cfg = _config_empty(empty=-5.0)
    cfg.env["should_use_nemo_gym"] = False
    with pytest.raises(ValueError, match="NeMo-Gym"):
        _resolve_ppo_message_level_advantage_penalties(cfg)
