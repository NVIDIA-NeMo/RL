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

"""The time-efficiency reward hook inside _postprocess_single_nemo_gym_group."""

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import _postprocess_single_nemo_gym_group
from nemo_rl.utils.time_efficiency import TimeEfficiencyConfig
from nemo_rl.utils.timer import Timer


class _FakeGeneration:
    cfg = {"max_total_sequence_length": 100}


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return {"<think>": [12], "</think>": [13]}[text]


def _result(reward, run_time_s, resolved):
    return {
        "full_result": {
            "reward": reward,
            "openhands_run_time": run_time_s,
            "resolved": resolved,
            "response": {"output": []},
        },
        "message_log": [
            {"role": "user", "token_ids": torch.tensor([3, 4])},
            {"role": "assistant", "token_ids": torch.tensor([1, 2])},
        ],
        "input_message_log": [{"role": "user", "token_ids": torch.tensor([3, 4])}],
    }


def _postprocess(results, time_efficiency_config, reward_penalty_config=None):
    return _postprocess_single_nemo_gym_group(
        nemo_gym_rows=[{"agent_ref": {"name": "swe_agents"}} for _ in results],
        results=results,
        timer=Timer(),
        timer_prefix="timing/test",
        policy_generation=_FakeGeneration(),
        input_batch=BatchedDataDict({"loss_multiplier": torch.ones(len(results))}),
        tokenizer=_FakeTokenizer(),
        log_full_result_tables=False,
        reward_penalty_config=reward_penalty_config,
        time_efficiency_config=time_efficiency_config,
    )


def test_deduction_reaches_total_reward_and_metrics():
    results = [_result(1.0, 1800.0, True), _result(0.0, 3600.0, False)]
    out = _postprocess(results, TimeEfficiencyConfig(enabled=True))

    # 1.0 - 30/60 = 0.5 and 0.0 - 60/60 = -1.0 -> mean -0.25
    assert out.rollout_metrics["total_reward/mean"] == pytest.approx(-0.25)
    assert torch.allclose(out.final_batch["total_reward"], torch.tensor([0.5, -1.0]))
    assert out.rollout_metrics["time_efficiency/minutes/mean"] == pytest.approx(45.0)
    assert out.rollout_metrics["time_efficiency/deduction/max"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "apply_to, expected_rewards, expected_deduction_max",
    [("all", [-0.5, -1.0], 1.0), ("correct", [0.0, 0.0], 0.0)],
)
def test_deduction_runs_after_reward_zeroing_penalties(
    apply_to, expected_rewards, expected_deduction_max
):
    # Both rollouts have an empty response.output, so penalize_empty_final_answer
    # zeroes them before the deduction runs. Under "all" the zeroed rollouts are
    # still charged for their wall time; under "correct" a zeroed rollout counts
    # as a failure and is not charged.
    results = [_result(1.0, 1800.0, True), _result(0.0, 3600.0, False)]
    out = _postprocess(
        results,
        TimeEfficiencyConfig(enabled=True, apply_to=apply_to),
        reward_penalty_config={"penalize_empty_final_answer": True},
    )

    assert out.rollout_metrics["empty_final_answer_rate"] == pytest.approx(1.0)
    assert torch.allclose(
        out.final_batch["total_reward"], torch.tensor(expected_rewards)
    )
    assert out.rollout_metrics["time_efficiency/deduction/max"] == pytest.approx(
        expected_deduction_max
    )


def test_disabled_leaves_rewards_and_metrics_untouched():
    results = [_result(1.0, 1800.0, True), _result(0.0, 3600.0, False)]
    out = _postprocess(results, None)

    assert out.rollout_metrics["total_reward/mean"] == pytest.approx(0.5)
    assert not any(k.startswith("time_efficiency/") for k in out.rollout_metrics)
