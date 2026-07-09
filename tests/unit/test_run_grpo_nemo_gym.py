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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from examples.nemo_gym import run_grpo_nemo_gym


def _rollout_result(instance_id: str, reward: int, rollout_time: float):
    full_result = {
        "responses_create_params": {"metadata": {"instance_id": instance_id}},
        "reward": reward,
    }
    return SimpleNamespace(
        rollout_metrics={
            "timing/rollout/total": rollout_time,
            "timing/rollout/run_rollouts": rollout_time - 1,
            "mean_gen_tokens_per_sample": 4.0,
            "swe_agent/full_result": SimpleNamespace(data=[[json.dumps(full_result)]]),
        }
    )


def test_collect_trajectories_logs_rollout_and_generation_metrics(monkeypatch):
    refit_policy_generation = MagicMock()
    run_rollout = MagicMock(
        side_effect=[
            _rollout_result("instance-1", reward=1, rollout_time=12.0),
            _rollout_result("instance-2", reward=0, rollout_time=18.0),
        ]
    )
    log_generation_metrics = MagicMock()
    monkeypatch.setattr(
        run_grpo_nemo_gym, "refit_policy_generation", refit_policy_generation
    )
    monkeypatch.setattr(run_grpo_nemo_gym, "run_async_nemo_gym_rollout", run_rollout)
    monkeypatch.setattr(
        run_grpo_nemo_gym,
        "log_generation_metrics_to_wandb",
        log_generation_metrics,
    )

    generation_metrics = [
        {"inflight_batch_sizes": {0: [1, 0]}},
        {"inflight_batch_sizes": {0: [2, 0]}},
    ]
    policy_generation = MagicMock()
    policy_generation.get_logger_metrics.side_effect = generation_metrics
    logger = MagicMock()
    master_config = SimpleNamespace(
        policy={
            "generation": {
                "colocated": {"enabled": False},
                "vllm_cfg": {
                    "enable_vllm_metrics_logger": True,
                    "vllm_metrics_logger_interval": 0.5,
                },
            },
            "max_total_sequence_length": 4096,
        },
        grpo={"max_val_samples": 2},
        logger={"wandb_enabled": True},
    )

    run_grpo_nemo_gym.collect_trajectories(
        policy=MagicMock(),
        policy_generation=policy_generation,
        val_dataloader=[object(), object()],
        tokenizer=MagicMock(),
        val_task_to_env={"nemo_gym": MagicMock()},
        logger=logger,
        master_config=master_config,
    )

    assert policy_generation.clear_logger_metrics.call_count == 2
    assert policy_generation.get_logger_metrics.call_count == 2
    policy_generation.finish_generation.assert_called_once_with()
    assert logger.log_string_list_as_jsonl.call_count == 2
    logged_batches = [
        [json.loads(row) for row in log_call.args[0]]
        for log_call in logger.log_string_list_as_jsonl.call_args_list
    ]
    assert [
        batch[0]["trajectory_collection_batch_index"] for batch in logged_batches
    ] == [
        0,
        1,
    ]
    assert all(
        batch[0]["trajectory_collection_batch_position"] == 0
        for batch in logged_batches
    )
    assert all(
        batch[0]["trajectory_collection_batch_size"] == 1 for batch in logged_batches
    )

    rollout_log_calls = [
        log_call
        for log_call in logger.log_metrics.call_args_list
        if log_call.kwargs.get("prefix") == "train"
    ]
    assert [
        log_call.args[0]["timing/rollout/total"] for log_call in rollout_log_calls
    ] == [12.0, 18.0]
    assert [log_call.args[1] for log_call in rollout_log_calls] == [1, 2]
    assert all(
        "full_result" not in key
        for log_call in rollout_log_calls
        for key in log_call.args[0]
    )

    assert log_generation_metrics.call_args_list == [
        call(generation_metrics[0], 1, 0.5, logger),
        call(generation_metrics[1], 2, 0.5, logger),
    ]
    collection_log_calls = [
        log_call
        for log_call in logger.log_metrics.call_args_list
        if log_call.kwargs.get("prefix") == "trajectory_collection"
    ]
    assert collection_log_calls[-1].args[0] == {
        "accuracy": 0.5,
        "num_resolved": 1,
        "num_trajectories": 2,
    }
    assert collection_log_calls[-1].args[1] == 2
    assert collection_log_calls[-1].kwargs["step_finished"] is True
