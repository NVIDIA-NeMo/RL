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

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

RUNNER = Path(__file__).resolve().parents[2] / "examples/nemo_gym/run_grpo_nemo_gym.py"


def _load_collect_trajectories() -> tuple[object, dict[str, object]]:
    tree = ast.parse(RUNNER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "collect_trajectories"
    )
    namespace: dict[str, object] = {
        name: object
        for name in (
            "ColocatablePolicyInterface",
            "GenerationInterface",
            "StatefulDataLoader",
            "TokenizerType",
            "EnvironmentInterface",
            "Logger",
            "MasterConfig",
            "Table",
        )
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER), "exec"),
        namespace,
    )
    return namespace["collect_trajectories"], namespace


def test_trajectory_collection_logs_spec_decode_metrics_per_batch(monkeypatch) -> None:
    collect_trajectories, namespace = _load_collect_trajectories()
    policy_generation = Mock()
    generation_metrics = [
        {
            "vllm/spec_num_drafts": 10.0,
            "vllm/spec_acceptance_rate-pos-1": 0.8,
        },
        {
            "vllm/spec_num_drafts": 20.0,
            "vllm/spec_acceptance_rate-pos-1": 0.7,
        },
    ]
    policy_generation.get_step_metrics.side_effect = generation_metrics
    logger = Mock()
    monkeypatch.setitem(namespace, "refit_policy_generation", Mock())
    monkeypatch.setitem(
        namespace,
        "run_nemo_gym_rollout_sync",
        Mock(
            return_value=SimpleNamespace(
                rollout_metrics={"full_result": SimpleNamespace(data=[["row"]])}
            )
        ),
    )

    collect_trajectories(
        policy=Mock(),
        policy_generation=policy_generation,
        val_dataloader=[{"batch": 1}, {"batch": 2}],
        tokenizer=Mock(),
        val_task_to_env={},
        logger=logger,
        master_config=SimpleNamespace(
            policy={
                "generation": {"colocated": {"enabled": False}},
                "max_total_sequence_length": 4096,
            }
        ),
    )

    assert policy_generation.snapshot_step_metrics.call_count == 2
    assert policy_generation.get_step_metrics.call_count == 2
    assert logger.log_metrics.call_args_list == [
        (
            (generation_metrics[0],),
            {
                "step": 1,
                "prefix": "trajectory_collection",
            },
        ),
        (
            (generation_metrics[1],),
            {
                "step": 2,
                "prefix": "trajectory_collection",
            },
        ),
    ]
    assert logger.log_string_list_as_jsonl.call_count == 2
    policy_generation.finish_generation.assert_called_once_with()


def test_trajectory_is_persisted_before_metric_collection_failure(monkeypatch) -> None:
    collect_trajectories, namespace = _load_collect_trajectories()
    policy_generation = Mock()
    policy_generation.get_step_metrics.side_effect = RuntimeError("metrics failed")
    logger = Mock()
    monkeypatch.setitem(namespace, "refit_policy_generation", Mock())
    monkeypatch.setitem(
        namespace,
        "run_nemo_gym_rollout_sync",
        Mock(
            return_value=SimpleNamespace(
                rollout_metrics={"full_result": SimpleNamespace(data=[["saved-row"]])}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="metrics failed"):
        collect_trajectories(
            policy=Mock(),
            policy_generation=policy_generation,
            val_dataloader=[{"batch": 1}],
            tokenizer=Mock(),
            val_task_to_env={},
            logger=logger,
            master_config=SimpleNamespace(
                policy={
                    "generation": {"colocated": {"enabled": False}},
                    "max_total_sequence_length": 4096,
                }
            ),
        )

    logger.log_string_list_as_jsonl.assert_called_once_with(
        ["saved-row"], "trajectory_collection.jsonl"
    )
