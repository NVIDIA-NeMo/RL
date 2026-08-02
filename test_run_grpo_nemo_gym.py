"""Regression tests for trajectory collection in run_grpo_nemo_gym."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from wandb import Table

from examples.nemo_gym.run_grpo_nemo_gym import collect_trajectories


def test_collect_trajectories_logs_only_full_result_metrics():
    """Only rollout metrics whose keys contain 'full_result' are persisted."""
    full_table = Table(columns=["json"])
    full_table.add_data('{"a": 1}')
    full_table.add_data('{"b": 2}')

    prefix_full_table = Table(columns=["json"])
    prefix_full_table.add_data('{"c": 3}')

    ignored_table = Table(columns=["json"])
    ignored_table.add_data('{"ignored": true}')

    rollout_result = MagicMock()
    rollout_result.rollout_metrics = {
        "full_result_a": full_table,
        "other_metric": 123,
        "not_full_result_table": ignored_table,
        "prefix_full_result_b": prefix_full_table,
    }

    logger = MagicMock()
    policy_generation = MagicMock()
    master_config = {
        "policy": {
            "generation": {"colocated": {"enabled": False}},
            "max_total_sequence_length": 128,
        }
    }
    val_dataloader = iter([{}])
    tokenizer = MagicMock()
    val_task_to_env = {}

    with (
        patch(
            "examples.nemo_gym.run_grpo_nemo_gym.run_nemo_gym_rollout_sync",
            return_value=rollout_result,
        ),
        patch("examples.nemo_gym.run_grpo_nemo_gym.refit_policy_generation"),
    ):
        collect_trajectories(
            policy=MagicMock(),
            policy_generation=policy_generation,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            val_task_to_env=val_task_to_env,
            logger=logger,
            master_config=master_config,
        )

    logger.log_string_list_as_jsonl.assert_called_once_with(
        ['{"a": 1}', '{"b": 2}', '{"c": 3}'], "trajectory_collection.jsonl"
    )
