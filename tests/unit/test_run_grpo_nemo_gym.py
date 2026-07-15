import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from examples.nemo_gym import run_grpo_nemo_gym


def _rollout_result(
    rewards: list[float],
    rollout_time: float,
) -> SimpleNamespace:
    full_results = [json.dumps({"reward": reward}) for reward in rewards]
    return SimpleNamespace(
        rollout_metrics={
            "timing/rollout/total": rollout_time,
            "timing/rollout/run_rollouts": rollout_time - 1,
            "mean_gen_tokens_per_sample": 4.0,
            "test_agent/full_result": SimpleNamespace(
                data=[[result] for result in full_results]
            ),
        }
    )


def _master_config(
    *, expected_trajectories: int, wandb_enabled: bool = True
) -> SimpleNamespace:
    return SimpleNamespace(
        policy={
            "generation": {
                "colocated": {"enabled": False},
                "vllm_cfg": {
                    "async_engine": True,
                    "enable_vllm_metrics_logger": True,
                    "vllm_metrics_logger_interval": 0.5,
                },
            },
            "max_total_sequence_length": 4096,
        },
        grpo={"max_val_samples": expected_trajectories},
        logger={"wandb_enabled": wandb_enabled},
    )


def test_pop_trajectory_collection_settings() -> None:
    nemo_gym_config = {
        "is_trajectory_collection": True,
        "trajectory_collection_batch_size": 16,
        "config_paths": ["gym.yaml"],
    }

    settings = run_grpo_nemo_gym._pop_trajectory_collection_settings(nemo_gym_config)

    assert settings == (True, 16)
    assert nemo_gym_config == {"config_paths": ["gym.yaml"]}


@pytest.mark.parametrize("batch_size", [True, 0, -1, 1.5, "16"])
def test_pop_trajectory_collection_settings_rejects_invalid_batch_size(
    batch_size: object,
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        run_grpo_nemo_gym._pop_trajectory_collection_settings(
            {
                "is_trajectory_collection": True,
                "trajectory_collection_batch_size": batch_size,
            }
        )


def test_pop_trajectory_collection_settings_requires_collection_mode() -> None:
    with pytest.raises(ValueError, match="requires"):
        run_grpo_nemo_gym._pop_trajectory_collection_settings(
            {
                "is_trajectory_collection": False,
                "trajectory_collection_batch_size": 16,
            }
        )


def test_collect_trajectories_logs_each_batch_and_generation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_policy_generation = MagicMock()
    run_rollout = MagicMock(
        side_effect=[
            _rollout_result([1.0, 0.0], rollout_time=12.0),
            _rollout_result([0.5], rollout_time=18.0),
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

    run_grpo_nemo_gym.collect_trajectories(
        policy=MagicMock(),
        policy_generation=policy_generation,
        val_dataloader=[object(), object()],
        tokenizer=MagicMock(),
        val_task_to_env={"nemo_gym": MagicMock()},
        logger=logger,
        master_config=_master_config(expected_trajectories=3),
    )

    refit_policy_generation.assert_called_once()
    assert policy_generation.clear_logger_metrics.call_count == 2
    assert policy_generation.get_logger_metrics.call_count == 2
    policy_generation.finish_generation.assert_called_once_with()

    assert logger.log_string_list_as_jsonl.call_count == 2
    logged_batches = [
        [json.loads(row) for row in log_call.args[0]]
        for log_call in logger.log_string_list_as_jsonl.call_args_list
    ]
    assert [
        result["trajectory_collection_batch_index"]
        for batch in logged_batches
        for result in batch
    ] == [0, 0, 1]
    assert [
        result["trajectory_collection_batch_position"]
        for batch in logged_batches
        for result in batch
    ] == [0, 1, 0]
    assert [
        result["trajectory_collection_batch_size"]
        for batch in logged_batches
        for result in batch
    ] == [2, 2, 1]

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
        "mean_reward": 0.5,
        "num_trajectories": 3,
    }
    assert collection_log_calls[-1].args[1] == 2
    assert collection_log_calls[-1].kwargs["step_finished"] is True


def test_collect_trajectories_rejects_empty_validation_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_policy_generation = MagicMock()
    monkeypatch.setattr(
        run_grpo_nemo_gym, "refit_policy_generation", refit_policy_generation
    )

    with pytest.raises(ValueError, match="non-empty validation dataset"):
        run_grpo_nemo_gym.collect_trajectories(
            policy=MagicMock(),
            policy_generation=MagicMock(),
            val_dataloader=[],
            tokenizer=MagicMock(),
            val_task_to_env={"nemo_gym": MagicMock()},
            logger=MagicMock(),
            master_config=_master_config(expected_trajectories=0),
        )

    refit_policy_generation.assert_not_called()


def test_collect_trajectories_skips_generation_metrics_without_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_grpo_nemo_gym, "refit_policy_generation", MagicMock())
    monkeypatch.setattr(
        run_grpo_nemo_gym,
        "run_async_nemo_gym_rollout",
        MagicMock(return_value=_rollout_result([1.0], rollout_time=12.0)),
    )
    log_generation_metrics = MagicMock()
    monkeypatch.setattr(
        run_grpo_nemo_gym,
        "log_generation_metrics_to_wandb",
        log_generation_metrics,
    )
    policy_generation = MagicMock()

    run_grpo_nemo_gym.collect_trajectories(
        policy=MagicMock(),
        policy_generation=policy_generation,
        val_dataloader=[object()],
        tokenizer=MagicMock(),
        val_task_to_env={"nemo_gym": MagicMock()},
        logger=MagicMock(),
        master_config=_master_config(expected_trajectories=1, wandb_enabled=False),
    )

    policy_generation.clear_logger_metrics.assert_not_called()
    policy_generation.get_logger_metrics.assert_not_called()
    log_generation_metrics.assert_not_called()
    policy_generation.finish_generation.assert_called_once_with()


def test_collect_trajectories_preserves_completed_batches_before_incomplete_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_grpo_nemo_gym, "refit_policy_generation", MagicMock())
    monkeypatch.setattr(
        run_grpo_nemo_gym,
        "run_async_nemo_gym_rollout",
        MagicMock(return_value=_rollout_result([1.0], rollout_time=12.0)),
    )
    policy_generation = MagicMock()
    logger = MagicMock()

    with pytest.raises(RuntimeError, match="expected 2, got 1"):
        run_grpo_nemo_gym.collect_trajectories(
            policy=MagicMock(),
            policy_generation=policy_generation,
            val_dataloader=[object()],
            tokenizer=MagicMock(),
            val_task_to_env={"nemo_gym": MagicMock()},
            logger=logger,
            master_config=_master_config(expected_trajectories=2),
        )

    logger.log_string_list_as_jsonl.assert_called_once()
    policy_generation.finish_generation.assert_called_once_with()


def test_collect_trajectories_finishes_generation_after_rollout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_grpo_nemo_gym, "refit_policy_generation", MagicMock())
    monkeypatch.setattr(
        run_grpo_nemo_gym,
        "run_async_nemo_gym_rollout",
        MagicMock(side_effect=RuntimeError("rollout failed")),
    )
    policy_generation = MagicMock()

    with pytest.raises(RuntimeError, match="rollout failed"):
        run_grpo_nemo_gym.collect_trajectories(
            policy=MagicMock(),
            policy_generation=policy_generation,
            val_dataloader=[object()],
            tokenizer=MagicMock(),
            val_task_to_env={"nemo_gym": MagicMock()},
            logger=MagicMock(),
            master_config=_master_config(expected_trajectories=1),
        )

    policy_generation.finish_generation.assert_called_once_with()
