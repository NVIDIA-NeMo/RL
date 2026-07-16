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

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch
from wandb import Table

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.evals.eval import (
    EVAL_STEP_METRIC,
    NemoGymEvalDataConfig,
    _ensure_nemo_gym_generation_metrics,
    _get_num_generations_per_prompt,
    _generate_texts,
    _log_generation_metrics,
    _run_nemo_gym_eval_impl,
    _summarize_generation_metrics,
    _validate_nemo_gym_eval_config,
    eval_cons_k,
    eval_pass_k,
    run_env_eval,
    setup,
    setup_nemo_gym_environment,
)
from nemo_rl.utils.logger import Logger


def test_eval_pass_k_basic():
    """Test basic pass@k evaluation."""
    # Test case: 3 samples, 2 correct, k=1
    rewards = torch.tensor([1.0, 0.0, 1.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    group_size = len(rewards) / num_tests_per_prompt
    average_score = score / group_size
    expected = 2 / 3
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("generation_config", "expected_use_async"),
    [
        ({"backend": "sglang"}, False),
        ({"backend": "sglang", "use_async_rollouts": False}, False),
        ({"backend": "sglang", "use_async_rollouts": True}, True),
        ({"backend": "vllm", "vllm_cfg": {"async_engine": False}}, False),
        ({"backend": "vllm", "vllm_cfg": {"async_engine": True}}, True),
    ],
)
def test_run_env_eval_selects_backend_specific_async_path(
    monkeypatch, generation_config, expected_use_async
):
    captured = {}

    async def fake_run_env_eval_impl(
        vllm_generation,
        dataloader,
        env,
        master_config,
        use_async=False,
        tokenizer=None,
        logger=None,
    ):
        captured["use_async"] = use_async

    monkeypatch.setattr("nemo_rl.evals.eval._run_env_eval_impl", fake_run_env_eval_impl)

    run_env_eval(
        vllm_generation=object(),
        dataloader=object(),
        env=object(),
        master_config=SimpleNamespace(generation=generation_config, env={}),
    )

    assert captured["use_async"] is expected_use_async


def test_generate_texts_uses_generation_interface_for_sglang() -> None:
    generation = MagicMock()
    generation.generate.return_value = BatchedDataDict(
        {
            "output_ids": torch.tensor([[10, 11, 20, 21], [12, 30, 0, 0]]),
            "unpadded_sequence_lengths": torch.tensor([4, 2]),
        }
    )
    batch = BatchedDataDict(
        {
            "message_log": [
                [{"role": "user", "content": "a", "token_ids": torch.tensor([10, 11])}],
                [{"role": "user", "content": "b", "token_ids": torch.tensor([12])}],
            ]
        }
    )
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.decode.side_effect = lambda tokens, **_: ",".join(
        str(token) for token in tokens.tolist()
    )

    texts = asyncio.run(
        _generate_texts(
            generation,
            BatchedDataDict({"prompts": ["a", "b"]}),
            False,
            backend="sglang",
            batch=batch,
            tokenizer=tokenizer,
        )
    )

    assert texts == ["20,21", "30"]
    generation.generate.assert_called_once()
    generation_inputs = generation.generate.call_args.args[0]
    assert generation_inputs["input_ids"].tolist() == [[10, 11], [12, 0]]
    assert generation_inputs["input_lengths"].tolist() == [2, 1]
    assert generation.generate.call_args.kwargs == {"greedy": False}


def test_setup_dispatches_to_sglang_rollout_engine(monkeypatch) -> None:
    cluster = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.evals.eval.RayVirtualCluster", MagicMock(return_value=cluster)
    )
    generation = MagicMock()
    sglang_generation = MagicMock(return_value=generation)
    monkeypatch.setattr("nemo_rl.evals.eval.SGLangGeneration", sglang_generation)
    config = SimpleNamespace(
        eval={
            "metric": "mean_reward",
            "num_tests_per_prompt": 1,
            "seed": 42,
            "k_value": 1,
        },
        generation={
            "backend": "sglang",
            "model_name": "test-model",
            "temperature": 0.0,
            "top_k": None,
            "num_prompts_per_step": 2,
            "sglang_cfg": {},
        },
        cluster={"gpus_per_node": 2, "num_nodes": 1},
        env={},
        data={"dataset_name": "AIME2024"},
        policy=None,
    )

    configured_generation, dataloader, returned_config = setup(
        config, MagicMock(), [object(), object(), object()]
    )

    assert configured_generation is generation
    assert len(dataloader) == 2
    assert returned_config is config
    assert config.generation["sglang_cfg"]["model_path"] == "test-model"
    sglang_generation.assert_called_once_with(
        cluster=cluster, sglang_cfg=config.generation
    )


def _nemo_gym_eval_config(
    *,
    metrics_enabled: bool = True,
    num_generations_per_prompt: int = 1,
    num_prompts_per_step: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        eval={
            "metric": "mean_reward",
            "num_tests_per_prompt": num_generations_per_prompt,
            "k_value": 1,
            "seed": 42,
            "save_path": None,
        },
        generation={
            "backend": "vllm",
            "model_name": "test-model",
            "max_new_tokens": 128,
            "temperature": 1.0 if num_generations_per_prompt > 1 else 0.0,
            "top_p": 1.0,
            "top_k": None,
            "num_prompts_per_step": num_prompts_per_step,
            "stop_strings": None,
            "stop_token_ids": None,
            "vllm_cfg": {
                "async_engine": True,
                "expose_http_server": True,
                "enable_vllm_metrics_logger": metrics_enabled,
                "vllm_metrics_logger_interval": 0.5,
            },
        },
        data=NemoGymEvalDataConfig(
            dataset_name="NemoGymDataset",
            data_path="eval.jsonl",
            processor="nemo_gym_data_processor",
            env_name="nemo_gym",
        ),
        env={"should_use_nemo_gym": True, "nemo_gym": {}},
        logger={"wandb_enabled": True},
    )


def test_nemo_gym_eval_can_derive_metrics_without_backend_telemetry() -> None:
    _validate_nemo_gym_eval_config(_nemo_gym_eval_config(metrics_enabled=False))


def test_nemo_gym_eval_accepts_megatron_rollout_engine() -> None:
    config = _nemo_gym_eval_config()
    config.generation = {
        **config.generation,
        "backend": "megatron",
        "mcore_generation_config": {
            "async_engine": True,
            "expose_http_server": True,
        },
    }

    _validate_nemo_gym_eval_config(config)


def test_setup_nemo_gym_environment_uses_rollout_engine_endpoints(monkeypatch) -> None:
    create_actor = MagicMock(return_value=object())
    monkeypatch.setattr("nemo_rl.evals.eval.create_nemo_gym_actor", create_actor)
    config = _nemo_gym_eval_config()
    config.env["nemo_gym"] = MagicMock()
    config.env["nemo_gym"].model_dump.return_value = {}
    generation = SimpleNamespace(dp_openai_server_base_urls=["http://worker-0:8000/v1"])

    result = setup_nemo_gym_environment(generation, config)

    assert result is create_actor.return_value
    create_actor.assert_called_once_with(
        model_name="test-model",
        base_urls=["http://worker-0:8000/v1"],
        nemo_gym_config={},
    )


def test_eval_generation_count_uses_num_tests_per_prompt() -> None:
    assert _get_num_generations_per_prompt({"num_tests_per_prompt": 2}) == 2


@pytest.mark.parametrize("value", [None, 0, -1, True])
def test_eval_generation_count_must_be_positive_integer(value) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _get_num_generations_per_prompt({"num_tests_per_prompt": value})


def test_nemo_gym_eval_requires_matching_dataset_switch() -> None:
    config = _nemo_gym_eval_config()
    config.env["should_use_nemo_gym"] = False

    with pytest.raises(ValueError, match="env.should_use_nemo_gym=true"):
        _validate_nemo_gym_eval_config(config)


def test_nemo_gym_eval_requires_null_top_k() -> None:
    config = _nemo_gym_eval_config()
    config.generation["top_k"] = 0

    with pytest.raises(ValueError, match="generation.top_k=null"):
        _validate_nemo_gym_eval_config(config)


def test_run_env_eval_dispatches_to_nemo_gym_rollout(monkeypatch) -> None:
    expected_result = object()
    run_nemo_gym_eval = MagicMock(return_value=expected_result)
    monkeypatch.setattr("nemo_rl.evals.eval._run_nemo_gym_eval_impl", run_nemo_gym_eval)

    vllm_generation = object()
    dataloader = object()
    nemo_gym = object()
    tokenizer = MagicMock()
    logger = MagicMock()
    config = _nemo_gym_eval_config()

    result = run_env_eval(
        vllm_generation=vllm_generation,
        dataloader=dataloader,
        env=nemo_gym,
        master_config=config,
        tokenizer=tokenizer,
        logger=logger,
    )

    assert result is expected_result
    logger.use_batch_steps.assert_called_once_with(
        EVAL_STEP_METRIC, ("eval/*", "generation_metrics/*", "ray/*")
    )
    run_nemo_gym_eval.assert_called_once()
    call_kwargs = run_nemo_gym_eval.call_args.kwargs
    assert call_kwargs["vllm_generation"] is vllm_generation
    assert call_kwargs["dataloader"] is dataloader
    assert call_kwargs["nemo_gym"] is nemo_gym
    assert call_kwargs["tokenizer"] is tokenizer
    assert call_kwargs["logger"] is logger
    assert call_kwargs["master_config"] is config


def test_run_env_eval_requires_runtime_logger_for_nemo_gym() -> None:
    with pytest.raises(
        ValueError,
        match="requires a logger to persist generation metrics",
    ):
        run_env_eval(
            vllm_generation=object(),
            dataloader=object(),
            env=object(),
            master_config=_nemo_gym_eval_config(),
            tokenizer=MagicMock(),
        )


def test_nemo_gym_eval_persists_raw_generation_metrics(tmp_path) -> None:
    config = _nemo_gym_eval_config()
    config.logger = {
        "log_dir": str(tmp_path),
        "wandb_enabled": False,
        "tensorboard_enabled": False,
        "mlflow_enabled": False,
        "swanlab_enabled": False,
        "monitor_gpus": False,
        "wandb": {},
        "tensorboard": {},
        "mlflow": {},
        "swanlab": {},
        "gpu_monitoring": {"collection_interval": 10, "flush_interval": 10},
    }
    metrics = {"inflight_batch_sizes": {0: [2, 1, 0]}}

    _log_generation_metrics(
        generation_metrics=metrics,
        step=3,
        generation_config=config.generation,
        logger_config=config.logger,
        require_numeric_metrics=True,
        logger=Logger(config.logger),
    )

    rows = (tmp_path / "generation_metrics.jsonl").read_text().splitlines()
    assert [json.loads(row) for row in rows] == [
        {
            "step": 3,
            "metrics": {"inflight_batch_sizes": {"0": [2, 1, 0]}},
        }
    ]


def test_nemo_gym_eval_rejects_empty_generation_metrics() -> None:
    with pytest.raises(
        RuntimeError,
        match="did not produce any numeric generation metrics",
    ):
        _log_generation_metrics(
            generation_metrics={},
            step=1,
            generation_config=_nemo_gym_eval_config().generation,
            logger_config=_nemo_gym_eval_config().logger,
            require_numeric_metrics=True,
            logger=MagicMock(),
        )


def test_eval_summarizes_generation_metrics_per_batch() -> None:
    metrics = {
        "generation_tokens": {0: [1, 3], 1: [2, 6]},
        "kv_cache_usage_perc": {0: [0.1, 0.2], 1: []},
        "non_numeric": {0: ["ignored"]},
    }

    assert _summarize_generation_metrics(metrics) == pytest.approx(
        {
            "generation_tokens/mean": 3.0,
            "generation_tokens/max": 6.0,
            "generation_tokens/last_mean": 4.5,
            "generation_tokens/last_sum": 9.0,
            "kv_cache_usage_perc/mean": 0.15,
            "kv_cache_usage_perc/max": 0.2,
            "kv_cache_usage_perc/last_mean": 0.2,
            "kv_cache_usage_perc/last_sum": 0.2,
        }
    )


def test_nemo_gym_eval_derives_backend_neutral_generation_metrics() -> None:
    assert _ensure_nemo_gym_generation_metrics(
        {},
        {
            "gen_tokens_per_sample/mean": 12.5,
            "timing/rollout/total": 2.0,
        },
        4,
    ) == {
        "completed_generations": {0: [4.0]},
        "generated_tokens_per_sample": {0: [12.5]},
        "generated_tokens": {0: [50.0]},
        "rollout_seconds": {0: [2.0]},
        "generated_tokens_per_rollout_second": {0: [25.0]},
    }


def test_nemo_gym_eval_logs_results_and_generation_metrics(monkeypatch) -> None:
    rollout_results = [
        SimpleNamespace(
            final_batch={"total_reward": torch.tensor([1.0, 0.0])},
            rollout_metrics={
                "timing/rollout/total": 2.0,
                "agent/full_result": Table(
                    columns=["Full result"],
                    data=[
                        [json.dumps({"reward": 1.0})],
                        [json.dumps({"reward": 0.0})],
                    ],
                ),
            },
        ),
        SimpleNamespace(
            final_batch={"total_reward": torch.tensor([0.5])},
            rollout_metrics={
                "timing/rollout/total": 1.0,
                "agent/full_result": Table(
                    columns=["Full result"],
                    data=[[json.dumps({"reward": 0.5})]],
                ),
            },
        ),
    ]
    run_rollout = MagicMock(side_effect=rollout_results)
    monkeypatch.setattr("nemo_rl.evals.eval.run_async_nemo_gym_rollout", run_rollout)
    monkeypatch.setattr("nemo_rl.evals.eval._print_results", MagicMock())
    monkeypatch.setattr("nemo_rl.evals.eval.ray.get", lambda value: value)
    log_generation_metrics = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.evals.eval.log_generation_metrics_to_wandb",
        log_generation_metrics,
    )

    vllm_generation = MagicMock()
    per_batch_generation_metrics = [
        {"inflight_batch_sizes": {0: [1, 0]}},
        {"inflight_batch_sizes": {0: [2, 0]}},
    ]
    vllm_generation.get_logger_metrics.side_effect = per_batch_generation_metrics
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([object(), object()])
    dataloader.__len__.return_value = 2
    dataloader.dataset = [object(), object(), object()]
    nemo_gym = MagicMock()
    logger = MagicMock()

    result = _run_nemo_gym_eval_impl(
        vllm_generation=vllm_generation,
        dataloader=dataloader,
        nemo_gym=nemo_gym,
        tokenizer=MagicMock(),
        master_config=_nemo_gym_eval_config(),
        logger=logger,
    )

    assert result.average_score == pytest.approx(0.5)
    assert result.num_samples == 3
    assert result.generation_metrics == [
        {"step": 1, "metrics": per_batch_generation_metrics[0]},
        {"step": 2, "metrics": per_batch_generation_metrics[1]},
    ]
    assert vllm_generation.clear_logger_metrics.call_count == 2
    assert vllm_generation.get_logger_metrics.call_count == 2
    vllm_generation.shutdown.assert_called_once_with()
    nemo_gym.shutdown.remote.assert_called_once_with()
    assert log_generation_metrics.call_args_list == [
        call(
            per_batch_generation_metrics[0],
            1,
            0.5,
            logger,
            step_metric=EVAL_STEP_METRIC,
        ),
        call(
            per_batch_generation_metrics[1],
            2,
            0.5,
            logger,
            step_metric=EVAL_STEP_METRIC,
        ),
    ]
    assert logger.flush_system_metrics.call_args_list == [call(1), call(2)]
    committed_batch_logs = [
        log_call
        for log_call in logger.log_metrics.call_args_list
        if log_call.kwargs.get("step_finished")
    ]
    assert [
        log_call.args[0][EVAL_STEP_METRIC] for log_call in committed_batch_logs
    ] == [
        1,
        2,
    ]
    assert committed_batch_logs[-1].args[0]["score"] == pytest.approx(0.5)

    generation_jsonl_calls = [
        log_call
        for log_call in logger.log_string_list_as_jsonl.call_args_list
        if log_call.args[1] == "generation_metrics.jsonl"
    ]
    result_jsonl_calls = [
        log_call
        for log_call in logger.log_string_list_as_jsonl.call_args_list
        if log_call.args[1] == "nemo_gym_eval_results.jsonl"
    ]
    assert len(generation_jsonl_calls) == 2
    assert len(result_jsonl_calls) == 2
    assert [
        (json.loads(row)["eval_batch_index"], json.loads(row)["eval_step"])
        for log_call in result_jsonl_calls
        for row in log_call.args[0]
    ] == [(0, 1), (0, 1), (1, 2)]


def test_nemo_gym_eval_tracks_multiple_generations_per_prompt(monkeypatch) -> None:
    rewards = [0.0, 1.0, 0.5, 1.0]
    rollout_result = SimpleNamespace(
        final_batch={"total_reward": torch.tensor(rewards)},
        rollout_metrics={
            "agent/full_result": Table(
                columns=["Full result"],
                data=[[json.dumps({"reward": reward})] for reward in rewards],
            ),
        },
    )
    run_rollout = MagicMock(return_value=rollout_result)
    monkeypatch.setattr("nemo_rl.evals.eval.run_async_nemo_gym_rollout", run_rollout)
    monkeypatch.setattr("nemo_rl.evals.eval._print_results", MagicMock())
    monkeypatch.setattr("nemo_rl.evals.eval.ray.get", lambda value: value)

    vllm_generation = MagicMock()
    vllm_generation.get_logger_metrics.return_value = {
        "inflight_batch_sizes": {0: [4, 0]}
    }
    original_batch = MagicMock()
    repeated_batch = MagicMock()
    original_batch.repeat_interleave.return_value = repeated_batch
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([original_batch])
    dataloader.__len__.return_value = 1
    dataloader.dataset = [object()]
    nemo_gym = MagicMock()
    logger = MagicMock()

    result = _run_nemo_gym_eval_impl(
        vllm_generation=vllm_generation,
        dataloader=dataloader,
        nemo_gym=nemo_gym,
        tokenizer=MagicMock(),
        master_config=_nemo_gym_eval_config(
            num_generations_per_prompt=4,
            num_prompts_per_step=1,
        ),
        logger=logger,
    )

    assert result.average_score == pytest.approx(0.625)
    original_batch.repeat_interleave.assert_called_once_with(4)
    assert run_rollout.call_args.kwargs["input_batch"] is repeated_batch
    result_jsonl_call = next(
        log_call
        for log_call in logger.log_string_list_as_jsonl.call_args_list
        if log_call.args[1] == "nemo_gym_eval_results.jsonl"
    )
    result_rows = [json.loads(row) for row in result_jsonl_call.args[0]]
    assert [row["prompt_index"] for row in result_rows] == [0, 0, 0, 0]
    assert [row["generation_index"] for row in result_rows] == [0, 1, 2, 3]
    assert all(row["num_generations_per_prompt"] == 4 for row in result_rows)
    committed_batch_log = next(
        log_call
        for log_call in logger.log_metrics.call_args_list
        if log_call.kwargs.get("step_finished")
    )
    assert committed_batch_log.args[0]["batch/num_generations_per_prompt"] == 4


def test_nemo_gym_eval_rejects_incomplete_full_results(monkeypatch) -> None:
    rollout_result = SimpleNamespace(
        final_batch={"total_reward": torch.tensor([1.0, 0.0])},
        rollout_metrics={
            "agent/full_result": Table(
                columns=["Full result"],
                data=[[json.dumps({"reward": 1.0})]],
            ),
        },
    )
    monkeypatch.setattr(
        "nemo_rl.evals.eval.run_async_nemo_gym_rollout",
        MagicMock(return_value=rollout_result),
    )
    monkeypatch.setattr("nemo_rl.evals.eval.ray.get", lambda value: value)

    vllm_generation = MagicMock()
    vllm_generation.get_logger_metrics.return_value = {
        "inflight_batch_sizes": {0: [1, 0]}
    }
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([object()])
    dataloader.dataset = [object(), object()]
    nemo_gym = MagicMock()

    with pytest.raises(
        RuntimeError,
        match="batch 0 returned 1 full results for 2 rewards",
    ):
        _run_nemo_gym_eval_impl(
            vllm_generation=vllm_generation,
            dataloader=dataloader,
            nemo_gym=nemo_gym,
            tokenizer=MagicMock(),
            master_config=_nemo_gym_eval_config(),
            logger=MagicMock(),
        )

    vllm_generation.shutdown.assert_called_once_with()
    nemo_gym.shutdown.remote.assert_called_once_with()


def test_eval_pass_k_all_correct():
    """Test pass@k when all samples are correct."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    group_size = len(rewards) / num_tests_per_prompt
    average_score = score / group_size
    expected = 1.0
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_none_correct():
    """Test pass@k when no samples are correct."""
    rewards = torch.tensor([0.0, 0.0, 0.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    average_score = score / (len(rewards) / num_tests_per_prompt)
    expected = 0.0
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_multiple_groups():
    """Test pass@k with multiple groups."""
    # Two groups: [1,0,1] and [0,1,0]
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    average_score = score / (len(rewards) / num_tests_per_prompt)
    expected = 0.5
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_cons_k_basic():
    """Test basic cons@k evaluation."""
    rewards = torch.tensor([1.0, 0.0, 1.0])
    extracted_answers = ["A", "B", "A"]
    num_tests_per_prompt = 3
    group_size = len(rewards) / num_tests_per_prompt
    score = eval_cons_k(
        rewards,
        num_tests_per_prompt=num_tests_per_prompt,
        k=1,
        extracted_answers=extracted_answers,
    )
    average_score = score / group_size
    expected = 2 / 3
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_cons_k_multiple_groups():
    """Test cons@k with multiple groups."""
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    num_tests_per_prompt = 5
    extracted_answers = [
        "Correct",
        "Wrong1",
        "Correct",
        "Wrong2",
        "Correct",
        "Wrong3",
        "Correct",
        "Wrong4",
        "Correct",
        "Wrong4",
    ]
    group_size = len(rewards) / num_tests_per_prompt
    score = eval_cons_k(
        rewards,
        num_tests_per_prompt=num_tests_per_prompt,
        k=3,
        extracted_answers=extracted_answers,
    )
    average_score = score / group_size

    """
    For the first group, the extracted answers are [Correct, Wrong1, Correct, Wrong2, Correct]
    When calculating unbiased estimate of cons@3(k=3), we need to consider the majority vote of all Combination(5, 3) = 10 cases.
    The 10 cases are:
    - Correct, Wrong1, Correct      Majority: Correct
    - Correct, Wrong1, Wrong2       Majority: Correct(Choose the first one when there is a tie)
    - Correct, Wrong1, Correct      Majority: Correct
    - Correct, Correct, Wrong2      Majority: Correct
    - Correct, Correct, Correct     Majority: Correct
    - Correct, Wrong2, Correct      Majority: Correct
    - Wrong1, Correct, Wrong2       Majority: Wrong1 (Choose the first one when there is a tie)
    - Wrong1, Correct, Correct      Majority: Correct
    - Wrong1, Wrong2, Correct       Majority: Wrong1 (Choose the first one when there is a tie)
    - Correct, Wrong2, Correct      Majority: Correct
    The final result is 8/10.

    For the second group, the extracted answers are [Wrong3, Correct, Wrong4, Correct, Wrong4]
    When calculating unbiased estimate of cons@3(k=3), we need to consider the majority vote of all Combination(5, 3) = 10 cases.
    The 10 cases are:
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Correct, Correct      Majority: Correct
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Wrong4, Correct       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Wrong4, Wrong4        Majority: Wrong4
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Correct, Wrong4, Correct      Majority: Correct
    - Correct, Wrong4, Wrong4       Majority: Wrong4 (Choose the first one when there is a tie)
    - Correct, Correct, Wrong4      Majority: Correct
    - Wrong4, Correct, Wrong4       Majority: Wrong4
    The final result is 3/10.
    Since there len(rewards)/num_tests_per_prompt = 10/5 = 2 groups
    The final result is( 8/10 + 3/10 ) / 2 = 11/20 = 0.55
    """
    expected = 11 / 20
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)
