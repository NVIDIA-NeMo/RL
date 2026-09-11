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

from typing import Any

from nemo_rl.algorithms.mlperf_grpo_logging import (
    MLPerfGRPOLogger,
    create_mlperf_logger,
    mlperf_enabled,
)


class _FakeConstants:
    ADAMW = "adamw"

    def __getattr__(self, name: str) -> str:
        return name.lower()


class _FakeMLLogger:
    def __init__(self) -> None:
        self.constants = _FakeConstants()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def start(self, **kwargs: Any) -> None:
        self.calls.append(("start", kwargs))

    def end(self, **kwargs: Any) -> None:
        self.calls.append(("end", kwargs))

    def event(self, **kwargs: Any) -> None:
        self.calls.append(("event", kwargs))

    def log_init_stop_run_start(self) -> None:
        self.calls.append(("init_stop_run_start", {}))

    def mlperf_submission_log(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("submission", {"args": args, "kwargs": kwargs}))


def _config() -> dict[str, Any]:
    return {
        "logger": {
            "mlperf_enabled": True,
            "mlperf": {
                "benchmark": "grpo_nemo_gym",
                "target_accuracy": 0.75,
                "force_success_status": False,
            },
        },
        "cluster": {"num_nodes": 4},
        "loss_fn": {
            "truncated_importance_sampling_ratio_min": 0.999,
            "truncated_importance_sampling_ratio": 1.002,
            "truncated_importance_sampling_type": "seq-mask-tis",
        },
        "grpo": {
            "seed": 42,
            "max_num_steps": 4,
            "val_period": 2,
            "num_prompts_per_step": 2,
            "num_generations_per_prompt": 4,
            "validation_generation": {"temperature": 0.0, "top_p": 1.0},
        },
        "policy": {
            "train_global_batch_size": 8,
            "train_micro_batch_size": 1,
            "max_total_sequence_length": 1024,
            "precision": "bfloat16",
            "megatron_cfg": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 4,
                "optimizer": {
                    "lr": 5e-6,
                    "min_lr": 5e-6,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_eps": 1e-8,
                    "weight_decay": 0.0,
                    "clip_grad": 1.0,
                },
                "scheduler": {
                    "lr_warmup_iters": 3,
                    "lr_decay_iters": 100,
                    "lr_decay_style": "constant",
                },
            },
            "generation": {
                "backend": "vllm",
                "temperature": 1.0,
                "top_p": 1.0,
                "vllm_cfg": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                    "expert_parallel_size": 4,
                },
            },
        },
    }


def test_mlperf_enabled_accepts_both_supported_flags() -> None:
    assert not mlperf_enabled({"logger": {}})
    assert mlperf_enabled({"logger": {"mlperf_enabled": True}})
    assert mlperf_enabled({"logger": {"mlperf": {"enabled": True}}})
    assert create_mlperf_logger({"logger": {}}) is None


def test_mlperf_grpo_logger_tracks_lifecycle_and_target() -> None:
    config = _config()
    fake = _FakeMLLogger()
    logger = MLPerfGRPOLogger(config, mllogger=fake)

    logger.log_init_start()
    logger.log_hyperparams(train_dataset=range(3), val_dataset=range(2))
    logger.log_hyperparams_after_setup(data_parallel_size=2)
    logger.log_init_stop_run_start()

    logger.start_eval(0)
    logger.end_eval(
        0,
        {"accuracy": 0.5},
        {"total_validation_time": 1.25},
    )
    assert logger.block_started

    logger.observe_metrics(
        {"loss": 0.2, "reward": 0.4, "grad_norm": 0.6},
        step=1,
        prefix="train",
    )
    logger.start_eval(2)
    logger.end_eval(2, {"accuracy": 0.8})

    # The train block is open, so the eval is held until the step's train
    # metrics are logged: they land before BLOCK_STOP and run_stop.
    assert not logger.run_stopped
    logger.observe_metrics(
        {"loss": 0.1, "reward": 0.1},
        step=2,
        prefix="train",
    )
    logger.observe_metrics(
        {"total_step_time": 1.0},
        step=2,
        prefix="timing/train",
        step_finished=True,
    )

    assert logger.target_reached
    assert logger.run_stopped
    assert not logger.block_started
    assert config["grpo"]["max_num_steps"] == 2

    run_stop_calls = [
        kwargs
        for method, kwargs in fake.calls
        if method == "end" and kwargs.get("key") == "run_stop"
    ]
    assert len(run_stop_calls) == 1
    assert run_stop_calls[0]["metadata"] == {
        "samples_count": 16,
        "status": "success",
    }

    tracked_stats = [
        kwargs["value"]
        for method, kwargs in fake.calls
        if method == "event" and kwargs.get("key") == "tracked_stats"
    ]
    assert {"reduced_train_loss": 0.2, "reward": 0.4, "grad_norm": 0.6} in tracked_stats

    importance_sampling_calls = {
        kwargs["key"]: kwargs["value"]
        for method, kwargs in fake.calls
        if method == "event"
        and kwargs.get("key")
        in {
            "truncated_importance_sampling_ratio_min",
            "truncated_importance_sampling_ratio",
            "truncated_importance_sampling_type",
        }
    }
    assert importance_sampling_calls == {
        "truncated_importance_sampling_ratio_min": 0.999,
        "truncated_importance_sampling_ratio": 1.002,
        "truncated_importance_sampling_type": "seq-mask-tis",
    }

    gradient_accumulation_calls = [
        kwargs
        for method, kwargs in fake.calls
        if method == "event" and kwargs.get("key") == "gradient_accumulation_steps"
    ]
    assert gradient_accumulation_calls == [
        {
            "key": "gradient_accumulation_steps",
            "value": 4,
            "metadata": {},
        }
    ]


def test_mlperf_grpo_logger_allows_disabled_target() -> None:
    config = _config()
    config["logger"]["mlperf"]["target_accuracy"] = None
    fake = _FakeMLLogger()
    logger = MLPerfGRPOLogger(config, mllogger=fake)
    logger.log_init_stop_run_start()

    logger.start_eval(2)
    logger.end_eval(2, {"accuracy": 1.0})

    assert logger.target_accuracy is None
    assert not logger.target_reached
    assert not logger.run_stopped
    assert config["grpo"]["max_num_steps"] == 4


def test_mlperf_train_block_start_and_stop_sample_counts_match() -> None:
    config = _config()
    config["grpo"].update(
        {
            "max_num_steps": 4,
            "val_period": 1,
            "val_start_at": 3,
        }
    )
    fake = _FakeMLLogger()
    logger = MLPerfGRPOLogger(config, mllogger=fake)

    logger.start_train_block(0)
    logger.stop_train_block(3)
    logger.start_train_block(3)
    logger.stop_train_block(4)

    block_start_counts = [
        kwargs["metadata"]["samples_count"]
        for method, kwargs in fake.calls
        if method == "start" and kwargs.get("key") == "block_start"
    ]
    block_stop_counts = [
        kwargs["metadata"]["samples_count"]
        for method, kwargs in fake.calls
        if method == "end" and kwargs.get("key") == "block_stop"
    ]
    assert block_start_counts == block_stop_counts == [24, 8]


def test_mlperf_final_eval_defers_run_stop_until_train_metrics() -> None:
    config = _config()
    fake = _FakeMLLogger()
    logger = MLPerfGRPOLogger(config, mllogger=fake)
    logger.log_init_stop_run_start()

    final_step = config["grpo"]["max_num_steps"]
    logger.start_eval(final_step)
    logger.end_eval(final_step, {"accuracy": 0.5})

    assert not logger.run_stopped
    assert logger.pending_run_stop_status == "aborted"

    logger.observe_metrics(
        {"loss": 0.2, "reward": 0.4},
        step=final_step,
        prefix="train",
    )
    logger.finalize()

    tracked_index = next(
        index
        for index, (method, kwargs) in enumerate(fake.calls)
        if method == "event"
        and kwargs.get("key") == "tracked_stats"
        and kwargs.get("value") == {"reduced_train_loss": 0.2, "reward": 0.4}
    )
    run_stop_index = next(
        index
        for index, (method, kwargs) in enumerate(fake.calls)
        if method == "end" and kwargs.get("key") == "run_stop"
    )
    assert tracked_index < run_stop_index
    assert logger.run_stopped
    assert logger.pending_run_stop_status is None
    assert fake.calls[run_stop_index][1]["metadata"] == {
        "samples_count": final_step * config["policy"]["train_global_batch_size"],
        "status": "aborted",
    }


def test_mlperf_held_eval_orders_train_stats_before_block_stop() -> None:
    """A target-reaching eval must not suppress the final step's train stats."""
    config = _config()
    fake = _FakeMLLogger()
    logger = MLPerfGRPOLogger(config, mllogger=fake)
    logger.log_init_stop_run_start()
    logger.start_train_block(0)

    # The trainers validate mid-step and log the step's train metrics after.
    logger.start_eval(2)
    assert not any(
        method == "start" and kwargs.get("key") == "eval_start"
        for method, kwargs in fake.calls
    ), "held eval must not open the eval interval while a train block is open"
    logger.end_eval(2, {"accuracy": 0.9}, {"total_validation_time": 2.0})
    logger.observe_metrics(
        {"loss": 0.2, "reward": 0.4}, step=2, prefix="train"
    )
    logger.observe_metrics(
        {"total_step_time": 3.0}, step=2, prefix="timing/train", step_finished=True
    )

    def _index(method: str, key: str) -> int:
        return next(
            index
            for index, (m, kwargs) in enumerate(fake.calls)
            if m == method and kwargs.get("key") == key
        )

    train_stats_index = next(
        index
        for index, (m, kwargs) in enumerate(fake.calls)
        if m == "event"
        and kwargs.get("key") == "tracked_stats"
        and kwargs.get("value", {}).get("reward") == 0.4
    )
    block_stop_index = _index("end", "block_stop")
    assert train_stats_index < block_stop_index
    assert block_stop_index < _index("start", "eval_start")
    assert _index("event", "eval_accuracy") < _index("end", "eval_stop")
    assert _index("end", "eval_stop") < _index("end", "run_stop")
    # Held eval events are backdated by the validation duration.
    assert fake.calls[_index("start", "eval_start")][1].get("time_ms")
    assert fake.calls[block_stop_index][1].get("time_ms")
    assert logger.run_stopped
    assert fake.calls[-1] == (
        "end",
        {"key": "run_stop", "metadata": {"samples_count": 16, "status": "success"}},
    )


def test_mlperf_defer_run_stop_suppresses_terminal_events() -> None:
    """Deferred (offline) evaluation: training emits no run_stop."""
    config = _config()
    config["logger"]["mlperf"]["defer_run_stop"] = True
    fake = _FakeMLLogger()
    logger = MLPerfGRPOLogger(config, mllogger=fake)
    logger.log_init_stop_run_start()
    logger.start_train_block(0)
    logger.observe_metrics({"loss": 0.2, "reward": 0.4}, step=1, prefix="train")

    # Trainer epilogue: finalize is a no-op while deferred.
    logger.finalize()
    assert not logger.run_stopped
    assert logger.block_started

    # The driver closes the train block at the final weight-update time; the
    # offline evaluator owns run_stop.
    logger.stop_train_block(3, time_ms=1_700_000_000_000)
    block_stop = fake.calls[-1]
    assert block_stop[0] == "end" and block_stop[1]["key"] == "block_stop"
    assert block_stop[1]["time_ms"] == 1_700_000_000_000
    assert block_stop[1]["metadata"]["samples_count"] == 24
    assert not any(
        method == "end" and kwargs.get("key") == "run_stop"
        for method, kwargs in fake.calls
    )

    # Failure path: the driver flips the flag and finalizes normally.
    logger.defer_run_stop = False
    logger.finalize()
    assert fake.calls[-1] == (
        "end",
        {"key": "run_stop", "metadata": {"samples_count": 24, "status": "aborted"}},
    )
