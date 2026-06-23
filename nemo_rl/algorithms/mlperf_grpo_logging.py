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

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional

try:
    from mlperf_common.frameworks.pyt import PyTCommunicationHandler
    from mlperf_common.logging import MLLoggerWrapper
except ImportError:  # pragma: no cover - exercised without mlperf_common installed
    PyTCommunicationHandler = None  # type: ignore[assignment]
    MLLoggerWrapper = None  # type: ignore[assignment]


def mlperf_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether MLPerf logging is enabled in a NeMo-RL config."""
    logger_cfg = config.get("logger", {})
    if not isinstance(logger_cfg, Mapping):
        return False

    mlperf_cfg = logger_cfg.get("mlperf", {})
    if not isinstance(mlperf_cfg, Mapping):
        mlperf_cfg = {}

    return bool(logger_cfg.get("mlperf_enabled", mlperf_cfg.get("enabled", False)))


def create_mlperf_logger(
    config: dict[str, Any],
    mllogger: Optional[Any] = None,
) -> Optional["MLPerfGRPOLogger"]:
    """Create an MLPerf GRPO logger when enabled, otherwise return None."""
    if not mlperf_enabled(config):
        return None
    return MLPerfGRPOLogger(config=config, mllogger=mllogger)


class MLPerfGRPOLogger:
    """Small direct MLPerf lifecycle logger for GRPO training."""

    def __init__(self, config: dict[str, Any], mllogger: Optional[Any] = None) -> None:
        self.config = config
        self.mlperf_config = self._get_mlperf_config(config)
        self.mllogger = mllogger or self._create_mllogger()

        self.global_batch_size = int(
            config["policy"].get(
                "train_global_batch_size",
                config["grpo"]["num_prompts_per_step"]
                * config["grpo"]["num_generations_per_prompt"],
            )
        )
        self.target_accuracy = float(self.mlperf_config.get("target_accuracy", 1.0))
        self.force_success_status = bool(
            self.mlperf_config.get("force_success_status", False)
        )

        self.run_started = False
        self.run_stopped = False
        self.target_reached = False
        self.block_started = False
        self.block_start_step = 0
        self.last_step = 0

        log_file = self.mlperf_config.get("log_file")
        if log_file:
            self._configure_output(str(log_file))

    @staticmethod
    def _get_mlperf_config(config: Mapping[str, Any]) -> dict[str, Any]:
        logger_cfg = config.get("logger", {})
        if isinstance(logger_cfg, Mapping) and isinstance(
            logger_cfg.get("mlperf"), Mapping
        ):
            return dict(logger_cfg["mlperf"])
        return {}

    @staticmethod
    def _create_mllogger() -> Any:
        if MLLoggerWrapper is None or PyTCommunicationHandler is None:
            raise ImportError(
                "MLPerf logging is enabled, but mlperf_common is not installed. "
                "Install git+https://github.com/NVIDIA/mlperf-common.git before running."
            )
        return MLLoggerWrapper(PyTCommunicationHandler())

    @staticmethod
    def _configure_output(log_file: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        try:
            from mlperf_logging import mllog
        except ImportError:
            return

        try:
            mllog.config(filename=log_file)
        except TypeError:
            mllog.config(filename=log_file, root_dir=os.getcwd())

    @property
    def constants(self) -> Any:
        return self.mllogger.constants

    def _call(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        method = getattr(self.mllogger, method_name)
        try:
            method(*args, **kwargs)
        except TypeError:
            kwargs.pop("unique", None)
            method(*args, **kwargs)

    def _start(self, key: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self._call("start", key=key, metadata=metadata or {})

    def _end(self, key: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self._call("end", key=key, metadata=metadata or {})

    def _event(
        self,
        key: str,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
        unique: Optional[bool] = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "key": key,
            "value": value,
            "metadata": metadata or {},
        }
        if unique is not None:
            kwargs["unique"] = unique
        self._call("event", **kwargs)

    def sample_count(self, step: int) -> int:
        return int(step) * self.global_batch_size

    def block_size_samples(self) -> int:
        val_period = int(self.config["grpo"].get("val_period", 0))
        if val_period <= 0:
            val_period = int(self.config["grpo"].get("max_num_steps", 0))
        return val_period * self.global_batch_size

    def log_init_start(self) -> None:
        self._start(self.constants.INIT_START)

    def log_hyperparams(
        self,
        train_dataset: Any = None,
        val_dataset: Any = None,
    ) -> None:
        cfg = self.config
        grpo_cfg = cfg["grpo"]
        policy_cfg = cfg["policy"]
        megatron_cfg = policy_cfg.get("megatron_cfg", {})
        optimizer_cfg = megatron_cfg.get("optimizer", {})
        scheduler_cfg = megatron_cfg.get("scheduler", {})
        generation_cfg = policy_cfg.get("generation", {})
        backend_cfg = generation_cfg.get("vllm_cfg", {})

        benchmark = str(self.mlperf_config.get("benchmark", "grpo_nemo_gym"))

        if num_nodes := cfg.get("cluster", {}).get("num_nodes"):
            self.mllogger.mlperf_submission_log(
                benchmark=benchmark,
                num_nodes=num_nodes,
            )
        else:
            self.mllogger.mlperf_submission_log(benchmark)

        train_samples = self._dataset_len(train_dataset)
        if train_samples is not None:
            train_samples *= int(grpo_cfg["num_generations_per_prompt"])
        else:
            train_samples = int(grpo_cfg["max_num_steps"]) * self.global_batch_size
        eval_samples = self._dataset_len(val_dataset)
        if eval_samples is None:
            eval_samples = grpo_cfg.get("max_val_samples")

        # fmt: off
        logging_configs = {
            self.constants.SEED: grpo_cfg.get("seed"),
            self.constants.MAX_STEPS: grpo_cfg.get("max_num_steps"),
            self.constants.GLOBAL_BATCH_SIZE: self.global_batch_size,
            self.constants.MICRO_BATCH_SIZE: policy_cfg.get("train_micro_batch_size"),
            self.constants.MAX_SEQUENCE_LENGTH: policy_cfg.get("max_total_sequence_length"),
            self.constants.TRAIN_SAMPLES: train_samples,
            self.constants.EVAL_SAMPLES: eval_samples,
            self.constants.INIT_CHECKPOINT_STEP: 0,
            self.constants.OPT_NAME: self.constants.ADAMW,
            self.constants.OPT_BASE_LR: optimizer_cfg.get("lr"),
            self.constants.OPT_END_LR: optimizer_cfg.get("min_lr"),
            self.constants.OPT_ADAMW_BETA_1: optimizer_cfg.get("adam_beta1"),
            self.constants.OPT_ADAMW_BETA_2: optimizer_cfg.get("adam_beta2"),
            self.constants.OPT_ADAMW_EPSILON: optimizer_cfg.get("adam_eps"),
            self.constants.OPT_ADAMW_WEIGHT_DECAY: optimizer_cfg.get("weight_decay"),
            self.constants.OPT_GRADIENT_CLIP_NORM: optimizer_cfg.get("clip_grad"),
            self.constants.OPT_LR_WARMUP_STEPS: scheduler_cfg.get("lr_warmup_iters"),
            self.constants.OPT_LR_DECAY_STEPS: scheduler_cfg.get("lr_decay_iters"),
            self.constants.OPT_LR_DECAY_SCHEDULE: scheduler_cfg.get("lr_decay_style"),
            # Training step parallelism
            self.constants.TENSOR_PARALLELISM: megatron_cfg.get("tensor_model_parallel_size"),
            self.constants.PIPELINE_PARALLELISM: megatron_cfg.get("pipeline_model_parallel_size"),
            self.constants.CONTEXT_PARALLELISM: megatron_cfg.get("context_parallel_size"),
            self.constants.EXPERT_PARALLELISM: megatron_cfg.get("expert_model_parallel_size"),
            # Generation parallelism
            "generation_backend": generation_cfg.get("backend"),
            "generation_tensor_parallelism": backend_cfg.get("tensor_parallel_size"),
            "generation_pipeline_parallelism": backend_cfg.get("pipeline_parallel_size"),
            "generation_expert_parallelism": backend_cfg.get("expert_parallel_size"),
            "generation_training_rollout_temperature": generation_cfg.get("temperature"),
            "generation_training_rollout_top_p": generation_cfg.get("top_p"),
            "generation_validation_rollout_temperature": grpo_cfg.get("validation_generation", {}).get("temperature"),
            "generation_validation_rollout_top_p": grpo_cfg.get("validation_generation", {}).get( "top_p"),
            "num_prompts_per_step": grpo_cfg.get("num_prompts_per_step"),
            "num_generations_per_prompt": grpo_cfg.get("num_generations_per_prompt"),
            "target_accuracy": self.target_accuracy,
        }
        # fmt: on

        for key, value in logging_configs.items():
            if value is not None:
                self._event(key=key, value=value)

    @staticmethod
    def _dataset_len(dataset: Any) -> Optional[int]:
        if dataset is None:
            return None
        if isinstance(dataset, Mapping):
            return sum(len(ds) for ds in dataset.values())
        return len(dataset)

    def log_init_stop_run_start(self) -> None:
        if self.run_started:
            return
        self.mllogger.log_init_stop_run_start()
        self.run_started = True

    def start_train_block(self, step: int) -> None:
        if self.run_stopped or self.block_started:
            return
        self.last_step = max(self.last_step, int(step))
        self.block_started = True
        self.block_start_step = int(step)
        self._start(
            self.constants.BLOCK_START,
            metadata={
                self.constants.SAMPLES_COUNT: self.block_size_samples(),
                "step": int(step),
            },
        )

    def stop_train_block(self, step: int) -> None:
        if not self.block_started:
            return
        step = int(step)
        self.last_step = max(self.last_step, step)
        block_samples = max(0, step - self.block_start_step) * self.global_batch_size
        self._end(
            self.constants.BLOCK_STOP,
            metadata={
                self.constants.SAMPLES_COUNT: block_samples,
                "step": step,
            },
        )
        self.block_started = False

    def start_eval(self, step: int) -> None:
        step = int(step)
        self.stop_train_block(step)
        self._start(
            self.constants.EVAL_START,
            metadata={
                self.constants.SAMPLES_COUNT: self.sample_count(step),
                "step": step,
            },
        )

    def end_eval(
        self,
        step: int,
        val_metrics: Mapping[str, Any],
        validation_timings: Optional[Mapping[str, Any]] = None,
    ) -> None:
        step = int(step)
        self.last_step = max(self.last_step, step)
        samples_count = self.sample_count(step)
        accuracy = self._to_scalar(val_metrics.get("accuracy", 0.0))

        if validation_timings:
            validation_time = self._to_scalar(
                validation_timings.get("total_validation_time")
            )
            if validation_time is not None:
                self._event(
                    key="tracked_stats",
                    metadata={"step": step},
                    value={"validation_time": validation_time},
                    unique=False,
                )

        self._event(
            key=self.constants.EVAL_ACCURACY,
            metadata={self.constants.SAMPLES_COUNT: samples_count},
            value=accuracy,
        )
        self._end(
            self.constants.EVAL_STOP,
            metadata={
                self.constants.SAMPLES_COUNT: samples_count,
                "step": step,
            },
        )

        if accuracy is not None and accuracy >= self.target_accuracy:
            self.config["grpo"]["max_num_steps"] = step
            self.stop_run(status="success", samples_count=samples_count)
        elif step >= int(self.config["grpo"].get("max_num_steps", step)):
            self.stop_run(status="aborted", samples_count=samples_count)
        else:
            self.start_train_block(step)

    def end_eval_with_error(self, step: int) -> None:
        self._end(
            self.constants.EVAL_STOP,
            metadata={
                self.constants.SAMPLES_COUNT: self.sample_count(int(step)),
                "step": int(step),
            },
        )
        self.stop_run(status="aborted", samples_count=self.sample_count(int(step)))

    def observe_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        prefix: str,
        step_finished: bool = False,
    ) -> None:
        step = int(step)
        self.last_step = max(self.last_step, step)
        if not self.run_started or self.run_stopped:
            return

        if prefix == "train":
            tracked = self._extract_train_stats(metrics)
        elif prefix == "timing/train":
            tracked = self._extract_timing_stats(metrics)
            if not step_finished and not tracked:
                return
        else:
            return

        if tracked:
            self._event(
                key="tracked_stats",
                metadata={
                    self.constants.SAMPLES_COUNT: self.sample_count(step),
                    "step": step,
                },
                value=tracked,
                unique=False,
            )

    def _extract_train_stats(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        key_map = {
            "loss": "reduced_train_loss",
            "reward": "reward",
            "grad_norm": "grad_norm",
            "global_valid_toks": "global_valid_toks",
            "global_valid_seqs": "global_valid_seqs",
        }
        tracked = {}
        for source_key, output_key in key_map.items():
            if source_key in metrics:
                tracked[output_key] = self._to_scalar(metrics[source_key])
        return {k: v for k, v in tracked.items() if v is not None}

    def _extract_timing_stats(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        key_map = {
            "total_step_time": "train_step_time",
            "policy_training": "policy_training_time",
            "generation": "generation_time",
            "exposed_generation": "exposed_generation_time",
            "weight_sync": "weight_sync_time",
            "valid_tokens_per_sec_per_gpu": "valid_tokens_per_sec_per_gpu",
        }
        tracked = {}
        for source_key, output_key in key_map.items():
            if source_key in metrics:
                tracked[output_key] = self._to_scalar(metrics[source_key])
        return {k: v for k, v in tracked.items() if v is not None}

    @staticmethod
    def _to_scalar(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float, bool, str)):
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        if hasattr(value, "mean"):
            try:
                mean_value = value.mean()
                return mean_value.item() if hasattr(mean_value, "item") else mean_value
            except (TypeError, ValueError):
                return None
        return None

    def stop_run(self, status: str, samples_count: Optional[int] = None) -> None:
        if self.run_stopped:
            return
        if status != "success" and self.force_success_status:
            status = "success"
        if samples_count is None:
            samples_count = self.sample_count(self.last_step)
        self._end(
            self.constants.RUN_STOP,
            metadata={
                self.constants.SAMPLES_COUNT: samples_count,
                "status": status,
            },
        )
        self.target_reached = status == "success"
        self.run_stopped = True
        self.block_started = False

    def finalize(self, status: str = "aborted") -> None:
        if not self.run_started or self.run_stopped:
            return
        self.stop_train_block(self.last_step)
        self.stop_run(status=status, samples_count=self.sample_count(self.last_step))
