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
import os
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
import ray
import torch
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from wandb import Table

from nemo_rl.algorithms.utils import log_generation_metrics_to_wandb, set_seed
from nemo_rl.data import EvalDataConfigType
from nemo_rl.data.collate_fn import eval_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.math_environment import MathEnvConfig
from nemo_rl.environments.nemo_gym import (
    DEFAULT_GYM_PORT_RANGE_HIGH,
    DEFAULT_GYM_PORT_RANGE_LOW,
    create_nemo_gym_actor,
)
from nemo_rl.environments.vlm_environment import VLMEnvConfig
from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout
from nemo_rl.models.generation.interfaces import GenerationConfig, GenerationInterface
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy import PolicyConfig, TokenizerConfig
from nemo_rl.utils.logger import Logger, LoggerConfig

EVAL_STEP_METRIC = "eval_step"
EVAL_STEP_PATTERNS = ("eval/*", "generation_metrics/*", "ray/*")

# ===============================================================================
# Configuration
# ===============================================================================


class EvalConfig(TypedDict):
    metric: str
    num_tests_per_prompt: int
    seed: int
    k_value: int
    save_path: str | None


class NemoGymEvalDataConfig(BaseModel, extra="allow"):
    """Dataset configuration for NeMo Gym evaluation."""

    dataset_name: Literal["NemoGymDataset"]
    data_path: str
    processor: Literal["nemo_gym_data_processor"]
    env_name: Literal["nemo_gym"]
    max_input_seq_length: int | None = None
    repeat: int = 1
    prompt_file: str | None = None
    system_prompt_file: str | None = None


class NemoGymEvalEnvConfig(BaseModel, extra="allow"):
    """NeMo RL control fields plus pass-through NeMo Gym configuration."""

    config_paths: list[str]
    port_range_low: int = DEFAULT_GYM_PORT_RANGE_LOW
    port_range_high: int = DEFAULT_GYM_PORT_RANGE_HIGH
    rollout_max_attempts_to_avoid_lp_nan: int = 1
    invalid_tool_call_patterns: list[str] | None = None
    thinking_tags: list[str] | None = None


# TODO: this should updated, but is left to avoid breaking changes
class _PassThroughEnvConfig(TypedDict):
    math: NotRequired[MathEnvConfig]
    mmau: NotRequired[VLMEnvConfig]
    should_use_nemo_gym: NotRequired[bool]
    nemo_gym: NotRequired[NemoGymEvalEnvConfig]


class MasterConfig(BaseModel, extra="allow"):
    eval: EvalConfig
    generation: GenerationConfig  # Fixed: was 'generate'
    tokenizer: TokenizerConfig  # Added missing tokenizer key
    data: NemoGymEvalDataConfig | EvalDataConfigType
    env: _PassThroughEnvConfig
    logger: LoggerConfig | None = None
    cluster: ClusterConfig
    policy: PolicyConfig | None = None


@dataclass
class EvalRunResult:
    """Programmatic result returned by an evaluation run."""

    average_score: float
    num_samples: int
    generation_metrics: list[dict[str, Any]]
    rollout_metrics: list[dict[str, Any]]


def _get_num_generations_per_prompt(eval_config: EvalConfig) -> int:
    """Validate and return the number of eval samples per prompt."""
    value = eval_config.get("num_tests_per_prompt")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            "eval.num_tests_per_prompt must be an integer greater than or equal to 1"
        )
    return value


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def should_use_nemo_gym(master_config: MasterConfig) -> bool:
    """Return whether the evaluation should use NeMo Gym rollouts."""
    return bool(master_config.env.get("should_use_nemo_gym"))


def _validate_nemo_gym_generation_config(
    generation_config: GenerationConfig,
) -> None:
    """Validate generation settings shared by all NeMo Gym eval callers."""
    if generation_config["top_k"] is not None:
        raise ValueError("NeMo Gym evaluation requires generation.top_k=null")
    if generation_config["stop_strings"] or generation_config["stop_token_ids"]:
        raise ValueError(
            "NeMo Gym evaluation does not support stop strings or stop token IDs"
        )

    backend = generation_config["backend"]
    if backend == "vllm":
        backend_config = generation_config["vllm_cfg"]
        config_path = "generation.vllm_cfg"
    elif backend == "megatron":
        backend_config = generation_config["mcore_generation_config"]
        config_path = "generation.mcore_generation_config"
    else:
        raise ValueError(
            "NeMo Gym evaluation supports the vLLM and Megatron rollout backends"
        )
    if not backend_config["async_engine"] or not backend_config.get(
        "expose_http_server"
    ):
        raise ValueError(
            f"NeMo Gym evaluation requires {config_path}.async_engine=true "
            "and expose_http_server=true"
        )

    if backend == "vllm" and backend_config.get("enable_vllm_metrics_logger"):
        metrics_interval = backend_config.get("vllm_metrics_logger_interval")
        if not isinstance(metrics_interval, (int, float)) or metrics_interval <= 0:
            raise ValueError(
                "generation.vllm_cfg.vllm_metrics_logger_interval must be "
                "positive when vLLM metrics logging is enabled"
            )


def _validate_nemo_gym_eval_config(master_config: MasterConfig) -> None:
    """Fail early when a NeMo Gym evaluation cannot produce valid metrics."""
    use_nemo_gym = should_use_nemo_gym(master_config)
    if isinstance(master_config.data, NemoGymEvalDataConfig) and not use_nemo_gym:
        raise ValueError(
            "data.dataset_name=NemoGymDataset requires env.should_use_nemo_gym=true"
        )
    if not use_nemo_gym:
        return

    if "nemo_gym" not in master_config.env:
        raise ValueError(
            "env.should_use_nemo_gym=true requires an env.nemo_gym config block"
        )
    if not isinstance(master_config.data, NemoGymEvalDataConfig):
        raise ValueError(
            "NeMo Gym evaluation requires data.dataset_name=NemoGymDataset"
        )
    if master_config.logger is None:
        raise ValueError(
            "NeMo Gym evaluation requires a logger config to persist generation metrics"
        )
    if master_config.eval["metric"] == "cons@k":
        raise ValueError(
            "cons@k is not supported for NeMo Gym evaluation because Gym does "
            "not expose a uniform extracted-answer field; use mean_reward or pass@k"
        )
    if master_config.eval["metric"] not in {"mean_reward", "pass@k"}:
        raise ValueError(
            "NeMo Gym evaluation supports only mean_reward and pass@k metrics"
        )

    _validate_nemo_gym_generation_config(master_config.generation)


def setup_nemo_gym_environment(
    vllm_generation: GenerationInterface,
    master_config: MasterConfig,
) -> Any:
    """Start NeMo Gym against the eval generation server endpoints."""
    _validate_nemo_gym_eval_config(master_config)
    base_urls = getattr(vllm_generation, "dp_openai_server_base_urls", None)
    if not isinstance(base_urls, list) or not base_urls:
        raise RuntimeError(
            f"The {master_config.generation['backend']} rollout engine did not "
            "expose any OpenAI server endpoints for NeMo Gym"
        )
    nemo_gym_config = master_config.env["nemo_gym"].model_dump(exclude_none=True)
    return create_nemo_gym_actor(
        model_name=master_config.generation["model_name"],
        base_urls=base_urls,
        nemo_gym_config=nemo_gym_config,
    )


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
) -> tuple[
    GenerationInterface,
    DataLoader,
    MasterConfig,
]:
    """Set up components for model evaluation.

    Initializes the VLLM model and data loader.

    Args:
        master_config: Configuration settings.
        dataset: Dataset to evaluate on.

    Returns:
        VLLM model, data loader, and config.
    """
    # Extract individual configs for easier access
    eval_config = master_config.eval
    generation_config = master_config.generation
    cluster_config = master_config.cluster

    _validate_nemo_gym_eval_config(master_config)

    # Set seed for reproducibility
    set_seed(eval_config["seed"])

    # Check settings
    metric = eval_config["metric"]
    k_value = eval_config["k_value"]
    num_generations_per_prompt = _get_num_generations_per_prompt(eval_config)
    temperature = generation_config["temperature"]
    top_k = generation_config["top_k"]

    # Validate metrics
    assert metric in ["mean_reward", "pass@k", "cons@k"], f"Invalid metric: {metric}"
    if num_generations_per_prompt > 1:
        assert temperature > 0 and top_k != 1, (
            "temperature > 0 and top_k != 1 are required for multiple samples"
        )

    assert k_value >= 1, "k_value must be greater than or equal to 1"
    assert num_generations_per_prompt >= k_value, (
        "num_generations_per_prompt must be greater than or equal to k_value "
    )

    # ==========================
    #           Data
    # ==========================
    if generation_config["num_prompts_per_step"] == -1:
        generation_config["num_prompts_per_step"] = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=generation_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    print(f"  ✓ Evaluation dataset loaded with {len(dataset)} samples")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="eval_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #           Model
    # ==========================
    print("\n▶ Setting up model...")
    # Initialize the configured rollout engine.
    backend = generation_config["backend"]
    if backend == "vllm":
        policy_generation: GenerationInterface = VllmGeneration(
            cluster=cluster, config=generation_config
        )
    elif backend == "sglang":
        sglang_config = generation_config["sglang_cfg"]
        if "model_path" not in sglang_config:
            sglang_config["model_path"] = generation_config["model_name"]
        policy_generation = SGLangGeneration(
            cluster=cluster, sglang_cfg=generation_config
        )
    elif backend == "megatron":
        if master_config.policy is None:
            raise ValueError(
                "Megatron evaluation requires the full policy config; use the "
                "GRPO-to-eval entrypoint or provide eval.policy"
            )
        master_config.policy["generation"] = generation_config
        policy_generation = MegatronGeneration(
            cluster=cluster,
            config=master_config.policy,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unsupported evaluation generation backend: {backend}")
    print(
        f"  ✓ Using {backend} backend for generation with "
        f"{generation_config['model_name']}"
    )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy_generation,
        dataloader,
        master_config,
    )


# ===============================================================================
# Evaluation
# ===============================================================================


def eval_pass_k(rewards: torch.Tensor, num_tests_per_prompt: int, k: int) -> float:
    """Evaluate pass@k score using an unbiased estimator.

    Reference: https://github.com/huggingface/evaluate/blob/32546aafec25cdc2a5d7dd9f941fc5be56ba122f/metrics/code_eval/code_eval.py#L198-L213
    Args:
        rewards: Tensor of shape (batch_size * num_tests_per_prompt)
        k: int (pass@k value)

    Returns:
        pass_k_score: float
    """

    def eval_single_chunk(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return float(1.0 - torch.prod(1.0 - k / torch.arange(n - c + 1, n + 1)).item())

    # rewards is a 1d tensor of size (batch_size * num_tests_per_prompt)
    group_rewards = rewards.split(num_tests_per_prompt)
    pass_k_score = 0.0
    for group_reward in group_rewards:
        num_correct = group_reward.sum().item()
        pass_k_score += eval_single_chunk(num_tests_per_prompt, num_correct, k)

    return pass_k_score


def eval_cons_k(
    rewards: torch.Tensor,
    num_tests_per_prompt: int,
    k: int,
    extracted_answers: list[str | None],
) -> float:
    """Evaluate cons@k score using an unbiased estimator.

    Args:
        rewards: Tensor of shape (batch_size * num_tests_per_prompt)
        num_tests_per_prompt: int
        k: int
        extracted_answers: list[str| None]

    Returns:
        cons_k_score: float
    """

    def majority_vote(answers: list[str | None]) -> str | None:
        """Find the most common answer in the list of answers."""
        if not answers:
            return None
        # To fix@rayentian: How to deal with the case that there are multiple most common answers? Now we just return the first one.
        return Counter(answers).most_common(1)[0][0]

    def eval_single_cons_k(
        chunk_rewards: torch.Tensor, chunk_answers: list[str | None], n: int, k: int
    ) -> float:
        if chunk_answers is None or n == 0 or k > n:
            return 0.0

        total_subsets = 0
        correct_subsets = 0
        # For each subset of k answers, we vote for the most common answer.
        # If the most common answer is the same as the gold answer, we consider the subset as correct.
        for subset_indices in combinations(range(n), k):
            subset_answers = [chunk_answers[i] for i in subset_indices]
            majority_answer = majority_vote(subset_answers)
            reward_idx = chunk_answers.index(majority_answer)
            reward = chunk_rewards[reward_idx].item()
            total_subsets += 1
            if reward == 1.0:
                correct_subsets += 1

        return correct_subsets / total_subsets

    assert len(extracted_answers) == len(rewards), (
        "The number of extracted answers must be the same as the number of rewards"
    )
    # Split the rewards and extracted answers into groups of num_tests_per_prompt.
    group_rewards = rewards.split(num_tests_per_prompt)
    group_extracted_answers = [
        extracted_answers[i : i + num_tests_per_prompt]
        for i in range(0, len(extracted_answers), num_tests_per_prompt)
    ]
    assert len(group_rewards) == len(group_extracted_answers), (
        "The number of rewards and extracted answers must be the same"
    )
    num_groups = len(group_rewards)
    cons_k_score = 0.0
    # For each group of num_tests_per_prompt rewards and extracted answers, we evaluate the cons@k score.
    for i in range(num_groups):
        chunk_rewards = group_rewards[i]
        chunk_answers = group_extracted_answers[i]
        assert len(chunk_rewards) == len(chunk_answers), (
            "The number of rewards and extracted answers must be the same"
        )
        cons_k_score += eval_single_cons_k(
            chunk_rewards, chunk_answers, len(chunk_answers), k
        )

    return cons_k_score


def run_env_eval(
    vllm_generation: GenerationInterface,
    dataloader: DataLoader,
    env: Any,
    master_config: MasterConfig,
    *,
    tokenizer: AutoTokenizer | None = None,
    logger: Logger | None = None,
) -> EvalRunResult:
    """Main entry point for running evaluation using environment.

    Generates model responses and evaluates them by env.

    Args:
        vllm_generation: Model for generating responses.
        dataloader: Data loader with evaluation samples.
        env: Environment that scores responses.
        master_config: Configuration settings.
    """
    if should_use_nemo_gym(master_config):
        if tokenizer is None:
            raise ValueError("NeMo Gym evaluation requires a tokenizer")
        if logger is None:
            raise ValueError(
                "NeMo Gym evaluation requires a logger to persist generation metrics"
            )
        _validate_nemo_gym_eval_config(master_config)
        logger.use_batch_steps(EVAL_STEP_METRIC, EVAL_STEP_PATTERNS)
        return _run_nemo_gym_eval_impl(
            vllm_generation=vllm_generation,
            dataloader=dataloader,
            nemo_gym=env,
            tokenizer=tokenizer,
            master_config=master_config,
            logger=logger,
        )

    if logger is not None:
        logger.use_batch_steps(EVAL_STEP_METRIC, EVAL_STEP_PATTERNS)
    generation_config = master_config.generation
    backend = generation_config.get("backend", "")
    if backend == "sglang":
        use_async = bool(generation_config.get("use_async_rollouts", False))
    elif backend == "vllm":
        use_async = bool(
            generation_config.get("vllm_cfg", {}).get("async_engine", False)
        )
    elif backend == "megatron":
        use_async = bool(
            generation_config.get("mcore_generation_config", {}).get(
                "async_engine", False
            )
        )
    else:
        use_async = False
    return asyncio.run(
        _run_env_eval_impl(
            vllm_generation,
            dataloader,
            env,
            master_config,
            use_async=use_async,
            tokenizer=tokenizer,
            logger=logger,
        )
    )


def _log_generation_metrics(
    *,
    generation_metrics: dict[str, Any],
    step: int,
    generation_config: GenerationConfig,
    logger_config: LoggerConfig | None,
    require_numeric_metrics: bool,
    logger: Logger | None,
) -> None:
    """Persist raw generation metrics and log batch-step summaries and plots."""
    if logger is None or logger_config is None:
        return

    summary = _summarize_generation_metrics(generation_metrics)
    if not summary:
        if require_numeric_metrics:
            raise RuntimeError(
                f"NeMo Gym eval batch step {step} did not produce any numeric "
                "generation metrics"
            )
        return
    logger.log_string_list_as_jsonl(
        [json.dumps({"step": step, "metrics": generation_metrics})],
        "generation_metrics.jsonl",
    )
    logger.log_metrics(
        {EVAL_STEP_METRIC: step, **summary},
        step,
        prefix="generation_metrics",
        step_metric=EVAL_STEP_METRIC,
    )
    if generation_metrics and logger_config["wandb_enabled"]:
        timeline_interval = 1.0
        if generation_config["backend"] == "vllm":
            timeline_interval = generation_config["vllm_cfg"].get(
                "vllm_metrics_logger_interval", timeline_interval
            )
        log_generation_metrics_to_wandb(
            generation_metrics,
            step,
            timeline_interval,
            logger,
            step_metric=EVAL_STEP_METRIC,
        )


def _summarize_generation_metrics(
    generation_metrics: dict[str, Any],
) -> dict[str, float]:
    """Reduce per-worker time series to bounded scalar metrics for one batch."""
    summary: dict[str, float] = {}
    for metric_name, per_worker in generation_metrics.items():
        worker_series = (
            per_worker.values() if isinstance(per_worker, dict) else [per_worker]
        )
        all_values: list[float] = []
        worker_last_values: list[float] = []
        for series in worker_series:
            values = series if isinstance(series, (list, tuple)) else [series]
            numeric_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, bool)
                and np.isfinite(value)
            ]
            if numeric_values:
                all_values.extend(numeric_values)
                worker_last_values.append(numeric_values[-1])

        if not all_values:
            continue
        summary[f"{metric_name}/mean"] = float(np.mean(all_values))
        summary[f"{metric_name}/max"] = float(np.max(all_values))
        summary[f"{metric_name}/last_mean"] = float(np.mean(worker_last_values))
        summary[f"{metric_name}/last_sum"] = float(np.sum(worker_last_values))
    return summary


def _ensure_nemo_gym_generation_metrics(
    generation_metrics: dict[str, Any],
    rollout_metrics: dict[str, Any],
    num_generations: int,
) -> dict[str, Any]:
    """Use backend telemetry when available, otherwise derive batch telemetry."""
    if _summarize_generation_metrics(generation_metrics):
        return generation_metrics

    fallback_metrics: dict[str, Any] = {
        "completed_generations": {0: [float(num_generations)]}
    }
    mean_tokens = rollout_metrics.get("gen_tokens_per_sample/mean")
    if isinstance(mean_tokens, (int, float)) and np.isfinite(mean_tokens):
        fallback_metrics["generated_tokens_per_sample"] = {0: [float(mean_tokens)]}
        fallback_metrics["generated_tokens"] = {
            0: [float(mean_tokens * num_generations)]
        }

    rollout_seconds = rollout_metrics.get("timing/rollout/total")
    if isinstance(rollout_seconds, (int, float)) and np.isfinite(rollout_seconds):
        fallback_metrics["rollout_seconds"] = {0: [float(rollout_seconds)]}
        if rollout_seconds > 0 and isinstance(mean_tokens, (int, float)):
            fallback_metrics["generated_tokens_per_rollout_second"] = {
                0: [float(mean_tokens * num_generations / rollout_seconds)]
            }
    return fallback_metrics


def _log_eval_batch_metrics(
    logger: Logger | None,
    *,
    step: int,
    metrics: dict[str, int | float],
) -> None:
    """Commit generation, rollout, and system metrics at one eval batch step."""
    if logger is None:
        return
    logger.flush_system_metrics(step)
    logger.log_metrics(
        {EVAL_STEP_METRIC: step, **metrics},
        step,
        prefix="eval",
        step_metric=EVAL_STEP_METRIC,
        step_finished=True,
    )


def _run_nemo_gym_eval_impl(
    *,
    vllm_generation: GenerationInterface,
    dataloader: DataLoader,
    nemo_gym: Any,
    tokenizer: AutoTokenizer,
    master_config: MasterConfig,
    logger: Logger,
) -> EvalRunResult:
    """Run NeMo Gym rollouts and aggregate eval and generation metrics."""
    _validate_nemo_gym_eval_config(master_config)
    generation_config = master_config.generation
    metric = master_config.eval["metric"]
    num_generations_per_prompt = _get_num_generations_per_prompt(master_config.eval)
    num_prompts_per_step = generation_config["num_prompts_per_step"]
    k_value = master_config.eval["k_value"]

    score = 0.0
    evaluation_data: list[dict[str, Any]] = []
    generation_metrics: list[dict[str, Any]] = []
    rollout_metrics: list[dict[str, Any]] = []
    dataset_size = len(dataloader.dataset)
    expected_results = dataset_size * num_generations_per_prompt
    processed_prompts = 0

    try:
        if expected_results <= 0:
            raise ValueError("NeMo Gym evaluation requires a non-empty dataset")

        for batch_idx, batch in enumerate(dataloader):
            batch_step = batch_idx + 1
            if num_generations_per_prompt > 1:
                batch = batch.repeat_interleave(num_generations_per_prompt)

            vllm_generation.clear_logger_metrics()
            rollout_result = run_async_nemo_gym_rollout(
                policy_generation=vllm_generation,
                input_batch=batch,
                tokenizer=tokenizer,
                task_to_env={"nemo_gym": nemo_gym},
                generation_config=generation_config,
                max_seq_len=None,
                max_rollout_turns=None,
                greedy=False,
            )
            rewards = rollout_result.final_batch["total_reward"]
            batch_generation_metrics = _ensure_nemo_gym_generation_metrics(
                vllm_generation.get_logger_metrics(),
                rollout_result.rollout_metrics,
                rewards.numel(),
            )
            generation_metrics.append(
                {"step": batch_step, "metrics": batch_generation_metrics}
            )
            _log_generation_metrics(
                generation_metrics=batch_generation_metrics,
                step=batch_step,
                generation_config=generation_config,
                logger_config=master_config.logger,
                require_numeric_metrics=True,
                logger=logger,
            )

            if rewards.numel() % num_generations_per_prompt != 0:
                raise RuntimeError(
                    f"NeMo Gym eval batch {batch_idx} returned {rewards.numel()} "
                    "rewards, which is not divisible by "
                    f"num_generations_per_prompt={num_generations_per_prompt}"
                )
            if metric == "mean_reward":
                batch_score = rewards.sum().item() / num_generations_per_prompt
            else:
                batch_score = eval_pass_k(rewards, num_generations_per_prompt, k_value)
            score += batch_score
            batch_num_prompts = rewards.numel() // num_generations_per_prompt
            if batch_num_prompts > num_prompts_per_step:
                raise RuntimeError(
                    f"NeMo Gym eval batch {batch_idx} returned {batch_num_prompts} "
                    "prompts, which exceeds "
                    f"num_prompts_per_step={num_prompts_per_step}"
                )
            if (
                batch_step < len(dataloader)
                and batch_num_prompts != num_prompts_per_step
            ):
                raise RuntimeError(
                    f"NeMo Gym eval batch {batch_idx} returned {batch_num_prompts} "
                    "prompts before the final batch; expected "
                    f"num_prompts_per_step={num_prompts_per_step}"
                )
            prompt_index_offset = processed_prompts
            processed_prompts += batch_num_prompts

            scalar_rollout_metrics = {
                key: value
                for key, value in rollout_result.rollout_metrics.items()
                if isinstance(value, (bool, int, float))
            }
            rollout_metrics.append(scalar_rollout_metrics)
            logger.log_metrics(
                {EVAL_STEP_METRIC: batch_step, **scalar_rollout_metrics},
                batch_step,
                prefix="eval/rollout",
                step_metric=EVAL_STEP_METRIC,
            )

            serialized_results: list[str] = []
            for key, value in rollout_result.rollout_metrics.items():
                if "full_result" not in key or not isinstance(value, Table):
                    continue
                serialized_results.extend(row[0] for row in value.data)

            if len(serialized_results) != rewards.numel():
                raise RuntimeError(
                    f"NeMo Gym eval batch {batch_idx} returned "
                    f"{len(serialized_results)} full results for "
                    f"{rewards.numel()} rewards"
                )

            attributed_results: list[str] = []
            for batch_position, serialized_result in enumerate(serialized_results):
                full_result = json.loads(serialized_result)
                prompt_position = batch_position // num_generations_per_prompt
                sample = {
                    "full_result": full_result,
                    "reward": float(full_result["reward"]),
                    "sample_index": len(evaluation_data),
                    "prompt_index": prompt_index_offset + prompt_position,
                    "generation_index": (batch_position % num_generations_per_prompt),
                    "num_generations_per_prompt": num_generations_per_prompt,
                    "eval_batch_index": batch_idx,
                    "eval_step": batch_step,
                    "eval_batch_position": batch_position,
                    "eval_batch_size": len(serialized_results),
                }
                evaluation_data.append(sample)
                attributed_results.append(json.dumps(sample, separators=(",", ":")))

            if not attributed_results:
                raise RuntimeError(
                    f"NeMo Gym eval batch {batch_idx} did not contain any full results"
                )
            logger.log_string_list_as_jsonl(
                attributed_results, "nemo_gym_eval_results.jsonl"
            )
            batch_metrics: dict[str, int | float] = {
                "batch/reward_mean": float(rewards.float().mean().item()),
                "batch/reward_sum": float(rewards.sum().item()),
                "batch/score": float(batch_score / batch_num_prompts),
                "batch/num_prompts": batch_num_prompts,
                "batch/num_generations": rewards.numel(),
                "batch/num_generations_per_prompt": num_generations_per_prompt,
                "num_prompts_total": processed_prompts,
                "score_running": float(score / processed_prompts),
            }
            if batch_step == len(dataloader):
                batch_metrics["score"] = float(score / dataset_size)
            _log_eval_batch_metrics(
                logger,
                step=batch_step,
                metrics=batch_metrics,
            )
    finally:
        try:
            ray.get(nemo_gym.shutdown.remote())
        finally:
            vllm_generation.shutdown()

    if len(evaluation_data) != expected_results:
        raise RuntimeError(
            "NeMo Gym evaluation was incomplete: "
            f"expected {expected_results} full results, got {len(evaluation_data)}"
        )

    save_path = master_config.eval.get("save_path")
    if evaluation_data and save_path is not None:
        _save_evaluation_data_to_json(
            evaluation_data,
            _master_config_data(master_config),
            save_path,
        )

    _print_results(
        generation_config,
        score,
        dataset_size,
        metric,
        k_value,
        num_generations_per_prompt,
        dataset_name=_master_config_dataset_name(master_config),
        seed=master_config.eval["seed"],
    )
    average_score = score / dataset_size
    return EvalRunResult(
        average_score=average_score,
        num_samples=dataset_size,
        generation_metrics=generation_metrics,
        rollout_metrics=rollout_metrics,
    )


async def _run_env_eval_impl(
    vllm_generation: GenerationInterface,
    dataloader: DataLoader,
    env: Any,
    master_config: MasterConfig,
    use_async: bool = False,
    tokenizer: AutoTokenizer | None = None,
    logger: Logger | None = None,
) -> EvalRunResult:
    """Unified implementation for both sync and async evaluation."""
    # Extract for easier access
    generation_config = master_config.generation
    eval_config = master_config.eval
    metric = eval_config["metric"]
    num_generations_per_prompt = _get_num_generations_per_prompt(eval_config)
    k_value = eval_config["k_value"]

    # List to collect evaluation data for parquet file
    evaluation_data = []
    generation_metrics: list[dict[str, Any]] = []
    processed_prompts = 0

    # Run evaluation loop
    score = 0.0
    for batch_idx, batch in enumerate(dataloader):
        batch_step = batch_idx + 1
        batch_num_prompts = len(batch["message_log"])
        vllm_generation.clear_logger_metrics()

        # measure multiple samples
        if num_generations_per_prompt > 1:
            batch = batch.repeat_interleave(num_generations_per_prompt)

        # get input prompt from message_log
        is_multimodal = "vllm_content" in batch
        prompts = []
        prompts_for_display = []
        for i, message_log in enumerate(batch["message_log"]):
            if is_multimodal and batch["vllm_content"][i] is not None:
                vllm_content = batch["vllm_content"][i]
                prompt_dict = {"prompt": vllm_content}
                multi_modal_data = {}
                audios = batch.get("vllm_audios", None)
                if audios is not None and len(audios[i]) > 0:
                    multi_modal_data["audio"] = (
                        audios[i][0] if len(audios[i]) == 1 else audios[i]
                    )
                images = batch.get("vllm_images", None)
                if images is not None and len(images[i]) > 0:
                    multi_modal_data["image"] = (
                        images[i][0] if len(images[i]) == 1 else images[i]
                    )
                videos = batch.get("vllm_videos", None)
                if videos is not None and len(videos[i]) > 0:
                    multi_modal_data["video"] = (
                        videos[i][0] if len(videos[i]) == 1 else videos[i]
                    )
                if multi_modal_data:
                    prompt_dict["multi_modal_data"] = multi_modal_data
                prompts.append(prompt_dict)
                prompts_for_display.append(vllm_content)
            else:
                # Text-only fallback: use raw prompt strings (vLLM will tokenize them).
                # Note: utils.py's format_prompt_for_vllm_generation uses pre-tokenized
                # prompt_token_ids instead, since the training pipeline already has
                # input_ids tensors. Both are valid vLLM inputs but may tokenize
                # slightly differently.
                content = [message["content"] for message in message_log]
                content = "\n".join(content)
                prompts.append(content)
                prompts_for_display.append(content)

        # generate by vllm
        inputs = BatchedDataDict({"prompts": prompts})
        outputs = await _generate_texts(
            vllm_generation,
            inputs,
            use_async,
            backend=generation_config["backend"],
            batch=batch,
            tokenizer=tokenizer,
        )
        batch_generation_metrics = vllm_generation.get_logger_metrics()
        generation_metrics.append(
            {"step": batch_step, "metrics": batch_generation_metrics}
        )
        _log_generation_metrics(
            generation_metrics=batch_generation_metrics,
            step=batch_step,
            generation_config=generation_config,
            logger_config=master_config.logger,
            require_numeric_metrics=False,
            logger=logger,
        )

        # append to message_log
        for idx, output in enumerate(outputs):
            batch["message_log"][idx].append(
                {
                    "role": "assistant",
                    "content": output,
                }
            )

        # evaluate generations with the environment
        to_env = [
            get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
            for i in range(len(batch["message_log"]))
        ]

        env_return = ray.get(env.step.remote(to_env, batch["extra_env_info"], True))
        rewards = env_return.rewards

        # Collect data for JSON file
        for i, (prompt, output, message_log, reward, extra_info) in enumerate(
            zip(
                prompts_for_display,
                outputs,
                batch["message_log"],
                rewards.tolist(),
                batch["extra_env_info"],
            )
        ):
            evaluation_data.append(
                {
                    "prompt": prompt,
                    "response": output,
                    "reward": reward,
                    "message_log": message_log,
                    "extra_env_info": extra_info,
                    "sample_index": len(evaluation_data),
                    "prompt_index": processed_prompts + i // num_generations_per_prompt,
                    "generation_index": i % num_generations_per_prompt,
                    "num_generations_per_prompt": num_generations_per_prompt,
                    "eval_batch_index": batch_idx,
                    "eval_step": batch_step,
                }
            )

        # update stats
        if metric == "mean_reward":
            batch_score = rewards.sum().item() / num_generations_per_prompt
        elif metric == "pass@k":
            batch_score = eval_pass_k(rewards, num_generations_per_prompt, k_value)
        elif metric == "cons@k":
            extracted_answers = env_return.answers
            batch_score = eval_cons_k(
                rewards, num_generations_per_prompt, k_value, extracted_answers
            )
        else:
            raise ValueError(f"Invalid metric: {metric}")
        score += batch_score
        processed_prompts += batch_num_prompts
        batch_metrics = {
            "batch/reward_mean": float(rewards.float().mean().item()),
            "batch/reward_sum": float(rewards.sum().item()),
            "batch/score": float(batch_score / batch_num_prompts),
            "batch/num_prompts": batch_num_prompts,
            "batch/num_generations": rewards.numel(),
            "batch/num_generations_per_prompt": num_generations_per_prompt,
            "num_prompts_total": processed_prompts,
            "score_running": float(score / processed_prompts),
        }
        if batch_step == len(dataloader):
            batch_metrics["score"] = float(score / len(dataloader.dataset))
        _log_eval_batch_metrics(
            logger,
            step=batch_step,
            metrics=batch_metrics,
        )

    # Cleanup before printing results
    ray.get(env.shutdown.remote())
    vllm_generation.shutdown()

    # Save evaluation data to JSON file if save_path is specified
    save_path = eval_config.get("save_path")
    if evaluation_data and save_path is not None:
        _save_evaluation_data_to_json(
            evaluation_data,
            _master_config_data(master_config),
            save_path,
        )

    # Print results
    _print_results(
        generation_config,
        score,
        len(dataloader.dataset),
        metric,
        k_value,
        num_generations_per_prompt,
        dataset_name=_master_config_dataset_name(master_config),
        seed=master_config.eval["seed"],
    )
    dataset_size = len(dataloader.dataset)
    return EvalRunResult(
        average_score=score / dataset_size,
        num_samples=dataset_size,
        generation_metrics=generation_metrics,
        rollout_metrics=[],
    )


def _build_generation_inputs(
    batch: BatchedDataDict,
    tokenizer: AutoTokenizer,
) -> BatchedDataDict:
    """Build backend-neutral token inputs from an eval message-log batch."""
    input_ids = []
    for sample_idx, message_log in enumerate(batch["message_log"]):
        token_ids = [message.get("token_ids") for message in message_log]
        if not token_ids or any(
            not isinstance(tokens, torch.Tensor) for tokens in token_ids
        ):
            raise ValueError(
                "Non-vLLM evaluation requires token_ids on every message; "
                f"sample {sample_idx} is missing them"
            )
        input_ids.append(torch.cat(token_ids))

    input_lengths = torch.tensor(
        [tokens.numel() for tokens in input_ids], dtype=torch.int32
    )
    return BatchedDataDict(
        {
            "input_ids": pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "input_lengths": input_lengths,
        }
    )


def _decode_generation_outputs(
    outputs: BatchedDataDict,
    input_lengths: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> list[str]:
    """Decode only newly generated tokens from GenerationInterface outputs."""
    texts = []
    for sample_idx, input_length in enumerate(input_lengths.tolist()):
        total_length = int(outputs["unpadded_sequence_lengths"][sample_idx].item())
        generated_ids = outputs["output_ids"][
            sample_idx, int(input_length) : total_length
        ]
        texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return texts


async def _generate_texts(
    vllm_generation: GenerationInterface,
    inputs: BatchedDataDict,
    use_async: bool,
    *,
    backend: str,
    batch: BatchedDataDict,
    tokenizer: AutoTokenizer | None,
) -> list[str]:
    """Generate text through vLLM's text API or the common engine interface."""
    if backend == "vllm" and isinstance(vllm_generation, VllmGeneration):
        if use_async:
            # generate_text_async accepts one sample per call; fan out and gather.
            async def _generate_single_sample(i):
                single = inputs.slice(i, i + 1)
                async for _, result in vllm_generation.generate_text_async(single):
                    return (i, result["texts"][0])
                raise RuntimeError(f"No output produced for sample {i}")

            results = await asyncio.gather(
                *(_generate_single_sample(i) for i in range(inputs.size))
            )
            results.sort(key=lambda x: x[0])
            return [text for _, text in results]
        return vllm_generation.generate_text(inputs)["texts"]

    if tokenizer is None:
        raise ValueError(f"{backend} evaluation requires a tokenizer")
    generation_inputs = _build_generation_inputs(batch, tokenizer)
    if use_async:
        # GenerationInterface async methods accept one sample per call.
        async def _generate_single_sample(i: int) -> tuple[int, str]:
            single = generation_inputs.slice(i, i + 1)
            async for _, result in vllm_generation.generate_async(single, greedy=False):
                return (
                    i,
                    _decode_generation_outputs(
                        result, single["input_lengths"], tokenizer
                    )[0],
                )
            raise RuntimeError(f"No output produced for sample {i}")

        results = await asyncio.gather(
            *(_generate_single_sample(i) for i in range(generation_inputs.size))
        )
        results.sort(key=lambda x: x[0])
        return [text for _, text in results]

    outputs = vllm_generation.generate(generation_inputs, greedy=False)
    return _decode_generation_outputs(
        outputs, generation_inputs["input_lengths"], tokenizer
    )


def _master_config_dataset_name(master_config: MasterConfig) -> str:
    """Return the configured dataset name for output metadata."""
    return (
        master_config.data.dataset_name
        if isinstance(master_config.data, NemoGymEvalDataConfig)
        else master_config.data["dataset_name"]
    )


def _master_config_data(master_config: MasterConfig) -> dict[str, Any]:
    """Build serializable output metadata for a standalone eval config."""
    return {
        "model_name": master_config.generation["model_name"],
        "dataset_name": _master_config_dataset_name(master_config),
        "metric": master_config.eval["metric"],
        "k_value": master_config.eval["k_value"],
        "num_generations_per_prompt": _get_num_generations_per_prompt(
            master_config.eval
        ),
        "temperature": master_config.generation["temperature"],
        "top_p": master_config.generation["top_p"],
        "top_k": master_config.generation["top_k"],
    }


def _save_evaluation_data_to_json(
    evaluation_data: list[dict[str, Any]],
    config_data: dict[str, Any],
    save_path: str,
) -> None:
    """Save evaluation data to a JSON file.

    Args:
        evaluation_data: List of evaluation samples
        config_data: Resolved configuration metadata to persist.
        save_path: Path to save evaluation results. Set to null to disable saving.
                  Example: "results/eval_output" or "/path/to/evaluation_results"
    """
    # Create directory if it doesn't exist
    save_dir = save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Generate file paths within the directory
    eval_data_path = os.path.join(save_dir, "evaluation_data.json")
    config_path = os.path.join(save_dir, "config.json")

    # Prepare the data to save
    data_to_save = {"evaluation_data": evaluation_data}

    # Save configuration to separate JSON file
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"\n✓ Configuration saved to: {config_path}")

    # Process data to make it JSON serializable
    processed_data = []
    for sample in evaluation_data:
        processed_sample = sample.copy()
        # Convert non-serializable objects to strings
        if "message_log" in sample:
            processed_sample["message_log"] = str(sample["message_log"])
        if "extra_env_info" in sample:
            processed_sample["extra_env_info"] = str(sample["extra_env_info"])
        processed_data.append(processed_sample)

    # Update data to save with processed version
    data_to_save["evaluation_data"] = processed_data

    # Save to JSON file
    with open(eval_data_path, "w") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"\n✓ Evaluation data saved to: {eval_data_path}")
    print(f"  Total samples: {len(evaluation_data)}")
    print(f"  File size: {os.path.getsize(eval_data_path) / 1024 / 1024:.2f} MB")


def _print_results(
    generation_config: GenerationConfig,
    score: float,
    dataset_size: int,
    metric: str,
    k_value: int,
    num_generations_per_prompt: int,
    *,
    dataset_name: str,
    seed: int,
) -> None:
    """Print evaluation results."""
    dataset_name = os.path.basename(dataset_name)
    model_name = os.path.basename(generation_config["model_name"])
    max_new_tokens = generation_config["max_new_tokens"]
    temperature = generation_config["temperature"]
    top_p = generation_config["top_p"]
    top_k = generation_config["top_k"]
    average_score = score / dataset_size

    print("\n" + "=" * 60)
    print(f"{model_name=} {dataset_name=}")
    print(f"{max_new_tokens=} {temperature=} {top_p=} {top_k=} {seed=}\n")
    metric_name = f"{metric[:-1]}{k_value}" if metric.endswith("@k") else metric
    print(f"metric={metric_name} {num_generations_per_prompt=}\n")
    print(f"score={average_score:.4f} ({score}/{dataset_size})")
    print("=" * 60 + "\n")
