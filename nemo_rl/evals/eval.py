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
from itertools import combinations
from typing import Any, Literal, NotRequired, TypedDict

import ray
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import EvalDataConfigType
from nemo_rl.data.collate_fn import eval_collate_fn, rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.math_environment import MathEnvConfig
from nemo_rl.environments.vlm_environment import VLMEnvConfig
from nemo_rl.experience.rollouts import (
    AsyncNemoGymRolloutResult,
    run_async_nemo_gym_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy import TokenizerConfig

# ===============================================================================
# Configuration
# ===============================================================================

SINGLE_TURN_ROLLOUT_MODE = "single_turn"
NEMO_GYM_ROLLOUT_MODE = "nemo_gym"
SUPPORTED_ROLLOUT_MODES = (SINGLE_TURN_ROLLOUT_MODE, NEMO_GYM_ROLLOUT_MODE)
SINGLE_TURN_METRICS = ("pass@k", "cons@k")
NEMO_GYM_METRICS = ("mean_reward", "pass@k")


class EvalConfig(TypedDict):
    # Supported values: "single_turn" for the existing one-step env flow,
    # "nemo_gym" for NeMo-Gym-managed multi-turn rollouts.
    rollout_mode: Literal["single_turn", "nemo_gym"]
    # Maximum turns for native eval rollouts. NeMo-Gym owns turn limits itself,
    # so this must be null when rollout_mode is "nemo_gym".
    max_rollout_turns: int | None
    metric: str
    num_tests_per_prompt: int
    seed: int
    k_value: int
    save_path: str | None
    save_full_gym_result: bool


# TODO: this should updated, but is left to avoid breaking changes
class _PassThroughEnvConfig(TypedDict):
    math: NotRequired[MathEnvConfig]
    mmau: NotRequired[VLMEnvConfig]
    nemo_gym: NotRequired[dict[str, Any]]


class MasterConfig(TypedDict):
    eval: EvalConfig
    generation: GenerationConfig  # Fixed: was 'generate'
    tokenizer: TokenizerConfig  # Added missing tokenizer key
    data: EvalDataConfigType
    env: _PassThroughEnvConfig
    cluster: ClusterConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def _validate_eval_config(master_config: MasterConfig) -> None:
    """Validate eval settings that depend on rollout mode."""
    eval_config = master_config["eval"]
    generation_config = master_config["generation"]

    rollout_mode = eval_config["rollout_mode"]
    assert rollout_mode in SUPPORTED_ROLLOUT_MODES, (
        f"Invalid rollout_mode: {rollout_mode}. "
        f"Supported modes: {SUPPORTED_ROLLOUT_MODES}"
    )

    metric = eval_config["metric"]
    k_value = eval_config["k_value"]
    num_tests_per_prompt = eval_config["num_tests_per_prompt"]
    temperature = generation_config["temperature"]
    top_k = generation_config["top_k"]

    if rollout_mode == SINGLE_TURN_ROLLOUT_MODE:
        assert metric in SINGLE_TURN_METRICS, (
            f"Invalid metric for single-turn eval: {metric}. "
            f"Supported metrics: {SINGLE_TURN_METRICS}"
        )
        assert eval_config["max_rollout_turns"] == 1, (
            "eval.max_rollout_turns must be 1 for rollout_mode='single_turn'. "
            "Use rollout_mode='nemo_gym' for NeMo-Gym-managed multi-turn eval."
        )
        if num_tests_per_prompt > 1:
            assert temperature > 0 and top_k != 1, (
                "temperature > 0 and top_k != 1 are required for multiple samples"
            )
    elif rollout_mode == NEMO_GYM_ROLLOUT_MODE:
        assert metric in NEMO_GYM_METRICS, (
            f"Invalid metric for NeMo-Gym eval: {metric}. "
            f"Supported metrics: {NEMO_GYM_METRICS}"
        )
        assert eval_config["max_rollout_turns"] is None, (
            "eval.max_rollout_turns must be null for rollout_mode='nemo_gym'. "
            "Configure turn limits inside the NeMo-Gym task/environment config."
        )
        assert num_tests_per_prompt == 1, (
            "eval.num_tests_per_prompt > 1 is not supported for "
            "rollout_mode='nemo_gym'. Repeat rows in the NeMo-Gym dataset if "
            "multiple independent rollouts per prompt are required."
        )
        assert generation_config["backend"] == "vllm", (
            "Only vLLM backend is supported for NeMo-Gym evaluation"
        )
        assert generation_config["vllm_cfg"]["async_engine"], (
            "rollout_mode='nemo_gym' requires generation.vllm_cfg.async_engine=true"
        )
        assert generation_config["vllm_cfg"].get("expose_http_server", None), (
            "rollout_mode='nemo_gym' requires "
            "generation.vllm_cfg.expose_http_server=true"
        )
        assert generation_config["top_k"] is None, (
            "rollout_mode='nemo_gym' requires generation.top_k=null because "
            "top-k is not OpenAI-compatible"
        )
        assert generation_config["stop_strings"] is None, (
            "rollout_mode='nemo_gym' requires generation.stop_strings=null"
        )
        assert generation_config["stop_token_ids"] is None, (
            "rollout_mode='nemo_gym' requires generation.stop_token_ids=null"
        )

    assert k_value >= 1, "k_value must be greater than or equal to 1"
    assert num_tests_per_prompt >= k_value, (
        "num_tests_per_prompt must be greater than or equal to k_value "
    )


def _get_eval_collate_fn(rollout_mode: str):
    """Return the collate function required by the selected eval rollout mode."""
    if rollout_mode == NEMO_GYM_ROLLOUT_MODE:
        return rl_collate_fn
    return eval_collate_fn


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
) -> tuple[
    VllmGeneration,
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
    eval_config = master_config["eval"]
    generation_config = master_config["generation"]
    cluster_config = master_config["cluster"]

    # Set seed for reproducibility
    set_seed(eval_config["seed"])

    _validate_eval_config(master_config)

    # ==========================
    #           Data
    # ==========================
    if generation_config["num_prompts_per_step"] == -1:
        generation_config["num_prompts_per_step"] = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=generation_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=_get_eval_collate_fn(eval_config["rollout_mode"]),
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
    # check backend
    backend = generation_config["backend"]
    assert backend == "vllm", "Only vLLM backend is supported for evaluation"

    # initialize vllm generation
    vllm_generation = VllmGeneration(cluster=cluster, config=generation_config)
    print(
        f"  ✓ Using vLLM backend for generation with {generation_config['model_name']}"
    )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        vllm_generation,
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


def run_env_eval(vllm_generation, dataloader, env, master_config, tokenizer):
    """Main entry point for running evaluation using environment.

    Generates model responses and evaluates them by env.

    Args:
        vllm_generation: Model for generating responses.
        dataloader: Data loader with evaluation samples.
        env: Environment that scores responses.
        master_config: Configuration settings.
        tokenizer: Tokenizer used by NeMo-Gym result postprocessing.
    """
    # Check if async engine is enabled and run appropriate version
    if master_config["generation"]["vllm_cfg"]["async_engine"]:
        asyncio.run(
            _run_env_eval_impl(
                vllm_generation,
                dataloader,
                env,
                master_config,
                tokenizer,
                use_async=True,
            )
        )
    else:
        asyncio.run(
            _run_env_eval_impl(
                vllm_generation,
                dataloader,
                env,
                master_config,
                tokenizer,
                use_async=False,
            )
        )


def _score_batch_rewards(
    rewards: torch.Tensor,
    metric: str,
    num_tests_per_prompt: int,
    k_value: int,
    extracted_answers: list[str | None] | None = None,
) -> float:
    """Score one eval batch according to the selected metric."""
    if metric == "mean_reward":
        return rewards.sum().item()
    if metric == "pass@k":
        return eval_pass_k(rewards, num_tests_per_prompt, k_value)
    if metric == "cons@k":
        assert extracted_answers is not None, (
            "extracted_answers are required for cons@k"
        )
        return eval_cons_k(rewards, num_tests_per_prompt, k_value, extracted_answers)
    raise ValueError(f"Invalid metric: {metric}")


def _collect_nemo_gym_evaluation_data(
    batch: BatchedDataDict,
    rollout_result: AsyncNemoGymRolloutResult,
    include_full_gym_result: bool,
    start_sample_index: int,
) -> list[dict[str, Any]]:
    """Build serializable per-sample records for NeMo-Gym eval results."""
    final_batch = rollout_result.final_batch
    raw_results = rollout_result.raw_results or [{} for _ in final_batch["message_log"]]
    samples = []

    for i, (message_log, reward, agent_ref, extra_info, raw_result) in enumerate(
        zip(
            final_batch["message_log"],
            final_batch["total_reward"].tolist(),
            final_batch["agent_ref"],
            batch["extra_env_info"],
            raw_results,
        )
    ):
        sample = {
            "reward": reward,
            "message_log": message_log,
            "agent_ref": agent_ref,
            "extra_env_info": extra_info,
            "sample_index": start_sample_index + i,
        }
        if include_full_gym_result and raw_result.get("full_result") is not None:
            sample["full_result"] = raw_result["full_result"]
        samples.append(sample)

    return samples


async def _run_env_eval_impl(
    vllm_generation, dataloader, env, master_config, tokenizer, use_async=False
):
    """Unified implementation for both sync and async evaluation."""
    # Extract for easier access
    generation_config = master_config["generation"]
    eval_config = master_config["eval"]
    metric = eval_config["metric"]
    num_tests_per_prompt = eval_config["num_tests_per_prompt"]
    k_value = eval_config["k_value"]
    rollout_mode = eval_config["rollout_mode"]

    # List to collect evaluation data for JSON output.
    evaluation_data = []

    # Run evaluation loop
    score = 0.0
    for batch in dataloader:
        if rollout_mode == NEMO_GYM_ROLLOUT_MODE:
            rollout_result = run_async_nemo_gym_rollout(
                policy_generation=vllm_generation,
                input_batch=batch,
                tokenizer=tokenizer,
                task_to_env={NEMO_GYM_ROLLOUT_MODE: env},
                generation_config=generation_config,
                max_seq_len=None,
                max_rollout_turns=None,
                greedy=False,
            )
            rewards = rollout_result.final_batch["total_reward"]
            evaluation_data.extend(
                _collect_nemo_gym_evaluation_data(
                    batch,
                    rollout_result,
                    eval_config["save_full_gym_result"],
                    len(evaluation_data),
                )
            )
            score += _score_batch_rewards(
                rewards, metric, num_tests_per_prompt, k_value
            )
            continue

        # measure multiple samples
        if num_tests_per_prompt > 1:
            batch = batch.repeat_interleave(num_tests_per_prompt)

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
        outputs = await _generate_texts(vllm_generation, inputs, use_async)

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
                }
            )

        # update stats
        extracted_answers = env_return.answers if metric == "cons@k" else None
        score += _score_batch_rewards(
            rewards, metric, num_tests_per_prompt, k_value, extracted_answers
        )

    # Cleanup before printing results
    ray.get(env.shutdown.remote())
    vllm_generation.shutdown()

    # Save evaluation data to JSON file if save_path is specified
    save_path = eval_config.get("save_path")
    if evaluation_data and save_path is not None:
        _save_evaluation_data_to_json(evaluation_data, master_config, save_path)

    # Print results
    _print_results(
        master_config,
        generation_config,
        score,
        len(dataloader.dataset),
        metric,
        k_value,
        num_tests_per_prompt,
    )


async def _generate_texts(vllm_generation, inputs, use_async):
    """Generate texts using either sync or async method."""
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
        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])
        return [text for _, text in results]
    else:
        # Use sync generation
        return vllm_generation.generate_text(inputs)["texts"]


def _save_evaluation_data_to_json(evaluation_data, master_config, save_path):
    """Save evaluation data to a JSON file.

    Args:
        evaluation_data: List of evaluation samples
        master_config: Configuration dictionary
        save_path: Path to save evaluation results. Set to null to disable saving.
                  Example: "results/eval_output" or "/path/to/evaluation_results"
    """
    # Extract configuration information
    config_data = {
        "model_name": master_config["generation"]["model_name"],
        "dataset_name": master_config["data"]["dataset_name"],
        "rollout_mode": master_config["eval"]["rollout_mode"],
        "metric": master_config["eval"]["metric"],
        "k_value": master_config["eval"]["k_value"],
        "num_tests_per_prompt": master_config["eval"]["num_tests_per_prompt"],
        "max_rollout_turns": master_config["eval"]["max_rollout_turns"],
        "save_full_gym_result": master_config["eval"]["save_full_gym_result"],
        "temperature": master_config["generation"]["temperature"],
        "top_p": master_config["generation"]["top_p"],
        "top_k": master_config["generation"]["top_k"],
    }

    # Create directory if it doesn't exist
    save_dir = save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Generate file paths within the directory
    eval_data_path = os.path.join(save_dir, "evaluation_data.json")
    config_path = os.path.join(save_dir, "config.json")

    # Save configuration to separate JSON file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"\n✓ Configuration saved to: {config_path}")

    # Process data to make it JSON serializable
    processed_data = [_make_json_serializable(sample) for sample in evaluation_data]

    # Save to JSON file
    with open(eval_data_path, "w", encoding="utf-8") as f:
        json.dump({"evaluation_data": processed_data}, f, indent=2)

    print(f"\n✓ Evaluation data saved to: {eval_data_path}")
    print(f"  Total samples: {len(evaluation_data)}")
    print(f"  File size: {os.path.getsize(eval_data_path) / 1024 / 1024:.2f} MB")


def _make_json_serializable(value):
    """Convert nested tensors and other values into JSON-serializable objects."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_make_json_serializable(v) for v in value]

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _format_metric_name(metric: str, k_value: int) -> str:
    """Format metric labels for terminal output."""
    if metric.endswith("@k"):
        return f"{metric[:-1]}{k_value}"
    return metric


def _print_results(
    master_config,
    generation_config,
    score,
    dataset_size,
    metric,
    k_value,
    num_tests_per_prompt,
):
    """Print evaluation results."""
    dataset_name = os.path.basename(master_config["data"]["dataset_name"])
    model_name = os.path.basename(generation_config["model_name"])
    max_new_tokens = generation_config["vllm_cfg"]["max_model_len"]
    seed = master_config["eval"]["seed"]
    temperature = generation_config["temperature"]
    top_p = generation_config["top_p"]
    top_k = generation_config["top_k"]
    average_score = score / dataset_size
    metric_name = _format_metric_name(metric, k_value)

    print("\n" + "=" * 60)
    print(f"{model_name=} {dataset_name=}")
    print(f"{max_new_tokens=} {temperature=} {top_p=} {top_k=} {seed=}\n")
    print(f"metric={metric_name} {num_tests_per_prompt=}\n")
    print(f"score={average_score:.4f} ({score}/{dataset_size})")
    print("=" * 60 + "\n")
