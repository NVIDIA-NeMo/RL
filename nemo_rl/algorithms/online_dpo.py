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
import copy
import gc
import os
import time
import warnings
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import (
    _should_log_nemo_gym_responses,
    _should_use_async_rollouts,
    refit_policy_generation,
)
from nemo_rl.algorithms.loss import DPOLossFn
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import preference_collate_fn, rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import LLMMessageLogType, PreferenceDatumSpec
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.sglang import SGLangConfig, SGLangGeneration
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig, print_message_log_samples
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class OnlineDPOConfig(TypedDict):
    """Configuration for online DPO.

    Required keys:
        num_prompts_per_step: Number of prompts sampled for each online generation
            batch. Recommended default: equal to policy.train_global_batch_size.
        num_generations_per_prompt: Number of responses sampled per prompt. Online
            DPO currently requires exactly 2.
        max_num_epochs: Maximum passes over the prompt dataloader.
        max_num_steps: Maximum policy optimizer steps.
        max_rollout_turns: Maximum environment turns per generated response. Online
            DPO currently requires 1 so the final trainable target is the sampled
            assistant response.
        val_period: Run environment rollout validation every N optimizer steps. Use
            0 to disable periodic validation.
        val_batch_size: Number of prompts per validation rollout batch.
        val_at_start: Whether to run validation before the first optimizer step.
        val_at_end: Whether to run validation on the final optimizer step.
        max_val_samples: Maximum validation samples to roll out when validation runs.
        seed: Random seed for data, model, and generation setup.
        min_reward_margin: Minimum absolute reward difference required to keep a
            generated pair. Recommended default: 0.0.
        drop_truncated_pairs: Whether to drop pairs where either response hit a
            rollout truncation condition. Recommended default: true.
        reference_policy_kl_penalty: DPO beta parameter.
        preference_average_log_probs: Whether to average DPO logprob differences
            over response tokens.
        sft_average_log_probs: Whether to average the auxiliary SFT term over
            response tokens.
        preference_loss_weight: Weight for the DPO preference loss.
        sft_loss_weight: Weight for the auxiliary SFT loss on chosen responses.
    """

    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_epochs: int
    max_num_steps: int
    max_rollout_turns: int
    val_period: int
    val_batch_size: int
    val_at_start: bool
    val_at_end: bool
    max_val_samples: int
    seed: int
    min_reward_margin: float
    drop_truncated_pairs: bool
    reference_policy_kl_penalty: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool
    preference_loss_weight: float
    sft_loss_weight: float


class OnlineDPOSaveState(TypedDict):
    consumed_samples: int
    current_step: int
    current_prompt_batch: int
    current_epoch: int
    total_steps: int
    total_valid_tokens: int
    val_reward: NotRequired[float]


class OnlineDPOLoggerConfig(LoggerConfig):
    num_val_samples_to_print: int


class MasterConfig(TypedDict):
    policy: PolicyConfig
    env: dict[str, Any]
    data: DataConfig
    online_dpo: OnlineDPOConfig
    logger: OnlineDPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


def _default_online_dpo_save_state() -> OnlineDPOSaveState:
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_prompt_batch": 0,
        "current_epoch": 0,
        "total_steps": 0,
        "total_valid_tokens": 0,
        "val_reward": -99999999.0,
    }


def _validate_online_dpo_config(
    master_config: MasterConfig, processor: Optional[AutoProcessor]
) -> None:
    online_dpo_config = master_config["online_dpo"]
    policy_config = master_config["policy"]
    data_config = master_config["data"]

    assert online_dpo_config["num_generations_per_prompt"] == 2, (
        "Online DPO currently requires online_dpo.num_generations_per_prompt=2 "
        "so each prompt produces one chosen/rejected pair."
    )
    assert online_dpo_config["max_rollout_turns"] == 1, (
        "Online DPO currently supports one rollout turn. Multi-turn trajectories "
        "need a richer pair construction policy before they can be trained safely."
    )
    assert online_dpo_config["num_prompts_per_step"] > 0
    assert online_dpo_config["max_num_epochs"] > 0
    assert online_dpo_config["max_num_steps"] > 0
    assert online_dpo_config["val_period"] >= 0
    assert online_dpo_config["val_batch_size"] > 0
    assert online_dpo_config["max_val_samples"] >= 0
    assert online_dpo_config["min_reward_margin"] >= 0
    assert not data_config["use_multiple_dataloader"], (
        "Online DPO currently supports a single prompt dataloader."
    )
    assert processor is None, "Online DPO currently supports text-only LLM data."
    assert not policy_config["dynamic_batching"]["enabled"], (
        "Dynamic batching is currently not supported with Online DPO because "
        "DPO relies on stable chosen/rejected ordering within each pair."
    )
    assert not policy_config["sequence_packing"]["enabled"], (
        "Sequence packing is currently not supported with Online DPO because "
        "DPO relies on stable chosen/rejected ordering within each pair."
    )
    assert not bool(master_config["env"].get("should_use_nemo_gym")), (
        "Online DPO currently uses NeMo RL EnvironmentInterface environments, "
        "not NeMo-Gym rollouts."
    )

    generation_config = policy_config["generation"]
    assert generation_config is not None, (
        "A generation config in policy.generation is required for Online DPO."
    )
    assert generation_config["colocated"]["enabled"], (
        "Online DPO currently supports colocated generation only."
    )
    assert (
        policy_config["train_global_batch_size"]
        == online_dpo_config["num_prompts_per_step"]
    ), (
        "For the first Online DPO implementation, policy.train_global_batch_size "
        "must equal online_dpo.num_prompts_per_step. This keeps each optimizer "
        "step at one full batch of usable preference pairs."
    )


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
    processor: Optional[AutoProcessor] = None,
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    DPOLossFn,
    Logger,
    CheckpointManager,
    OnlineDPOSaveState,
    MasterConfig,
]:
    """Set up Online DPO training components."""
    setup_start_time = time.perf_counter()
    _validate_online_dpo_config(master_config, processor)

    policy_config = master_config["policy"]
    generation_config = policy_config["generation"]
    online_dpo_config = master_config["online_dpo"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    assert generation_config is not None

    set_seed(online_dpo_config["seed"])

    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    online_dpo_save_state: Optional[OnlineDPOSaveState] = cast(
        Optional[OnlineDPOSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if online_dpo_save_state is None:
        online_dpo_save_state = _default_online_dpo_save_state()

    train_dataloader = StatefulDataLoader(
        dataset,
        batch_size=online_dpo_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)
    print(f"  - Online DPO train dataloader loaded with {len(dataset)} prompts")

    val_dataloader: Optional[StatefulDataLoader] = None
    if (
        online_dpo_config["val_period"] > 0
        or online_dpo_config["val_at_start"]
        or online_dpo_config["val_at_end"]
    ):
        assert val_dataset is not None, (
            "Validation dataset is required when Online DPO validation is enabled."
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=online_dpo_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
            drop_last=False,
            num_workers=data_config["num_workers"],
        )
        print(
            f"  - Online DPO validation dataloader loaded with {len(val_dataset)} prompts"
        )

    print("\nSetting up compute cluster...", flush=True)
    backend = generation_config["backend"]
    cluster = RayVirtualCluster(
        name="online_dpo_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1 if backend == "megatron" else 2,
    )
    print(
        f"  - Ray cluster initialized with {cluster_config['num_nodes']} node(s)",
        flush=True,
    )

    print("\nSetting up model and generation...", flush=True)
    generation_config["model_name"] = policy_config["model_name"]
    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)

    if policy_config["megatron_cfg"]["enabled"]:
        total_train_iters = min(
            online_dpo_config["max_num_steps"],
            online_dpo_config["max_num_epochs"] * len(train_dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters * 2
        if "scheduler" in policy_config["megatron_cfg"]:
            for key in policy_config["megatron_cfg"]["scheduler"]:
                if "iters" in key:
                    policy_config["megatron_cfg"]["scheduler"][key] *= 2

    policy_generation: Optional[GenerationInterface]
    if backend == "megatron":
        policy_generation = None
        print(
            f"  - Using {backend} backend for generation with {policy_config['model_name']}",
            flush=True,
        )
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if "hf_config_overrides" in policy_config:
            generation_config["vllm_kwargs"]["hf_overrides"] = policy_config[
                "hf_config_overrides"
            ]
        policy_generation = VllmGeneration(cluster=cluster, config=generation_config)
        policy_generation.finish_generation()
        print(
            f"  - Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )
    elif backend == "sglang":
        generation_config = cast(SGLangConfig, generation_config)
        if "model_path" not in generation_config["sglang_cfg"]:
            generation_config["sglang_cfg"]["model_path"] = policy_config["model_name"]
        policy_generation = SGLangGeneration(cluster=cluster, config=generation_config)
        policy_generation.finish_generation()
        print(
            f"  - Using SGLang backend for generation with {policy_config['model_name']}",
            flush=True,
        )
    else:
        raise ValueError(f"Unsupported generation backend for Online DPO: {backend}")

    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=True,
    )
    policy.print_node_ip_and_gpu_id()

    state_dict_info = policy.prepare_refit_info()
    if policy_generation is not None:
        policy_generation.prepare_refit_info(state_dict_info)

    loss_fn = DPOLossFn(
        online_dpo_config,
        use_linear_ce_fusion=policy_config["megatron_cfg"]["enabled"]
        and policy_config["megatron_cfg"]["use_linear_ce_fusion_loss"],
    )

    print(f"  - Setup complete in {time.perf_counter() - setup_start_time:.2f}s")

    return (
        policy,
        policy_generation,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        online_dpo_save_state,
        master_config,
    )


def strip_trailing_environment_messages(
    message_log: LLMMessageLogType,
) -> LLMMessageLogType:
    """Return a copy of a rollout log ending at the last assistant response.

    Rollout helpers append the environment observation after scoring. DPO should
    optimize the sampled assistant response, not the trailing environment message.
    """
    last_assistant_idx = None
    for idx in range(len(message_log) - 1, -1, -1):
        if message_log[idx]["role"] == "assistant":
            last_assistant_idx = idx
            break
    if last_assistant_idx is None:
        raise ValueError(
            "Cannot build an Online DPO pair without an assistant response."
        )
    return copy.deepcopy(message_log[: last_assistant_idx + 1])


def _message_log_length(message_log: LLMMessageLogType) -> int:
    length = 0
    for message in message_log:
        token_ids = message.get("token_ids")
        if token_ids is None:
            raise ValueError("Online DPO message logs must contain token_ids.")
        assert isinstance(token_ids, torch.Tensor)
        length += int(token_ids.numel())
    return length


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def build_preference_datums_from_rollouts(
    rollout_batch: BatchedDataDict,
    *,
    min_reward_margin: float,
    drop_truncated_pairs: bool,
    max_pairs: Optional[int] = None,
) -> tuple[list[PreferenceDatumSpec], dict[str, float]]:
    """Create chosen/rejected Online DPO records from a two-sample rollout batch."""
    if rollout_batch.size % 2 != 0:
        raise ValueError(
            f"Online DPO expects an even rollout batch size, got {rollout_batch.size}."
        )
    if "total_reward" not in rollout_batch:
        raise ValueError("Online DPO rollout batch is missing total_reward.")

    preference_datums: list[PreferenceDatumSpec] = []
    reward_margins = []
    chosen_rewards = []
    rejected_rewards = []
    tie_pairs = 0
    truncated_pairs = 0
    dropped_pairs = 0
    processed_pairs = 0

    truncated = rollout_batch.get("truncated")

    for pair_start in range(0, rollout_batch.size, 2):
        processed_pairs += 1
        first_idx = pair_start
        second_idx = pair_start + 1

        if drop_truncated_pairs and truncated is not None:
            first_truncated = bool(truncated[first_idx])
            second_truncated = bool(truncated[second_idx])
            if first_truncated or second_truncated:
                truncated_pairs += 1
                dropped_pairs += 1
                continue

        first_reward = _as_float(rollout_batch["total_reward"][first_idx])
        second_reward = _as_float(rollout_batch["total_reward"][second_idx])
        margin = abs(first_reward - second_reward)
        if margin <= min_reward_margin:
            tie_pairs += 1
            dropped_pairs += 1
            continue

        if first_reward > second_reward:
            chosen_idx, rejected_idx = first_idx, second_idx
            chosen_reward, rejected_reward = first_reward, second_reward
        else:
            chosen_idx, rejected_idx = second_idx, first_idx
            chosen_reward, rejected_reward = second_reward, first_reward

        chosen_log = strip_trailing_environment_messages(
            cast(LLMMessageLogType, rollout_batch["message_log"][chosen_idx])
        )
        rejected_log = strip_trailing_environment_messages(
            cast(LLMMessageLogType, rollout_batch["message_log"][rejected_idx])
        )

        loss_multiplier = min(
            _as_float(rollout_batch["loss_multiplier"][chosen_idx]),
            _as_float(rollout_batch["loss_multiplier"][rejected_idx]),
        )
        datum: dict[str, Any] = {
            "message_log_chosen": chosen_log,
            "message_log_rejected": rejected_log,
            "length_chosen": _message_log_length(chosen_log),
            "length_rejected": _message_log_length(rejected_log),
            "loss_multiplier": loss_multiplier,
            "idx": _as_int(rollout_batch["idx"][pair_start]),
        }
        if "task_name" in rollout_batch:
            datum["task_name"] = rollout_batch["task_name"][pair_start]
        preference_datums.append(cast(PreferenceDatumSpec, datum))
        reward_margins.append(margin)
        chosen_rewards.append(chosen_reward)
        rejected_rewards.append(rejected_reward)

        if max_pairs is not None and len(preference_datums) >= max_pairs:
            break

    generated_pairs = rollout_batch.size // 2
    discarded_pairs = max(0, generated_pairs - processed_pairs)
    metrics = {
        "generated_pairs": float(generated_pairs),
        "usable_pairs": float(len(preference_datums)),
        "dropped_pairs": float(dropped_pairs),
        "discarded_pairs": float(discarded_pairs),
        "tie_pairs": float(tie_pairs),
        "truncated_pairs": float(truncated_pairs),
        "chosen_reward": float(np.mean(chosen_rewards)) if chosen_rewards else 0.0,
        "rejected_reward": float(np.mean(rejected_rewards))
        if rejected_rewards
        else 0.0,
        "reward_margin": float(np.mean(reward_margins)) if reward_margins else 0.0,
    }
    return preference_datums, metrics


def collate_preference_datums(
    preference_datums: list[PreferenceDatumSpec],
    tokenizer: TokenizerType,
    make_sequence_length_divisible_by: int,
) -> BatchedDataDict:
    """Collate Online DPO preference records with the offline DPO collator."""
    return preference_collate_fn(
        preference_datums,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
        add_loss_mask=True,
    )


def add_reference_logprobs_to_preference_batch(
    preference_batch: BatchedDataDict,
    policy: ColocatablePolicyInterface,
    *,
    micro_batch_size: int,
    timer: Optional[Timer] = None,
) -> BatchedDataDict:
    """Append rolled reference-policy logprobs required by DPOLossFn."""
    reference_logprobs = policy.get_reference_policy_logprobs(
        preference_batch,
        micro_batch_size=micro_batch_size,
        timer=timer,
    )["reference_logprobs"]
    preference_batch["reference_policy_logprobs"] = torch.roll(
        reference_logprobs, -1, dims=-1
    )
    return preference_batch


def _run_rollout(
    policy_generation: GenerationInterface,
    batch: BatchedDataDict,
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    master_config: MasterConfig,
) -> tuple[BatchedDataDict, dict[str, Any]]:
    if _should_use_async_rollouts(master_config):
        return run_async_multi_turn_rollout(
            policy_generation=policy_generation,
            input_batch=batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=master_config["policy"]["max_total_sequence_length"],
            max_rollout_turns=master_config["online_dpo"]["max_rollout_turns"],
            greedy=False,
        )
    return run_multi_turn_rollout(
        policy_generation=policy_generation,
        input_batch=batch,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=master_config["policy"]["max_total_sequence_length"],
        max_rollout_turns=master_config["online_dpo"]["max_rollout_turns"],
        greedy=False,
    )


def _aggregate_numeric_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    metric_values: dict[str, list[float]] = {}
    for metric_dict in metrics:
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))
    for key, values in metric_values.items():
        aggregated[key] = float(np.mean(values))
    return aggregated


def _aggregate_pair_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    count_keys = {
        "generated_pairs",
        "usable_pairs",
        "dropped_pairs",
        "discarded_pairs",
        "tie_pairs",
        "truncated_pairs",
    }
    aggregated = {key: 0.0 for key in count_keys}
    weighted_keys = {"chosen_reward", "rejected_reward", "reward_margin"}
    weighted_values = {key: 0.0 for key in weighted_keys}
    total_usable_pairs = 0.0

    for metric_dict in metrics:
        for key in count_keys:
            aggregated[key] += metric_dict.get(key, 0.0)
        usable_pairs = metric_dict.get("usable_pairs", 0.0)
        total_usable_pairs += usable_pairs
        for key in weighted_keys:
            weighted_values[key] += metric_dict.get(key, 0.0) * usable_pairs

    for key in weighted_keys:
        aggregated[key] = (
            weighted_values[key] / total_usable_pairs if total_usable_pairs > 0 else 0.0
        )

    return aggregated


def _prepare_generation(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    *,
    need_refit: bool,
    generation_stale: bool,
    colocated_inference: bool,
    timer: Optional[Timer] = None,
) -> bool:
    if need_refit and generation_stale:
        refit_policy_generation(
            policy,
            policy_generation,
            colocated_inference,
            timer=timer,
        )
        return False

    if colocated_inference and need_refit:
        policy.offload_after_refit()
    policy_generation.prepare_for_generation()
    return generation_stale


def _collect_preference_batch(
    dataloader_iter,
    policy_generation: GenerationInterface,
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    master_config: MasterConfig,
    timer: Timer,
) -> tuple[Optional[BatchedDataDict], dict[str, float], dict[str, Any], int, int]:
    online_dpo_config = master_config["online_dpo"]
    target_pairs = master_config["policy"]["train_global_batch_size"]
    preference_datums: list[PreferenceDatumSpec] = []
    pair_metrics = []
    rollout_metrics = []
    consumed_prompts = 0
    consumed_prompt_batches = 0

    while len(preference_datums) < target_pairs:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break

        consumed_prompts += batch.size
        consumed_prompt_batches += 1
        repeated_batch = batch.repeat_interleave(
            online_dpo_config["num_generations_per_prompt"]
        )

        with timer.time("generation"):
            rollout_batch, current_rollout_metrics = _run_rollout(
                policy_generation,
                repeated_batch,
                tokenizer,
                task_to_env,
                master_config,
            )
        rollout_metrics.append(current_rollout_metrics)

        remaining_pairs = target_pairs - len(preference_datums)
        new_datums, current_pair_metrics = build_preference_datums_from_rollouts(
            rollout_batch,
            min_reward_margin=online_dpo_config["min_reward_margin"],
            drop_truncated_pairs=online_dpo_config["drop_truncated_pairs"],
            max_pairs=remaining_pairs,
        )
        preference_datums.extend(new_datums)
        pair_metrics.append(current_pair_metrics)

        del repeated_batch
        del rollout_batch

    if len(preference_datums) < target_pairs:
        return (
            None,
            _aggregate_pair_metrics(pair_metrics),
            {},
            consumed_prompts,
            consumed_prompt_batches,
        )

    preference_batch = collate_preference_datums(
        preference_datums,
        tokenizer,
        master_config["policy"]["make_sequence_length_divisible_by"],
    )
    preference_batch.to("cpu")

    return (
        preference_batch,
        _aggregate_pair_metrics(pair_metrics),
        _aggregate_numeric_metrics(rollout_metrics),
        consumed_prompts,
        consumed_prompt_batches,
    )


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
    logger: Optional[Logger] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run environment rollout validation for Online DPO."""
    if val_dataloader is None:
        assert (
            val_dataloader is not None or master_config["online_dpo"]["val_period"] == 0
        ), "val_dataloader is None, so online_dpo.val_period must be 0"
        print("  - No validation dataloader provided, skipping validation", flush=True)
        return {}, {}
    assert val_task_to_env is not None

    timer = Timer()
    total_rewards = []
    total_lengths = []
    all_message_logs = []
    rollout_metric_batches = []

    with timer.time("total_validation_time"):
        print(f"Starting Online DPO validation at step {step}...", flush=True)
        online_dpo_config = master_config["online_dpo"]
        max_batches = (
            max(
                1,
                (
                    online_dpo_config["max_val_samples"]
                    + online_dpo_config["val_batch_size"]
                    - 1
                )
                // online_dpo_config["val_batch_size"],
            )
            if online_dpo_config["max_val_samples"] > 0
            else len(val_dataloader)
        )

        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            val_batch, rollout_metrics = _run_rollout(
                policy_generation,
                val_batch,
                tokenizer,
                val_task_to_env,
                master_config,
            )
            rollout_metric_batches.append(rollout_metrics)
            total_rewards.extend(val_batch["total_reward"].tolist())
            total_lengths.append(rollout_metrics["mean_gen_tokens_per_sample"])
            all_message_logs.extend(
                [
                    get_keys_from_message_log(
                        val_batch["message_log"][i], ["role", "content"]
                    )
                    for i in range(len(val_batch["message_log"]))
                ]
            )

    num_samples = len(total_rewards)
    reward = (
        float(torch.tensor(total_rewards, dtype=torch.float32).mean().item())
        if num_samples
        else 0.0
    )
    avg_length = (
        float(sum(total_lengths) / len(total_lengths)) if total_lengths else 0.0
    )
    val_metrics = {
        "reward": reward,
        "avg_length": avg_length,
        "num_samples": float(num_samples),
        **_aggregate_numeric_metrics(rollout_metric_batches),
    }

    try:
        print_message_log_samples(
            all_message_logs,
            total_rewards,
            num_samples=min(
                master_config["logger"]["num_val_samples_to_print"],
                len(all_message_logs),
            ),
            step=step,
        )
    except Exception as e:
        print(f"Error displaying validation samples: {str(e)}", flush=True)

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    print("\nOnline DPO validation results:")
    print(f"  - Reward: {reward:.4f}")
    print(f"  - Average response length: {avg_length:.1f} tokens")
    print(f"  - Samples processed: {num_samples}", flush=True)

    if logger is not None:
        logger.log_batched_dict_as_jsonl(
            {"content": all_message_logs, "rewards": total_rewards},
            f"val_data_step{step}.jsonl",
        )

    timer.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return val_metrics, timing_metrics


def _aggregate_train_metrics(train_results: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "loss": train_results["loss"].numpy(),
        "grad_norm": train_results["grad_norm"].numpy(),
    }
    if "moe_metrics" in train_results:
        metrics.update({f"moe/{k}": v for k, v in train_results["moe_metrics"].items()})
    metrics.update(train_results["all_mb_metrics"])

    aggregated = {}
    for key, value in metrics.items():
        if key in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
            aggregated[key] = float(np.mean(value).item())
        else:
            aggregated[key] = float(np.sum(value).item())
    return aggregated


def online_dpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    train_dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DPOLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    online_dpo_save_state: OnlineDPOSaveState,
    master_config: MasterConfig,
) -> None:
    """Run Online DPO training."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    need_refit = True
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        need_refit = False
    assert policy_generation is not None
    generation_stale = True
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    current_step = online_dpo_save_state["current_step"]
    current_prompt_batch = online_dpo_save_state.get(
        "current_prompt_batch", current_step
    )
    total_steps = online_dpo_save_state["total_steps"]
    current_epoch = online_dpo_save_state["current_epoch"]
    consumed_samples = online_dpo_save_state["consumed_samples"]
    total_valid_tokens = online_dpo_save_state.get("total_valid_tokens", 0)

    online_dpo_config = master_config["online_dpo"]
    max_num_steps = online_dpo_config["max_num_steps"]
    max_num_epochs = online_dpo_config["max_num_epochs"]
    val_period = online_dpo_config["val_period"]

    if online_dpo_config["val_at_start"] and total_steps == 0:
        print("\nRunning initial Online DPO validation...", flush=True)
        generation_stale = _prepare_generation(
            policy,
            policy_generation,
            need_refit=need_refit,
            generation_stale=generation_stale,
            colocated_inference=colocated_inference,
        )
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        dataloader_iter = iter(train_dataloader)

        while total_steps < max_num_steps:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), max_num_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, total_steps + 1)

            val_metrics, validation_timings = None, None
            with timer.time("total_step_time"):
                print("Preparing online preference batch...", flush=True)
                with timer.time("prepare_for_generation/total"):
                    generation_stale = _prepare_generation(
                        policy,
                        policy_generation,
                        need_refit=need_refit,
                        generation_stale=generation_stale,
                        colocated_inference=colocated_inference,
                        timer=timer,
                    )

                (
                    preference_batch,
                    pair_metrics,
                    rollout_metrics,
                    consumed_prompts,
                    consumed_prompt_batches,
                ) = _collect_preference_batch(
                    dataloader_iter,
                    policy_generation,
                    tokenizer,
                    task_to_env,
                    master_config,
                    timer,
                )
                current_prompt_batch += consumed_prompt_batches
                policy_generation.finish_generation()
                if preference_batch is None:
                    if consumed_prompts > 0:
                        warnings.warn(
                            "Discarding the final partial Online DPO batch because it "
                            "did not contain enough usable preference pairs."
                        )
                    timer.reset()
                    break

                logger.log_metrics(rollout_metrics, total_steps + 1, prefix="train")

                print("Computing reference policy logprobs...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()
                with timer.time("reference_policy_logprobs"):
                    preference_batch = add_reference_logprobs_to_preference_batch(
                        preference_batch,
                        policy,
                        micro_batch_size=master_config["policy"][
                            "train_micro_batch_size"
                        ]
                        * 2,
                        timer=timer,
                    )

                print("Training policy with DPO loss...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    generation_stale = True
                with timer.time("policy_training"):
                    train_results = policy.train(
                        preference_batch,
                        loss_fn,
                        eval_mode=False,
                        gbs=master_config["policy"]["train_global_batch_size"] * 2,
                        mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                        timer=timer,
                    )

                is_last_step = total_steps + 1 >= max_num_steps or (
                    current_epoch + 1 == max_num_epochs
                    and current_prompt_batch >= len(train_dataloader)
                )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    online_dpo_config["val_at_end"] and is_last_step
                ):
                    generation_stale = _prepare_generation(
                        policy,
                        policy_generation,
                        need_refit=need_refit,
                        generation_stale=generation_stale,
                        colocated_inference=colocated_inference,
                    )
                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                        logger=logger,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                metrics = _aggregate_train_metrics(train_results)
                metrics.update({f"online_dpo/{k}": v for k, v in pair_metrics.items()})
                metrics.update(rollout_metrics)
                total_valid_tokens += metrics["global_valid_toks"]

                consumed_samples += consumed_prompts
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()
                    online_dpo_save_state["current_step"] = current_step + 1
                    online_dpo_save_state["current_prompt_batch"] = current_prompt_batch
                    online_dpo_save_state["total_steps"] = total_steps + 1
                    online_dpo_save_state["current_epoch"] = current_epoch
                    online_dpo_save_state["total_valid_tokens"] = total_valid_tokens
                    online_dpo_save_state["consumed_samples"] = consumed_samples
                    if val_metrics is not None:
                        online_dpo_save_state["val_reward"] = val_metrics["reward"]
                    elif "val_reward" in online_dpo_save_state:
                        del online_dpo_save_state["val_reward"]

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with "
                            "'val:' or 'train:'."
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"No {prefix} metrics were collected for "
                                f"checkpoint metric {metric_name}.",
                                stacklevel=2,
                            )
                            if full_metric_name in online_dpo_save_state:
                                del online_dpo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            online_dpo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, online_dpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            timing_metrics = timer.get_timing_metrics(reduction_op="sum")
            total_time = timing_metrics.get("total_step_time", 0)
            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
                if total_time > 0
                else 0.0
            )

            print("\nOnline DPO training results:")
            print(f"  - Loss: {metrics['loss']:.4f}")
            print(f"  - Preference loss: {metrics['preference_loss']:.4f}")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - Usable pairs: {metrics['online_dpo/usable_pairs']:.0f}")
            print(f"  - Dropped pairs: {metrics['online_dpo/dropped_pairs']:.0f}")
            print(f"  - Reward margin: {metrics['online_dpo/reward_margin']:.4f}")
            print("\nTiming:")
            print(f"  - Total step time: {total_time:.2f}s")
            for key, value in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if key != "total_step_time":
                    percent = (value / total_time * 100) if total_time > 0 else 0
                    print(f"  - {key}: {value:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(
                timing_metrics,
                total_steps + 1,
                prefix="timing/train",
                step_finished=True,
            )

            if not _should_log_nemo_gym_responses(master_config):
                logger.log_batched_dict_as_jsonl(
                    {
                        "input_lengths": preference_batch["input_lengths"].tolist(),
                        "token_ids": preference_batch["input_ids"].tolist(),
                        "token_loss_mask": preference_batch["token_mask"].tolist(),
                        "sample_loss_mask": preference_batch["sample_mask"].tolist(),
                    },
                    f"train_data_step{total_steps + 1}.jsonl",
                )

            del preference_batch
            del train_results
            del metrics
            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_num_steps:
                print("Max number of steps has been reached, stopping training early")
                return

        current_epoch += 1
        current_step = 0
        current_prompt_batch = 0
