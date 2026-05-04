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
"""GRPO trainer — TransferQueue-mediated path (sync).

Sibling fork of ``nemo_rl.algorithms.grpo``. Mirrors verl's split between
``main_ppo.py`` (legacy) and ``main_ppo_sync.py`` (TQ-only): each file
has zero internal branching on whether TQ is engaged, and the example
script chooses one or the other.

Setup, helpers, and ``validate`` are re-imported from ``grpo``; only the
training loop body is duplicated here so the per-step lifecycle hooks
(register / seed-put / per-rank fetch / clear) can live in straight
sequential code.

Parity with the legacy path is verified by running the same config
against both entrypoints and diffing the wandb runs (Stage 5 of the
data-plane integration plan).
"""

from __future__ import annotations

import asyncio
import os
import warnings
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

# Re-imports from grpo so this file is a thin trainer-only fork.
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    MasterConfig,
    _create_advantage_estimator,
    _extract_prompt_only_messages,
    _log_mixed_rewards_and_advantages_information,
    _should_log_nemo_gym_responses,
    _should_use_async_rollouts,
    _should_use_nemo_gym,
    compute_and_apply_seq_logprob_error_masking,
    dynamic_sampling,
    refit_policy_generation,
    scale_rewards,
    validate,
)
from nemo_rl.algorithms.loss import (
    ClippedPGLossDataDict,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.reward_functions import apply_reward_shaping
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    log_generation_metrics_to_wandb,
    print_performance_metrics,
)
from nemo_rl.data.dataloader import MultipleDataloaderWrapper
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data_plane import (
    KVBatchMeta,
    build_data_plane_client,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.memory_tracker import MemoryTracker
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# Tensor fields of ``train_data`` we seed into the partition. The set must
# match FIELD_SCHEMA in nemo_rl/data_plane/schema.py once Stage 2 lands.
_DP_SEED_FIELDS = (
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "prev_logprobs",
    "reference_policy_logprobs",
    "advantages",
    "token_mask",
    "sample_mask",
)


def grpo_train_sync(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    wrapped_dataloader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
) -> None:
    """Run GRPO training algorithm — TransferQueue-mediated.

    Lifecycle per training step:
      1. ``register_partition`` once we have a complete batch.
      2. After ``train_data`` is assembled, ``kv_batch_put`` seeds the
         partition; build a ``KVBatchMeta`` carrying keys + per-sample
         seqlens.
      3. ``policy.train_from_dp_meta(meta)`` — driver fans out the
         per-rank meta only; each worker fetches its own slice from TQ
         (1-hop, no tensor data through the driver).
      4. ``kv_clear`` at end of step before the next register reuses the
         partition.

    Drops the legacy ``policy.train(BatchedDataDict)`` call entirely —
    parity test runs this trainer alongside ``grpo.grpo_train`` for the
    baseline.
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()
    memory_tracker = MemoryTracker()

    kv_scales_cache = None  # Cache reused for computed kv scales

    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert policy_generation is not None  # for mypy type check

    if master_config["grpo"].get("skip_reference_policy_logprobs_calculation"):
        assert master_config["loss_fn"]["reference_policy_kl_penalty"] == 0
        print(
            "Reference policy logprob calculation will be skipped since `grpo.skip_reference_policy_logprobs_calculation` is set to True and `loss_fn.reference_policy_kl_penalty` is 0."
        )

    sync_kv_scales = getattr(policy_generation, "requires_kv_scale_sync", False)

    current_step = grpo_save_state["current_step"]
    total_steps = grpo_save_state["total_steps"]
    max_num_steps = master_config["grpo"]["max_num_steps"]
    current_epoch = grpo_save_state["current_epoch"]
    max_num_epochs = master_config["grpo"]["max_num_epochs"]
    consumed_samples = grpo_save_state["consumed_samples"]
    total_valid_tokens = grpo_save_state.get("total_valid_tokens", 0)
    val_at_start = master_config["grpo"]["val_at_start"]
    val_at_end = master_config["grpo"]["val_at_end"]
    val_period = master_config["grpo"]["val_period"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    adv_estimator = _create_advantage_estimator(master_config)

    # ── Data-plane setup (mandatory in the sync trainer) ───────────────
    dp_cfg = master_config.get("data_plane")
    if not dp_cfg or not dp_cfg.get("enabled", False):
        raise ValueError(
            "grpo_train_sync requires master_config['data_plane']['enabled']=True. "
            "Use the legacy nemo_rl.algorithms.grpo.grpo_train trainer if you don't "
            "want TransferQueue."
        )
    dp_client = build_data_plane_client(dp_cfg)
    if hasattr(policy, "setup_data_plane"):
        # Workers attach to the (already-bootstrapped) controller via
        # bootstrap=False; train_from_dp_meta below relies on this.
        policy.setup_data_plane(dp_cfg)

    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        memory_tracker.snapshot_start_of_stage("Initial validation", dir())

        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
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

    if master_config["data"]["use_multiple_dataloader"]:
        warnings.warn(
            "When using multiple dataloaders, MultipleDataloaderWrapper operates as an infinite iterator. "
            "As a result, grpo.max_num_epochs will be ignored, and only grpo.max_num_steps will be used."
        )

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        memory_tracker.snapshot_start_of_stage("Preparing batch", dir())
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        batch_cache: Optional[BatchedDataDict[DatumSpec]] = None
        dynamic_sampling_num_gen_batches = 0

        for batch in wrapped_dataloader:
            metrics_logging_data: dict = {}
            metrics: dict = {}

            if master_config["data"]["use_multiple_dataloader"]:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/{max_num_steps} {'=' * 25}",
                    flush=True,
                )
            else:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/{min(len(wrapped_dataloader), max_num_steps)} {'=' * 25}",
                    flush=True,
                )

            maybe_gpu_profile_step(policy, total_steps + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config["grpo"]["num_generations_per_prompt"]
                        )
                    )
                    batched_flat, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    input_ids = batched_flat["token_ids"]

                memory_tracker.snapshot_start_of_stage("Generation", dir())
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        if sync_kv_scales and kv_scales_cache is None:
                            print("▶ Computing KV cache scales...", flush=True)
                            policy.prepare_for_lp_inference()
                            calib_flat, calib_input_lengths = (
                                batched_message_log_to_flat_message(
                                    repeated_batch["message_log"],
                                    pad_value_dict={
                                        "token_ids": tokenizer.pad_token_id
                                    },
                                    make_sequence_length_divisible_by=master_config[
                                        "policy"
                                    ]["make_sequence_length_divisible_by"],
                                )
                            )
                            calibration_data = BatchedDataDict[ClippedPGLossDataDict](
                                {
                                    "input_ids": calib_flat["token_ids"],
                                    "input_lengths": calib_input_lengths,
                                }
                            )
                            calibration_data.update(
                                calib_flat.get_multimodal_dict(as_tensors=False)
                            )
                            calibration_data.to("cpu")
                            kv_scales_cache = policy.calibrate_qkv_fp8_scales(
                                calibration_data, include_q=True
                            )["layers"]

                        refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                            timer=timer,
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()

                dynamic_sampling_num_gen_batches += 1
                if dynamic_sampling_num_gen_batches == 1 and hasattr(
                    policy_generation, "snapshot_step_metrics"
                ):
                    policy_generation.snapshot_step_metrics()
                with timer.time("generation"):
                    if policy_generation is not None:
                        policy_generation.clear_logger_metrics()
                    if _should_use_nemo_gym(master_config):
                        generation_config = master_config["policy"]["generation"]
                        nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=None,
                            generation_config=generation_config,
                            max_rollout_turns=None,
                            greedy=False,
                        )
                        input_ids = nemo_gym_rollout_result.input_ids
                        repeated_batch = nemo_gym_rollout_result.final_batch
                        rollout_metrics = nemo_gym_rollout_result.rollout_metrics
                        del nemo_gym_rollout_result

                        if not _should_log_nemo_gym_responses(master_config):
                            for key in list(rollout_metrics):
                                if "full_result" in key:
                                    rollout_metrics.pop(key)

                    elif _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["grpo"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["grpo"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    policy_generation.finish_generation()
                    if policy_generation is not None:
                        generation_logger_metrics = (
                            policy_generation.get_logger_metrics()
                        )

                    metrics_logging_data["mean_gen_tokens_per_sample"] = (
                        rollout_metrics["mean_gen_tokens_per_sample"]
                    )
                    logger.log_metrics(rollout_metrics, total_steps + 1, prefix="train")

                repeated_batch = scale_rewards(
                    repeated_batch, master_config["grpo"]["reward_scaling"]
                )
                if master_config["grpo"]["reward_shaping"]["enabled"]:
                    repeated_batch = apply_reward_shaping(
                        repeated_batch, master_config["grpo"]["reward_shaping"]
                    )

                memory_tracker.snapshot_start_of_stage("Processing rewards", dir())
                print("▶ Processing rewards...,", flush=True)
                with timer.time("reward_calculation"):
                    rewards = repeated_batch["total_reward"]

                    print("▶ Computing advantages...", flush=True)
                    if master_config["grpo"].get("calculate_advantages_on_gpu"):
                        print("Computing advantages on GPU!")
                        device_id = 0
                        baseline, std = calculate_baseline_and_std_per_prompt(
                            input_ids.cuda(device_id),
                            rewards.cuda(device_id),
                            torch.ones_like(rewards).cuda(device_id),
                            leave_one_out_baseline=master_config["grpo"][
                                "use_leave_one_out_baseline"
                            ],
                        )
                        baseline = baseline.cpu()
                        std = std.cpu()
                    else:
                        baseline, std = calculate_baseline_and_std_per_prompt(
                            input_ids,
                            rewards,
                            torch.ones_like(rewards),
                            leave_one_out_baseline=master_config["grpo"][
                                "use_leave_one_out_baseline"
                            ],
                        )

                    repeated_batch, is_batch_complete, batch_cache, ds_metrics = (
                        dynamic_sampling(
                            repeated_batch,
                            std,
                            baseline,
                            dynamic_sampling_num_gen_batches,
                            master_config,
                            timer,
                            batch_cache,
                        )
                    )
                    if ds_metrics:
                        ds_metrics["dynamic_sampling_num_gen_batches"] = (
                            dynamic_sampling_num_gen_batches
                        )
                    rewards = (
                        repeated_batch["total_reward"]
                        if not master_config["grpo"]["use_dynamic_sampling"]
                        else repeated_batch["filtered_reward"]
                    )
                    baseline = repeated_batch["baseline"]
                    std = repeated_batch["std"]

                    if not is_batch_complete:
                        continue

                    # ── Stage 0/Stage 1: register the per-step partition.
                    # Static "train" id (verl-style); cleared and reused
                    # each step.
                    dp_client.register_partition(
                        partition_id="train",
                        fields=list(_DP_SEED_FIELDS),
                        num_samples=int(repeated_batch["loss_multiplier"].shape[0]),
                        consumer_tasks=["prev_lp", "ref_lp", "train"],
                        grpo_group_size=master_config["grpo"][
                            "num_generations_per_prompt"
                        ],
                    )

                    gen_step_metrics = {}
                    if hasattr(policy_generation, "get_step_metrics"):
                        gen_step_metrics = policy_generation.get_step_metrics()

                    baseline_for_log = baseline.clone()

                    prompt_only_message_logs = _extract_prompt_only_messages(
                        repeated_batch["message_log"]
                    )
                    prompt_batched_flat, _ = batched_message_log_to_flat_message(
                        prompt_only_message_logs,
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    prompt_ids_for_adv = prompt_batched_flat["token_ids"]
                    del prompt_only_message_logs
                    del prompt_batched_flat
                    del input_ids
                    del baseline
                    del std

                with timer.time("data_processing"):
                    use_overlong_filtering = master_config["grpo"]["overlong_filtering"]
                    if use_overlong_filtering:
                        loss_multiplier = repeated_batch["loss_multiplier"].clone()
                        truncated = repeated_batch["truncated"]

                        if isinstance(truncated, list):
                            truncated = torch.tensor(truncated, dtype=torch.bool)

                        loss_multiplier[truncated] = 0
                        repeated_batch["loss_multiplier"] = loss_multiplier
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        for j, message in enumerate(message_log):
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    extra_multimodal_data = flat_messages.get_multimodal_dict(
                        as_tensors=False
                    )
                    train_data.update(extra_multimodal_data)
                    train_data.to("cpu")

                    metrics_logging_data["content"] = flat_messages["content"]

                memory_tracker.snapshot_start_of_stage("Computing logprobs", dir())
                print("▶ Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    logprob_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                            **extra_multimodal_data,
                        }
                    )
                    train_data["prev_logprobs"] = policy.get_logprobs(
                        logprob_data, timer=timer
                    )["logprobs"]

                    if not master_config["grpo"].get(
                        "skip_reference_policy_logprobs_calculation"
                    ):
                        train_data["reference_policy_logprobs"] = (
                            policy.get_reference_policy_logprobs(
                                logprob_data,
                                timer=timer,
                            )["reference_logprobs"]
                        )

                    del logprob_data
                    del extra_multimodal_data

                    (
                        max_seq_mult_prob_error,
                        num_masked_seqs,
                        masked_correct_pct,
                    ) = compute_and_apply_seq_logprob_error_masking(
                        train_data=train_data,
                        rewards=rewards,
                        seq_logprob_error_threshold=master_config["grpo"][
                            "seq_logprob_error_threshold"
                        ],
                    )

                with timer.time("advantage_calculation"):
                    print("▶ Computing advantages...", flush=True)
                    token_mask = train_data["token_mask"]
                    sample_mask = train_data["sample_mask"]
                    mask = token_mask * sample_mask.unsqueeze(-1)

                    train_data["advantages"] = adv_estimator.compute_advantage(
                        prompt_ids=prompt_ids_for_adv,
                        rewards=rewards,
                        mask=mask,
                        repeated_batch=repeated_batch,
                        logprobs_policy=train_data["prev_logprobs"],
                        logprobs_reference=train_data.get("reference_policy_logprobs"),
                    )
                    del prompt_ids_for_adv

                    _log_mixed_rewards_and_advantages_information(
                        logger=logger,
                        total_steps=total_steps,
                        metrics=metrics,
                        baseline=baseline_for_log,
                        advantages=train_data["advantages"],
                    )
                    del baseline_for_log

                # ── Driver-side balanced packing (mirrors legacy lm_policy.train).
                # ``shard_by_batch_size(shards=DP_world, sequence_packing_args=...)``
                # uses ``bin_count_multiple=DP_world``, which is what guarantees
                # every DP rank ends up with the same number of microbatches —
                # without it, sequence-packing / dynamic-batching produce
                # variable per-rank bin counts and Megatron diverges on its
                # first cross-DP collective. Pre-shard here, then fan out a
                # ``list[KVBatchMeta]`` with each shard's pre-computed
                # micro_batch_indices/lengths in ``extra_info``.
                policy_cfg = master_config["policy"]
                dp_world = policy.sharding_annotations.get_axis_size(
                    "data_parallel"
                )
                gbs = policy_cfg["train_global_batch_size"]
                seqpack_cfg = policy_cfg.get("sequence_packing", {}) or {}
                dynbatch_cfg = policy_cfg.get("dynamic_batching", {}) or {}

                spa: Optional[dict[str, Any]] = None
                dba: Optional[dict[str, Any]] = None
                if dynbatch_cfg.get("enabled", False):
                    dba = {
                        "input_key": "input_ids",
                        "input_lengths_key": "input_lengths",
                        "sequence_length_round": dynbatch_cfg[
                            "sequence_length_round"
                        ],
                        "max_tokens_per_microbatch": dynbatch_cfg[
                            "train_mb_tokens"
                        ],
                    }
                elif seqpack_cfg.get("enabled", False):
                    spa = {
                        "algorithm": seqpack_cfg["algorithm"],
                        "input_key": "input_ids",
                        "input_lengths_key": "input_lengths",
                        "sequence_length_pad_multiple": policy_cfg[
                            "make_sequence_length_divisible_by"
                        ],
                        "max_tokens_per_microbatch": seqpack_cfg[
                            "train_mb_tokens"
                        ],
                    }

                if dba is not None:
                    pre_shards, _ = train_data.shard_by_batch_size(
                        dp_world,
                        batch_size=gbs,
                        dynamic_batching_args=dba,
                    )
                elif spa is not None:
                    pre_shards, _ = train_data.shard_by_batch_size(
                        dp_world,
                        batch_size=gbs,
                        sequence_packing_args=spa,
                    )
                else:
                    pre_shards = train_data.shard_by_batch_size(
                        dp_world,
                        batch_size=gbs,
                    )

                dp_metas: list[KVBatchMeta] = []
                for dp_rank, shard in enumerate(pre_shards):
                    n_shard = int(shard["sample_mask"].shape[0])
                    shard_keys = [
                        f"step{total_steps}_dp{dp_rank}_s{i}"
                        for i in range(n_shard)
                    ]
                    shard_field_names = [
                        f
                        for f in _DP_SEED_FIELDS
                        if f in shard and isinstance(shard[f], torch.Tensor)
                    ]
                    shard_fields = TensorDict(
                        {
                            f: shard[f].detach().contiguous()
                            for f in shard_field_names
                        },
                        batch_size=[n_shard],
                    )
                    asyncio.run(
                        dp_client.kv_batch_put(
                            keys=shard_keys,
                            partition_id="train",
                            fields=shard_fields,
                        )
                    )
                    extra: dict[str, Any] = {}
                    if (
                        getattr(shard, "micro_batch_indices", None) is not None
                        and getattr(shard, "micro_batch_lengths", None) is not None
                    ):
                        extra["micro_batch_indices"] = shard.micro_batch_indices
                        extra["micro_batch_lengths"] = shard.micro_batch_lengths
                        ecpg = getattr(shard, "elem_counts_per_gb", None)
                        if ecpg is not None:
                            extra["elem_counts_per_gb"] = ecpg
                    dp_metas.append(
                        KVBatchMeta(
                            partition_id="train",
                            task_name="train",
                            keys=shard_keys,
                            fields=shard_field_names,
                            sequence_lengths=[
                                int(s) for s in shard["input_lengths"].tolist()
                            ],
                            extra_info=extra,
                        )
                    )

                memory_tracker.snapshot_start_of_stage("Policy train", dir())
                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    # 1-hop: driver fans out the per-rank pre-balanced meta
                    # list; the @dp_dispatch decorator on Policy.train detects
                    # the list[KVBatchMeta] input and routes through worker
                    # `train_presharded`, which fetches its slice from TQ.
                    train_results = policy.train(
                        dp_metas,
                        loss_fn=loss_fn,
                        timer=timer,
                    )

                if sync_kv_scales:
                    with timer.time("recompute_kv_scales"):
                        print(
                            "▶ Recomputing KV cache scales after policy update...",
                            flush=True,
                        )
                        kv_scales_cache = policy.calibrate_qkv_fp8_scales(
                            train_data, include_q=True
                        )["layers"]
                        POLICY_GENERATION_STALE = True

                is_last_step = total_steps + 1 >= max_num_steps
                if not master_config["data"]["use_multiple_dataloader"]:
                    is_last_step = is_last_step or (
                        (current_epoch + 1 == max_num_epochs)
                        and (current_step + 1 == len(wrapped_dataloader))
                    )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    memory_tracker.snapshot_start_of_stage("Validation", dir())
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()
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

                flat_advantages = train_data["advantages"]
                flat_token_mask = flat_messages["token_loss_mask"]

                response_advantages = torch.masked_select(
                    flat_advantages, flat_token_mask.bool()
                )

                memory_tracker.snapshot_start_of_stage("Metrics", dir())
                metrics = {
                    **metrics,
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "reward": rewards.numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                    "advantages/mean": torch.mean(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    "advantages/max": torch.max(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    "advantages/min": torch.min(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    **ds_metrics,
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                if master_config["grpo"]["use_dynamic_sampling"]:
                    metrics["filtered_reward"] = rewards.numpy()
                    metrics["reward"] = repeated_batch["total_reward"].numpy()

                metrics.update(train_results["all_mb_metrics"])
                metrics.update(gen_step_metrics)
                for k, v in metrics.items():
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k] = (
                            np.min(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k] = (
                            np.max(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {
                        "lr",
                        "wd",
                        "reward",
                        "filtered_reward",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    elif isinstance(v, (np.ndarray, list)):
                        metrics[k] = np.sum(v).item()
                    else:
                        print(f"Skipping aggregation for {k} ({type(v)})")

                metrics.update(rollout_metrics)
                metrics["generation_logger_metrics"] = generation_logger_metrics
                total_valid_tokens += metrics["global_valid_toks"]

                metrics["max_seq_mult_prob_error"] = max_seq_mult_prob_error
                metrics["num_masked_seqs_by_logprob_error"] = num_masked_seqs
                metrics["masked_correct_pct"] = masked_correct_pct

                consumed_samples += master_config["grpo"]["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                memory_tracker.snapshot_start_of_stage("Checkpointing", dir())
                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()

                    grpo_save_state["current_step"] = current_step + 1
                    grpo_save_state["total_steps"] = total_steps + 1
                    grpo_save_state["current_epoch"] = current_epoch
                    grpo_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        grpo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in grpo_save_state:
                        del grpo_save_state["val_reward"]
                    grpo_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. ",
                                stacklevel=2,
                            )
                            if full_metric_name in grpo_save_state:
                                del grpo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            grpo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, grpo_save_state, master_config
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
                        if master_config["data"]["use_multiple_dataloader"]:
                            for (
                                task_name,
                                task_dataloader,
                            ) in wrapped_dataloader.dataloaders.items():
                                torch.save(
                                    task_dataloader.state_dict(),
                                    os.path.join(
                                        checkpoint_path,
                                        f"train_dataloader_{task_name}.pt",
                                    ),
                                )
                        else:
                            torch.save(
                                wrapped_dataloader.state_dict(),
                                os.path.join(checkpoint_path, "train_dataloader.pt"),
                            )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            memory_tracker.snapshot_start_of_stage("Logging", dir())
            if not _should_log_nemo_gym_responses(master_config):
                log_data: dict = {}
                if "agent_ref" in repeated_batch:
                    log_data["agent_ref"] = repeated_batch["agent_ref"]
                log_data["content"] = flat_messages["content"]
                log_data["rewards"] = rewards.tolist()
                if master_config["grpo"]["use_dynamic_sampling"]:
                    log_data["filtered_rewards"] = rewards.tolist()
                    log_data["rewards"] = repeated_batch["total_reward"].tolist()
                log_data["input_lengths"] = input_lengths.tolist()
                log_data["token_ids"] = train_data["input_ids"].tolist()
                log_data["token_loss_mask"] = train_data["token_mask"].tolist()
                log_data["sample_loss_mask"] = train_data["sample_mask"].tolist()
                log_data["advantages"] = train_data["advantages"].tolist()
                log_data["generation_logprobs"] = train_data[
                    "generation_logprobs"
                ].tolist()
                log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()

                logger.log_batched_dict_as_jsonl(
                    log_data, f"train_data_step{total_steps + 1}.jsonl"
                )
                del log_data
            del flat_messages

            timing_metrics: dict = timer.get_timing_metrics(reduction_op="sum")  # type: ignore
            if metrics["token_mult_prob_error"] > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths": repeated_batch["length"],
                        "full_lengths": input_lengths,
                        "generation_logprobs": train_data["generation_logprobs"],
                        "prev_logprobs": train_data["prev_logprobs"],
                        "token_mask": train_data["token_mask"],
                        "sample_mask": train_data["sample_mask"],
                    },
                    total_steps + 1,
                    name="train/token_mult_prob_error_plot_sample",
                )
            del train_data
            if master_config["policy"]["generation"].get("vllm_cfg", {}).get(
                "enable_vllm_metrics_logger", False
            ) and master_config.get("logger", {}).get("wandb_enabled", False):
                log_generation_metrics_to_wandb(
                    generation_logger_metrics,
                    total_steps + 1,
                    master_config["policy"]["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            if (
                master_config["policy"]["generation"]
                .get("vllm_cfg", {})
                .get("async_engine", False)
            ):
                for metric_name in metrics.keys():
                    if metric_name.startswith("histogram/"):
                        logger.log_histogram(
                            metrics[metric_name],
                            total_steps + 1,
                            f"generation_metrics/{metric_name}",
                        )

            print("\n📊 Training Results:")
            print(f"  • Loss: {metrics['loss']:.4f}")
            if "draft_loss" in metrics:
                print(f"  • Draft Loss: {metrics['draft_loss']:.4f}")
            print(f"  • Generation KL Error: {metrics['gen_kl_error']:.4f}")
            if master_config["grpo"]["use_dynamic_sampling"]:
                print(f"  • Avg Filtered Reward: {np.mean(rewards.numpy()):.4f}")
                print(
                    f"  • Avg Total Reward: {np.mean(repeated_batch['total_reward'].numpy()):.4f}"
                )
            else:
                print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(
                f"  • Mean Generation Length: {metrics_logging_data['mean_gen_tokens_per_sample']:.4f}",
                flush=True,
            )

            print("\n⏱️  Timing:", flush=True)
            total_time = timing_metrics.get("total_step_time", 0)

            number_of_samples_per_step = (
                master_config["grpo"]["num_prompts_per_step"]
                * master_config["grpo"]["num_generations_per_prompt"]
            )
            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)

            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            performance_metrics = print_performance_metrics(
                train_results, metrics, timing_metrics, master_config
            )

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(
                performance_metrics, total_steps + 1, prefix="performance"
            )
            logger.log_metrics(
                timing_metrics,
                total_steps + 1,
                prefix="timing/train",
                step_finished=True,
            )

            batch_cache = None
            dynamic_sampling_num_gen_batches = 0

            memory_tracker.snapshot_start_of_stage("After CPU memory clear", dir())

            del repeated_batch
            del rewards
            del metrics
            if "val_metrics" in dir():
                del val_metrics

            # Stage 7: clear the partition before the next step's register
            # reuses the same id.
            dp_client.kv_clear(keys=None, partition_id="train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                memory_tracker.snapshot_start_of_stage("", dir())
                print("Timeout has been reached, stopping training early", flush=True)
                dp_client.close()
                return
            if total_steps >= max_num_steps:
                memory_tracker.snapshot_start_of_stage("", dir())
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                dp_client.close()
                return

        current_epoch += 1
        current_step = 0

    dp_client.close()
