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

Sibling fork of ``nemo_rl.algorithms.grpo``. Each file has zero
internal branching on whether TQ is engaged; the example script
chooses one or the other based on ``data_plane.enabled``.

Setup and helpers are re-imported from ``grpo``; the training loop body
is duplicated here so the per-step lifecycle hooks (register / seed-put
/ per-rank fetch / clear) can live in straight sequential code.
Validation is implemented locally as :func:`validate_sync` — a
TQ-mediated sibling of :func:`nemo_rl.algorithms.grpo.validate` that
routes val rollouts through ``SyncRolloutActor.rollout_to_tq`` into a
per-batch ``"val"`` partition.

Parity with the legacy path is verified by running the same config
against both entrypoints and diffing the wandb runs.
"""

from __future__ import annotations

import gc
import os
import warnings
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nemo_rl.models.policy.tq_policy import TQPolicy

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

# Re-imports from grpo so this file is a thin trainer-only fork.
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    MasterConfig,
    _clip_grpo_advantages,
    _create_advantage_estimator,
    _log_mixed_rewards_and_advantages_information,
    _should_log_nemo_gym_responses,
    _should_use_nemo_gym,
    compute_and_apply_seq_logprob_error_masking,
    scale_rewards,
)
from nemo_rl.algorithms.loss import (
    ClippedPGLossDataDict,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.reward_functions import apply_reward_shaping
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    get_gdpo_reward_component_keys,
    log_generation_metrics_to_wandb,
    print_performance_metrics,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import DP_CALIB_INPUT_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.sync_rollout_actor import SyncRolloutActor
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger, print_message_log_samples
from nemo_rl.utils.memory_tracker import MemoryTracker
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer
from nemo_rl.utils.venvs import make_actor_runtime_env
from nemo_rl.weight_sync import WeightSynchronizer

# ── DAPO non-zero-std dynamic sampling, slice-only ─────────────────────
# Slice-only formulation of nemo_rl.algorithms.grpo.dynamic_sampling: filter
# on std != 0, accumulate survivors across iterations, slice on overflow.
# Bulk in TQ untouched except for clear_samples of dropped/discarded uids.


def _apply_dynamic_sampling(
    *,
    meta: KVBatchMeta,
    driver_carry: BatchedDataDict,
    pending_meta: Optional[KVBatchMeta],
    pending_carry: Optional[BatchedDataDict],
    pending_unfiltered_rewards: list[torch.Tensor],
    train_prompts_size: int,
    num_gen_batches: int,
    max_gen_batches: int,
    policy: "TQPolicy",
) -> tuple[
    Optional[KVBatchMeta],
    Optional[BatchedDataDict],
    list[torch.Tensor],
    bool,
    dict[str, Any],
    Optional[torch.Tensor],
]:
    """Process one dynamic-sampling iteration.

    Drops zero-std (filtered) keys, merges survivors into the running
    pending cache, and reports whether the cache has reached
    ``train_prompts_size``. When complete, the returned ``pending_*`` IS
    the training batch.

    Args:
        meta: This iteration's ``KVBatchMeta``.
        driver_carry: Per-row driver-local tensors for this iteration
            (rewards, masks, prompt_ids_for_adv, baseline/std, …).
        pending_meta: Survivors accumulated from prior iterations.
        pending_carry: ``driver_carry`` rows aligned to ``pending_meta``.
        pending_unfiltered_rewards: All iterations' rewards pre-filter,
            for legacy reward metric parity.
        train_prompts_size: Target batch size.
        num_gen_batches: Iteration counter (1-based).
        max_gen_batches: Upper bound on iterations before raising.
        policy: TQPolicy whose ``discard_samples`` is used to drop filtered keys.

    Returns:
        ``(pending_meta, pending_carry, pending_rewards, is_complete,
        ds_metrics, unfiltered_for_log)``.
    """
    # Cumulative unfiltered total_reward for legacy metrics["reward"]
    # parity. Reference-only append (no copy) — slice tensors are
    # produced fresh per iteration, not aliased to TQ-owned bulk.
    pending_unfiltered_rewards.append(driver_carry["total_reward"])

    # Filter input comes from ``meta.tags`` so the filter decision is
    # meta-only — no tensor data needed. The driver mirrored ``std``
    # into tags right after baseline/std compute.
    if meta.tags is None:
        raise ValueError(
            "_apply_dynamic_sampling: meta.tags is None — driver must "
            "stamp 'std' into meta.tags before this call."
        )
    keep_idx = [i for i, t in enumerate(meta.tags) if t["std"] != 0.0]
    drop_keys = [k for k, t in zip(meta.sample_ids, meta.tags) if t["std"] == 0.0]
    if drop_keys:
        policy.discard_samples(drop_keys, meta.partition_id)

    # Subset survivors and merge into the running cache.
    if keep_idx:
        survivors_meta = meta.subset(keep_idx)
        survivors_carry = driver_carry.select_indices(keep_idx)
        survivors_carry["filtered_reward"] = survivors_carry["total_reward"]
        if pending_meta is None:
            pending_meta, pending_carry = survivors_meta, survivors_carry
        else:
            assert pending_carry is not None
            pending_meta = pending_meta.concat(survivors_meta)
            pending_carry = BatchedDataDict.from_batches(
                [pending_carry, survivors_carry]
            )

    n = len(pending_meta.sample_ids) if pending_meta is not None else 0
    if n < train_prompts_size:
        if num_gen_batches > max_gen_batches:
            raise ValueError(
                f"Dynamic sampling reached max_gen_batches={max_gen_batches}. "
                f"Increase grpo.dynamic_sampling_max_gen_batches or revisit "
                f"data diversity / num_prompts_per_step / num_generations_per_prompt."
            )
        return pending_meta, pending_carry, pending_unfiltered_rewards, False, {}, None

    ds_metrics: dict[str, Any] = {"dynamic_sampling_num_gen_batches": num_gen_batches}
    assert pending_meta is not None and pending_carry is not None
    if n > train_prompts_size:
        policy.discard_samples(
            list(pending_meta.sample_ids[train_prompts_size:]),
            pending_meta.partition_id,
        )
        pending_meta = pending_meta.slice(0, train_prompts_size)
        pending_carry = pending_carry.slice(0, train_prompts_size)
        ds_metrics["dynamic_sampling_num_discarded_valid_samples"] = (
            n - train_prompts_size
        )

    unfiltered_for_log = torch.cat(pending_unfiltered_rewards)[:train_prompts_size]
    return pending_meta, pending_carry, [], True, ds_metrics, unfiltered_for_log


def validate_sync(
    *,
    rollout_actor: SyncRolloutActor,
    policy: "TQPolicy",
    val_dataloader: Optional[StatefulDataLoader],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
    logger: Optional[Logger] = None,
    partition_id: str = "val",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """TQ-mediated counterpart to :func:`nemo_rl.algorithms.grpo.validate`.

    Per-batch: register the val partition → ``rollout_to_tq`` →
    ``policy.read_from_dataplane`` for message logs → ``policy.finish_step``.
    Caller owns ``policy_generation.prepare_for_generation`` /
    ``finish_generation`` around the call; the actor's per-rollout
    ``finish_generation`` is suppressed so inference state stays warm
    across batches.
    """
    if val_dataloader is None:
        assert master_config.grpo["val_period"] == 0, (
            "val_dataloader is None, so grpo.val_period must be 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    timer = Timer()
    total_rewards: list[float] = []
    total_lengths: list[float] = []
    all_message_logs: list[list[dict[str, str]]] = []
    additional_metrics: dict[str, Any] = {}
    capture_extras = _should_use_nemo_gym(master_config)

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)
        max_batches = (
            master_config.grpo["max_val_samples"]
            // master_config.grpo["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break
            n_prompts = int(val_batch.size)
            policy.prepare_val_partition(n_prompts, partition_id=partition_id)
            meta, driver_carry, rollout_metrics, _ = ray.get(
                rollout_actor.rollout_to_tq.remote(
                    val_batch,
                    partition_id=partition_id,
                    first_iter=False,
                    finish_generation=False,
                    task_to_env_override=val_task_to_env,
                    carry_keys=["total_reward", "turn_roles", "turn_contents"],
                )
            )
            roles = driver_carry["turn_roles"]
            contents = driver_carry["turn_contents"]
            total_rewards.extend(driver_carry["total_reward"].tolist())
            total_lengths.append(rollout_metrics["mean_gen_tokens_per_sample"])
            all_message_logs.extend(
                [{"role": r, "content": c} for r, c in zip(roles[i], contents[i])]
                for i in range(n_prompts)
            )
            if capture_extras:
                additional_metrics = rollout_metrics
            policy.finish_step(meta)

        accuracy = (
            torch.tensor(total_rewards, dtype=torch.float32).mean().item()
            if total_rewards
            else 0.0
        )
        avg_length = sum(total_lengths) / len(total_lengths) if total_lengths else 0.0
        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
            **additional_metrics,
        }
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config.logger["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    print(
        f"\n📊 Validation Results:\n"
        f"    • Accuracy: {accuracy:.4f}\n"
        f"    • Average response length: {avg_length:.1f} tokens\n"
        f"    • Samples processed: {len(total_rewards)}\n"
        f"  ⏱️  Total validation time: "
        f"{timing_metrics.get('total_validation_time', 0):.2f}s",
        flush=True,
    )
    if logger is not None:
        logger.log_batched_dict_as_jsonl(
            {"content": all_message_logs, "rewards": total_rewards},
            f"val_data_step{step}.jsonl",
        )
    timer.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return val_metrics, timing_metrics


def _compute_seq_logprob_error_metrics(
    *,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    prev_logprobs: torch.Tensor,
    generation_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    seq_logprob_error_threshold: Optional[float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    # Thin BDD for the data-driven masking call: take
    # the slice you need, transform, write delta back.
    masking_data = BatchedDataDict[ClippedPGLossDataDict](
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "prev_logprobs": prev_logprobs,
            "generation_logprobs": generation_logprobs,
        }
    )
    seq_error_result = compute_and_apply_seq_logprob_error_masking(
        train_data=masking_data,
        rewards=rewards,
        seq_logprob_error_threshold=seq_logprob_error_threshold,
    )
    seq_logprob_error_metrics = seq_error_result
    if "num_masked_seqs" in seq_logprob_error_metrics:
        seq_logprob_error_metrics["num_masked_seqs_by_logprob_error"] = (
            seq_logprob_error_metrics.pop("num_masked_seqs")
        )
    return masking_data["sample_mask"], seq_logprob_error_metrics


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
    weight_sync: Optional[WeightSynchronizer] = None,
) -> None:
    """Run GRPO training algorithm — TransferQueue-mediated.

    Body mirrors :func:`nemo_rl.algorithms.grpo.grpo_train` with TQ-mediated
    Policy methods substituting the in-memory dispatch. The TQ lifecycle
    (controller bootstrap, worker attach, partition register, fan-out,
    drain, close) is fully encapsulated in
    :class:`nemo_rl.models.policy.tq_policy.TQPolicy` — this trainer just
    calls ``policy.prepare_step``, ``policy.get_logprobs``,
    ``policy.get_reference_policy_logprobs``, and ``policy.train``.

    Parity with the legacy path is verified by running the same config
    against both entrypoints and diffing the wandb runs.
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()
    memory_tracker = MemoryTracker()

    kv_scales_cache = None  # Cache reused for computed kv scales

    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
    assert policy_generation is not None

    if master_config.grpo.get("skip_reference_policy_logprobs_calculation"):
        assert master_config.loss_fn.reference_policy_kl_penalty == 0
        print(
            "Reference policy logprob calculation will be skipped since `grpo.skip_reference_policy_logprobs_calculation` is set to True and `loss_fn.reference_policy_kl_penalty` is 0."
        )

    sync_kv_scales = getattr(policy_generation, "requires_kv_scale_sync", False)

    current_step = grpo_save_state["current_step"]
    total_steps = grpo_save_state["total_steps"]
    max_num_steps = master_config.grpo["max_num_steps"]
    current_epoch = grpo_save_state["current_epoch"]
    max_num_epochs = master_config.grpo["max_num_epochs"]
    consumed_samples = grpo_save_state["consumed_samples"]
    total_valid_tokens = grpo_save_state.get("total_valid_tokens", 0)
    val_at_start = master_config.grpo["val_at_start"]
    val_at_end = master_config.grpo["val_at_end"]
    val_period = master_config.grpo["val_period"]
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]

    adv_estimator = _create_advantage_estimator(master_config)

    # ── Data-plane setup (mandatory in the sync trainer) ───────────────
    # Sync trainer requires a TQ-mediated policy. The TQPolicy actor
    # bootstraps the controller and attaches workers; ``policy.dp_cfg``
    # is the public marker. The explicit master_config check is the
    # entry-guard so users running this trainer with the legacy policy
    # see a clear error rather than an opaque AttributeError.
    dp_cfg = master_config.data_plane
    if not dp_cfg or not dp_cfg["enabled"]:
        raise ValueError(
            "grpo_train_sync requires master_config['data_plane']['enabled']=True. "
            "Use the legacy nemo_rl.algorithms.grpo.grpo_train trainer if you don't "
            "want TransferQueue."
        )

    # Driver-side pad-value dict for materialize() — the wire emits
    # jagged tensors for variable-length token fields (input_ids,
    # prompt_ids_for_adv); other fields default to pad=0.
    _pad_dict = {
        "input_ids": tokenizer.pad_token_id,
        "prompt_ids_for_adv": tokenizer.pad_token_id,
    }
    if not hasattr(policy, "dp_cfg"):
        raise ValueError(
            "grpo_train_sync requires a TQ-mediated policy "
            "(nemo_rl.models.policy.tq_policy.TQPolicy). examples/run_grpo.py "
            "constructs it via the policy_factory when data_plane.enabled=True."
        )

    # TQ-resident tensors live on CPU; baseline/std are computed on the
    # slice without a CUDA hop. The flag is a no-op here — warn so users
    # don't expect it to do anything.
    if master_config.grpo.get("calculate_advantages_on_gpu"):
        warnings.warn(
            "grpo.calculate_advantages_on_gpu has no effect when "
            "data_plane.enabled=true; baseline/std are computed on CPU "
            "because TQ-resident tensors are CPU-side.",
            stacklevel=2,
        )

    # ── Sync rollout actor (rollout 1-hop put) ──────────────────────
    # The actor owns the multi-turn rollout loop AND post-rollout
    # flatten / mask construction / prompt extraction / baseline-std /
    # TQ first-write. Bulk tensors stay actor-side until put_samples;
    # driver receives only KVBatchMeta + small slice via Ray.
    rollout_actor = SyncRolloutActor.options(
        runtime_env=make_actor_runtime_env(
            "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
        ),
    ).remote(
        policy_generation=policy_generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        master_config=master_config,
        dp_cfg=dp_cfg,
    )

    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        memory_tracker.snapshot_start_of_stage("Initial validation", dir())

        if weight_sync is not None and weight_sync.is_stale:
            weight_sync.sync_weights()
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate_sync(
            rollout_actor=rollout_actor,
            policy=policy,
            val_dataloader=val_dataloader,
            val_task_to_env=val_task_to_env,
            step=0,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")

    if master_config.data["use_multiple_dataloader"]:
        warnings.warn(
            "When using multiple dataloaders, MultipleDataloaderWrapper operates as an infinite iterator. "
            "As a result, grpo.max_num_epochs will be ignored, and only grpo.max_num_steps will be used."
        )

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        memory_tracker.snapshot_start_of_stage("Preparing batch", dir())
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        # 1-hop cross-iteration cache for dynamic_sampling: across
        # multiple inner iterations we accumulate non-zero-std prompts
        # until we have enough for a full training batch. The TQ
        # payload of pending uids remains alive until either consumed
        # by training (clear_samples at step end) or evicted on overflow.
        # ``pending_unfiltered_rewards`` is logging-only — preserves
        # legacy ``metrics["reward"]`` semantics (cumulative unfiltered
        # total_reward across all contributing iterations).
        pending_meta = None
        pending_carry: Optional[BatchedDataDict] = None
        pending_unfiltered_rewards: list[torch.Tensor] = []
        dynamic_sampling_num_gen_batches = 0

        for batch in wrapped_dataloader:
            metrics_logging_data: dict = {}
            metrics: dict = {}

            if master_config.data["use_multiple_dataloader"]:
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
                            master_config.grpo["num_generations_per_prompt"]
                        )
                    )

                memory_tracker.snapshot_start_of_stage("Generation", dir())
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    if weight_sync is not None and weight_sync.is_stale:
                        if sync_kv_scales and kv_scales_cache is None:
                            # KV-scale calibration uses message_log of the
                            # current step's PROMPTS (pre-generation), which
                            # is small and lives on the driver naturally.
                            # Unrelated to the rollout 1-hop put.
                            print("▶ Computing KV cache scales...", flush=True)
                            policy.prepare_for_lp_inference()
                            calib_flat, calib_input_lengths = (
                                batched_message_log_to_flat_message(
                                    repeated_batch["message_log"],
                                    pad_value_dict={
                                        "token_ids": tokenizer.pad_token_id
                                    },
                                    make_sequence_length_divisible_by=master_config.policy[
                                        "make_sequence_length_divisible_by"
                                    ],
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

                        weight_sync.sync_weights(
                            timer=timer,
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()

                # ── Per-step TQ partition register ─────────────────────
                # Done before the rollout actor's put_samples so the
                # partition exists with the expected schema.
                policy.prepare_step(
                    num_samples=int(repeated_batch.size),
                    group_size=master_config.grpo["num_generations_per_prompt"],
                )

                # ── Rollout 1-hop put: actor runs rollout + flatten +
                # mask construction + prompt extraction + baseline/std,
                # writes bulk to TQ in one flat put_samples, returns
                # only meta + small slice. Bulk never visits the driver.
                dynamic_sampling_num_gen_batches += 1
                with timer.time("generation"):
                    # Single Ray RPC: rollout + flatten + mask + prompt
                    # extraction + baseline/std + put_samples + finish
                    # generation + logger metrics — all bundled into one
                    # round-trip.
                    # ``first_iter`` is the actor's signal to call
                    # ``policy_generation.snapshot_step_metrics()``.
                    # ``dynamic_sampling_num_gen_batches`` is incremented
                    # to 1 just above before this branch — keep these in
                    # sync if either is renamed.
                    (
                        meta,
                        driver_carry,
                        rollout_metrics,
                        generation_logger_metrics,
                    ) = ray.get(
                        rollout_actor.rollout_to_tq.remote(
                            repeated_batch,
                            partition_id=policy.tq_partition_id,
                            group_size=master_config.grpo["num_generations_per_prompt"],
                            first_iter=(dynamic_sampling_num_gen_batches == 1),
                        )
                    )

                    if not _should_log_nemo_gym_responses(master_config):
                        for key in list(rollout_metrics):
                            if "full_result" in key:
                                rollout_metrics.pop(key)

                    metrics_logging_data["mean_gen_tokens_per_sample"] = (
                        rollout_metrics["mean_gen_tokens_per_sample"]
                    )
                    logger.log_metrics(rollout_metrics, total_steps + 1, prefix="train")

                # ── Per-sample driver compute on slice ────────────────
                # scale_rewards / apply_reward_shaping / overlong filter
                # / baseline-std all operate on small per-sample
                # tensors. Mirrors grpo_sync.py legacy layout — they
                # used to be on the driver, were briefly on the actor,
                # now back on the driver where they belong (no bulk
                # touched by any of these ops).
                with timer.time("reward_calculation"):
                    driver_carry = scale_rewards(
                        driver_carry,
                        master_config.grpo["reward_scaling"],
                    )
                    if master_config.grpo["reward_shaping"]["enabled"]:
                        driver_carry = apply_reward_shaping(
                            driver_carry,
                            master_config.grpo["reward_shaping"],
                        )
                    driver_carry["baseline"], driver_carry["std"] = (
                        calculate_baseline_and_std_per_prompt(
                            driver_carry["prompt_ids_for_adv"],
                            driver_carry["total_reward"],
                            torch.ones_like(driver_carry["total_reward"]),
                            leave_one_out_baseline=master_config.grpo[
                                "use_leave_one_out_baseline"
                            ],
                        )
                    )
                    # Mirror std onto meta so dynamic_sampling can filter
                    # without fetching tensor data.
                    meta.stamp_tags(
                        {
                            "std": driver_carry["std"].tolist(),
                            "baseline": driver_carry["baseline"].tolist(),
                        }
                    )

                # ── Dynamic sampling (DAPO non-zero-std filter) ────────
                # Slice-only; bulk in TQ untouched except for clear_samples
                # of dropped / overflow-discarded uids.
                ds_metrics: dict = {}
                unfiltered_rewards_for_logging: Optional[torch.Tensor] = None
                if master_config.grpo["use_dynamic_sampling"]:
                    with timer.time("dynamic_sampling"):
                        train_prompts_size = (
                            master_config.grpo["num_prompts_per_step"]
                            * master_config.grpo["num_generations_per_prompt"]
                        )
                        (
                            pending_meta,
                            pending_carry,
                            pending_unfiltered_rewards,
                            is_complete,
                            ds_metrics,
                            unfiltered_rewards_for_logging,
                        ) = _apply_dynamic_sampling(
                            meta=meta,
                            driver_carry=driver_carry,
                            pending_meta=pending_meta,
                            pending_carry=pending_carry,
                            pending_unfiltered_rewards=pending_unfiltered_rewards,
                            train_prompts_size=train_prompts_size,
                            num_gen_batches=dynamic_sampling_num_gen_batches,
                            max_gen_batches=master_config.grpo[
                                "dynamic_sampling_max_gen_batches"
                            ],
                            policy=policy,
                        )
                        if not is_complete:
                            current_size = (
                                len(pending_meta.sample_ids)
                                if pending_meta is not None
                                else 0
                            )
                            print(
                                f"Dynamic sampling: {current_size}/{train_prompts_size} "
                                f"non-zero-std prompts after batch "
                                f"{dynamic_sampling_num_gen_batches}; sampling more.",
                                flush=True,
                            )
                            continue

                        # Adopt the now-complete cache as this step's batch.
                        meta = pending_meta
                        driver_carry = pending_carry
                        pending_meta = None
                        pending_carry = None

                # Mirrors legacy ``grpo.py:1707-1716`` — applied on the
                # post-DS survivors so dropped rows don't affect this set.
                if master_config.grpo["overlong_filtering"]:
                    lm = driver_carry["loss_multiplier"].clone()
                    lm[driver_carry["truncated"]] = 0
                    driver_carry["loss_multiplier"] = lm

                # ── Unpack slice (small per-sample tensors) ────────────
                rewards = (
                    driver_carry["filtered_reward"]
                    if master_config.grpo["use_dynamic_sampling"]
                    else driver_carry["total_reward"]
                )
                baseline = driver_carry["baseline"]
                std = driver_carry["std"]
                input_lengths = driver_carry["input_lengths"]
                prompt_ids_for_adv = driver_carry["prompt_ids_for_adv"]
                loss_multiplier = driver_carry["loss_multiplier"]
                truncated = driver_carry["truncated"]
                length = driver_carry["length"]

                gen_step_metrics = {}
                if hasattr(policy_generation, "get_step_metrics"):
                    gen_step_metrics = policy_generation.get_step_metrics()
                baseline_for_log = baseline.clone()

                memory_tracker.snapshot_start_of_stage("Computing logprobs", dir())
                print("▶ Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    # Meta-driven worker dispatch. Workers fetch their
                    # slice from TQ and write ``prev_logprobs`` /
                    # ``reference_policy_logprobs`` columns back to TQ
                    # under ``meta.sample_ids``. The Ray return is
                    # discarded — driver reads from TQ below in one
                    # batched fetch to avoid double-shipping the per-token
                    # tensor through Ray's plasma store on top of the TQ
                    # writeback.
                    policy.get_logprobs_from_meta(meta, timer=timer)
                    compute_ref = not master_config.grpo.get(
                        "skip_reference_policy_logprobs_calculation"
                    )
                    if compute_ref:
                        policy.get_reference_policy_logprobs_from_meta(
                            meta,
                            timer=timer,
                        )

                    # Driver pulls only the per-token columns it needs
                    # for masking / advantage. Bulk (input_ids, multimodal,
                    # output_ids, attention_mask, position_ids) stays in
                    # TQ — workers will fetch it via ``train_presharded``.
                    extras_bdd = policy.read_from_dataplane(
                        meta,
                        select_fields=[
                            "prev_logprobs",
                            "generation_logprobs",
                            "token_mask",
                            *(["reference_policy_logprobs"] if compute_ref else []),
                        ],
                        pad_value_dict=_pad_dict,
                    )
                    prev_logprobs = extras_bdd["prev_logprobs"]
                    generation_logprobs = extras_bdd["generation_logprobs"]
                    token_mask = extras_bdd["token_mask"]
                    reference_policy_logprobs = (
                        extras_bdd["reference_policy_logprobs"] if compute_ref else None
                    )

                    sample_mask, seq_logprob_error_metrics = (
                        _compute_seq_logprob_error_metrics(
                            token_mask=token_mask,
                            sample_mask=loss_multiplier,
                            prev_logprobs=prev_logprobs,
                            generation_logprobs=generation_logprobs,
                            rewards=rewards,
                            seq_logprob_error_threshold=master_config.grpo[
                                "seq_logprob_error_threshold"
                            ],
                        )
                    )

                with timer.time("advantage_calculation"):
                    print("▶ Computing advantages...", flush=True)
                    mask = token_mask * sample_mask.unsqueeze(-1)

                    # GRPO / Reinforce++ ignore ``repeated_batch`` (it's
                    # swallowed via ``**kwargs``); GDPO reads the
                    # per-component reward keys returned by
                    # ``get_gdpo_reward_component_keys``. The actor stashes
                    # those keys into ``driver_carry`` — same payload as
                    # legacy passing the full repeated_batch.
                    adv_inputs = BatchedDataDict(
                        {
                            "total_reward": rewards,
                            "baseline": baseline,
                            "std": std,
                        }
                    )
                    for k in get_gdpo_reward_component_keys(driver_carry):
                        adv_inputs[k] = driver_carry[k]
                    advantages = adv_estimator.compute_advantage(
                        prompt_ids=prompt_ids_for_adv,
                        rewards=rewards,
                        mask=mask,
                        repeated_batch=adv_inputs,
                        logprobs_policy=prev_logprobs,
                        logprobs_reference=reference_policy_logprobs,
                    )
                    del prompt_ids_for_adv

                    _log_mixed_rewards_and_advantages_information(
                        logger=logger,
                        total_steps=total_steps,
                        metrics=metrics,
                        baseline=baseline_for_log,
                        advantages=advantages,
                    )
                    del baseline_for_log

                # ── Driver delta-write: advantages + (post-masking)
                # sample_mask under the same meta.sample_ids so workers fetch
                # the union via train_presharded.
                advantages = _clip_grpo_advantages(advantages, master_config.grpo)
                policy.write_to_dataplane(
                    meta,
                    fields={
                        "advantages": advantages,
                        "sample_mask": sample_mask,
                    },
                )

                memory_tracker.snapshot_start_of_stage("Policy train", dir())
                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    if weight_sync is not None:
                        weight_sync.mark_stale()

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    # Meta-driven train: workers fetch the union of
                    # rollout + driver-written + worker-written columns
                    # from TQ, train, return aggregated metrics via Ray.
                    train_results = policy.train_from_meta(
                        meta,
                        loss_fn=loss_fn,
                        timer=timer,
                    )

                if sync_kv_scales:
                    with timer.time("recompute_kv_scales"):
                        print(
                            "▶ Recomputing KV cache scales after policy update...",
                            flush=True,
                        )
                        # Positive include-list — calibration only consumes
                        # seq-dim tensor inputs. Train-side deltas
                        # (logprobs/advantages/masks) and wire-only message
                        # log bulk fields are skipped by virtue of not being
                        # in DP_CALIB_INPUT_FIELDS.
                        _calib_fields = [
                            f for f in (meta.fields or []) if f in DP_CALIB_INPUT_FIELDS
                        ]
                        calibration_data = policy.read_from_dataplane(
                            meta,
                            select_fields=_calib_fields,
                            pad_value_dict=_pad_dict,
                        )
                        kv_scales_cache = policy.calibrate_qkv_fp8_scales(
                            calibration_data,
                            include_q=True,
                        )["layers"]
                        if weight_sync is not None:
                            weight_sync.mark_stale()

                # Stash input_ids and content before clear_samples so the
                # late log_data jsonl block can use them. The clear below
                # removes meta.sample_ids from TQ, so any post-clear
                # read_columns on this meta would fail. ``content`` is a
                # decoded object array (list[str]); read_columns decodes
                # the NonTensorStack wire field via materialize.
                _log_input_ids: Optional[torch.Tensor] = None
                _log_content: Optional[np.ndarray] = None
                if not _should_log_nemo_gym_responses(master_config):
                    _log_select = ["input_ids"]
                    if "content" in (meta.fields or []):
                        _log_select.append("content")
                    _log_extras = policy.read_from_dataplane(
                        meta,
                        select_fields=_log_select,
                        pad_value_dict=_pad_dict,
                    )
                    _log_input_ids = _log_extras["input_ids"]
                    _log_content = _log_extras.get("content")

                # ── Step-end TQ cleanup ────────────────────────────────
                policy.finish_step(meta)

                is_last_step = total_steps + 1 >= max_num_steps
                if not master_config.data["use_multiple_dataloader"]:
                    is_last_step = is_last_step or (
                        (current_epoch + 1 == max_num_epochs)
                        and (current_step + 1 == len(wrapped_dataloader))
                    )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    memory_tracker.snapshot_start_of_stage("Validation", dir())
                    if weight_sync is not None and weight_sync.is_stale:
                        weight_sync.sync_weights(
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate_sync(
                        rollout_actor=rollout_actor,
                        policy=policy,
                        val_dataloader=val_dataloader,
                        val_task_to_env=val_task_to_env,
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

                # advantages and token_mask are in scope from the
                # advantage / masking blocks above. No need to re-fetch.
                response_advantages = torch.masked_select(advantages, token_mask.bool())

                memory_tracker.snapshot_start_of_stage("Metrics", dir())
                metrics = {
                    **metrics,
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "reward": rewards.numpy(),
                    "mean_prompt_length": length.numpy(),
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
                # Cumulative unfiltered total_reward across all DS iterations
                # (sliced to train_prompts_size). Falls back to filtered
                # rewards if apply_dynamic_sampling didn't provide it
                # (mid-step path). Hoisted once for reuse in metrics, jsonl,
                # and the per-step print below.
                unfiltered_rewards = (
                    unfiltered_rewards_for_logging
                    if unfiltered_rewards_for_logging is not None
                    else rewards
                )
                if master_config.grpo["use_dynamic_sampling"]:
                    metrics["filtered_reward"] = rewards.numpy()
                    metrics["reward"] = unfiltered_rewards.numpy()

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

                metrics.update(seq_logprob_error_metrics)

                consumed_samples += master_config.grpo["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                memory_tracker.snapshot_start_of_stage("Checkpointing", dir())
                if master_config.checkpointing["enabled"] and (
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

                    full_metric_name = master_config.checkpointing["metric_name"]
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
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        if master_config.data["use_multiple_dataloader"]:
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
            # Per-step log_data jsonl. The 1-hop driver holds per-token
            # slices it computed against (advantages, sample_mask,
            # prev_logprobs, generation_logprobs, token_mask). For
            # ``token_ids`` we fetch the small ``input_ids`` column from
            # TQ at log time — same data-driven slice pattern as masking
            # / KV calibration.
            if not _should_log_nemo_gym_responses(master_config):
                log_data: dict = {}
                if "agent_ref" in repeated_batch:
                    log_data["agent_ref"] = repeated_batch["agent_ref"]
                if master_config.grpo["use_dynamic_sampling"]:
                    # Legacy semantics: ``rewards`` is unfiltered total_reward,
                    # ``filtered_rewards`` is the kept slice that's trained on.
                    log_data["rewards"] = unfiltered_rewards.tolist()
                    log_data["filtered_rewards"] = rewards.tolist()
                else:
                    log_data["rewards"] = rewards.tolist()
                log_data["input_lengths"] = input_lengths.tolist()
                log_data["token_loss_mask"] = token_mask.tolist()
                log_data["sample_loss_mask"] = sample_mask.tolist()
                log_data["advantages"] = advantages.tolist()
                log_data["generation_logprobs"] = generation_logprobs.tolist()
                log_data["prev_logprobs"] = prev_logprobs.tolist()
                # input_ids was stashed before the step-end clear_samples (the
                # keys are no longer in TQ at this point); ``_log_input_ids``
                # is None when nemo_gym-responses logging path skipped the
                # outer ``if not _should_log_nemo_gym_responses`` branch.
                if _log_input_ids is not None:
                    log_data["token_ids"] = _log_input_ids.tolist()
                # ``content`` (raw assistant text) is fetched from TQ as
                # an object-array column above (stashed before clear_samples).
                if _log_content is not None:
                    log_data["content"] = _log_content.tolist()
                logger.log_batched_dict_as_jsonl(
                    log_data, f"train_data_step{total_steps + 1}.jsonl"
                )
                del log_data

            timing_metrics: dict = timer.get_timing_metrics(reduction_op="sum")  # type: ignore
            if metrics["token_mult_prob_error"] > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths": length,
                        "full_lengths": input_lengths,
                        "generation_logprobs": generation_logprobs,
                        "prev_logprobs": prev_logprobs,
                        "token_mask": token_mask,
                        "sample_mask": sample_mask,
                    },
                    total_steps + 1,
                    name="train/token_mult_prob_error_plot_sample",
                )
            if master_config.policy["generation"].get("vllm_cfg", {}).get(
                "enable_vllm_metrics_logger", False
            ) and master_config.logger.get("wandb_enabled", False):
                log_generation_metrics_to_wandb(
                    generation_logger_metrics,
                    total_steps + 1,
                    master_config.policy["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            if (
                master_config.policy["generation"]
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
            if master_config.grpo["use_dynamic_sampling"]:
                print(f"  • Avg Filtered Reward: {np.mean(rewards.numpy()):.4f}")
                print(
                    f"  • Avg Total Reward: {np.mean(unfiltered_rewards.numpy()):.4f}"
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
                master_config.grpo["num_prompts_per_step"]
                * master_config.grpo["num_generations_per_prompt"]
            )
            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
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

            dynamic_sampling_num_gen_batches = 0

            memory_tracker.snapshot_start_of_stage("After CPU memory clear", dir())

            del repeated_batch
            del rewards
            del metrics
            if "val_metrics" in dir():
                del val_metrics

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                memory_tracker.snapshot_start_of_stage("", dir())
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_num_steps:
                memory_tracker.snapshot_start_of_stage("", dir())
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0
