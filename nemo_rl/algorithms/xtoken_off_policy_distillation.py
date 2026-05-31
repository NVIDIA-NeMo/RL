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
"""Single-teacher cross-tokenizer off-policy distillation.

Training-loop layout mirrors ``run_distillation.py`` /
``nemo_rl/algorithms/distillation.py`` minus the on-policy bits (no env, no
rollout, no generation). Per step:

    1. Pull a collated batch (student & teacher token ids + alignment).
    2. Run teacher forward via ``Policy.get_topk_logits`` on TEACHER token
       ids — gives top-k teacher logits at teacher positions.
    3. Pack alignment payload + teacher topk into a student-side
       ``train_data`` dict.
    4. ``student_policy.train(train_data, loss_fn)`` — student forward +
       loss + backward + optimizer step happens inside the dtensor v2
       worker.

The collator and aligner do all the CPU-side cross-tokenizer work; the
loss function does only loss math; this module is just plumbing.
"""

from __future__ import annotations

import os
from typing import Any, NotRequired, Optional, TypedDict, cast

from pydantic import BaseModel

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.loss_functions import (
    CrossTokenizerDistillationLossConfig,
    CrossTokenizerDistillationLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.algorithms.x_token import TokenAligner
from nemo_rl.data import DataConfig
from nemo_rl.data.cross_tokenizer_collate import CrossTokenizerCollator
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# Keys packed into the student-side `train_data` BatchedDataDict whose dim 1
# is NOT the student sequence axis. They ride along on the dict so the loss
# fn can index them per-microbatch, but the worker's `check_sequence_dim`
# pre-flight (which assumes [B, student_seq, ...] for every 2+D tensor) must
# skip them. Sources:
#   - teacher_full_logits_ipc: list[B] of CUDA IPC handle dicts produced by
#     FullLogitsPostProcessor in dtensor_policy_worker_v2.get_full_logits_ipc.
#     Not a tensor at all — list of dicts — but listed here so the worker's
#     dict-level dim check skips it.
#   - teacher_input_ids/teacher_token_mask + alignment_*: produced by
#     CrossTokenizerCollator (in DataLoader workers).
# alignment_student_chunk_id and alignment_student_exact_partition_mask are
# [B, T_s] and DO follow the student-seq invariant, so they are NOT listed.
XTOKEN_NON_STUDENT_SEQ_KEYS: frozenset[str] = frozenset(
    {
        "teacher_full_logits_ipc",
        "teacher_input_ids",
        "teacher_token_mask",
        "alignment_pair_valid",
        "alignment_pair_is_correct",
        "alignment_teacher_exact_partition_mask",
        "alignment_teacher_chunk_id",
    }
)

# ===============================================================================
# Configuration
# ===============================================================================


class OffPolicyDistillationConfig(TypedDict):
    """Top-level distillation algo config.

    Attributes:
        num_prompts_per_step: Global batch size at the dataloader level.
        max_num_steps: Max training steps before early stop.
        max_num_epochs: Max passes over the training dataset.
        seed: RNG seed.
        val_period: Validation cadence in steps. ``0`` disables validation.
        val_at_start: Run validation before training begins.
        val_at_end: Run validation on the final step.
    """

    num_prompts_per_step: int
    max_num_steps: int
    max_num_epochs: int
    seed: int
    val_period: int
    val_at_start: bool
    val_at_end: bool


class OffPolicyDistillationSaveState(TypedDict):
    current_epoch: int
    current_step: int
    total_steps: int
    consumed_samples: int
    total_valid_tokens: int
    val_loss: NotRequired[float]


def _default_off_policy_distillation_save_state() -> OffPolicyDistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig    # student
    teacher: PolicyConfig
    loss_fn: CrossTokenizerDistillationLossConfig
    data: DataConfig
    distillation: OffPolicyDistillationConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup
# ===============================================================================


def setup(
    master_config: MasterConfig,
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    Policy,                         # student
    Policy,                         # teacher
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    CrossTokenizerDistillationLossFn,
    Logger,
    CheckpointManager,
    OffPolicyDistillationSaveState,
    MasterConfig,
]:
    """Construct cluster, dataloaders, policies, and loss fn for the run."""
    policy_config = master_config.policy
    teacher_config = master_config.teacher
    loss_config = master_config.loss_fn
    distillation_config = master_config.distillation
    data_config = master_config.data
    logger_config = master_config.logger
    cluster_config = master_config.cluster

    # Backend gate: this code path is DTensor V2 only by design.
    assert policy_config["dtensor_cfg"]["enabled"] and policy_config["dtensor_cfg"].get(
        "_v2", False
    ), "xtoken distillation requires policy.dtensor_cfg.enabled=true and _v2=true."
    assert teacher_config["dtensor_cfg"]["enabled"] and teacher_config["dtensor_cfg"].get(
        "_v2", False
    ), "xtoken distillation requires teacher.dtensor_cfg.enabled=true and _v2=true."

    set_seed(distillation_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config.model_dump())

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config.checkpointing)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    off_policy_distillation_state: Optional[OffPolicyDistillationSaveState] = cast(
        Optional[OffPolicyDistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if off_policy_distillation_state is None:
        off_policy_distillation_state = _default_off_policy_distillation_save_state()

    # ==========================
    #     Aligner + Collator
    # ==========================
    print("\n▶ Building token aligner and cross-tokenizer collator...", flush=True)
    aligner = TokenAligner(
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        projection_matrix_path=loss_config["projection_matrix_path"],
    )

    collator = CrossTokenizerCollator(
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        aligner=aligner,
        ctx_length_student=policy_config["max_total_sequence_length"],
        ctx_length_teacher=teacher_config["max_total_sequence_length"],
        make_seq_div_by_student=policy_config["make_sequence_length_divisible_by"],
        make_seq_div_by_teacher=teacher_config["make_sequence_length_divisible_by"],
    )

    # ==========================
    #           Data
    # ==========================
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=collator,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    if last_checkpoint_path is not None:
        dataloader_state = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state)
    print(
        f"  ✓ Training dataloader loaded with {len(train_dataset)} samples",
        flush=True,
    )

    val_dataloader: Optional[StatefulDataLoader] = None
    if val_dataset is not None and (
        distillation_config["val_period"] > 0
        or distillation_config["val_at_start"]
        or distillation_config["val_at_end"]
    ):
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["num_prompts_per_step"],
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
            num_workers=data_config["num_workers"],
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    cluster = RayVirtualCluster(
        name="xtoken_off_policy_distillation_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=2,
    )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up teacher policy...", flush=True)
    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=cluster,
        config=teacher_config,
        tokenizer=teacher_tokenizer,
        weights_path=None,
        optimizer_path=None,
        init_optimizer=False,
        init_reference_model=False,
    )
    teacher_policy.offload_after_refit()

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up student policy...", flush=True)
    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)
    student_policy = Policy(
        name_prefix="student",
        cluster=cluster,
        config=policy_config,
        tokenizer=student_tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )

    # ==========================
    #         Loss
    # ==========================
    # Inject both tokenizer vocab sizes so the projection matrix's V_s
    # and V_t axes match `logits.shape[-1]` exactly, instead of being
    # recovered from the highest ids that happen to appear in the sparse
    # projection file. `len(tokenizer)` is what HF treats as the
    # embedding / lm_head dim.
    loss_config = {
        **loss_config,
        "student_vocab_size": len(student_tokenizer),
        "teacher_vocab_size": len(teacher_tokenizer),
    }
    loss_fn = CrossTokenizerDistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policy,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        off_policy_distillation_state,
        master_config,
    )


# ===============================================================================
# Train loop
# ===============================================================================


def xtoken_off_policy_distillation_train(
    student_policy: Policy,
    teacher_policy: Policy,
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    loss_fn: CrossTokenizerDistillationLossFn,
    logger: Logger,
    checkpointer: CheckpointManager,
    off_policy_distillation_state: OffPolicyDistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Off-policy CT distillation training loop."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    distill_cfg = master_config.distillation
    current_epoch = off_policy_distillation_state["current_epoch"]
    current_step = off_policy_distillation_state["current_step"]
    total_steps = off_policy_distillation_state["total_steps"]
    consumed_samples = off_policy_distillation_state["consumed_samples"]
    total_valid_tokens = off_policy_distillation_state["total_valid_tokens"]
    val_period = distill_cfg["val_period"]
    val_at_start = distill_cfg["val_at_start"]
    val_at_end = distill_cfg["val_at_end"]
    max_epochs = distill_cfg["max_num_epochs"]
    max_steps = distill_cfg["max_num_steps"]

    if val_at_start and total_steps == 0 and val_dataloader is not None:
        val_metrics, val_timings = validate(
            student_policy,
            teacher_policy,
            val_dataloader,
            loss_fn,
            master_config,
            timer=timer,
        )
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(val_timings, total_steps, prefix="timing/validation")

    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )
        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/"
                f"{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)

            with timer.time("total_step_time"):
                with timer.time("teacher_forward_prep"):
                    teacher_policy.prepare_for_lp_inference()

                with timer.time("teacher_forward"):
                    teacher_data = BatchedDataDict(
                        input_ids=batch["teacher_input_ids"],
                        input_lengths=batch["teacher_input_lengths"],
                        token_mask=batch["teacher_token_mask"],
                        sample_mask=batch["sample_mask"],
                    )
                    # IPC transport: per-sample [T_t, V_t] full-vocab logit
                    # views are exported as CUDA IPC handles and consumed
                    # by the student in-process. The loss fn either uses
                    # full vocab (gold path) or derives a microbatch-global
                    # top-k from this inline (P-KL path).
                    teacher_handles = teacher_policy.get_full_logits_ipc(
                        teacher_data, timer=timer
                    )
                    # Model offload frees the teacher's PARAMS to CPU; the
                    # IPC-stashed logit tensors live in worker Python state
                    # and survive this call.
                    teacher_policy.offload_after_refit()

                # Pack student-side training data with teacher logits and
                # the alignment payload the loss fn will index into.
                train_data: BatchedDataDict[Any] = BatchedDataDict(
                    input_ids=batch["input_ids"],
                    input_lengths=batch["input_lengths"],
                    token_mask=batch["token_mask"],
                    sample_mask=batch["sample_mask"],
                    teacher_full_logits_ipc=teacher_handles,
                    alignment_pair_valid=batch["alignment_pair_valid"],
                    alignment_pair_is_correct=batch["alignment_pair_is_correct"],
                    alignment_student_exact_partition_mask=(
                        batch["alignment_student_exact_partition_mask"]
                    ),
                    alignment_teacher_exact_partition_mask=(
                        batch["alignment_teacher_exact_partition_mask"]
                    ),
                    alignment_student_chunk_id=batch["alignment_student_chunk_id"],
                    alignment_teacher_chunk_id=batch["alignment_teacher_chunk_id"],
                    alignment_num_chunks=batch["alignment_num_chunks"],
                )
                # `.to("cpu")` is a no-op on the IPC handle list (lists are
                # not tensors).
                train_data.to("cpu")

                with timer.time("training_prep"):
                    student_policy.prepare_for_training()

                with timer.time("policy_training"):
                    try:
                        train_results = student_policy.train(
                            train_data,
                            loss_fn,
                            timer=timer,
                            skip_keys=XTOKEN_NON_STUDENT_SEQ_KEYS,
                        )
                    finally:
                        # Producer-side CUDA tensors must be freed before
                        # the next teacher forward — otherwise memory grows
                        # unboundedly. Always release, even on student
                        # failure.
                        teacher_policy.release_ipc_buffer()

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                val_metrics: dict[str, Any] | None = None
                if val_dataloader is not None and (
                    (val_period > 0 and (total_steps + 1) % val_period == 0)
                    or (val_at_end and is_last_step)
                ):
                    val_metrics, val_timings = validate(
                        student_policy,
                        teacher_policy,
                        val_dataloader,
                        loss_fn,
                        master_config,
                        timer=timer,
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )
                    logger.log_metrics(
                        val_timings, total_steps + 1, prefix="timing/validation"
                    )

                metrics: dict[str, Any] = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                }
                metrics.update(train_results["all_mb_metrics"])
                # Reduce per-microbatch metrics to per-step scalars. The
                # P-KL path emits kl_loss/ce_loss/kl_loss_scale/proj_accuracy;
                # the gold-loss path emits kl_common/l1_uncommon. Either set
                # may be present — reduce both via the same rules.
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "accuracy",
                        "proj_accuracy",
                        "kl_loss_scale",
                        "kl_common",
                        "l1_uncommon",
                    }:
                        metrics[k] = float(np.mean(v))
                    else:
                        metrics[k] = float(np.sum(v))
                if "global_valid_toks" in metrics:
                    total_valid_tokens += int(metrics["global_valid_toks"])

                consumed_samples += distill_cfg["num_prompts_per_step"]
                timeout.mark_iteration()

                # ===== Checkpointing =====
                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1)
                    % master_config.checkpointing["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()
                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()
                    off_policy_distillation_state["current_epoch"] = current_epoch
                    off_policy_distillation_state["current_step"] = current_step + 1
                    off_policy_distillation_state["total_steps"] = total_steps + 1
                    off_policy_distillation_state["total_valid_tokens"] = total_valid_tokens
                    off_policy_distillation_state["consumed_samples"] = consumed_samples
                    if val_metrics is not None and "loss" in val_metrics:
                        off_policy_distillation_state["val_loss"] = float(val_metrics["loss"])
                    elif "val_loss" in off_policy_distillation_state:
                        del off_policy_distillation_state["val_loss"]

                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        prefix, metric_name = full_metric_name.split(":", 1)
                        source = metrics if prefix == "train" else (val_metrics or {})
                        if metric_name in source:
                            off_policy_distillation_state[full_metric_name] = float(source[metric_name])

                    with timer.time("checkpointing"):
                        ckpt_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, off_policy_distillation_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                ckpt_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                ckpt_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                ckpt_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(ckpt_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(ckpt_path)

            # ===== Logging =====
            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore
            # `metrics["loss"]` and the SUM-reduced terms (kl_loss, ce_loss
            # for the P-KL path) are SUM across all DP ranks AND microbatches
            # (= dp_size * local_mbs values summed). PT logs rank-0
            # per-microbatch raw, so for apples-to-apples we also print
            # per-MB-mean. n_mb = len of the flat list of per-MB metrics.
            n_mb = max(len(train_results["all_mb_metrics"].get("loss", [])), 1)
            print(
                f"  • Loss: {metrics['loss']:.4f} "
                f"(per-MB-mean: {metrics['loss'] / n_mb:.4f})",
                flush=True,
            )
            # P-KL path metrics — only printed when they're present.
            if "kl_loss" in metrics:
                kl_sum = float(metrics["kl_loss"])
                print(
                    f"  • KL:   {kl_sum:.4f} "
                    f"(per-MB-mean: {kl_sum / n_mb:.4f})",
                    flush=True,
                )
            if "ce_loss" in metrics:
                ce_sum = float(metrics["ce_loss"])
                print(
                    f"  • CE:   {ce_sum:.4f} "
                    f"(per-MB-mean: {ce_sum / n_mb:.4f})",
                    flush=True,
                )
            # Gold-loss path metrics — kl_common/l1_uncommon are already
            # per-MB means (np.mean branch above), so no /n_mb division.
            if "kl_common" in metrics:
                print(
                    f"  • KL(common):  {metrics['kl_common']:.4f}",
                    flush=True,
                )
            if "l1_uncommon" in metrics:
                print(
                    f"  • L1(uncommon): {metrics['l1_uncommon']:.4f}",
                    flush=True,
                )
            # Accuracy: P-KL emits next-token student accuracy + projection
            # top-1; gold emits top-1 common-vocab accuracy. Both arrive
            # under "accuracy" so the same line works.
            if "accuracy" in metrics:
                print(
                    f"  • Acc:  {metrics['accuracy'] * 100:.2f}%",
                    flush=True,
                )
            if "proj_accuracy" in metrics:
                print(
                    f"  • ProjAcc: {metrics['proj_accuracy'] * 100:.2f}%",
                    flush=True,
                )
            print(f"  • Total step time: {timing_metrics.get('total_step_time', 0):.2f}s", flush=True)
            for k, v in sorted(
                timing_metrics.items(), key=lambda kv: kv[1], reverse=True
            ):
                if k != "total_step_time":
                    print(f"  • {k}: {v:.2f}s", flush=True)

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                print("Timeout reached, stopping training early.", flush=True)
                return
            if total_steps >= max_steps:
                print("Max steps reached, stopping training.", flush=True)
                return

        current_epoch += 1
        current_step = 0


# ===============================================================================
# Validation
# ===============================================================================


def validate(
    student_policy: Policy,
    teacher_policy: Policy,
    val_dataloader: StatefulDataLoader,
    loss_fn: CrossTokenizerDistillationLossFn,
    master_config: MasterConfig,
    timer: Optional[Timer] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Held-out KL/CE on a validation dataloader.

    Reuses the same per-step path as training, but in eval mode so no
    backward / optimizer step runs. Returns mean train-style metrics.
    """
    distill_cfg = master_config.distillation
    timer = timer if timer is not None else Timer()

    losses: list[float] = []
    # The P-KL path emits kl_loss/ce_loss; the gold path emits
    # kl_common/l1_uncommon. Track both, only the ones the active loss
    # populates will end up in the returned metrics.
    kl_losses: list[float] = []
    ce_losses: list[float] = []
    kl_common_losses: list[float] = []
    l1_uncommon_losses: list[float] = []

    with timer.time("validation_total"):
        teacher_policy.prepare_for_lp_inference()
        for batch in val_dataloader:
            teacher_data = BatchedDataDict(
                input_ids=batch["teacher_input_ids"],
                input_lengths=batch["teacher_input_lengths"],
                token_mask=batch["teacher_token_mask"],
                sample_mask=batch["sample_mask"],
            )
            teacher_handles = teacher_policy.get_full_logits_ipc(teacher_data)

            train_data: BatchedDataDict[Any] = BatchedDataDict(
                input_ids=batch["input_ids"],
                input_lengths=batch["input_lengths"],
                token_mask=batch["token_mask"],
                sample_mask=batch["sample_mask"],
                teacher_full_logits_ipc=teacher_handles,
                alignment_pair_valid=batch["alignment_pair_valid"],
                alignment_pair_is_correct=batch["alignment_pair_is_correct"],
                alignment_student_exact_partition_mask=(
                    batch["alignment_student_exact_partition_mask"]
                ),
                alignment_teacher_exact_partition_mask=(
                    batch["alignment_teacher_exact_partition_mask"]
                ),
                alignment_student_chunk_id=batch["alignment_student_chunk_id"],
                alignment_teacher_chunk_id=batch["alignment_teacher_chunk_id"],
                alignment_num_chunks=batch["alignment_num_chunks"],
            )
            train_data.to("cpu")
            student_policy.prepare_for_training()
            try:
                results = student_policy.train(
                    train_data,
                    loss_fn,
                    eval_mode=True,
                    skip_keys=XTOKEN_NON_STUDENT_SEQ_KEYS,
                )
            finally:
                teacher_policy.release_ipc_buffer()
            losses.append(float(results["loss"].numpy()))
            mb_metrics = results.get("all_mb_metrics", {})
            if "kl_loss" in mb_metrics:
                kl_losses.append(float(np.mean(mb_metrics["kl_loss"])))
            if "ce_loss" in mb_metrics:
                ce_losses.append(float(np.mean(mb_metrics["ce_loss"])))
            if "kl_common" in mb_metrics:
                kl_common_losses.append(float(np.mean(mb_metrics["kl_common"])))
            if "l1_uncommon" in mb_metrics:
                l1_uncommon_losses.append(
                    float(np.mean(mb_metrics["l1_uncommon"]))
                )
        teacher_policy.offload_after_refit()

    metrics: dict[str, Any] = {
        "loss": float(np.mean(losses)) if losses else 0.0,
    }
    if kl_losses:
        metrics["kl_loss"] = float(np.mean(kl_losses))
    if ce_losses:
        metrics["ce_loss"] = float(np.mean(ce_losses))
    if kl_common_losses:
        metrics["kl_common"] = float(np.mean(kl_common_losses))
    if l1_uncommon_losses:
        metrics["l1_uncommon"] = float(np.mean(l1_uncommon_losses))

    return metrics, timer.get_timing_metrics(reduction_op="sum")  # type: ignore
