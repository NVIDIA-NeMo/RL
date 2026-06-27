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
"""Multi-teacher cross-tokenizer off-policy distillation.

Training-loop layout mirrors ``run_distillation.py`` /
``nemo_rl/algorithms/distillation.py`` minus the on-policy bits (no env, no
rollout, no generation). Per step:

    1. Pull a collated batch (student & teacher token ids + alignment).
    2. Run each teacher forward via ``Policy.get_full_logits_ipc`` on TEACHER
       token ids — every teacher ships full-vocab logits over CUDA IPC (no
       driver round-trip), reassembled across the teacher's TP/CP shards on
       the student side.
    3. Pack alignment payload + teacher IPC handles into a student-side
       ``train_data`` dict.
    4. ``student_policy.train(train_data, loss_fn)`` — student forward +
       loss + backward + optimizer step happens inside the dtensor v2
       worker.

The collator and aligner do all the CPU-side cross-tokenizer work; the
loss function does only loss math; this module is just plumbing.
"""

from __future__ import annotations

import math
import os
from typing import Any, NotRequired, Optional, TypedDict, cast

import numpy as np
import torch
from pydantic import BaseModel, Field
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.loss_functions import (
    CrossTokenizerDistillationLossConfig,
    CrossTokenizerDistillationLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.algorithms.x_token import TokenAligner
from nemo_rl.algorithms.x_token.utils import (
    assert_teacher_student_batch_grid,
    assert_xtoken_ipc_node_local,
    pad_distillation_val_batch,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.cross_tokenizer_collate import CrossTokenizerCollator
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.utils import load_dataloader_state
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
def xtoken_non_student_seq_keys(
    loss_fn: "CrossTokenizerDistillationLossFn",
) -> frozenset[str]:
    """Build the set of ``train_data`` keys whose dim 1 is NOT the student seq axis.

    These ride on the student-side ``train_data`` so the loss fn can index them
    per microbatch, but the worker's ``check_sequence_dim`` pre-flight (which
    assumes ``[B, student_seq, ...]`` for every 2+D tensor) must skip them. The
    set is teacher-dependent, so it is built from the loss fn's per-teacher
    metadata rather than a static constant. Every teacher ships full logits, so
    each contributes ``teacher_{i}_full_logits_ipc`` (a list of CUDA IPC handle
    dicts, not a tensor); a cross-tokenizer teacher additionally rides its
    teacher-seq tokenization (``teacher_{i}_input_ids`` / ``teacher_{i}_token_mask``,
    ``[B, T_t]``) and its teacher-seq / max_pairs ``alignment_{i}_*`` keys. The
    ``alignment_{i}_student_*`` (``[B, T_s]``) and ``alignment_{i}_num_chunks``
    (``[B]``) keys follow the student-seq invariant and are NOT skipped.
    """
    keys: set[str] = set()
    for i in range(loss_fn.num_teachers):
        keys.add(f"teacher_{i}_full_logits_ipc")
        if loss_fn.projection_matrix_paths[i] is not None:  # cross-tokenizer
            keys.add(f"teacher_{i}_input_ids")
            keys.add(f"teacher_{i}_token_mask")
            keys.add(f"alignment_{i}_pair_valid")
            keys.add(f"alignment_{i}_pair_is_correct")
            keys.add(f"alignment_{i}_teacher_exact_partition_mask")
            keys.add(f"alignment_{i}_teacher_chunk_id")
    return frozenset(keys)

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


class TeacherConfig(BaseModel, extra="allow"):
    """Per-teacher config for multi-teacher cross-tokenizer distillation.

    Carries the full ``PolicyConfig`` content (``model_name``, ``tokenizer``,
    ``dtensor_cfg``, …) as permitted extras, plus the cross-tokenizer knobs
    declared below. Use :meth:`policy_config` to recover the plain
    ``PolicyConfig`` dict for ``Policy`` construction.

    Attributes:
        projection_matrix_path: Path to this teacher's student->teacher
            projection matrix. ``None`` marks a *same-tokenizer* teacher:
            projection and alignment are skipped and the loss uses a direct
            per-position KL on the shared vocab.
        weight: Static loss weight for this teacher when several teachers are
            aggregated (``kd_loss_mode="sum"`` / the convex ``"averaged_logits"``
            mix). Single-teacher runs leave it at ``1.0``.
        gold_loss: Optional per-teacher override of ``loss_fn.gold_loss``.
            ``None`` falls back to the global value. Honored only in
            ``kd_loss_mode="sum"`` (other modes use the global).
        xtoken_loss: Optional per-teacher override of ``loss_fn.xtoken_loss``,
            same semantics as ``gold_loss``.
    """

    projection_matrix_path: Optional[str] = None
    weight: float = 1.0
    gold_loss: Optional[bool] = None
    xtoken_loss: Optional[bool] = None

    def policy_config(self) -> PolicyConfig:
        """Recover the plain ``PolicyConfig`` dict (cross-tokenizer knobs stripped)."""
        return cast(
            PolicyConfig,
            self.model_dump(
                exclude={"projection_matrix_path", "weight", "gold_loss", "xtoken_loss"}
            ),
        )


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig  # student
    teachers: list[TeacherConfig] = Field(min_length=1)
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
    teacher_tokenizers: list[PreTrainedTokenizerBase],
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    Policy,  # student
    list[Policy],  # teachers
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
    teachers = master_config.teachers
    teacher_configs = [t.policy_config() for t in teachers]
    loss_config = master_config.loss_fn
    distillation_config = master_config.distillation
    data_config = master_config.data
    logger_config = master_config.logger
    cluster_config = master_config.cluster

    assert len(teacher_tokenizers) == len(teachers), (
        f"expected one tokenizer per teacher; got {len(teacher_tokenizers)} "
        f"tokenizers for {len(teachers)} teachers."
    )

    # Backend gate: DTensor V2 only, for the student and every teacher. Unlike
    # the TP=CP=1 multi-teacher prototype, this path supports TP/CP/diff-DP
    # sharding (the loss is parallelism-invariant), so there is deliberately NO
    # tensor/context_parallel_size==1 assert.
    assert policy_config["dtensor_cfg"]["enabled"] and policy_config["dtensor_cfg"].get(
        "_v2"
    ), "xtoken distillation requires policy.dtensor_cfg.enabled=true and _v2=true."
    for i, tc in enumerate(teacher_configs):
        assert tc["dtensor_cfg"]["enabled"] and tc["dtensor_cfg"].get("_v2"), (
            f"xtoken distillation requires teachers[{i}].dtensor_cfg.enabled=true "
            "and _v2=true."
        )

    # A null projection path marks a same-vocab teacher (direct KL, no
    # projection/alignment); that only makes sense when it shares the student's
    # vocab, so check vocab SIZE (tokenizer names differ for same-vocab pairs
    # like Llama-3.2-3B vs -1B). Cross-tokenizer teachers (non-null path) are
    # validated against their projection matrix in the loss.
    student_vocab = len(student_tokenizer)
    for i, teacher in enumerate(teachers):
        if teacher.projection_matrix_path is None:
            assert len(teacher_tokenizers[i]) == student_vocab, (
                f"teachers[{i}] has projection_matrix_path=null (same-vocab "
                f"teacher) but its tokenizer vocab ({len(teacher_tokenizers[i])}) "
                f"!= student vocab ({student_vocab}). A same-vocab teacher must "
                "share the student tokenizer; set a projection_matrix_path to "
                "run it as a cross-tokenizer teacher."
            )

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
    print("\n▶ Building token aligners and cross-tokenizer collator...", flush=True)
    # One aligner per teacher; None for same-tokenizer teachers (no projection,
    # no alignment — the loss does a direct per-position KL there).
    aligners: list[Optional[TokenAligner]] = [
        None
        if teacher.projection_matrix_path is None
        else TokenAligner(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizers[i],
            projection_matrix_path=teacher.projection_matrix_path,
        )
        for i, teacher in enumerate(teachers)
    ]

    collator = CrossTokenizerCollator(
        student_tokenizer=student_tokenizer,
        teacher_tokenizers=list(teacher_tokenizers),
        aligners=aligners,
        ctx_length_student=policy_config["max_total_sequence_length"],
        ctx_length_teachers=[tc["max_total_sequence_length"] for tc in teacher_configs],
        make_seq_div_by_student=policy_config["make_sequence_length_divisible_by"],
        make_seq_div_by_teachers=[
            tc["make_sequence_length_divisible_by"] for tc in teacher_configs
        ],
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
        # Keep workers (and their collator: teacher tokenizers + aligner +
        # projection) alive across epochs. Without this, small datasets looped
        # via max_num_epochs respawn+re-init all workers at every epoch
        # boundary, which dominates step time when the epoch is short.
        persistent_workers=data_config["num_workers"] > 0,
    )
    if last_checkpoint_path:
        load_dataloader_state(train_dataloader, last_checkpoint_path, data_config)
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
            persistent_workers=data_config["num_workers"] > 0,
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
        # N teacher worker groups + 1 student, colocated and run serially.
        max_colocated_worker_groups=len(teachers) + 1,
    )

    # ==========================
    #      Teacher Policies
    # ==========================
    print(f"\n▶ Setting up {len(teachers)} teacher policies...", flush=True)
    teacher_policies: list[Policy] = []
    for i, tc in enumerate(teacher_configs):
        teacher_policy = Policy(
            name_prefix=f"teacher_{i}",
            cluster=cluster,
            config=tc,
            tokenizer=teacher_tokenizers[i],
            weights_path=None,
            optimizer_path=None,
            init_optimizer=False,
            init_reference_model=False,
        )
        teacher_policy.offload_after_refit()
        teacher_policies.append(teacher_policy)

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
    #   Teacher/student grid
    # ==========================
    # Teacher and student may differ in DP and MBS, but they must agree on the
    # global batch size (one dataloader batch feeds both, in the same global
    # order) and tile it cleanly into per-DP-rank chunks and whole microbatches.
    # assert_teacher_student_batch_grid checks both (GBS agreement + tiling).
    student_dp = student_policy.data_parallel_size
    student_tp = policy_config["dtensor_cfg"]["tensor_parallel_size"]
    student_cp = policy_config["dtensor_cfg"]["context_parallel_size"]
    # Each teacher may differ from the student (and from each other) in
    # DP/MBS/TP/CP, so check the batch grid and node-local IPC layout per
    # teacher. Train and validation share the grid (the student reuses its train
    # MBS in eval mode and each teacher's val export reuses its own train MBS),
    # so one check per teacher covers both.
    for i, (teacher_policy, tc) in enumerate(zip(teacher_policies, teacher_configs)):
        teacher_dp = teacher_policy.data_parallel_size
        assert_teacher_student_batch_grid(
            global_batch_size=distillation_config["num_prompts_per_step"],
            student_gbs=policy_config["train_global_batch_size"],
            teacher_gbs=tc["train_global_batch_size"],
            student_dp=student_dp,
            teacher_dp=teacher_dp,
            student_mbs=policy_config["train_micro_batch_size"],
            teacher_mbs=tc["train_micro_batch_size"],
        )
        # Node-local CUDA IPC: on >1 node it only works when teacher/student
        # share DP and a node-aligned model-parallel group, else a student rank
        # would read teacher shards from another node.
        assert_xtoken_ipc_node_local(
            num_nodes=cluster_config["num_nodes"],
            gpus_per_node=cluster_config["gpus_per_node"],
            student_tp=student_tp,
            student_cp=student_cp,
            teacher_tp=tc["dtensor_cfg"]["tensor_parallel_size"],
            teacher_cp=tc["dtensor_cfg"]["context_parallel_size"],
            student_dp=student_dp,
            teacher_dp=teacher_dp,
        )

    # ==========================
    #         Loss
    # ==========================
    # Inject both tokenizer vocab sizes so the projection matrix's V_s
    # and V_t axes match `logits.shape[-1]` exactly, instead of being
    # recovered from the highest ids that happen to appear in the sparse
    # projection file. `len(tokenizer)` is what HF treats as the
    # embedding / lm_head dim.
    # Per-teacher metadata is injected as parallel lists (one entry per
    # `teachers[i]`); the loss fn reads these lists directly. `len(tokenizer)`
    # is the HF embedding/lm_head dim, sizing each projection matrix's V_s/V_t
    # axes exactly. No `send_full_logits`: every teacher ships full logits and
    # the loss derives the top-k subset student-side.
    loss_config = {
        **loss_config,
        "student_vocab_size": len(student_tokenizer),
        "teacher_vocab_sizes": [len(tok) for tok in teacher_tokenizers],
        "projection_matrix_paths": [t.projection_matrix_path for t in teachers],
        "teacher_weights": [t.weight for t in teachers],
        "teacher_gold_loss": [t.gold_loss for t in teachers],
        "teacher_xtoken_loss": [t.xtoken_loss for t in teachers],
    }
    loss_fn = CrossTokenizerDistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policies,
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


def export_teacher_logits_and_pack(
    teacher_policies: list[Policy],
    loss_fn: CrossTokenizerDistillationLossFn,
    batch: BatchedDataDict[Any],
    teacher_mbs: list[int],
    *,
    timer: Optional[Timer] = None,
) -> BatchedDataDict[Any]:
    """Serially run each teacher's forward and pack the student ``train_data``.

    Teachers run one at a time (collocated): each is onloaded for inference,
    forwarded, then offloaded. Every teacher ships full-vocab logits over CUDA
    IPC (``teacher_{i}_full_logits_ipc``) — the loss derives the microbatch-global
    top-k subset student-side, so there is no top-K transport. A same-vocab
    teacher (``projection_matrix_paths[i] is None``) reuses the student tokens
    and rides no extra alignment; a cross-tokenizer teacher's own tokenization
    and ``alignment_{i}_*`` payload ride along (teacher-indexed). The persistent
    IPC buffers stay resident on the teacher GPUs until released by the caller
    after ``student.train``. Shared by the train loop and ``validate`` so the
    forward+pack sequence can't drift between them.
    """
    train_data: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "input_lengths": batch["input_lengths"],
        "token_mask": batch["token_mask"],
        "sample_mask": batch["sample_mask"],
    }
    for i, teacher_policy in enumerate(teacher_policies):
        same_vocab = loss_fn.projection_matrix_paths[i] is None
        if same_vocab:
            # Same tokenizer: the teacher forward reuses the student tokens.
            teacher_data: BatchedDataDict[Any] = BatchedDataDict(
                input_ids=batch["input_ids"],
                input_lengths=batch["input_lengths"],
                token_mask=batch["token_mask"],
                sample_mask=batch["sample_mask"],
            )
        else:
            teacher_data = BatchedDataDict(
                input_ids=batch[f"teacher_{i}_input_ids"],
                input_lengths=batch[f"teacher_{i}_input_lengths"],
                token_mask=batch[f"teacher_{i}_token_mask"],
                sample_mask=batch["sample_mask"],
            )
            # Cross-tokenizer teacher tokens + alignment payload ride along
            # (teacher-indexed); the loss fn indexes them per microbatch.
            train_data[f"teacher_{i}_input_ids"] = batch[f"teacher_{i}_input_ids"]
            train_data[f"teacher_{i}_token_mask"] = batch[f"teacher_{i}_token_mask"]
            for field in (
                "pair_valid",
                "pair_is_correct",
                "student_exact_partition_mask",
                "teacher_exact_partition_mask",
                "student_chunk_id",
                "teacher_chunk_id",
                "num_chunks",
            ):
                train_data[f"alignment_{i}_{field}"] = batch[f"alignment_{i}_{field}"]

        teacher_policy.prepare_for_lp_inference()
        handles = teacher_policy.get_full_logits_ipc(
            teacher_data, micro_batch_size=teacher_mbs[i], timer=timer
        )
        train_data[f"teacher_{i}_full_logits_ipc"] = handles
        # Free the teacher's PARAMS to CPU; the persistent IPC buffers live in
        # worker state and survive this call.
        teacher_policy.offload_after_refit()

    return BatchedDataDict(train_data)


def xtoken_off_policy_distillation_train(
    student_policy: Policy,
    teacher_policies: list[Policy],
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
    # Per-teacher export MBS (each teacher's own train MBS) and the
    # non-student-seq keys the worker's check_sequence_dim must skip
    # (teacher-count-dependent, so built from the loss fn).
    teacher_mbs = [
        t.policy_config()["train_micro_batch_size"] for t in master_config.teachers
    ]
    skip_keys = xtoken_non_student_seq_keys(loss_fn)

    if val_at_start and total_steps == 0 and val_dataloader is not None:
        val_metrics, val_timings = validate(
            student_policy,
            teacher_policies,
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
                with timer.time("teacher_forward"):
                    # Serial per-teacher forward; each teacher ships full-vocab
                    # logits over IPC (``teacher_{i}_full_logits_ipc``) and the
                    # loss derives the microbatch-global top-k subset
                    # student-side. Packs the per-teacher alignment payload too.
                    train_data = export_teacher_logits_and_pack(
                        teacher_policies, loss_fn, batch, teacher_mbs, timer=timer
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
                            check_dim_skip_keys=skip_keys,
                        )
                    except Exception:
                        # Free every teacher's producer IPC buffer before
                        # propagating so a failed step doesn't leak teacher
                        # logits. The happy path keeps the buffers persistent
                        # across steps (reused via copy_) and releases once at
                        # loop exit — releasing every step would free + realloc
                        # the large teacher logits buffers and fragment into OOM.
                        for teacher_policy in teacher_policies:
                            teacher_policy.release_ipc_buffer()
                        raise

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
                        teacher_policies,
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
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
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
                    off_policy_distillation_state["total_valid_tokens"] = (
                        total_valid_tokens
                    )
                    off_policy_distillation_state["consumed_samples"] = consumed_samples
                    if val_metrics is not None and "loss" in val_metrics:
                        off_policy_distillation_state["val_loss"] = float(
                            val_metrics["loss"]
                        )
                    elif "val_loss" in off_policy_distillation_state:
                        del off_policy_distillation_state["val_loss"]

                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        prefix, metric_name = full_metric_name.split(":", 1)
                        source = metrics if prefix == "train" else (val_metrics or {})
                        if metric_name in source:
                            off_policy_distillation_state[full_metric_name] = float(
                                source[metric_name]
                            )

                    with timer.time("checkpointing"):
                        ckpt_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1,
                            off_policy_distillation_state,
                            master_config,
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(ckpt_path, "policy", "weights"),
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
            # (= dp_size * local_mbs values summed). We also print the
            # per-MB-mean for a per-microbatch-comparable signal.
            # n_mb = len of the flat list of per-MB metrics.
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
                    f"  • KL:   {kl_sum:.4f} (per-MB-mean: {kl_sum / n_mb:.4f})",
                    flush=True,
                )
            if "ce_loss" in metrics:
                ce_sum = float(metrics["ce_loss"])
                print(
                    f"  • CE:   {ce_sum:.4f} (per-MB-mean: {ce_sum / n_mb:.4f})",
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
            print(
                f"  • Total step time: {timing_metrics.get('total_step_time', 0):.2f}s",
                flush=True,
            )
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
                for teacher_policy in teacher_policies:
                    teacher_policy.release_ipc_buffer()
                return
            if total_steps >= max_steps:
                print("Max steps reached, stopping training.", flush=True)
                for teacher_policy in teacher_policies:
                    teacher_policy.release_ipc_buffer()
                return

        current_epoch += 1
        current_step = 0
    for teacher_policy in teacher_policies:
        teacher_policy.release_ipc_buffer()


# ===============================================================================
# Validation
# ===============================================================================


def validate(
    student_policy: Policy,
    teacher_policies: list[Policy],
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

    # Teacher and student may differ in DP/MBS, and teachers may differ from
    # each other; the final val batch (drop_last=False) can be ragged. Pad each
    # batch up to the smallest size that tiles cleanly on the student grid and
    # every teacher's grid so the even-split path applies. Each teacher's val
    # export reuses its own train MBS (no separate val knob).
    student_dp = student_policy.data_parallel_size
    student_mbs = master_config.policy["train_micro_batch_size"]
    teacher_mbs = [
        t.policy_config()["train_micro_batch_size"] for t in master_config.teachers
    ]
    pad_quantum = math.lcm(
        student_dp * student_mbs,
        *[
            teacher_policy.data_parallel_size * teacher_mbs[i]
            for i, teacher_policy in enumerate(teacher_policies)
        ],
    )
    skip_keys = xtoken_non_student_seq_keys(loss_fn)

    with timer.time("validation_total"):
        for batch in val_dataloader:
            target_size = math.ceil(batch.size / pad_quantum) * pad_quantum
            batch = pad_distillation_val_batch(batch, target_size)

            train_data = export_teacher_logits_and_pack(
                teacher_policies, loss_fn, batch, teacher_mbs, timer=timer
            )
            train_data.to("cpu")
            student_policy.prepare_for_training()
            try:
                results = student_policy.train(
                    train_data,
                    loss_fn,
                    eval_mode=True,
                    check_dim_skip_keys=skip_keys,
                )
            except Exception:
                for teacher_policy in teacher_policies:
                    teacher_policy.release_ipc_buffer()
                raise
            losses.append(float(np.mean(results["loss"].numpy())))
            mb_metrics = results.get("all_mb_metrics", {})
            if "kl_loss" in mb_metrics:
                kl_losses.append(float(np.mean(mb_metrics["kl_loss"])))
            if "ce_loss" in mb_metrics:
                ce_losses.append(float(np.mean(mb_metrics["ce_loss"])))
            if "kl_common" in mb_metrics:
                kl_common_losses.append(float(np.mean(mb_metrics["kl_common"])))
            if "l1_uncommon" in mb_metrics:
                l1_uncommon_losses.append(float(np.mean(mb_metrics["l1_uncommon"])))
        for teacher_policy in teacher_policies:
            teacher_policy.release_ipc_buffer()
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
