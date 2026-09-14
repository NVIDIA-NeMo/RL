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
    2. Run each teacher forward on TEACHER token ids and ship either dense
       full-vocabulary or sparse top-k logits over CUDA IPC (no driver
       round-trip), reassembled across the teacher's TP/CP shards on the
       student side.
    3. Pack alignment payload + teacher IPC handles into a student-side
       ``train_data`` dict.
    4. ``student_policy.train(train_data, loss_fn)`` — student forward +
       loss + backward + optimizer step happens inside the policy worker.

The collator and aligner do all the CPU-side cross-tokenizer work; the
loss function does only loss math; this module is just plumbing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from itertools import count
from typing import Any, NotRequired, Optional, TypedDict, cast

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.loss_functions import (
    CrossTokenizerDistillationLossConfig,
    CrossTokenizerDistillationLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.algorithms.x_token import TokenAligner
from nemo_rl.algorithms.x_token.loss_utils import _chunk_ids_to_spans
from nemo_rl.algorithms.x_token.utils import (
    assert_teacher_student_batch_grid,
    assert_xtoken_ipc_node_local,
    pad_distillation_val_batch,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.cross_tokenizer_collate import CrossTokenizerCollator
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.packing import (
    LockstepPackingItem,
    LockstepPackingPlan,
    SidePackingSpec,
    build_lockstep_packing_plan,
)
from nemo_rl.data.utils import load_dataloader_state
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.megatron.router_replay import router_replay_enabled
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
#     FullLogitsPostProcessor in the DTensor/Megatron worker's get_full_logits_ipc.
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
    metadata rather than a static constant. Every teacher contributes exactly
    one of ``teacher_{i}_full_logits_ipc`` or
    ``teacher_{i}_sparse_logits_ipc`` (lists of CUDA IPC handle dicts, not
    tensors); both possible keys are skipped so the transport may be selected
    at runtime. A cross-tokenizer teacher additionally rides its
    teacher-seq tokenization (``teacher_{i}_input_ids`` / ``teacher_{i}_token_mask``,
    ``[B, T_t]``) and its teacher-seq / max_pairs ``alignment_{i}_*`` keys. The
    ``alignment_{i}_student_*`` (``[B, T_s]``) and ``alignment_{i}_num_chunks``
    (``[B]``) keys follow the student-seq invariant and are NOT skipped.
    """
    keys: set[str] = set()
    for i in range(loss_fn.num_teachers):
        keys.add(f"teacher_{i}_full_logits_ipc")
        keys.add(f"teacher_{i}_sparse_logits_ipc")
        if loss_fn.projection_matrix_paths[i] is not None:  # cross-tokenizer
            keys.add(f"teacher_{i}_input_ids")
            keys.add(f"teacher_{i}_token_mask")
            keys.add(f"alignment_{i}_pair_valid")
            keys.add(f"alignment_{i}_pair_is_correct")
            keys.add(f"alignment_{i}_teacher_exact_partition_mask")
            keys.add(f"alignment_{i}_teacher_chunk_id")
    return frozenset(keys)


# Per-microbatch metrics that are *intensive* — rates, ratios, per-chunk
# averages, and global-batch constants replicated onto every microbatch. Their
# per-step value is the mean over the flat per-(microbatch, DP rank) list;
# everything else is an *extensive* per-microbatch contribution (loss shares
# normalized by a global denominator, counts) whose per-step value is the sum.
MEAN_REDUCED_MB_METRICS = frozenset(
    {
        "lr",
        "wd",
        "global_valid_seqs",
        "global_valid_toks",
        "accuracy",
        "proj_accuracy",
        "kl_loss_scale",
        # Gold-loss path: per-mb local chunk average.
        "kl_common",
        "l1_uncommon",
        # v6 (prefix_bidir_partition_kl_v3) diagnostics. Each microbatch
        # reports an average over *its own* chunks, so summing them scales
        # with the microbatch count instead of staying per-chunk (which is
        # how top1_acc_per_chunk reached an impossible 110.60).
        "kl_common_per_chunk",
        "kl_partition_first_per_chunk",
        "kl_partition_last_per_chunk",
        "top1_acc_per_chunk",
    }
)

# Per-teacher metrics are suffixed ``_t{i}`` by the loss fn's aggregation.
_TEACHER_METRIC_SUFFIX = re.compile(r"_t\d+$")

_XTOKEN_LOCKSTEP_PACKING_ALGORITHM = "lockstep_first_fit_decreasing"
_UNSAFE_PACKED_MODEL_MARKERS = (
    "mamba",
    "nemotron-h",
    "nemotron_h",
    "nemotronh",
    "nano",
)

_XTOKEN_LOGICAL_BATCH_DIGEST_SCHEMA_VERSION = 1
_XTOKEN_LOGICAL_BATCH_BASE_FIELDS = (
    "input_ids",
    "input_lengths",
    "token_mask",
    "kd_token_mask",
    "sample_mask",
    "student_semantic_regions",
)
_XTOKEN_LOGICAL_BATCH_TEACHER_SUFFIXES = (
    "_input_ids",
    "_input_lengths",
    "_token_mask",
    "_semantic_regions",
)


def reduce_mb_metric(key: str, values: Any) -> float:
    """Reduce one metric's per-(microbatch, DP rank) list to its per-step scalar."""
    if _TEACHER_METRIC_SUFFIX.sub("", key) in MEAN_REDUCED_MB_METRICS:
        return float(np.mean(values))
    return float(np.sum(values))


def _log_dense_ipc_reconstruction_telemetry(
    all_mb_metrics: Mapping[str, Any],
    *,
    packing_plan: LockstepPackingPlan,
    num_teachers: int,
) -> None:
    """Validate and log consumer-measured packed dense-IPC fallbacks."""
    fallback_metric_keys = (
        "ipc_reconstruction_fallbacks",
        *(f"ipc_reconstruction_fallbacks_t{i}" for i in range(num_teachers)),
    )
    missing_fallback_metrics = [
        key for key in fallback_metric_keys if key not in all_mb_metrics
    ]
    if missing_fallback_metrics:
        raise RuntimeError(
            "Packed dense IPC reconstruction telemetry is missing consumer-side "
            f"metrics {missing_fallback_metrics}."
        )

    reduced_fallbacks = {
        key: reduce_mb_metric(key, all_mb_metrics[key]) for key in fallback_metric_keys
    }
    non_integral_fallbacks = {
        key: value
        for key, value in reduced_fallbacks.items()
        if not math.isfinite(value)
        or value < 0
        or not math.isclose(value, round(value), abs_tol=1.0e-6)
    }
    if non_integral_fallbacks:
        raise RuntimeError(
            "Packed dense IPC reconstruction telemetry must contain finite, "
            "non-negative integer counts after reduction; got "
            f"{non_integral_fallbacks}."
        )
    fallback_count = round(reduced_fallbacks["ipc_reconstruction_fallbacks"])
    per_teacher_fallback_counts = tuple(
        round(reduced_fallbacks[f"ipc_reconstruction_fallbacks_t{i}"])
        for i in range(num_teachers)
    )
    if fallback_count != sum(per_teacher_fallback_counts):
        raise RuntimeError(
            "Packed dense IPC reconstruction telemetry is internally "
            f"inconsistent: total={fallback_count}, "
            f"per_teacher={per_teacher_fallback_counts}."
        )
    teacher_counts = " ".join(
        f"teacher_{i}={count}" for i, count in enumerate(per_teacher_fallback_counts)
    )
    print(
        "XTOKEN_IPC_RECONSTRUCTION "
        f"batch_uid={packing_plan.batch_uid} "
        f"reconstruction_fallbacks={fallback_count} "
        f"{teacher_counts}",
        flush=True,
    )


def _canonical_logical_digest_json_value(value: Any, *, field: str) -> Any:
    """Normalize one non-tensor logical field for deterministic JSON hashing."""
    if isinstance(value, np.generic):
        return _canonical_logical_digest_json_value(value.item(), field=field)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"xToken logical digest field {field!r} contains {value!r}."
            )
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError(
                f"xToken logical digest field {field!r} has non-string "
                f"mapping keys {non_string_keys!r}."
            )
        for key in sorted(value):
            normalized[key] = _canonical_logical_digest_json_value(
                value[key], field=f"{field}.{key}"
            )
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _canonical_logical_digest_json_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"xToken logical digest field {field!r} has unsupported type "
        f"{type(value).__name__}."
    )


def _logical_digest_json_scalar_count(value: Any) -> int:
    """Count canonical JSON scalar leaves for a ragged logical field."""
    if isinstance(value, Mapping):
        return sum(_logical_digest_json_scalar_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_logical_digest_json_scalar_count(item) for item in value)
    return 1


def _logical_digest_field_record(value: Any, *, field: str) -> dict[str, Any]:
    """Return shape, dtype, count, and a framed SHA-256 for one field."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        metadata = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "count": int(tensor.numel()),
        }
        header = json.dumps(
            metadata,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest = hashlib.sha256(header + b"\n" + raw).hexdigest()
        return {**metadata, "sha256": digest}

    normalized = _canonical_logical_digest_json_value(value, field=field)
    shape = [len(normalized)] if isinstance(normalized, list) else []
    metadata = {
        "shape": shape,
        "dtype": "canonical-json",
        "count": _logical_digest_json_scalar_count(normalized),
    }
    payload = json.dumps(
        {"metadata": metadata, "value": normalized},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {**metadata, "sha256": hashlib.sha256(payload).hexdigest()}


def build_xtoken_logical_batch_digest_record(
    batch: BatchedDataDict[Any], *, batch_uid: int
) -> dict[str, Any]:
    """Build the arm-independent logical batch record logged before packing.

    The record intentionally contains hashes and structural metadata only. The
    frozen asset keeps the ordered sample IDs beside this record, while runtime
    logs can prove exact equality without exposing dataset text or identifiers.
    """
    if batch_uid < 0 or batch_uid >= 2**31:
        raise ValueError(
            f"batch_uid must fit the non-negative high 31 bits of int64, got {batch_uid}."
        )
    sample_ids_value = batch.get("sample_id")
    if torch.is_tensor(sample_ids_value):
        sample_ids: Any = sample_ids_value.detach().cpu().tolist()
    elif isinstance(sample_ids_value, Sequence) and not isinstance(
        sample_ids_value, (str, bytes)
    ):
        sample_ids = list(sample_ids_value)
    else:
        raise ValueError(
            "xToken logical batch digest requires an ordered sample_id sequence."
        )
    if len(sample_ids) != batch.size:
        raise ValueError(
            f"sample_id has {len(sample_ids)} entries for batch size {batch.size}."
        )
    normalized_sample_ids = _canonical_logical_digest_json_value(
        sample_ids, field="sample_id"
    )
    if any(
        sample_id is None or not str(sample_id).strip()
        for sample_id in normalized_sample_ids
    ):
        raise ValueError("xToken logical batch digest found an empty sample_id.")

    field_names = [key for key in _XTOKEN_LOGICAL_BATCH_BASE_FIELDS if key in batch]
    field_names.extend(
        sorted(
            key
            for key in batch
            if key.startswith("teacher_")
            and key.endswith(_XTOKEN_LOGICAL_BATCH_TEACHER_SUFFIXES)
        )
    )
    field_names.extend(sorted(key for key in batch if key.startswith("alignment_")))
    if not {"input_ids", "input_lengths", "token_mask", "sample_mask"}.issubset(
        field_names
    ):
        raise ValueError(
            "xToken logical batch digest is missing a required student field: "
            f"available={tuple(field_names)}."
        )
    if len(field_names) != len(set(field_names)):
        raise AssertionError(f"Duplicate xToken logical digest fields: {field_names}.")

    sample_id_record = _logical_digest_field_record(
        normalized_sample_ids, field="sample_id"
    )
    batch_item_ids = tuple((batch_uid << 32) | ordinal for ordinal in range(batch.size))
    record: dict[str, Any] = {
        "schema_version": _XTOKEN_LOGICAL_BATCH_DIGEST_SCHEMA_VERSION,
        "batch_uid": batch_uid,
        "logical_samples": batch.size,
        "sample_ids": sample_id_record,
        "batch_item_ids": _logical_digest_field_record(
            batch_item_ids, field="batch_item_id"
        ),
        "fields": {
            key: _logical_digest_field_record(batch[key], field=key)
            for key in field_names
        },
    }
    canonical_record = json.dumps(
        record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(canonical_record).hexdigest()
    return record


def log_xtoken_logical_batch_digest(
    batch: BatchedDataDict[Any], *, batch_uid: int
) -> dict[str, Any]:
    """Emit and return one compact arm-independent logical-batch record."""
    record = build_xtoken_logical_batch_digest_record(batch, batch_uid=batch_uid)
    print(
        "XTOKEN_LOGICAL_BATCH_DIGEST "
        + json.dumps(
            record,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return record


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
        offload_student_after_step: Offload the student model and optimizer to
            CPU after each optimizer step. Student and teachers are colocated on
            the same GPUs, so the next step's teacher forward otherwise competes
            with resident student optimizer state.
    """

    num_prompts_per_step: int
    max_num_steps: int
    max_num_epochs: int
    seed: int
    val_period: int
    val_at_start: bool
    val_at_end: bool
    offload_student_after_step: NotRequired[bool]


class OffPolicyDistillationSaveState(TypedDict):
    current_epoch: int
    current_step: int
    total_steps: int
    consumed_samples: int
    total_valid_tokens: int
    next_packing_batch_uid: NotRequired[int]
    val_loss: NotRequired[float]


def _default_off_policy_distillation_save_state() -> OffPolicyDistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
        "next_packing_batch_uid": 0,
    }


class TeacherAlignerConfig(BaseModel, extra="allow"):
    """Per-teacher token-alignment configuration.

    Attributes:
        projection_matrix_path: Path to this teacher's student-to-teacher
            projection matrix. ``None`` marks a same-tokenizer teacher, which
            bypasses projection and alignment and uses direct per-position KL.
        drop_first_assistant_chunk_kl: Whether chat-mode alignment drops the
            first content pair in each assistant message for this teacher. The
            CE token mask is unaffected.
        pseudo_target_path: Path to this teacher's forward pseudo-target table
            (student-to-teacher sub-token chains). ``None`` for a same-tokenizer
            teacher.
        reverse_pseudo_target_path: Path to this teacher's reverse pseudo-target
            table (teacher-to-student sub-token chains). ``None`` for a
            same-tokenizer teacher.
    """

    projection_matrix_path: Optional[str] = None
    drop_first_assistant_chunk_kl: bool = False
    pseudo_target_path: Optional[str] = None
    reverse_pseudo_target_path: Optional[str] = None


def _teacher_topk_ipc_keep_realized(
    loss_config: CrossTokenizerDistillationLossConfig,
) -> bool:
    """Whether sparse teacher IPC must retain realized teacher labels."""
    topk_k = int(loss_config["teacher_topk_ipc_k"])
    return topk_k > 0 and (
        bool(loss_config.get("prefix_bidir_v3_pure_alm", False))
        or topk_k == 1
        or bool(loss_config["teacher_topk_ipc_keep_realized"])
    )


def _build_teacher_force_include_token_ids(
    batch: BatchedDataDict[Any],
    *,
    teacher_idx: int,
    loss_config: CrossTokenizerDistillationLossConfig,
) -> Optional[torch.Tensor]:
    """Build realized teacher labels that sparse IPC must retain per position."""
    needs_force_ids = _teacher_topk_ipc_keep_realized(loss_config) or (
        int(loss_config["prefix_bidir_v3_noise_filter_topk"]) > 0
    )
    if not needs_force_ids:
        return None

    teacher_input_ids = batch[f"teacher_{teacher_idx}_input_ids"]
    if not torch.is_tensor(teacher_input_ids):
        return None

    pair_valid = batch[f"alignment_{teacher_idx}_pair_valid"].to(torch.bool)
    max_pairs = int(pair_valid.shape[1])
    student_spans = _chunk_ids_to_spans(
        batch[f"alignment_{teacher_idx}_student_chunk_id"], max_pairs
    )
    teacher_spans = _chunk_ids_to_spans(
        batch[f"alignment_{teacher_idx}_teacher_chunk_id"], max_pairs
    )
    force_ids = torch.full_like(teacher_input_ids, -1)
    kl_chunk_shift = bool(loss_config["kl_chunk_shift"])
    batch_size, teacher_seq_len = teacher_input_ids.shape

    for batch_idx in range(batch_size):
        for pair_idx in range(max_pairs):
            if not bool(pair_valid[batch_idx, pair_idx]):
                continue
            student_start = int(student_spans[batch_idx, pair_idx, 0])
            student_end = int(student_spans[batch_idx, pair_idx, 1])
            teacher_start = int(teacher_spans[batch_idx, pair_idx, 0])
            teacher_end = int(teacher_spans[batch_idx, pair_idx, 1])
            if student_end <= student_start or teacher_end <= teacher_start:
                continue
            if kl_chunk_shift and student_start > 0 and teacher_start > 0:
                predictor_start = teacher_start - 1
            else:
                predictor_start = teacher_start
            for offset in range(teacher_end - teacher_start):
                predictor_pos = predictor_start + offset
                label_pos = teacher_start + offset
                if (
                    0 <= predictor_pos < teacher_seq_len
                    and 0 <= label_pos < teacher_seq_len
                ):
                    force_ids[batch_idx, predictor_pos] = teacher_input_ids[
                        batch_idx, label_pos
                    ]
    return force_ids


def _get_teacher_logits_ipc(
    teacher_policy: Policy,
    teacher_data: BatchedDataDict[Any],
    loss_fn: CrossTokenizerDistillationLossFn,
    *,
    teacher_idx: int,
    micro_batch_size: int,
    force_include_token_ids: Optional[torch.Tensor],
    timer: Optional[Timer],
    packing_plan: Optional[LockstepPackingPlan] = None,
    packing_side_id: Optional[str] = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return the indexed transport suffix and IPC handles for one teacher."""
    topk_k = int(loss_fn.cfg["teacher_topk_ipc_k"])
    if packing_plan is not None and topk_k > 0:
        raise ValueError(
            "xToken lockstep packing currently supports dense teacher IPC only."
        )
    if topk_k > 0 and loss_fn.projection_matrix_paths[teacher_idx] is not None:
        noise_filter_topk = int(loss_fn.cfg["prefix_bidir_v3_noise_filter_topk"])
        if noise_filter_topk < 0:
            raise ValueError(
                "prefix_bidir_v3_noise_filter_topk must be >= 0, "
                f"got {noise_filter_topk}"
            )
        if force_include_token_ids is not None:
            teacher_data["force_include_token_ids"] = force_include_token_ids
        handles = teacher_policy.get_topk_logits_ipc(
            teacher_data,
            k=topk_k,
            temperature=float(loss_fn.cfg["temperature"]),
            vocab_size=int(loss_fn.teacher_vocab_sizes[teacher_idx]),
            micro_batch_size=micro_batch_size,
            support_mode=str(loss_fn.cfg["teacher_topk_ipc_support_mode"]),
            gt_filter_topk=(noise_filter_topk if noise_filter_topk > 0 else None),
            timer=timer,
        )
        return "sparse_logits_ipc", handles

    full_logits_kwargs: dict[str, Any] = {}
    if packing_plan is not None:
        full_logits_kwargs = {
            "packing_plan": packing_plan,
            "packing_side_id": packing_side_id,
        }
    handles = teacher_policy.get_full_logits_ipc(
        teacher_data,
        micro_batch_size=micro_batch_size,
        timer=timer,
        **full_logits_kwargs,
    )
    return "full_logits_ipc", handles


class TeacherConfig(BaseModel, extra="allow"):
    """Per-teacher config for multi-teacher cross-tokenizer distillation.

    Carries the full ``PolicyConfig`` content (``model_name``, ``tokenizer``,
    ``dtensor_cfg``, …) as permitted extras, plus the cross-tokenizer knobs
    declared below. Alignment-specific settings live in the typed ``aligner``
    block. Use :meth:`policy_config` to recover the plain ``PolicyConfig`` dict
    for ``Policy`` construction.

    Attributes:
        aligner: This teacher's projection and chat-alignment settings.
        weight: Static loss weight for this teacher when several teachers are
            aggregated (``kd_loss_mode="sum"`` / the convex ``"averaged_logits"``
            mix). Single-teacher runs leave it at ``1.0``.
    """

    aligner: TeacherAlignerConfig = Field(default_factory=TeacherAlignerConfig)
    weight: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_projection_matrix_path(cls, value: Any) -> Any:
        """Migrate the released root projection path into ``aligner``."""
        if not isinstance(value, dict):
            return value

        root_pseudo_paths = {
            key
            for key in ("pseudo_target_path", "reverse_pseudo_target_path")
            if key in value
        }
        if root_pseudo_paths:
            keys = ", ".join(f"teachers[i].{key}" for key in sorted(root_pseudo_paths))
            raise ValueError(
                f"{keys} must be nested under teachers[i].aligner; root-level "
                "pseudo-target paths are not supported"
            )

        if "projection_matrix_path" not in value:
            return value

        migrated = dict(value)
        legacy_path = migrated.pop("projection_matrix_path")
        if "aligner" not in migrated:
            migrated["aligner"] = {"projection_matrix_path": legacy_path}
        else:
            raw_aligner = migrated["aligner"]
            if isinstance(raw_aligner, TeacherAlignerConfig):
                if (
                    "projection_matrix_path" in raw_aligner.model_fields_set
                    and raw_aligner.projection_matrix_path != legacy_path
                ):
                    raise ValueError(
                        "conflicting projection matrix paths at "
                        "teachers[i].projection_matrix_path and "
                        "teachers[i].aligner.projection_matrix_path"
                    )
                aligner = raw_aligner.model_dump()
                aligner["projection_matrix_path"] = legacy_path
                migrated["aligner"] = aligner
            elif isinstance(raw_aligner, dict):
                aligner = dict(raw_aligner)
                if (
                    "projection_matrix_path" in aligner
                    and aligner["projection_matrix_path"] != legacy_path
                ):
                    raise ValueError(
                        "conflicting projection matrix paths at "
                        "teachers[i].projection_matrix_path and "
                        "teachers[i].aligner.projection_matrix_path"
                    )
                aligner["projection_matrix_path"] = legacy_path
                migrated["aligner"] = aligner
            else:
                raise ValueError(
                    "teachers[i].aligner must be a mapping when using the legacy "
                    "teachers[i].projection_matrix_path field"
                )

        warnings.warn(
            "teachers[i].projection_matrix_path is deprecated; use "
            "teachers[i].aligner.projection_matrix_path instead.",
            FutureWarning,
            stacklevel=2,
        )
        return migrated

    def policy_config(self) -> PolicyConfig:
        """Recover the plain ``PolicyConfig`` dict (cross-tokenizer knobs stripped)."""
        return cast(
            PolicyConfig,
            self.model_dump(exclude={"aligner", "weight"}),
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

    @model_validator(mode="before")
    @classmethod
    def _reject_shared_drop_first_assistant_chunk_kl(cls, value: Any) -> Any:
        """Reject the unreleased shared spelling instead of ignoring it."""
        if isinstance(value, dict):
            data = value.get("data")
            if isinstance(data, dict) and "drop_first_assistant_chunk_kl" in data:
                raise ValueError(
                    "data.drop_first_assistant_chunk_kl was replaced by the "
                    "per-teacher "
                    "teachers[i].aligner.drop_first_assistant_chunk_kl field"
                )
        return value


# ===============================================================================
# Setup
# ===============================================================================


def _xtoken_entity_parallelism(
    cfg: PolicyConfig, *, label: str
) -> tuple[int, int, int]:
    """Return ``(tensor_parallel, context_parallel, pipeline_parallel)`` for a config.

    xToken distillation accepts either the DTensor-V2 backend
    (``dtensor_cfg.enabled`` and ``dtensor_cfg._v2``) or the Megatron backend
    (``megatron_cfg.enabled``). Both support tensor and context parallelism (the
    loss is parallelism-invariant); Megatron additionally supports pipeline
    parallelism, where only the last stage holds logits and therefore only it
    contributes teacher full-logits IPC handles. DTensor has no pipeline axis,
    so it always reports ``pipeline_parallel == 1``.

    Args:
        cfg: The policy or teacher policy config.
        label: Human-readable name of the entity for error messages, e.g.
            ``"teachers[0]"``.

    Returns:
        The entity's tensor-, context- and pipeline-parallel sizes, read from
        whichever backend block is enabled.
    """
    if "megatron_cfg" in cfg and cfg["megatron_cfg"]["enabled"]:
        mcfg = cfg["megatron_cfg"]
        return (
            mcfg["tensor_model_parallel_size"],
            mcfg["context_parallel_size"],
            mcfg["pipeline_model_parallel_size"],
        )
    assert cfg["dtensor_cfg"]["enabled"] and cfg["dtensor_cfg"].get("_v2"), (
        f"xtoken distillation requires {label} to enable either DTensor-V2 "
        f"(dtensor_cfg.enabled=true and _v2=true) or Megatron "
        f"(megatron_cfg.enabled=true)."
    )
    return (
        cfg["dtensor_cfg"]["tensor_parallel_size"],
        cfg["dtensor_cfg"]["context_parallel_size"],
        1,
    )


def validate_xtoken_router_replay_setup(master_config: MasterConfig) -> None:
    """Fail before cluster creation when xToken replay cannot be consumed."""
    if not router_replay_enabled(master_config.policy):
        return
    megatron_cfg = master_config.policy.get("megatron_cfg") or {}
    if not megatron_cfg.get("enabled", False):
        raise ValueError(
            "policy.router_replay.enabled=true for xToken distillation requires "
            "the Megatron student policy backend; DTensor cannot consume "
            "routed_experts."
        )


def _sequence_packing_enabled(cfg: PolicyConfig) -> bool:
    packing = cfg.get("sequence_packing") or {}
    return bool(packing.get("enabled", False))


def _positive_config_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    return value


def _xtoken_configured_dp_size(
    cfg: PolicyConfig,
    *,
    label: str,
    world_size: int,
) -> int:
    tp, cp, pp = _xtoken_entity_parallelism(cfg, label=label)
    model_parallel_size = tp * cp * pp
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"xToken lockstep packing cannot form {label}'s DP grid: world_size="
            f"{world_size} is not divisible by tp*cp*pp={model_parallel_size}."
        )
    return world_size // model_parallel_size


def _load_xtoken_packed_model_config(cfg: PolicyConfig) -> Any:
    """Resolve architecture metadata without constructing a worker or model.

    A local tokenizer snapshot is preferred when the model is still named by a
    mutable Hub id and that snapshot also contains ``config.json``. This keeps
    qualification architecture checks on the same pinned revision used by the
    collator and backend tokenizer. Local model/checkpoint paths remain
    authoritative. Any format whose architecture cannot be resolved is denied
    below rather than silently admitted to recurrent-state packing.
    """
    from transformers import AutoConfig

    model_source = str(cfg.get("model_name", ""))
    tokenizer_source = str((cfg.get("tokenizer") or {}).get("name", ""))
    architecture_source = model_source
    if (
        not os.path.exists(model_source)
        and os.path.isdir(tokenizer_source)
        and os.path.isfile(os.path.join(tokenizer_source, "config.json"))
    ):
        architecture_source = tokenizer_source

    hf_config_overrides = cfg.get("hf_config_overrides") or {}
    return AutoConfig.from_pretrained(
        architecture_source,
        trust_remote_code=True,
        local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
        **hf_config_overrides,
    )


def _iter_nonempty_recurrent_config_fields(
    value: object, *, prefix: str = ""
) -> Iterator[str]:
    """Yield configured Mamba/SSM field paths from nested model metadata."""
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        key_lower = str(key).lower()
        if ("mamba" in key_lower or "ssm" in key_lower) and child not in (
            None,
            False,
            0,
            "",
            [],
            {},
        ):
            yield field
        yield from _iter_nonempty_recurrent_config_fields(child, prefix=field)


def _validate_xtoken_packed_model_architecture(
    cfg: PolicyConfig, *, label: str
) -> None:
    """Reject recurrent/hybrid architectures using resolved model metadata."""
    model_name = str(cfg.get("model_name", ""))
    try:
        model_config = _load_xtoken_packed_model_config(cfg)
    except Exception as error:
        raise ValueError(
            "xToken lockstep packing must resolve model architecture metadata "
            f"before cluster creation; could not inspect {label}.model_name="
            f"{model_name!r}: {type(error).__name__}: {error}"
        ) from error

    metadata = model_config.to_dict()
    identity_metadata = {
        "model_type": getattr(model_config, "model_type", None),
        "architectures": getattr(model_config, "architectures", None),
        "layer_types": metadata.get("layer_types"),
        "layers_block_type": metadata.get("layers_block_type"),
        "hybrid_override_pattern": metadata.get("hybrid_override_pattern"),
    }
    identity_text = repr(identity_metadata).lower()
    unsafe_marker = next(
        (marker for marker in _UNSAFE_PACKED_MODEL_MARKERS if marker in identity_text),
        None,
    )
    recurrent_fields = tuple(_iter_nonempty_recurrent_config_fields(metadata))
    hybrid_pattern = metadata.get("hybrid_override_pattern")
    if unsafe_marker is not None or recurrent_fields or hybrid_pattern:
        raise ValueError(
            "xToken lockstep packing does not yet support Nano/Mamba recurrent "
            f"state isolation for {label}.model_name={model_name!r}; resolved "
            f"architecture={identity_metadata!r}, recurrent_fields="
            f"{recurrent_fields!r}."
        )


def _validate_xtoken_packed_backend(cfg: PolicyConfig, *, label: str) -> None:
    """Validate the initial dense-transformer packing support matrix."""
    model_name = str(cfg.get("model_name", "")).lower()
    unsafe_marker = next(
        (marker for marker in _UNSAFE_PACKED_MODEL_MARKERS if marker in model_name),
        None,
    )
    if unsafe_marker is not None:
        raise ValueError(
            f"xToken lockstep packing does not yet support Nano/Mamba state "
            f"isolation; {label}.model_name={cfg.get('model_name')!r} matched "
            f"{unsafe_marker!r}."
        )
    _validate_xtoken_packed_model_architecture(cfg, label=label)

    make_divisible = _positive_config_int(
        cfg.get("make_sequence_length_divisible_by"),
        field=f"{label}.make_sequence_length_divisible_by",
    )
    tp, cp, pp = _xtoken_entity_parallelism(cfg, label=label)
    megatron_cfg = cfg.get("megatron_cfg") or {}
    if bool(megatron_cfg.get("enabled", False)):
        if pp != 1:
            raise ValueError(
                "xToken lockstep packing currently requires Megatron PP=1; "
                f"{label} has pipeline_model_parallel_size={pp}."
            )
        if cp not in (1, 2):
            raise ValueError(
                "xToken lockstep packing currently supports Megatron CP=1 or "
                f"CP=2; {label} has context_parallel_size={cp}."
            )
        fp8_cfg = megatron_cfg.get("fp8_cfg") or {}
        if bool(fp8_cfg.get("enabled", False)):
            raise ValueError(
                "xToken lockstep packing has not qualified Megatron FP8 packed-"
                f"total padding; disable {label}.megatron_cfg.fp8_cfg.enabled."
            )
        if (
            megatron_cfg.get("moe_token_dispatcher_type") == "flex"
            and megatron_cfg.get("moe_flex_dispatcher_backend") == "hybridep"
        ):
            raise ValueError(
                "xToken lockstep packing has not qualified HybridEP packed-total "
                f"padding for {label}."
            )
        minimum_divisor = cp * 2 if cp > 1 else 1
        if tp > 1 and bool(megatron_cfg.get("sequence_parallel", False)):
            minimum_divisor *= tp
        if make_divisible % minimum_divisor != 0:
            raise ValueError(
                f"{label}.make_sequence_length_divisible_by={make_divisible} "
                f"must be a multiple of {minimum_divisor} for Megatron "
                f"TP={tp}, CP={cp}, sequence_parallel="
                f"{bool(megatron_cfg.get('sequence_parallel', False))}."
            )
        return

    dtensor_cfg = cfg["dtensor_cfg"]
    if cp != 1:
        raise ValueError(
            "xToken lockstep packing currently supports DTensor-V2 CP=1 only; "
            f"{label} has context_parallel_size={cp}."
        )
    if bool(dtensor_cfg.get("sequence_parallel", False)):
        raise ValueError(
            "xToken lockstep packing currently requires DTensor-V2 sequence "
            f"parallelism to be disabled for {label}."
        )


def validate_xtoken_packing_setup(master_config: MasterConfig) -> Optional[int]:
    """Fail before cluster creation for unsupported xToken packing configs.

    Returns the common configured DP size when packing is enabled and ``None``
    when every side has packing disabled.
    """
    policy_config = master_config.policy
    teacher_configs = [teacher.policy_config() for teacher in master_config.teachers]
    side_configs: list[tuple[str, PolicyConfig]] = [
        ("policy", policy_config),
        *[
            (f"teachers[{teacher_idx}]", teacher_config)
            for teacher_idx, teacher_config in enumerate(teacher_configs)
        ],
    ]
    packing_modes = [_sequence_packing_enabled(config) for _, config in side_configs]
    if not any(packing_modes):
        return None
    if not all(packing_modes):
        enabled_labels = [
            label
            for (label, _), enabled in zip(side_configs, packing_modes, strict=True)
            if enabled
        ]
        disabled_labels = [
            label
            for (label, _), enabled in zip(side_configs, packing_modes, strict=True)
            if not enabled
        ]
        raise ValueError(
            "xToken lockstep packing must be enabled on the student and every "
            f"teacher; enabled={enabled_labels}, disabled={disabled_labels}."
        )

    data_config = master_config.data
    if data_config.get("num_packed_rows", 1) != 1:
        raise ValueError(
            "xToken lockstep packing keeps one source row per logical sample; "
            f"data.num_packed_rows must be 1, got "
            f"{data_config.get('num_packed_rows')!r}."
        )
    default_config = data_config.get("default") or {}
    for split_name in ("train", "validation"):
        split_configs = data_config.get(split_name)
        if split_configs is None:
            continue
        if isinstance(split_configs, Mapping):
            dataset_configs = [split_configs]
        elif isinstance(split_configs, list):
            dataset_configs = [
                cast(Mapping[str, Any], dataset_config)
                for dataset_config in split_configs
            ]
        else:
            raise ValueError(
                f"data.{split_name} must be a dataset config or list of configs, "
                f"got {type(split_configs).__name__}."
            )
        for dataset_idx, dataset_config in enumerate(dataset_configs):
            characters_per_sample = dataset_config.get(
                "characters_per_sample", default_config.get("characters_per_sample")
            )
            if characters_per_sample is not None:
                raise ValueError(
                    "xToken lockstep packing requires one Arrow/chat row per "
                    "logical sample; set "
                    f"data.{split_name}.characters_per_sample=null "
                    f"(dataset {dataset_idx} resolved to "
                    f"{characters_per_sample!r})."
                )

    algorithms: list[str] = []
    for label, config in side_configs:
        dynamic_batching = config.get("dynamic_batching") or {}
        if bool(dynamic_batching.get("enabled", False)):
            raise ValueError(
                "xToken lockstep packing is incompatible with independent "
                f"dynamic batching; disable {label}.dynamic_batching.enabled."
            )
        sequence_packing = config.get("sequence_packing") or {}
        algorithm = str(sequence_packing.get("algorithm", ""))
        algorithms.append(algorithm)
        if "fuse_loss" not in sequence_packing:
            raise ValueError(
                f"{label}.sequence_packing.fuse_loss must be set explicitly; "
                "xToken lockstep packing requires false."
            )
        if bool(sequence_packing["fuse_loss"]):
            raise ValueError(
                "xToken lockstep packing requires sequence_packing.fuse_loss=false; "
                f"it is enabled for {label}."
            )
        _positive_config_int(
            sequence_packing.get("train_mb_tokens"),
            field=f"{label}.sequence_packing.train_mb_tokens",
        )
        if label.startswith("teachers["):
            _positive_config_int(
                sequence_packing.get("logprob_mb_tokens"),
                field=f"{label}.sequence_packing.logprob_mb_tokens",
            )
        if config.get("train_micro_batch_size") != 1:
            raise ValueError(
                "xToken lockstep packing preserves MBS=1 logical loss cohorts; "
                f"{label}.train_micro_batch_size must be 1, got "
                f"{config.get('train_micro_batch_size')!r}."
            )
        _validate_xtoken_packed_backend(config, label=label)

    if set(algorithms) != {_XTOKEN_LOCKSTEP_PACKING_ALGORITHM}:
        raise ValueError(
            "xToken lockstep packing selects one controller algorithm for every "
            f"side; expected {_XTOKEN_LOCKSTEP_PACKING_ALGORITHM!r}, got "
            f"{dict(zip((label for label, _ in side_configs), algorithms, strict=True))}."
        )

    loss_config = master_config.loss_fn
    if int(loss_config.get("teacher_topk_ipc_k", 0)) != 0:
        raise ValueError(
            "xToken lockstep packing currently supports dense teacher IPC only; "
            "set loss_fn.teacher_topk_ipc_k=0."
        )
    if loss_config.get("kd_loss_mode") != "sum":
        raise ValueError(
            "xToken lockstep packing currently supports static additive teacher "
            "aggregation only; set loss_fn.kd_loss_mode='sum'."
        )
    if loss_config.get("sum_weights_metric") is not None:
        raise ValueError(
            "xToken lockstep packing does not support per-batch teacher scoring; "
            "set loss_fn.sum_weights_metric=null."
        )

    world_size = _positive_config_int(
        master_config.cluster["num_nodes"], field="cluster.num_nodes"
    ) * _positive_config_int(
        master_config.cluster["gpus_per_node"], field="cluster.gpus_per_node"
    )
    dp_sizes = [
        _xtoken_configured_dp_size(config, label=label, world_size=world_size)
        for label, config in side_configs
    ]
    if len(set(dp_sizes)) != 1:
        raise ValueError(
            "xToken lockstep packing initially requires equal student/teacher "
            f"DP sizes; got {dict(zip((label for label, _ in side_configs), dp_sizes, strict=True))}."
        )
    data_parallel_size = dp_sizes[0]
    global_batch_size = _positive_config_int(
        master_config.distillation["num_prompts_per_step"],
        field="distillation.num_prompts_per_step",
    )
    if global_batch_size % data_parallel_size != 0:
        raise ValueError(
            f"xToken logical GBS={global_batch_size} must be divisible by the "
            f"common DP size {data_parallel_size}."
        )
    for label, config in side_configs:
        if config.get("train_global_batch_size") != global_batch_size:
            raise ValueError(
                "xToken GBS counts logical samples and must match on every side; "
                f"distillation GBS={global_batch_size}, "
                f"{label}.train_global_batch_size="
                f"{config.get('train_global_batch_size')!r}."
            )

    student_tp, student_cp, student_pp = _xtoken_entity_parallelism(
        policy_config, label="policy"
    )
    for teacher_idx, teacher_config in enumerate(teacher_configs):
        teacher_tp, teacher_cp, teacher_pp = _xtoken_entity_parallelism(
            teacher_config, label=f"teachers[{teacher_idx}]"
        )
        assert_xtoken_ipc_node_local(
            num_nodes=master_config.cluster["num_nodes"],
            gpus_per_node=master_config.cluster["gpus_per_node"],
            student_tp=student_tp,
            student_cp=student_cp,
            student_pp=student_pp,
            teacher_tp=teacher_tp,
            teacher_cp=teacher_cp,
            teacher_pp=teacher_pp,
            student_dp=data_parallel_size,
            teacher_dp=data_parallel_size,
        )
    return data_parallel_size


def _freeze_tokenizer_value(value: object) -> object:
    """Convert tokenizer metadata into a deterministic comparison value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_tokenizer_value(item)) for key, item in value.items()
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_tokenizer_value(item) for item in value)
    added_token_fields = (
        "content",
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "special",
    )
    if all(hasattr(value, field) for field in added_token_fields):
        return tuple(
            (field, _freeze_tokenizer_value(getattr(value, field)))
            for field in added_token_fields
        )
    return repr(value)


def _tokenizer_reuse_signature(
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_config: Mapping[str, Any],
) -> dict[str, object]:
    """Describe every tokenizer semantic required for safe token-ID reuse."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception as error:
        raise ValueError(
            "same-tokenizer reuse requires tokenizer.get_vocab() to be available"
        ) from error
    if not isinstance(vocab, Mapping):
        raise ValueError(
            "same-tokenizer reuse requires tokenizer.get_vocab() to return a mapping"
        )
    try:
        vocab_items = tuple(
            sorted((str(token), int(idx)) for token, idx in vocab.items())
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "same-tokenizer reuse requires a string-to-integer vocabulary mapping"
        ) from error

    backend_signature: object = None
    is_fast = bool(getattr(tokenizer, "is_fast", False))
    if not is_fast:
        raise ValueError(
            "same-tokenizer reuse requires fast tokenizers with a complete "
            "serialized backend signature; slow-tokenizer normalization, "
            "pretokenization, and postprocessing cannot be proven equivalent"
        )
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is None:
        raise ValueError(
            "same-tokenizer reuse requires a serializable fast-tokenizer backend"
        )
    to_str = getattr(backend_tokenizer, "to_str", None)
    if not callable(to_str):
        raise ValueError(
            "same-tokenizer reuse requires fast tokenizer backend serialization"
        )
    backend_signature = to_str()

    return {
        "tokenizer_class": (
            type(tokenizer).__module__,
            type(tokenizer).__qualname__,
        ),
        "is_fast": is_fast,
        "vocab": vocab_items,
        "backend": backend_signature,
        "special_tokens_map": _freeze_tokenizer_value(
            getattr(tokenizer, "special_tokens_map_extended", {})
        ),
        "all_special_ids": tuple(int(idx) for idx in tokenizer.all_special_ids),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "chat_template_kwargs": _freeze_tokenizer_value(
            tokenizer_config.get("chat_template_kwargs") or {}
        ),
    }


def _assert_same_tokenizer_reuse_safe(
    *,
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    student_config: PolicyConfig,
    teacher_config: PolicyConfig,
    teacher_idx: int,
) -> None:
    student_signature = _tokenizer_reuse_signature(
        student_tokenizer, student_config["tokenizer"]
    )
    teacher_signature = _tokenizer_reuse_signature(
        teacher_tokenizer, teacher_config["tokenizer"]
    )
    differing_fields = sorted(
        field
        for field in student_signature
        if student_signature[field] != teacher_signature[field]
    )
    if differing_fields:
        raise ValueError(
            f"teachers[{teacher_idx}].aligner.projection_matrix_path is null, "
            "but safe "
            "student-token reuse requires identical tokenizer mapping, backend, "
            "special-token configuration, chat template, and template kwargs; "
            f"differing fields: {differing_fields}."
        )


def validate_xtoken_tokenizer_reuse(
    master_config: MasterConfig,
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizers: Sequence[PreTrainedTokenizerBase],
) -> None:
    """Validate every null-projection teacher before any worker is created."""
    if len(teacher_tokenizers) != len(master_config.teachers):
        raise ValueError(
            f"expected one tokenizer per teacher; got {len(teacher_tokenizers)} "
            f"tokenizers for {len(master_config.teachers)} teachers."
        )
    for teacher_idx, teacher in enumerate(master_config.teachers):
        if teacher.aligner.projection_matrix_path is None:
            _assert_same_tokenizer_reuse_safe(
                student_tokenizer=student_tokenizer,
                teacher_tokenizer=teacher_tokenizers[teacher_idx],
                student_config=master_config.policy,
                teacher_config=teacher.policy_config(),
                teacher_idx=teacher_idx,
            )


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

    validate_xtoken_router_replay_setup(master_config)

    # Backend gate. The student and each teacher may be DTensor V2 or Megatron.
    # Both support TP/CP/diff-DP sharding and Megatron additionally supports PP
    # (the loss is parallelism-invariant), so there is deliberately NO
    # tensor/context/pipeline_parallel_size==1 assert on either backend.
    # ``_xtoken_entity_parallelism`` validates each entity's backend as a side
    # effect (raising on an unsupported / misconfigured one).
    _xtoken_entity_parallelism(policy_config, label="policy")
    for i, tc in enumerate(teacher_configs):
        _xtoken_entity_parallelism(tc, label=f"teachers[{i}]")
    configured_packing_dp = validate_xtoken_packing_setup(master_config)

    teacher_topk_ipc_k = int(loss_config["teacher_topk_ipc_k"])
    if teacher_topk_ipc_k < 0:
        raise ValueError(
            f"loss_fn.teacher_topk_ipc_k must be >= 0, got {teacher_topk_ipc_k}"
        )
    teacher_topk_support_mode = str(loss_config["teacher_topk_ipc_support_mode"])
    if teacher_topk_support_mode != "row_topk":
        raise ValueError(
            "loss_fn.teacher_topk_ipc_support_mode currently supports only "
            f"'row_topk', got {teacher_topk_support_mode!r}"
        )
    if teacher_topk_ipc_k > 0:
        has_sparse_teacher = any(
            teacher.aligner.projection_matrix_path is not None for teacher in teachers
        )
        if has_sparse_teacher and loss_config["sum_weights_metric"] is not None:
            raise ValueError(
                "Sparse teacher IPC cannot be combined with sum_weights_metric; "
                "dynamic teacher scoring requires full logits."
            )
        if has_sparse_teacher and loss_config["kd_loss_mode"] == "select_teacher":
            raise ValueError(
                "Sparse teacher IPC cannot be combined with "
                "kd_loss_mode='select_teacher'; teacher selection requires full logits."
            )
        for i, (teacher, teacher_config) in enumerate(
            zip(teachers, teacher_configs, strict=True)
        ):
            if teacher.aligner.projection_matrix_path is None:
                continue
            dtensor_cfg = teacher_config["dtensor_cfg"]
            if not (dtensor_cfg["enabled"] and dtensor_cfg.get("_v2", False)):
                raise ValueError(
                    "Sparse teacher IPC currently requires DTensor-V2 for each "
                    f"cross-tokenizer teacher; teachers[{i}] is not DTensor-V2."
                )

    # A null projection path marks a same-tokenizer teacher (direct KL, no
    # projection/alignment). Reuse is safe only when the complete tokenizer
    # semantics match; equal vocabulary size alone does not prove that token IDs
    # or chat rendering have the same meaning.
    validate_xtoken_tokenizer_reuse(
        master_config, student_tokenizer, teacher_tokenizers
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
        if teacher.aligner.projection_matrix_path is None
        else TokenAligner(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizers[i],
            projection_matrix_path=teacher.aligner.projection_matrix_path,
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
        drop_first_assistant_chunk_kl_by_teacher=[
            teacher.aligner.drop_first_assistant_chunk_kl for teacher in teachers
        ],
        # Shared chat/instruct knobs; default to the raw-text path.
        mode=data_config.get("collator_mode", "text"),
        include_thinking_in_loss=data_config.get("include_thinking_in_loss", False),
        native_thinking_alignment=data_config.get("native_thinking_alignment", False),
        kd_alignment_regions=data_config.get("kd_alignment_regions", None),
        num_packed_rows=data_config.get("num_packed_rows", 1),
        require_routed_experts=router_replay_enabled(policy_config),
        student_chat_template_kwargs=policy_config["tokenizer"].get(
            "chat_template_kwargs"
        ),
        teacher_chat_template_kwargs=[
            tc["tokenizer"].get("chat_template_kwargs") or {} for tc in teacher_configs
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

    # Megatron uses train_iters to configure its scheduler horizon. Derive it
    # from the same bounds that terminate the training loop so it stays in sync
    # with the actual dataloader rather than becoming a stale YAML constant.
    total_train_iters = min(
        distillation_config["max_num_steps"],
        distillation_config["max_num_epochs"] * len(train_dataloader),
    )
    for tc in teacher_configs:
        if "megatron_cfg" in tc and tc["megatron_cfg"]["enabled"]:
            tc["megatron_cfg"]["train_iters"] = total_train_iters
    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

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
    # Finish the student's lazy worker/model construction while every teacher
    # is still offloaded. Without this barrier, colocated Megatron student
    # construction can overlap the first teacher forward and OOM at its peak.
    # Megatron-FSDP cannot CPU-offload its shards in the current MCore version,
    # but its steady-state resident shard is much smaller than construction.
    student_policy.prepare_for_training()
    if distillation_config.get("offload_student_after_step", False):
        # Backends that support student CPU offload should start the first
        # teacher phase in the same offloaded state used between later steps.
        student_policy.offload_after_refit()

    # ==========================
    #   Teacher/student grid
    # ==========================
    # Teacher and student may differ in DP and MBS, but they must agree on the
    # global batch size (one dataloader batch feeds both, in the same global
    # order) and tile it cleanly into per-DP-rank chunks and whole microbatches.
    # assert_teacher_student_batch_grid checks both (GBS agreement + tiling).
    student_dp = student_policy.data_parallel_size
    if configured_packing_dp is not None and student_dp != configured_packing_dp:
        raise RuntimeError(
            "xToken lockstep packing configured DP changed during worker setup: "
            f"expected {configured_packing_dp}, student reports {student_dp}."
        )
    student_tp, student_cp, student_pp = _xtoken_entity_parallelism(
        policy_config, label="policy"
    )
    # Each teacher may differ from the student (and from each other) in
    # DP/MBS/TP/CP, so check the batch grid and node-local IPC layout per
    # teacher. Train and validation share the grid (the student reuses its train
    # MBS in eval mode and each teacher's val export reuses its own train MBS),
    # so one check per teacher covers both.
    for i, (teacher_policy, tc) in enumerate(zip(teacher_policies, teacher_configs)):
        teacher_dp = teacher_policy.data_parallel_size
        if configured_packing_dp is not None and teacher_dp != configured_packing_dp:
            raise RuntimeError(
                "xToken lockstep packing configured DP changed during worker setup: "
                f"expected {configured_packing_dp}, teachers[{i}] reports "
                f"{teacher_dp}."
            )
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
        # would read teacher shards from another node. The teacher's TP/CP/PP
        # are read per backend (DTensor reports PP=1), and the model-parallel
        # group is tp * cp * pp — PP counts because the producing last stage
        # must land on the same node as the consuming student ranks.
        teacher_tp, teacher_cp, teacher_pp = _xtoken_entity_parallelism(
            tc, label=f"teachers[{i}]"
        )
        assert_xtoken_ipc_node_local(
            num_nodes=cluster_config["num_nodes"],
            gpus_per_node=cluster_config["gpus_per_node"],
            student_tp=student_tp,
            student_cp=student_cp,
            student_pp=student_pp,
            teacher_tp=teacher_tp,
            teacher_cp=teacher_cp,
            teacher_pp=teacher_pp,
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
    # axes exactly. Dense IPC ships full logits; sparse IPC ships teacher top-k
    # logits plus logZ and lets the loss reconstruct only the required support.
    loss_config = {
        **loss_config,
        "student_vocab_size": len(student_tokenizer),
        "teacher_vocab_sizes": [len(tok) for tok in teacher_tokenizers],
        "projection_matrix_paths": [
            teacher.aligner.projection_matrix_path for teacher in teachers
        ],
        "teacher_weights": [t.weight for t in teachers],
        # v6 pseudo-target tables (student<->teacher sub-token chains) per
        # cross-tokenizer teacher; None for same-tokenizer teachers.
        "pseudo_target_paths": [t.aligner.pseudo_target_path for t in teachers],
        "reverse_pseudo_target_paths": [
            t.aligner.reverse_pseudo_target_path for t in teachers
        ],
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


def _batch_length_tuple(batch: BatchedDataDict[Any], key: str) -> tuple[int, ...]:
    value = batch.get(key)
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError(
            f"xToken lockstep packing requires {key!r} as a rank-1 tensor of "
            f"exact untruncated lengths, got {type(value).__name__}."
        )
    lengths = tuple(int(length) for length in value.detach().cpu().tolist())
    if len(lengths) != batch.size or any(length <= 0 for length in lengths):
        raise ValueError(
            f"xToken lockstep packing received invalid {key}: {lengths}; expected "
            f"{batch.size} positive logical-sample lengths."
        )
    return lengths


def _effective_lengths(
    raw_lengths: tuple[int, ...], *, divisor: int
) -> tuple[int, ...]:
    divisor = _positive_config_int(divisor, field="sequence length divisor")
    return tuple(
        ((length + divisor - 1) // divisor) * divisor for length in raw_lengths
    )


def _fixed_tail_physical_size(
    effective_lengths: tuple[int, ...], *, capacity: int
) -> int:
    """Keep DTensor TP's fixed packed tail without hiding an oversized bin."""
    return max(capacity, sum(effective_lengths))


def _bind_semantic_regions_to_occurrences(
    batch: BatchedDataDict[Any], batch_item_ids: tuple[int, ...]
) -> None:
    """Attach occurrence identity to side-local chat semantic-region records."""
    for key in tuple(batch.keys()):
        if key != "student_semantic_regions" and not (
            key.startswith("teacher_") and key.endswith("_semantic_regions")
        ):
            continue
        per_sample = batch[key]
        if not isinstance(per_sample, Sequence) or isinstance(per_sample, (str, bytes)):
            raise ValueError(f"{key} must contain one semantic-region list per row")
        if len(per_sample) != len(batch_item_ids):
            raise ValueError(
                f"{key} has {len(per_sample)} rows for "
                f"{len(batch_item_ids)} batch occurrences"
            )
        bound_rows: list[tuple[tuple[object, ...], ...]] = []
        for batch_item_id, regions in zip(batch_item_ids, per_sample, strict=True):
            bound_regions: list[tuple[object, ...]] = []
            for region in regions:
                if not isinstance(region, Sequence) or isinstance(region, (str, bytes)):
                    raise ValueError(f"{key} contains a non-sequence region")
                region_tuple = tuple(region)
                if len(region_tuple) == 5:
                    bound_regions.append((batch_item_id, *region_tuple))
                elif len(region_tuple) == 6 and region_tuple[0] == batch_item_id:
                    bound_regions.append(region_tuple)
                else:
                    raise ValueError(
                        f"{key} region must be (turn, role, name, start, end) "
                        "before binding or begin with its exact batch_item_id"
                    )
            bound_rows.append(tuple(bound_regions))
        batch[key] = bound_rows


def build_xtoken_lockstep_packing_plan(
    batch: BatchedDataDict[Any],
    master_config: MasterConfig,
    *,
    batch_uid: int,
    data_parallel_size: int,
) -> Optional[LockstepPackingPlan]:
    """Assign occurrence IDs and build the one controller-owned packing plan."""
    if not _sequence_packing_enabled(master_config.policy):
        return None
    if batch_uid < 0 or batch_uid >= 2**31:
        raise ValueError(
            f"batch_uid must fit the non-negative high 31 bits of int64, got {batch_uid}."
        )

    sample_ids_value = batch.get("sample_id")
    if torch.is_tensor(sample_ids_value):
        sample_ids: Sequence[object] = sample_ids_value.detach().cpu().tolist()
    elif isinstance(sample_ids_value, Sequence) and not isinstance(
        sample_ids_value, (str, bytes)
    ):
        sample_ids = sample_ids_value
    else:
        raise ValueError(
            "xToken lockstep packing requires a durable sample_id for every "
            "logical row before occurrence IDs are assigned."
        )
    if len(sample_ids) != batch.size:
        raise ValueError(
            f"sample_id has {len(sample_ids)} entries for batch size {batch.size}."
        )

    batch_item_ids = tuple((batch_uid << 32) | ordinal for ordinal in range(batch.size))
    length_device = batch["input_lengths"].device
    batch["batch_item_id"] = torch.tensor(
        batch_item_ids, dtype=torch.long, device=length_device
    )
    _bind_semantic_regions_to_occurrences(batch, batch_item_ids)
    items = tuple(
        LockstepPackingItem(sample_id=sample_id, batch_item_id=batch_item_id)
        for sample_id, batch_item_id in zip(sample_ids, batch_item_ids, strict=True)
    )

    student_config = master_config.policy
    student_raw_lengths = _batch_length_tuple(batch, "input_lengths")
    student_divisor = int(student_config["make_sequence_length_divisible_by"])
    student_packing = student_config["sequence_packing"]
    student_capacity = int(student_packing["train_mb_tokens"])
    student_physical_size_fn = None
    if (
        student_config["dtensor_cfg"]["enabled"]
        and int(student_config["dtensor_cfg"]["tensor_parallel_size"]) > 1
    ):
        # Generic Automodel packing pads TP>1 training microbatches to the
        # configured token budget. Preserve that requirement in the shared
        # plan so controller materialization cannot bypass it.
        student_physical_size_fn = partial(
            _fixed_tail_physical_size, capacity=student_capacity
        )
    side_specs = [
        SidePackingSpec(
            side_id="student",
            capacity=student_capacity,
            raw_lengths=student_raw_lengths,
            effective_lengths=_effective_lengths(
                student_raw_lengths, divisor=student_divisor
            ),
            physical_size_fn=student_physical_size_fn,
        )
    ]
    for teacher_idx, teacher in enumerate(master_config.teachers):
        teacher_config = teacher.policy_config()
        if teacher.aligner.projection_matrix_path is None:
            teacher_raw_lengths = student_raw_lengths
        else:
            teacher_raw_lengths = _batch_length_tuple(
                batch, f"teacher_{teacher_idx}_input_lengths"
            )
        teacher_packing = teacher_config["sequence_packing"]
        side_specs.append(
            SidePackingSpec(
                side_id=f"teacher_{teacher_idx}",
                capacity=int(teacher_packing["logprob_mb_tokens"]),
                raw_lengths=teacher_raw_lengths,
                effective_lengths=_effective_lengths(
                    teacher_raw_lengths,
                    divisor=int(teacher_config["make_sequence_length_divisible_by"]),
                ),
            )
        )

    return build_lockstep_packing_plan(
        batch_uid=batch_uid,
        items=items,
        sides=side_specs,
        data_parallel_size=data_parallel_size,
    )


def log_xtoken_packing_telemetry(
    plan: LockstepPackingPlan, batch: BatchedDataDict[Any]
) -> None:
    """Emit controller-side packing evidence without conflating bins and samples."""
    sample_mask = batch["sample_mask"]
    if torch.is_tensor(sample_mask):
        masked_samples = int((sample_mask == 0).sum().item())
    else:
        masked_samples = sum(float(value) == 0.0 for value in sample_mask)
    multi_sample_bins = sum(len(bin_items) > 1 for bin_items in plan.bins)
    split_count = plan.equalization_splits
    print(
        "XTOKEN_PACKING "
        f"batch_uid={plan.batch_uid} logical_samples="
        f"{len(plan.canonical_batch_item_ids)} physical_bins={len(plan.bins)} "
        f"multi_sample_bins={multi_sample_bins} masked_logical_samples="
        f"{masked_samples} equalization_splits={split_count} transport=dense",
        flush=True,
    )
    for side_id, side in plan.sides.items():
        raw_tokens = sum(side.raw_lengths)
        effective_tokens = sum(side.effective_lengths)
        physical_tokens = sum(side.physical_tokens_by_bin)
        available_tokens = side.capacity * len(side.physical_tokens_by_bin)
        utilization = physical_tokens / available_tokens
        print(
            "XTOKEN_PACKING_SIDE "
            f"batch_uid={plan.batch_uid} side={side_id} bins_per_dp_rank="
            f"{tuple(len(indices) for indices in side.rank_bin_indices)} "
            f"raw_tokens={raw_tokens} effective_tokens={effective_tokens} "
            f"physical_tokens={physical_tokens} capacity_tokens={available_tokens} "
            f"utilization={utilization:.6f} padding_waste="
            f"{physical_tokens - raw_tokens} tail_padding_waste="
            f"{physical_tokens - effective_tokens}",
            flush=True,
        )


def _dense_ipc_logical_bytes(handles: Sequence[Mapping[str, Any]]) -> int:
    """Return the logical dense-logit bytes addressable by aggregated handles."""
    total = 0
    for sample_index, sample in enumerate(handles):
        shards = sample.get("teacher_shards")
        if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
            raise ValueError(
                "Dense IPC telemetry requires teacher_shards for sample "
                f"{sample_index}."
            )
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise TypeError(
                    "Dense IPC telemetry requires mapping shard records, got "
                    f"{type(shard).__name__}."
                )
            actual_shape = shard.get("actual_shape")
            dtype = shard.get("dtype")
            if (
                not isinstance(actual_shape, (list, tuple, torch.Size))
                or len(actual_shape) != 2
                or not isinstance(dtype, torch.dtype)
            ):
                raise ValueError(
                    "Dense IPC telemetry requires a two-dimensional actual_shape "
                    f"and torch dtype, got {actual_shape!r}/{dtype!r}."
                )
            total += (
                math.prod(int(value) for value in actual_shape)
                * torch.empty((), dtype=dtype).element_size()
            )
    return total


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
    packing_plan: Optional[LockstepPackingPlan] = None,
) -> BatchedDataDict[Any]:
    """Serially run each teacher's forward and pack the student ``train_data``.

    Teachers run one at a time (collocated): each is onloaded for inference,
    forwarded, then offloaded. Cross-tokenizer teachers use sparse top-k + logZ
    IPC when ``teacher_topk_ipc_k > 0`` and otherwise use full-vocab IPC;
    same-vocab teachers always retain full logits for their direct-KL path. A
    cross-tokenizer teacher's own tokenization and ``alignment_{i}_*`` payload
    ride along (teacher-indexed). The persistent IPC buffers stay resident on
    the teacher GPUs until released by the caller after ``student.train``.
    Shared by the train loop and ``validate`` so the forward+pack sequence can't
    drift between them.
    """
    train_data: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "input_lengths": batch["input_lengths"],
        "token_mask": batch["token_mask"],
        "kd_token_mask": batch.get("kd_token_mask", batch["token_mask"]),
        "sample_mask": batch["sample_mask"],
    }
    if "routed_experts" in batch:
        train_data["routed_experts"] = batch["routed_experts"]
    if packing_plan is not None:
        train_data["sample_id"] = batch["sample_id"]
        train_data["batch_item_id"] = batch["batch_item_id"]
        if "student_semantic_regions" in batch:
            train_data["student_semantic_regions"] = batch["student_semantic_regions"]
    for i, teacher_policy in enumerate(teacher_policies):
        same_vocab = loss_fn.projection_matrix_paths[i] is None
        semantic_key = f"teacher_{i}_semantic_regions"
        if packing_plan is not None and semantic_key in batch:
            train_data[semantic_key] = batch[semantic_key]
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

        if packing_plan is not None:
            teacher_data["sample_id"] = batch["sample_id"]
            teacher_data["batch_item_id"] = batch["batch_item_id"]

        teacher_policy.prepare_for_lp_inference()
        force_include_token_ids = (
            None
            if same_vocab
            else _build_teacher_force_include_token_ids(
                batch,
                teacher_idx=i,
                loss_config=loss_fn.cfg,
            )
        )
        ipc_suffix, handles = _get_teacher_logits_ipc(
            teacher_policy,
            teacher_data,
            loss_fn,
            teacher_idx=i,
            micro_batch_size=teacher_mbs[i],
            force_include_token_ids=force_include_token_ids,
            timer=timer,
            packing_plan=packing_plan,
            packing_side_id=f"teacher_{i}" if packing_plan is not None else None,
        )
        train_data[f"teacher_{i}_{ipc_suffix}"] = handles
        if packing_plan is not None:
            if ipc_suffix != "full_logits_ipc":
                raise ValueError(
                    "xToken lockstep telemetry encountered unsupported packed "
                    f"transport {ipc_suffix!r}."
                )
            ipc_bytes = _dense_ipc_logical_bytes(handles)
            print(
                "XTOKEN_IPC "
                f"batch_uid={packing_plan.batch_uid} teacher=teacher_{i} "
                f"transport=dense ipc_bytes={ipc_bytes}",
                flush=True,
            )
        # Free the teacher's PARAMS to CPU; the persistent IPC buffers live in
        # worker state and survive this call.
        teacher_policy.offload_after_refit()

    return BatchedDataDict(train_data)


def _packing_batch_uids_from_state(
    state: OffPolicyDistillationSaveState,
) -> Iterator[int]:
    """Yield occurrence namespaces from the checkpointed high-water mark.

    A single persisted counter is shared by training and every validation batch.
    This keeps ``batch_item_id`` values unique when validation contains several
    batches and when the loop resumes from a checkpoint. Older checkpoints do
    not carry the field, so they retain the former train-derived starting point.
    """
    if "next_packing_batch_uid" not in state:
        state["next_packing_batch_uid"] = int(state["total_steps"]) * 2
    while True:
        batch_uid = state["next_packing_batch_uid"]
        if isinstance(batch_uid, bool) or not isinstance(batch_uid, int):
            raise ValueError(
                f"next_packing_batch_uid must be an integer, got {batch_uid!r}."
            )
        if batch_uid < 0 or batch_uid >= 2**31:
            raise ValueError(
                "next_packing_batch_uid must fit the non-negative high 31 bits "
                f"of int64, got {batch_uid}."
            )
        state["next_packing_batch_uid"] = batch_uid + 1
        yield batch_uid


def _restore_student_between_teacher_state(
    student_policy: Policy,
    master_config: MasterConfig,
) -> None:
    """Restore the configured student residency between teacher forwards."""
    if master_config.distillation.get("offload_student_after_step", False):
        student_policy.offload_after_refit()
    elif master_config.policy.get("offload_optimizer_for_logprob", False):
        student_policy.offload_before_refit()


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
    offload_student_after_step = bool(
        distill_cfg.get("offload_student_after_step", False)
    )
    offload_student_optimizer_after_step = bool(
        master_config.policy.get("offload_optimizer_for_logprob", False)
    )
    if offload_student_after_step:
        print(
            "Student post-step offload enabled: the student model/optimizer are "
            "offloaded after each optimizer step, before the next teacher pass.",
            flush=True,
        )
    elif offload_student_optimizer_after_step:
        print(
            "Student post-step optimizer offload enabled: optimizer state is "
            "offloaded before the next teacher pass while model shards remain resident.",
            flush=True,
        )
    # Per-teacher export MBS (each teacher's own train MBS) and the
    # non-student-seq keys the worker's check_sequence_dim must skip
    # (teacher-count-dependent, so built from the loss fn).
    teacher_mbs = [
        t.policy_config()["train_micro_batch_size"] for t in master_config.teachers
    ]
    skip_keys = xtoken_non_student_seq_keys(loss_fn)
    # Training and validation draw from one checkpointed high-water mark. A
    # step-derived train counter is insufficient because one validation pass can
    # consume many global batches before the next checkpoint/resume boundary.
    packing_batch_uids = _packing_batch_uids_from_state(off_policy_distillation_state)

    if val_at_start and total_steps == 0 and val_dataloader is not None:
        val_metrics, val_timings = validate(
            student_policy,
            teacher_policies,
            val_dataloader,
            loss_fn,
            master_config,
            skip_keys=skip_keys,
            timer=timer,
            packing_batch_uids=packing_batch_uids,
        )
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(val_timings, total_steps, prefix="timing/validation")

    ft_save_period = master_config.checkpointing.get("ft_save_period")

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

            batch_uid = next(packing_batch_uids)
            log_xtoken_logical_batch_digest(batch, batch_uid=batch_uid)
            packing_plan = build_xtoken_lockstep_packing_plan(
                batch,
                master_config,
                batch_uid=batch_uid,
                data_parallel_size=student_policy.data_parallel_size,
            )
            if packing_plan is not None:
                log_xtoken_packing_telemetry(packing_plan, batch)

            with timer.time("total_step_time"):
                with timer.time("teacher_forward"):
                    # Serial per-teacher forward; each cross-tokenizer teacher
                    # selects dense or sparse IPC from the loss config while a
                    # same-vocab teacher keeps full logits. The per-teacher
                    # alignment payload is packed alongside the handles.
                    train_data = export_teacher_logits_and_pack(
                        teacher_policies,
                        loss_fn,
                        batch,
                        teacher_mbs,
                        timer=timer,
                        packing_plan=packing_plan,
                    )

                with timer.time("training_prep"):
                    student_policy.prepare_for_training()

                with timer.time("policy_training"):
                    try:
                        packing_kwargs: dict[str, Any] = {}
                        if packing_plan is not None:
                            packing_kwargs = {
                                "packing_plan": packing_plan,
                                "packing_side_id": "student",
                            }
                        train_results = student_policy.train(
                            train_data,
                            loss_fn,
                            timer=timer,
                            check_dim_skip_keys=skip_keys,
                            **packing_kwargs,
                        )
                        if packing_plan is not None:
                            _log_dense_ipc_reconstruction_telemetry(
                                train_results.get("all_mb_metrics", {}),
                                packing_plan=packing_plan,
                                num_teachers=loss_fn.num_teachers,
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

                if offload_student_after_step:
                    with timer.time("student_step_offload"):
                        _restore_student_between_teacher_state(
                            student_policy, master_config
                        )
                elif offload_student_optimizer_after_step:
                    with timer.time("student_optimizer_offload"):
                        _restore_student_between_teacher_state(
                            student_policy, master_config
                        )

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
                        skip_keys=skip_keys,
                        timer=timer,
                        packing_batch_uids=packing_batch_uids,
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
                # the gold-loss path emits kl_common/l1_uncommon; the v6 path
                # emits the *_per_chunk diagnostics. Any set may be present —
                # reduce all via the same rules.
                for k, v in metrics.items():
                    metrics[k] = reduce_mb_metric(k, v)
                if "global_valid_toks" in metrics:
                    total_valid_tokens += int(metrics["global_valid_toks"])

                consumed_samples += distill_cfg["num_prompts_per_step"]
                timeout.mark_iteration()

                # ===== Checkpointing =====
                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
                    or (
                        ft_save_period is not None
                        and (total_steps + 1) % ft_save_period == 0
                    )
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
                        checkpointer.begin_finalization(
                            ckpt_path,
                            wait_fn=student_policy.finalize_async_save,
                        )
                    _restore_student_between_teacher_state(
                        student_policy, master_config
                    )

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
            print(f"  • GradNorm: {metrics['grad_norm']:.4f}", flush=True)
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
                checkpointer.shutdown()
                print("Timeout reached, stopping training early.", flush=True)
                for teacher_policy in teacher_policies:
                    teacher_policy.release_ipc_buffer()
                return
            if total_steps >= max_steps:
                checkpointer.shutdown()
                print("Max steps reached, stopping training.", flush=True)
                for teacher_policy in teacher_policies:
                    teacher_policy.release_ipc_buffer()
                return

        current_epoch += 1
        current_step = 0
    # Flush the last checkpoint's background finalization on an epoch-bounded
    # exit. Reaching max_epochs falls through the while loop and bypasses the
    # inline shutdown() calls at the max_steps / timeout early returns, so
    # without this the daemon finalization thread could be killed before the
    # final tmp_step_N is renamed.
    checkpointer.shutdown()
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
    skip_keys: frozenset[str],
    timer: Optional[Timer] = None,
    packing_batch_uids: Optional[Iterator[int]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Held-out KL/CE on a validation dataloader.

    Reuses the same per-step path as training, but in eval mode so no
    backward / optimizer step runs. Returns mean train-style metrics.

    ``skip_keys`` is the non-student-seq-axis key set for the worker's
    ``check_sequence_dim`` pre-flight; built once in the train loop and threaded
    in so it can't drift from a parallel rebuild here.
    """
    distill_cfg = master_config.distillation
    timer = timer if timer is not None else Timer()
    if packing_batch_uids is None:
        packing_batch_uids = count()

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

    with timer.time("validation_total"):
        for batch in val_dataloader:
            target_size = math.ceil(batch.size / pad_quantum) * pad_quantum
            batch = pad_distillation_val_batch(batch, target_size)
            packing_plan = build_xtoken_lockstep_packing_plan(
                batch,
                master_config,
                batch_uid=next(packing_batch_uids),
                data_parallel_size=student_policy.data_parallel_size,
            )
            if packing_plan is not None:
                log_xtoken_packing_telemetry(packing_plan, batch)

            train_data = export_teacher_logits_and_pack(
                teacher_policies,
                loss_fn,
                batch,
                teacher_mbs,
                timer=timer,
                packing_plan=packing_plan,
            )
            student_policy.prepare_for_training()
            try:
                packing_kwargs: dict[str, Any] = {}
                if packing_plan is not None:
                    packing_kwargs = {
                        "packing_plan": packing_plan,
                        "packing_side_id": "student",
                    }
                results = student_policy.train(
                    train_data,
                    loss_fn,
                    eval_mode=True,
                    gbs=target_size,
                    check_dim_skip_keys=skip_keys,
                    **packing_kwargs,
                )
                _restore_student_between_teacher_state(student_policy, master_config)
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
