# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import collections
import os
from typing import Annotated, Any, Literal, Mapping, NotRequired, Optional, TypedDict

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from torch.distributed.tensor import DTensor

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType, LossType
from nemo_rl.algorithms.utils import masked_mean
from nemo_rl.algorithms.x_token.loss_utils import (
    DenseTeacherRowReader,
    LocalizedAlignment,
    SparseTeacherLogits,
    SparseTeacherRowReader,
    TeacherRowPayload,
    build_exact_token_map,
    ce_label_mask,
    cp_load_balanced_global_positions,
    next_token_accuracy,
    same_tokenizer_full_vocab_kl,
    same_tokenizer_topk_kl,
    student_next_token_ce,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_contiguous_tensor,
    chunked_vocab_parallel_logsumexp,
    distributed_selected_logprobs,
    get_next_token_logprobs_from_logits,
    group_all_reduce_sum,
    group_all_reduce_sum_with_grad,
    group_all_reduce_sum_with_grad_backward_sum,
    to_local_if_dtensor,
    vocab_parallel_argmax,
)

CSD_STUDENT_LOSS_CHUNK_SIZE = 256


def _generalized_jsd(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Per-element generalized Jensen-Shannon Divergence on log-probabilities.

    Mirrors TRL's ``GOLDTrainer.generalized_jsd_loss`` (huggingface/trl @v1.0.0
    ``trl/experimental/gold/gold_trainer.py``). Inputs are log-probs (may be
    sub-distribution slices - no normalization is applied here, so the
    divergence is on raw probability mass).

    beta semantics (matches TRL):
      * beta = 0   -> forward KL : KL(teacher || student)   (mean-seeking)
      * beta = 1   -> reverse KL : KL(student || teacher)   (mode-seeking; TRL's default)
      * beta = 0.5 -> symmetric JSD
      * 0 < beta < 1 -> blended KL on a (1-beta).student + beta.teacher mixture

    Numerics: computation is forced to fp32 even when inputs are bf16/fp16.
    bf16's ~7.8e-3 epsilon causes (student_log_probs - teacher_log_probs) to
    underflow as the student converges, collapsing the matched-term loss to
    exactly 0 mid-training. Cast back to the input dtype on return.

    Returns a tensor with the same shape as the inputs (per-element divergence,
    matching F.kl_div(reduction='none')). Caller does the chunk/vocab sum.
    """
    in_dtype = student_log_probs.dtype
    s_lp = student_log_probs.to(torch.float32)
    t_lp = teacher_log_probs.to(torch.float32)

    if beta == 0.0:
        out = torch.nn.functional.kl_div(
            s_lp,
            t_lp,
            reduction="none",
            log_target=True,
        )
        return out.to(in_dtype)
    if beta == 1.0:
        out = torch.nn.functional.kl_div(
            t_lp,
            s_lp,
            reduction="none",
            log_target=True,
        )
        return out.to(in_dtype)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=s_lp.device)
    # mixture_log_probs[v] = log( (1-beta)*p_s(v) + beta*p_t(v) )
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                s_lp + torch.log1p(-beta_t),
                t_lp + torch.log(beta_t),
            ]
        ),
        dim=0,
    )
    kl_teacher = torch.nn.functional.kl_div(
        mixture_log_probs,
        t_lp,
        reduction="none",
        log_target=True,
    )
    kl_student = torch.nn.functional.kl_div(
        mixture_log_probs,
        s_lp,
        reduction="none",
        log_target=True,
    )
    out = beta_t * kl_teacher + (1 - beta_t) * kl_student
    return out.to(in_dtype)


class CSDLossConfig(BaseModel, extra="allow"):
    """CSD loss options and runtime-injected per-teacher metadata.

    Forward tables map student tokens to teacher sub-token chains; reverse
    tables map teacher tokens to student chains. Optional projection matrices
    provide the common-token mapping when ``common_indices_from_subtoks=False``.
    """

    temperature: Annotated[float, Field(gt=0)] = 1.0
    vocab_topk: Annotated[int, Field(gt=0)] = 8192  # Same-tokenizer direct KL only.
    reverse_kl: bool = False
    kl_loss_weight: float = 1.0
    ce_loss_scale: float = 0.1
    dynamic_loss_scaling: bool = True
    kd_loss_mode: Literal["sum", "averaged_logits", "select_teacher"] = "sum"
    normalize_teacher_by_vocab: bool = False
    alpha: Annotated[float, Field(gt=0)] = 1.0
    sum_weights_metric: Optional[Literal["ce", "entropy", "max_prob"]] = None

    # Runtime-injected by setup; these are not user loss_fn fields.
    student_vocab_size: Optional[int] = None
    teacher_vocab_sizes: list[int] = Field(default_factory=list)
    projection_matrix_paths: Optional[list[Optional[str]]] = None
    teacher_weights: list[float] = Field(default_factory=list)
    student_to_teacher_table_paths: Optional[list[Optional[str]]] = None
    teacher_to_student_table_paths: Optional[list[Optional[str]]] = None

    # Shift each noninitial chunk to predictor positions before computing KD.
    kl_chunk_shift: bool = True
    # Build exact common-token pairs from forward tables instead of projections.
    common_indices_from_subtoks: bool = False
    csd_loss_fn: Literal["kl", "jsd", "bce"] = "jsd"
    csd_jsd_beta: Annotated[float, Field(ge=0, le=1)] = 0.5
    # None inherits csd_loss_fn for the last-position partition.
    csd_last_pos_loss_fn: Optional[Literal["kl", "jsd", "bce"]] = None
    # Multiplier on each mismatch (non-1-to-1) chunk loss relative to a
    # 1-to-1 chunk loss.
    csd_mismatch_loss_beta: Annotated[float, Field(ge=0)] = 5.0
    # Zero disables filtering; otherwise retain realized labels in teacher top-k.
    csd_noise_filter_topk: Annotated[int, Field(ge=0)] = 0
    # Compare only the realized chain probability and its complementary bucket.
    csd_pure_alm: bool = False
    # Score 1-to-many and many-to-1 chunks by their realized token chains
    # against REST instead of prefix-support alternatives.
    csd_realized_only_mismatch: bool = False
    csd_alm_bce_tau: Annotated[float, Field(gt=0)] = 1.0
    # Positive k sends each teacher row's top-k logits plus its full-vocab logZ
    # (the KD then compares top-k plus a REST bucket); zero sends dense logits.
    teacher_topk_ipc_k: Annotated[int, Field(ge=0)] = 128
    teacher_topk_ipc_support_mode: Literal["row_topk"] = "row_topk"
    teacher_topk_ipc_keep_realized: bool = True
    # Owner-local evaluation (eligible Megatron students): selected student or
    # teacher rows per tile, and alignment chunks evaluated per batch. Both
    # bound memory only; they do not change the loss.
    csd_selected_row_chunk_size: Annotated[int, Field(gt=0)] = 128
    csd_support_chunk_batch_size: Annotated[int, Field(gt=0)] = 16

    # Runtime-injected by setup: evaluate cross-tokenizer KD on the context-
    # parallel rank that owns each chunk and return a CP-partial loss.
    owner_local: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_position_0_keys(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for key in (
                "csd_position_0_kl",
                "csd_mismatch_pos0_alpha",
                "csd_mismatch_pos0_weight",
                "csd_mismatch_loss_scale",
            ):
                if key in value:
                    raise ValueError(
                        f"loss_fn.{key} was removed: there is no position-0 "
                        "term, and mismatch chunks are weighted by "
                        "csd_mismatch_loss_beta alone."
                    )
        return value


def owner_local_config_violations(cfg: CSDLossConfig) -> list[str]:
    """CSD loss settings that owner-local evaluation does not support.

    An empty list means the loss can run owner-local. Owner-local KD keeps a
    static-weight sum of per-teacher terms, evaluates common chunks over a
    multi-token support (so BCE, which needs a binary partition, is allowed
    only for realized-only mismatch chunks), and with sparse teacher IPC
    requires every request's realized label to be kept.
    """
    violations: list[str] = []
    if cfg.kd_loss_mode != "sum":
        violations.append(f"kd_loss_mode={cfg.kd_loss_mode!r} (needs 'sum')")
    if cfg.sum_weights_metric is not None:
        violations.append(
            f"sum_weights_metric={cfg.sum_weights_metric!r} (needs static weights)"
        )
    if cfg.csd_pure_alm:
        violations.append("csd_pure_alm=true")
    if cfg.csd_loss_fn == "bce":
        violations.append("csd_loss_fn='bce' (common chunks are not binary)")
    if cfg.csd_last_pos_loss_fn == "bce" and not cfg.csd_realized_only_mismatch:
        violations.append(
            "csd_last_pos_loss_fn='bce' without csd_realized_only_mismatch=true"
        )
    if (
        cfg.teacher_topk_ipc_k > 1
        and not cfg.teacher_topk_ipc_keep_realized
        and not cfg.csd_realized_only_mismatch
    ):
        violations.append(
            "teacher_topk_ipc_keep_realized=false with teacher_topk_ipc_k > 1"
        )
    return violations


class CSDLossDataDict(TypedDict):
    """Student-side keys are fixed; teacher-side keys are teacher-indexed.

    Only the student keys below are static. Each teacher ``i`` contributes a
    dynamic set of keys produced by ``CrossTokenizerCollator`` / the trainer and
    so cannot be enumerated here:

    - Every teacher contributes exactly one logits transport. Same-tokenizer
      teachers and dense cross-tokenizer teachers use
      ``teacher_{i}_full_logits_ipc`` (List[B] of CUDA IPC handle dicts with
      TP/CP shard metadata). Sparse cross-tokenizer teachers use
      ``teacher_{i}_sparse_logits_ipc`` with row-top-k logits/token ids, the
      exact full-vocabulary log-normalizer, and optional realized-label
      membership. ``prepare_loss_input`` P2P-reads and reassembles either form.
    - Cross-tokenizer teacher only: ``teacher_{i}_input_ids`` /
      ``teacher_{i}_token_mask`` ``[B, T_t]`` and ``alignment_{i}_*``
      (``pair_valid`` / ``pair_is_correct`` ``[B, max_pairs]``;
      ``student_chunk_id`` ``[B, T_s]``; ``teacher_chunk_id`` ``[B, T_t]``;
      partition masks; ``num_chunks``).
    - Same-tokenizer teacher: no ``teacher_{i}_input_ids`` / ``alignment_{i}_*``;
      it reuses the student tokenization (identity 1:1 aligned).
    """

    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    kd_token_mask: NotRequired[torch.Tensor]
    sample_mask: torch.Tensor


class CSDLossFn(LossFunction):
    """CSD cross-tokenizer distillation loss.

    A cross-tokenizer teacher's KD term is computed by ``_compute_csd_loss``: a common-vocab partition KL/JSD
    on the exact 1-to-1 token map plus, for non-1-to-1 chunks, a batched
    mismatch partition KL over prefix-support alternatives (MtoM-as-ALM + rest
    bucket) weighted by ``csd_mismatch_loss_beta``. It is combined with a
    next-token student CE term as ``kl_loss_weight * kd + ce_loss_scale * ce`` —
    or, when ``dynamic_loss_scaling`` is set, with the KD term rescaled each
    step to match the detached CE magnitude.

    Multi-teacher: ``setup`` injects per-teacher metadata (projection paths,
    weights, vocab sizes, pseudo-target-table paths). The per-teacher KD terms
    are aggregated by ``kd_loss_mode`` (``sum`` / ``averaged_logits`` /
    ``select_teacher``) and combined with a single student CE term. A teacher
    without CSD tables or a projection is a *same-tokenizer* teacher: alignment
    is skipped and its KD term is a direct top-k per-position KL on the
    shared vocab (top-k selected student-side over the reassembled full-vocab
    teacher logits). The single-teacher path is just ``num_teachers == 1``.

    Inputs (via ``LossInputType.DISTILLATION_CROSS_TOKENIZER``):
        logits: ``[B, T_s, V_s]`` raw student logits from the worker forward.
        student_logits_contig: CP-relaid contiguous student logits shared by
            every teacher's KD term.
        teacher_full_logits_by_idx: ``dict[int, [B, T, V_t]]`` full-vocab teacher
            logits for dense teachers, rebuilt from CUDA IPC handles by
            ``prepare_loss_input`` (see
            :func:`nemo_rl.algorithms.x_token.loss_utils.rebuild_teacher_full_logits_from_ipc`).
        teacher_sparse_logits_by_idx: Per-sparse-teacher row-top-k logits,
            token ids, exact full-vocabulary log-normalizers, and optional
            realized-label top-k membership, reconstructed for the full
            teacher sequence.
        teacher_row_payloads_by_idx: Owner-local evaluation only: each
            cross-tokenizer teacher's IPC records, read row by row by the rank
            that owns each chunk (see ``owner_local``).
        aligns_by_idx: ``dict[int, LocalizedAlignment]`` per teacher (cross-tok:
            localized chunk alignment; same-tok: thin, student fields only).

    Inputs (via ``data: BatchedDataDict``):
        See :class:`CSDLossDataDict`.

    Returns:
        ``(loss, metrics)``. Aggregate metrics: ``loss``, ``kl_loss`` (the
        aggregated KD term), ``ce_loss``, ``kl_loss_scale``, ``accuracy``,
        ``num_valid_samples``. Per-teacher metrics are suffixed ``_t{i}`` (e.g.
        ``kl_loss_t0``, ``top1_acc_per_chunk_t0``, ``weight_t0``);
        routing observability is reported as ``teacher_{i}/routed_samples``,
        ``teacher_{i}/routed_tokens``, and ``teacher_{i}/weighted_kl``;
        ``select_teacher`` additionally reports ``selected_teacher``.
    """

    # Group the global normalizers (_dp_global_masked_mean, the CSD
    # global_valid_chunks) reduce over; set per call by __call__. Declared at
    # class scope so it exists even for instances built without __init__ (the
    # unit tests construct bare instances to exercise single methods). None =>
    # the legacy WORLD reduce; see loss_replica_group for why WORLD deadlocks
    # under pipeline parallelism.
    _dp_cp_group: Optional[torch.distributed.ProcessGroup] = None
    # Whether per-window same-tokenizer KD terms are summed across CP; set per
    # call by __call__ (see ``cp_normalized_by_caller``) like ``_dp_cp_group``.
    _sum_window_kd_over_cp: bool = False

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION_CROSS_TOKENIZER

    def __init__(self, cfg: CSDLossConfig | Mapping[str, Any]):
        cfg = (
            cfg if isinstance(cfg, CSDLossConfig) else CSDLossConfig.model_validate(cfg)
        )
        # Dynamic teacher weighting (sum_weights_metric) and normalize_teacher_by_vocab
        # are only applied in kd_loss_mode="sum"; reject the combo instead of
        # silently ignoring them under the other modes.
        if cfg.sum_weights_metric is not None and cfg.kd_loss_mode != "sum":
            raise ValueError(
                f"sum_weights_metric={cfg.sum_weights_metric!r} is only applied "
                f"in kd_loss_mode='sum'; it is ignored by '{cfg.kd_loss_mode}'. "
                "Unset one of them."
            )
        if cfg.normalize_teacher_by_vocab and cfg.kd_loss_mode != "sum":
            raise ValueError(
                "normalize_teacher_by_vocab is only applied in kd_loss_mode='sum'; "
                f"it is ignored by '{cfg.kd_loss_mode}'. Unset one of them."
            )
        # averaged_logits forms a convex combination of teacher logits
        # (weight_i / sum(weights)); a zero weight-sum (all zeros, or a signed
        # set cancelling to 0) makes that division undefined. Reject it here
        # rather than fail with a deep ZeroDivisionError mid-step.
        _weights = cfg.teacher_weights
        if (
            cfg.kd_loss_mode == "averaged_logits"
            and _weights is not None
            and sum(_weights) == 0
        ):
            raise ValueError(
                "teacher_weights must not sum to zero in "
                "kd_loss_mode='averaged_logits' (they form a convex combination "
                f"of teacher logits); got teacher_weights={list(_weights)}."
            )
        if cfg.student_vocab_size is None:
            raise ValueError("CSD requires the runtime student_vocab_size")
        if cfg.owner_local:
            violations = owner_local_config_violations(cfg)
            if violations:
                raise ValueError(
                    "Owner-local CSD evaluation does not support "
                    + "; ".join(violations)
                    + "."
                )
        # Evaluate cross-tokenizer KD on CP owner ranks (CP-partial loss).
        self.owner_local = cfg.owner_local
        # Global loss knobs (shared across all teachers).
        self.temperature = cfg.temperature
        # vocab_topk / reverse_kl are consumed by the same-vocab direct-KL path
        # (same_tokenizer_topk_kl / same_tokenizer_full_vocab_kl); the
        # cross-tokenizer path is CSD.
        self.vocab_topk = cfg.vocab_topk
        self.reverse_kl = cfg.reverse_kl
        self.kl_loss_weight = cfg.kl_loss_weight
        self.ce_loss_scale = cfg.ce_loss_scale
        self.dynamic_loss_scaling = cfg.dynamic_loss_scaling
        self.student_vocab_size = cfg.student_vocab_size
        # Multi-teacher aggregation knobs.
        self.kd_loss_mode = cfg.kd_loss_mode
        self.normalize_teacher_by_vocab = cfg.normalize_teacher_by_vocab
        self.alpha = cfg.alpha
        # None selects the static per-teacher weights.
        self.sum_weights_metric = cfg.sum_weights_metric
        # Per-teacher metadata: parallel lists (one entry per ``teachers[i]``),
        # injected by ``xtoken_off_policy_distillation.setup``. Every teacher ships
        # full-vocab logits (the loss derives the top-k subset student-side), so
        # there is no per-teacher ``send_full_logits`` flag.
        self.projection_matrix_paths: list[Optional[str]] = (
            list(cfg.projection_matrix_paths)
            if cfg.projection_matrix_paths is not None
            else [None for _ in cfg.teacher_vocab_sizes]
        )
        self.teacher_vocab_sizes = list(cfg.teacher_vocab_sizes)
        self.teacher_weights = list(cfg.teacher_weights)
        # Every per-teacher list must have the same length (one entry per
        # teacher); a mismatch would otherwise surface as a deep IndexError
        # mid-training instead of a clear error here.
        per_teacher_lens = {
            "projection_matrix_paths": len(self.projection_matrix_paths),
            "teacher_vocab_sizes": len(self.teacher_vocab_sizes),
            "teacher_weights": len(self.teacher_weights),
        }
        if len(set(per_teacher_lens.values())) != 1:
            raise ValueError(
                f"per-teacher lists must be equal length, got {per_teacher_lens}"
            )
        self.num_teachers = len(self.projection_matrix_paths)
        # ------------------------------------------------------------------
        # CSD state. Each cross-tokenizer teacher's
        # KD term is the prefix-bidir partition KL; same-vocab teachers keep the
        # direct-KL path and hold no CSD state. Per-teacher artifact caches
        # (common-index set, prefix-support index, pseudo-target tables) are
        # keyed by teacher index so any number of cross-tok teachers compose.
        # ------------------------------------------------------------------
        self.common_indices_from_subtoks = bool(cfg.common_indices_from_subtoks)
        # Per-teacher pseudo-target table paths (forward: student->teacher
        # chains; reverse: teacher->student chains). None for same-vocab
        # teachers. Runtime-injected symmetrically to projection_matrix_paths.
        self.student_to_teacher_table_paths: list[Optional[str]] = (
            list(cfg.student_to_teacher_table_paths)
            if cfg.student_to_teacher_table_paths is not None
            else [None for _ in range(self.num_teachers)]
        )
        self.teacher_to_student_table_paths: list[Optional[str]] = (
            list(cfg.teacher_to_student_table_paths)
            if cfg.teacher_to_student_table_paths is not None
            else [None for _ in range(self.num_teachers)]
        )
        if (
            len(self.student_to_teacher_table_paths) != self.num_teachers
            or len(self.teacher_to_student_table_paths) != self.num_teachers
        ):
            raise ValueError("CSD table paths must contain one entry per teacher")
        # Optional per-microbatch loss dump for parity comparison
        # (NRL_CSD_LOSS_DUMP_DIR). Raw floats from the loss-compute site.
        self._loss_dump_dir = os.environ.get("NRL_CSD_LOSS_DUMP_DIR")
        self._loss_dump_records: list[dict[str, Any]] = []
        self._loss_dump_call_idx = 0
        # Keep the validated loss settings alongside per-teacher metadata.
        self.cfg = cfg
        # Lazy CSD caches, populated on first call, keyed by (device, teacher_idx).
        self._csd_common_indices_per_device: dict[
            tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._csd_teacher_to_common_student_per_device: dict[
            tuple[torch.device, int], torch.Tensor
        ] = {}
        self._csd_prefix_index_per_device: dict[
            tuple[torch.device, int], dict[str, Any]
        ] = {}
        # Per-teacher pseudo-target tables, loaded lazily. idx -> (subtoks,
        # lengths); forward = student->teacher chains, reverse = teacher->student.
        self._csd_fwd_subtoks_per_teacher: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._csd_rev_subtoks_per_teacher: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # The materialized projection matrix and the derived exact-map
        # partition both live in process-local caches in
        # ``x_token.loss_utils`` (see ``get_sparse_projection_matrix``,
        # ``get_topk_projection``, ``build_exact_token_map``), not on
        # this instance. That keeps the driver-side ``loss_fn`` free of
        # any large CUDA tensors and lets multiple loss instances on
        # the same worker share one load.

    def _teacher_is_same_vocab(self, i: int) -> bool:
        """Same-tokenizer teachers supply neither CSD tables nor a projection."""
        return (
            self.projection_matrix_paths[i] is None
            and self.student_to_teacher_table_paths[i] is None
        )

    @staticmethod
    def _student_kd_mask(align: LocalizedAlignment) -> torch.Tensor:
        """Return the semantic KD mask, with legacy text-batch fallback."""
        mask = align.student_kd_token_mask
        if mask is None:
            mask = align.student_token_mask
        if mask is None:
            raise ValueError("localized alignment is missing a student KD token mask")
        return to_local_if_dtensor(mask)

    def __call__(
        self,
        data: BatchedDataDict[CSDLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        logits: torch.Tensor,
        student_logits_contig: Optional[torch.Tensor],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        *,
        student_next_token_logprobs: Optional[torch.Tensor] = None,
        student_next_token_mask: Optional[torch.Tensor] = None,
        teacher_sparse_logits_by_idx: Optional[dict[int, SparseTeacherLogits]] = None,
        teacher_row_payloads_by_idx: Optional[dict[int, TeacherRowPayload]] = None,
        dense_reconstruction_fallbacks_by_idx: Optional[dict[int, int]] = None,
        global_valid_chunks_by_idx: Optional[dict[int, torch.Tensor]] = None,
        global_valid_kd_toks: Optional[torch.Tensor] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_normalized_by_caller: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the (multi-teacher) cross-tokenizer distillation loss.

        Per-teacher KD terms are aggregated per ``kd_loss_mode`` and combined
        with a single student next-token CE term — dynamic-scaled when
        ``dynamic_loss_scaling`` is set (KD rescaled to match the detached CE
        magnitude, ``kl_loss_weight`` / ``ce_loss_scale`` ignored), else
        fixed-weighted. The single-teacher path is just ``num_teachers == 1``.

        ``student_logits_contig`` (CP-relaid), the per-teacher ``aligns_by_idx``,
        and the dense and sparse teacher-logit dictionaries are precomputed in
        ``prepare_loss_input``. The Automodel CP path also supplies its
        sequence-local CE inputs; Megatron keeps the raw TP/CP-sharded
        ``logits`` for its memory-efficient CE path. Production callers provide
        ``global_valid_chunks_by_idx`` from the full batch before microbatching.

        ``dp_cp_group`` is the group the global normalizers reduce over. It is
        stashed on the instance rather than threaded through the four levels of
        per-teacher dispatch below; ``__call__`` is the single entry point and is
        not re-entrant, so the value is always the one for the call in flight.
        ``None`` keeps the legacy ``WORLD`` behaviour (single-process tests, and
        the DTensor path, which has no pipeline axis to deadlock on).

        ``cp_normalized_by_caller`` is set when the caller divides the returned
        loss by the CP size (the replicated Megatron path). A same-tokenizer KD
        term covers only this rank's contiguous window, so it is then summed
        across CP (with a SUM backward) to keep its full weight after the
        division.

        With ``owner_local`` (eligible Megatron students), cross-tokenizer
        teachers arrive as ``teacher_row_payloads_by_idx`` and ``logits`` stays
        in its CP shard. CE and every KD term are then CP-partial: each CP rank
        returns its own share, the caller must not divide by the CP size, and
        dynamic scaling and the reported metrics use detached CP sums.
        """
        self._dp_cp_group = dp_cp_group
        self._sum_window_kd_over_cp = cp_normalized_by_caller
        teacher_sparse_logits_by_idx = teacher_sparse_logits_by_idx or {}
        dense_reconstruction_fallbacks_by_idx = (
            dense_reconstruction_fallbacks_by_idx or {}
        )
        kd_normalizer = (
            global_valid_toks if global_valid_kd_toks is None else global_valid_kd_toks
        )
        if self.owner_local:
            ce_loss = self._compute_owner_local_ce(
                logits, data, global_valid_toks, tp_group=tp_group, cp_group=cp_group
            )
            total_kd, per_teacher_metrics = self._owner_local_sum_kd(
                logits,
                student_logits_contig,
                teacher_full_logits_by_idx,
                teacher_row_payloads_by_idx or {},
                aligns_by_idx,
                kd_normalizer,
                global_valid_chunks_by_idx,
                tp_group=tp_group,
                cp_group=cp_group,
            )
            reported_kd = self._detached_cp_sum(total_kd, cp_group)
            reported_ce = self._detached_cp_sum(ce_loss, cp_group)
            loss, reported_loss, kl_scale = self._combine_kd_and_ce(
                total_kd, ce_loss, reported_kd, reported_ce
            )
            accuracy = self._owner_local_next_token_accuracy(
                logits, data, tp_group=tp_group, cp_group=cp_group
            )
            metrics = self._loss_metrics(
                data,
                loss=reported_loss,
                kd=reported_kd,
                ce=reported_ce,
                kl_scale=kl_scale,
                accuracy=accuracy,
                dense_reconstruction_fallbacks_by_idx=dense_reconstruction_fallbacks_by_idx,
            )
            metrics.update(per_teacher_metrics)
            metrics.update(self._teacher_routing_metrics(data))
            return loss, metrics

        if student_logits_contig is None:
            raise ValueError("Replicated CSD evaluation needs the student window.")
        ce_loss = self._compute_ce(
            logits,
            data,
            global_valid_toks,
            student_next_token_logprobs=student_next_token_logprobs,
            student_next_token_mask=student_next_token_mask,
            student_logits_contig=student_logits_contig,
            tp_group=tp_group,
            cp_group=cp_group,
        )

        if self.kd_loss_mode == "sum":
            total_kd, per_teacher_metrics = self._sum_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                kd_normalizer,
                teacher_sparse_logits_by_idx=teacher_sparse_logits_by_idx,
                global_valid_chunks_by_idx=global_valid_chunks_by_idx,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        elif self.kd_loss_mode == "averaged_logits":
            total_kd, per_teacher_metrics = self._averaged_logits_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                kd_normalizer,
                teacher_sparse_logits_by_idx=teacher_sparse_logits_by_idx,
                global_valid_chunks_by_idx=global_valid_chunks_by_idx,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        elif self.kd_loss_mode == "select_teacher":
            total_kd, per_teacher_metrics = self._select_teacher_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                kd_normalizer,
                teacher_sparse_logits_by_idx=teacher_sparse_logits_by_idx,
                global_valid_chunks_by_idx=global_valid_chunks_by_idx,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        else:
            raise ValueError(f"Unknown kd_loss_mode: {self.kd_loss_mode!r}")

        # Combine the aggregated KD term with the single student CE term.
        loss, _, kl_scale = self._combine_kd_and_ce(
            total_kd, ce_loss, total_kd.detach(), ce_loss.detach()
        )

        # Next-token accuracy on the student side (quick per-step signal), masked
        # to valid tokens. Computed once on the student from the shared CP-relaid
        # fields (carried on every teacher's align); the CP-aware shift pairs
        # predictors with the right labels under load-balanced sharding.
        align0 = aligns_by_idx[0]
        assert align0.student_input_ids is not None
        assert align0.student_token_mask is not None
        accuracy = next_token_accuracy(
            student_logits_contig,
            input_ids=align0.student_input_ids,
            token_mask=align0.student_token_mask,
            sample_mask=data["sample_mask"],
            tp_group=tp_group,
            cp_group=cp_group,
        )

        metrics = self._loss_metrics(
            data,
            loss=loss,
            kd=total_kd,
            ce=ce_loss,
            kl_scale=kl_scale,
            accuracy=accuracy,
            dense_reconstruction_fallbacks_by_idx=dense_reconstruction_fallbacks_by_idx,
        )
        metrics.update(per_teacher_metrics)
        metrics.update(self._teacher_routing_metrics(data))
        return loss, metrics

    def _combine_kd_and_ce(
        self,
        total_kd: torch.Tensor,
        ce_loss: torch.Tensor,
        reported_kd: torch.Tensor,
        reported_ce: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(loss, reported_loss, kl_scale)``.

        ``reported_kd`` / ``reported_ce`` are the detached complete-batch terms
        (CP sums when the terms are CP-partial); the dynamic KD scale is taken
        from them so every CP rank applies the same ratio.
        """
        if self.dynamic_loss_scaling:
            # loss = sg(ce/kd) * kd + ce; user kl_loss_weight / ce_loss_scale
            # are intentionally ignored in this branch.
            kd_detached = reported_kd.abs()
            ce_detached = reported_ce.abs()
            kl_scale = torch.where(
                kd_detached > 0,
                ce_detached / kd_detached,
                torch.ones_like(kd_detached),
            )
            return (
                kl_scale * total_kd + ce_loss,
                kl_scale * reported_kd + reported_ce,
                kl_scale,
            )
        kl_scale = torch.tensor(1.0, device=total_kd.device, dtype=total_kd.dtype)
        return (
            self.kl_loss_weight * total_kd + self.ce_loss_scale * ce_loss,
            self.kl_loss_weight * reported_kd + self.ce_loss_scale * reported_ce,
            kl_scale,
        )

    @staticmethod
    def _loss_metrics(
        data: BatchedDataDict[CSDLossDataDict],
        *,
        loss: torch.Tensor,
        kd: torch.Tensor,
        ce: torch.Tensor,
        kl_scale: torch.Tensor,
        accuracy: torch.Tensor,
        dense_reconstruction_fallbacks_by_idx: dict[int, int],
    ) -> dict[str, Any]:
        """Aggregate loss metrics shared by the replicated and owner-local paths."""
        metrics: dict[str, Any] = {
            "loss": loss.item(),
            # Aggregate KD term (kept under ``kl_loss`` so existing trainer
            # metric handling continues to work); per-teacher terms are suffixed
            # ``_t{i}``.
            "kl_loss": kd.item(),
            "ce_loss": ce.item(),
            "kl_loss_scale": kl_scale.item(),
            "accuracy": accuracy.item(),
            "num_valid_samples": to_local_if_dtensor(data["sample_mask"]).sum().item(),
            "ipc_reconstruction_fallbacks": sum(
                dense_reconstruction_fallbacks_by_idx.values()
            ),
        }
        for teacher_idx, fallback_count in sorted(
            dense_reconstruction_fallbacks_by_idx.items()
        ):
            metrics[f"ipc_reconstruction_fallbacks_t{teacher_idx}"] = fallback_count
        return metrics

    def _teacher_routing_metrics(
        self,
        data: BatchedDataDict[CSDLossDataDict],
    ) -> dict[str, int]:
        """Count the logical samples and input tokens routed to each teacher.

        Every teacher forward receives the microbatch, regardless of how its KD
        term is aggregated later. Cross-tokenizer teachers are counted on their
        own tokenization; same-tokenizer teachers reuse the student tokenization.
        Masked padding rows and their tokens do not contribute.
        """
        sample_mask = to_local_if_dtensor(data["sample_mask"]).bool()
        routed_samples = int(sample_mask.sum().item())
        metrics: dict[str, int] = {}
        for teacher_idx in range(self.num_teachers):
            token_mask_key = (
                "token_mask"
                if self._teacher_is_same_vocab(teacher_idx)
                else f"teacher_{teacher_idx}_token_mask"
            )
            token_mask = to_local_if_dtensor(data[token_mask_key]).bool()
            routed_tokens = int((token_mask & sample_mask.unsqueeze(-1)).sum().item())
            metrics[f"teacher_{teacher_idx}/routed_samples"] = routed_samples
            metrics[f"teacher_{teacher_idx}/routed_tokens"] = routed_tokens
        return metrics

    # ------------------------------------------------------------------ #
    # Multi-teacher aggregation
    # ------------------------------------------------------------------ #
    def _compute_teacher_kd(
        self,
        i: int,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CSDLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        teacher_sparse_logits_by_idx: dict[int, SparseTeacherLogits],
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
        global_valid_chunks_by_idx: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """KD term for teacher ``i`` plus its (unsuffixed) metrics.

        Dispatches on tokenizer kind: same-vocab -> direct top-k per-position KL;
        cross-tokenizer -> the CSD prefix-bidir partition KL over teacher ``i``'s
        localized alignment. Both consume the shared CP-relaid student logits and
        route TP/CP through the parameterized loss-mode helpers.
        """
        has_full_logits = i in teacher_full_logits_by_idx
        has_sparse_logits = i in teacher_sparse_logits_by_idx
        if has_full_logits == has_sparse_logits:
            raise ValueError(
                f"Teacher {i} must provide exactly one dense or sparse logits "
                f"payload; dense={has_full_logits}, sparse={has_sparse_logits}."
            )
        if self._teacher_is_same_vocab(i):
            if has_sparse_logits:
                raise ValueError(
                    f"Same-vocab teacher {i} requires full teacher logits."
                )
            return self._compute_same_vocab_kl(
                i,
                student_logits_contig,
                teacher_full_logits_by_idx[i],
                aligns_by_idx[i],
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )

        global_valid_chunks = (
            global_valid_chunks_by_idx[i]
            if global_valid_chunks_by_idx is not None
            else None
        )

        kd, csd_metrics = self._compute_csd_loss(
            i,
            student_logits_contig,
            teacher_full_logits_by_idx.get(i),
            aligns_by_idx[i],
            teacher_sparse_payload=teacher_sparse_logits_by_idx.get(i),
            teacher_vocab_size=self.teacher_vocab_sizes[i],
            tp_group=tp_group,
            cp_group=cp_group,
            global_valid_chunks=global_valid_chunks,
        )
        # Surface the KD value under the shared per-teacher metric key and keep
        # the CSD chunk diagnostics; the dispatcher-level loss keys are dropped
        # (the aggregate ``loss`` / ``kl_loss`` are set by ``__call__``).
        metrics: dict[str, Any] = {"kl_loss": kd.item()}
        for key in (
            "kl_common_per_chunk",
            "kl_partition_last_per_chunk",
            "top1_acc_per_chunk",
            "num_common_chunks",
            "num_mismatch_chunks",
        ):
            if key in csd_metrics:
                metrics[key] = csd_metrics[key]
        return kd, metrics

    def _compute_same_vocab_kl(
        self,
        i: int,
        student_logits_contig: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Direct top-k per-position KL for a same-tokenizer teacher.

        Identical tokenizer => teacher tokens == student tokens (identity
        position alignment), so no projection / no chunk-averaging. The top-k
        vocab list is chosen from KD positions only (the semantic KD mask). The
        reduction is the KD-position mean normalized by ``global_valid_toks``
        (the KD-token count), scaled by ``T**2``.
        """
        kd = same_tokenizer_topk_kl(
            student_logits_contig,
            teacher_full_logits,
            kd_mask=self._student_kd_mask(align),
            sample_mask=align.sample_mask,
            num_kd_tokens=global_valid_toks,
            vocab_size=self.student_vocab_size,
            vocab_topk=self.vocab_topk,
            temperature=self.temperature,
            reverse_kl=self.reverse_kl,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        if self._sum_window_kd_over_cp:
            kd = group_all_reduce_sum_with_grad_backward_sum(kd, cp_group)
        return kd, {"kl_loss": kd.item()}

    def _sum_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CSDLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        teacher_sparse_logits_by_idx: dict[int, SparseTeacherLogits],
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
        global_valid_chunks_by_idx: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Weighted sum: ``total_kd = Σ_i weight_i · KD_i``.

        Weights are static (config ``weight``) or dynamic (``sum_weights_metric``).
        When ``normalize_teacher_by_vocab`` is set, each teacher's KD is
        additionally scaled by ``log(V_t_i) / log(min_j V_t_j)``.
        """
        device = student_logits_contig.device
        if self.sum_weights_metric is not None:
            if teacher_sparse_logits_by_idx:
                raise ValueError(
                    "sum_weights_metric requires full teacher logits and cannot "
                    "be used with sparse teacher IPC."
                )
            weights = self._compute_dynamic_weights(
                data, teacher_full_logits_by_idx, aligns_by_idx
            )
        else:
            weights = [
                torch.tensor(
                    self.teacher_weights[i],
                    device=device,
                    dtype=student_logits_contig.dtype,
                )
                for i in range(self.num_teachers)
            ]

        if self.normalize_teacher_by_vocab:
            temp_weight = torch.log(
                torch.tensor(float(min(self.teacher_vocab_sizes)), device=device)
            )

        total_kd: Optional[torch.Tensor] = None
        per_metrics: dict[str, Any] = {}
        # Deterministic teacher order: each teacher's KD fires its own
        # collectives, so the order must match across ranks.
        for i in range(self.num_teachers):
            kd_i, m_i = self._compute_teacher_kd(
                i,
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                teacher_sparse_logits_by_idx=teacher_sparse_logits_by_idx,
                global_valid_chunks_by_idx=global_valid_chunks_by_idx,
                tp_group=tp_group,
                cp_group=cp_group,
            )
            weighted = kd_i * weights[i]
            if self.normalize_teacher_by_vocab:
                v_scale = (
                    torch.log(
                        torch.tensor(float(self.teacher_vocab_sizes[i]), device=device)
                    )
                    / temp_weight
                )
                weighted = weighted * v_scale
            total_kd = weighted if total_kd is None else total_kd + weighted
            per_metrics[f"teacher_{i}/weighted_kl"] = float(weighted.item())
            for k, v in m_i.items():
                per_metrics[f"{k}_t{i}"] = v
            per_metrics[f"weight_t{i}"] = float(weights[i].item())
        assert total_kd is not None
        return total_kd, per_metrics

    def _averaged_logits_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CSDLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        teacher_sparse_logits_by_idx: dict[int, SparseTeacherLogits],
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
        global_valid_chunks_by_idx: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convex-weighted average of teacher logits, then one direct KL.

        Valid only when all teachers are same-tokenizer (no projection) and ship
        full logits of identical shape. Otherwise falls back to a plain
        static-weight sum (no dynamic weights, no ``normalize_teacher_by_vocab``).
        """
        full = [teacher_full_logits_by_idx.get(i) for i in range(self.num_teachers)]
        # Direct per-position KL is only valid when every teacher shares the
        # student's tokenizer (no projection matrix) *and* ships full logits of
        # identical shape. Two cross-tokenizer teachers can have matching shapes
        # yet still need the projection/alignment path, so the shape check alone
        # is insufficient.
        same_tokenizer = all(
            self._teacher_is_same_vocab(i) for i in range(self.num_teachers)
        )
        same_shape = all(f is not None for f in full) and (
            len({tuple(f.shape) for f in full if f is not None}) == 1
        )
        if not (same_tokenizer and same_shape):
            total_kd: Optional[torch.Tensor] = None
            per_metrics: dict[str, Any] = {}
            for i in range(self.num_teachers):
                kd_i, m_i = self._compute_teacher_kd(
                    i,
                    student_logits_contig,
                    data,
                    teacher_full_logits_by_idx,
                    aligns_by_idx,
                    global_valid_toks,
                    teacher_sparse_logits_by_idx=teacher_sparse_logits_by_idx,
                    global_valid_chunks_by_idx=global_valid_chunks_by_idx,
                    tp_group=tp_group,
                    cp_group=cp_group,
                )
                w = self.teacher_weights[i]
                weighted = kd_i * w
                total_kd = weighted if total_kd is None else total_kd + weighted
                per_metrics[f"teacher_{i}/weighted_kl"] = float(weighted.item())
                for k, v in m_i.items():
                    per_metrics[f"{k}_t{i}"] = v
                per_metrics[f"weight_t{i}"] = float(w)
            assert total_kd is not None
            return total_kd, per_metrics

        total_w = sum(self.teacher_weights)
        avg: Optional[torch.Tensor] = None
        for i, f in enumerate(full):
            assert f is not None
            contrib = f.float() * (self.teacher_weights[i] / total_w)
            avg = contrib if avg is None else avg + contrib
        assert avg is not None
        align = aligns_by_idx[0]
        kd = same_tokenizer_full_vocab_kl(
            student_logits_contig,
            avg,
            kd_mask=self._student_kd_mask(align),
            sample_mask=align.sample_mask,
            num_kd_tokens=global_valid_toks,
            vocab_size=self.student_vocab_size,
            temperature=self.temperature,
            reverse_kl=self.reverse_kl,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        if self._sum_window_kd_over_cp:
            kd = group_all_reduce_sum_with_grad_backward_sum(kd, cp_group)
        per_metrics: dict[str, Any] = {"kl_loss": kd.item()}
        for i, weight in enumerate(self.teacher_weights):
            per_metrics[f"teacher_{i}/weighted_kl"] = float(
                kd.item() * weight / total_w
            )
        return kd, per_metrics

    def _dp_global_masked_mean(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked mean of ``values`` over the *process-global* valid count.

        Teacher selection / dynamic weighting must be identical on every rank: a
        rank-local mean lets ranks pick a different teacher / different weights,
        and the per-teacher KD's collectives then see divergent participation
        (deadlock when one rank's choice fires a collective another's does not).
        All-reduce the masked sum and the mask count over the full group so every
        rank gets the same score. The result is detached (it gates selection /
        weighting and is not back-propagated).

        Reduces over ``self._dp_cp_group`` when set. That group is confined to
        one pipeline stage; ``WORLD`` would deadlock under PP, where only the
        last stage runs the loss and reaches this collective at all.
        """
        group = (
            self._dp_cp_group
            if self._dp_cp_group is not None
            else torch.distributed.group.WORLD
        )
        num = group_all_reduce_sum((values * mask).sum(), group=group)
        den = group_all_reduce_sum(mask.sum(), group=group).clamp(min=1.0)
        return num / den

    def _select_teacher_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CSDLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        teacher_sparse_logits_by_idx: dict[int, SparseTeacherLogits],
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
        global_valid_chunks_by_idx: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Use only the teacher with the lowest next-token CE on its own tokens."""
        if teacher_sparse_logits_by_idx:
            raise ValueError(
                "kd_loss_mode='select_teacher' requires full teacher logits and "
                "cannot be used with sparse teacher IPC."
            )
        with torch.no_grad():
            ces: list[float] = []
            for i in range(self.num_teachers):
                t_logits, t_ids, t_mask = self._teacher_score_inputs(
                    i, data, teacher_full_logits_by_idx, aligns_by_idx
                )
                ce_pos = torch.nn.functional.cross_entropy(
                    t_logits[:, :-1].reshape(-1, t_logits.shape[-1]).float(),
                    t_ids[:, 1:].reshape(-1),
                    reduction="none",
                )
                mask = (
                    t_mask[:, 1:].float() * data["sample_mask"].unsqueeze(-1).float()
                ).reshape(-1)
                ces.append(self._dp_global_masked_mean(ce_pos, mask).item())
            best = int(min(range(self.num_teachers), key=lambda j: ces[j]))

        kd, m = self._compute_teacher_kd(
            best,
            student_logits_contig,
            data,
            teacher_full_logits_by_idx,
            aligns_by_idx,
            global_valid_toks,
            teacher_sparse_logits_by_idx=teacher_sparse_logits_by_idx,
            global_valid_chunks_by_idx=global_valid_chunks_by_idx,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        per_metrics: dict[str, Any] = {f"{k}_t{best}": v for k, v in m.items()}
        for i in range(self.num_teachers):
            per_metrics[f"teacher_{i}/weighted_kl"] = 0.0
        per_metrics[f"teacher_{best}/weighted_kl"] = float(kd.item())
        per_metrics["selected_teacher"] = best
        return kd, per_metrics

    def _teacher_score_inputs(
        self,
        i: int,
        data: BatchedDataDict[CSDLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, input_ids, token_mask)`` for teacher ``i``'s CE / weight-metric scores.

        The token mask is over the tokenization the score is computed on: the
        shared student tokens (CP-relaid) for a same-vocab teacher, teacher ``i``'s
        own otherwise. Every teacher ships full logits, so the full distribution
        is always available.
        """
        if self._teacher_is_same_vocab(i):
            align = aligns_by_idx[i]
            assert align.student_input_ids is not None
            ids = to_local_if_dtensor(align.student_input_ids)
            token_mask = self._student_kd_mask(align)
        else:
            ids = to_local_if_dtensor(data[f"teacher_{i}_input_ids"])
            token_mask = to_local_if_dtensor(data[f"teacher_{i}_token_mask"])
        return teacher_full_logits_by_idx[i], ids, token_mask

    def _compute_dynamic_weights(
        self,
        data: BatchedDataDict[CSDLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
    ) -> list[torch.Tensor]:
        """Sequence-level dynamic teacher weights via ``sum_weights_metric``.

        Per teacher computes a scalar score (``ce`` -> -CE, ``entropy`` ->
        -entropy, ``max_prob`` -> max prob; higher = more trusted), optionally
        rescaled by ``log(V_t_i)/log(min_j V_t_j)``, then ``softmax(alpha *
        scores)`` across teachers.
        """
        device = data["input_ids"].device
        if self.normalize_teacher_by_vocab:
            temp_weight = torch.log(
                torch.tensor(float(min(self.teacher_vocab_sizes)), device=device)
            )
        scores: list[torch.Tensor] = []
        for i in range(self.num_teachers):
            t_logits, t_ids, t_mask = self._teacher_score_inputs(
                i, data, teacher_full_logits_by_idx, aligns_by_idx
            )
            score = self._teacher_weight_score(
                t_logits, t_ids, t_mask, data["sample_mask"]
            )
            if self.normalize_teacher_by_vocab:
                v_log = torch.log(
                    torch.tensor(float(self.teacher_vocab_sizes[i]), device=device)
                )
                score = score * (v_log / temp_weight)
            scores.append(score)
        weights = torch.softmax(self.alpha * torch.stack(scores), dim=0)
        return [weights[i] for i in range(self.num_teachers)]

    def _teacher_weight_score(
        self,
        t_logits: torch.Tensor,
        t_ids: torch.Tensor,
        t_mask: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar weight-metric score for one teacher (higher = more trusted).

        Padded positions and masked-out samples are excluded, so long-padded
        batches don't let near-uniform padding logits dominate the score.
        """
        samp = to_local_if_dtensor(sample_mask).unsqueeze(-1).float()
        if self.sum_weights_metric == "ce":
            ce_pos = torch.nn.functional.cross_entropy(
                t_logits[:, :-1].reshape(-1, t_logits.shape[-1]).float(),
                t_ids[:, 1:].reshape(-1),
                reduction="none",
            )
            mask = (t_mask[:, 1:].float() * samp).reshape(-1)
            return -self._dp_global_masked_mean(ce_pos, mask)
        mask = t_mask.float() * samp
        if self.sum_weights_metric == "entropy":
            probs = torch.softmax(t_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            return -self._dp_global_masked_mean(entropy, mask)
        if self.sum_weights_metric == "max_prob":
            probs = torch.softmax(t_logits.float(), dim=-1)
            return self._dp_global_masked_mean(probs.max(dim=-1).values, mask)
        raise ValueError(f"Unknown sum_weights_metric: {self.sum_weights_metric!r}")

    # ------------------------------------------------------------------ #
    # Loss-mode implementations
    # ------------------------------------------------------------------ #
    def _compute_ce(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict[CSDLossDataDict],
        global_valid_toks: torch.Tensor,
        *,
        student_next_token_logprobs: Optional[torch.Tensor] = None,
        student_next_token_mask: Optional[torch.Tensor] = None,
        student_logits_contig: Optional[torch.Tensor] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Next-token CE on the student side (TP/CP handled by the helpers).

        The Automodel CP path supplies sequence-local target log-probabilities,
        avoiding full-tensor gathers. DTensor logits otherwise carry their own
        device mesh, so the vocab-parallel log-prob helper resolves TP/CP from
        it. Megatron returns a *plain* tensor that is vocab-sharded (TP) and
        load-balanced sequence-sharded (CP). Compute only the target
        log-probabilities in fixed sequence chunks and gather the resulting
        ``[B, S]`` scalars across CP; never materialize a rank-3 full-sequence or
        full-vocabulary tensor. Ordinary unsharded tensors keep their existing
        path.
        """
        if student_next_token_logprobs is not None:
            assert student_next_token_mask is not None
            label_mask = student_next_token_mask.to(
                student_next_token_logprobs.dtype
            ) * to_local_if_dtensor(data["sample_mask"]).to(
                student_next_token_logprobs.device
            ).unsqueeze(-1)
            local_ce = masked_mean(
                -student_next_token_logprobs,
                label_mask,
                global_normalization_factor=global_valid_toks,
            )
            # Forward SUM gives every CP rank the same full-sequence CE (and
            # therefore the same dynamic KD scale). Identity backward keeps the
            # disjoint rank-local windows at a single gradient fanout.
            return group_all_reduce_sum_with_grad(local_ce, cp_group)

        tp_size = (
            torch.distributed.get_world_size(tp_group)
            if tp_group is not None and torch.distributed.is_initialized()
            else 1
        )
        cp_size = (
            torch.distributed.get_world_size(cp_group)
            if cp_group is not None and torch.distributed.is_initialized()
            else 1
        )
        if (
            not isinstance(logits, DTensor)
            and tp_group is not None
            and (tp_size > 1 or cp_size > 1)
        ):
            vocab_parallel_rank = torch.distributed.get_rank(tp_group)
            per_token_ce = -get_next_token_logprobs_from_logits(
                input_ids=data["input_ids"],
                next_token_logits=logits,
                seq_index=data.get("seq_index"),
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=tp_group,
                context_parallel_group=cp_group,
                chunk_size=CSD_STUDENT_LOSS_CHUNK_SIZE,
            )
        else:
            per_token_ce = student_next_token_ce(
                logits,
                input_ids=data["input_ids"],
                seq_index=data.get("seq_index"),
            )
        label_mask = ce_label_mask(
            token_mask=data["token_mask"],
            sample_mask=data["sample_mask"],
            ce_seq_len=per_token_ce.shape[1],
            dtype=per_token_ce.dtype,
        )
        return masked_mean(
            per_token_ce,
            label_mask,
            global_normalization_factor=global_valid_toks,
        )

    # CSD partition-loss helpers.
    def _get_csd_teacher_to_common_student(
        self,
        device: torch.device,
        i: int,
        v_t: int,
        common_student_idx_t: torch.Tensor,
        common_teacher_idx_t: torch.Tensor,
    ) -> torch.Tensor:
        cache_key = (device, i)
        cached = self._csd_teacher_to_common_student_per_device.get(cache_key)
        if cached is not None:
            return cached
        mapping = torch.full(
            (int(v_t),),
            -1,
            dtype=torch.long,
            device=device,
        )
        valid = common_teacher_idx_t < int(v_t)
        mapping[common_teacher_idx_t[valid]] = common_student_idx_t[valid]
        self._csd_teacher_to_common_student_per_device[cache_key] = mapping
        return mapping

    @staticmethod
    def _lookup_sparse_teacher_logp(
        topk_logits: torch.Tensor,
        topk_indices: torch.Tensor,
        teacher_log_z: torch.Tensor,
        b_idx: torch.Tensor,
        pos: torch.Tensor,
        token_ids: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lookup teacher log-probs for explicit token ids in sparse top-k rows.

        ``topk_indices`` must be sorted ascending along the last dimension.
        ``pos`` and ``token_ids`` may be either ``[K]`` or ``[K, N]``.
        Missing ids return an arbitrary log-prob value and ``found=False``;
        callers must mask them into the rest bucket.
        """
        squeeze = False
        if pos.dim() == 1:
            pos = pos.unsqueeze(-1)
            token_ids = token_ids.unsqueeze(-1)
            squeeze = True

        original_shape = token_ids.shape
        b_flat = b_idx.unsqueeze(-1).expand_as(pos).reshape(-1)
        pos_flat = pos.reshape(-1)
        token_flat = token_ids.reshape(-1).to(topk_indices.dtype)
        out_logp = torch.empty(
            (token_flat.numel(),),
            device=topk_logits.device,
            dtype=topk_logits.dtype,
        )
        out_found = torch.empty(
            (token_flat.numel(),),
            device=topk_indices.device,
            dtype=torch.bool,
        )

        k_top = topk_indices.shape[-1]
        row_chunk = int(os.environ.get("NRL_CSD_SPARSE_LOOKUP_ROWS", "2048"))
        row_chunk = max(row_chunk, 1)
        for start in range(0, token_flat.numel(), row_chunk):
            end = min(start + row_chunk, token_flat.numel())
            rows_idx = topk_indices[b_flat[start:end], pos_flat[start:end]]
            rows_vals = topk_logits[b_flat[start:end], pos_flat[start:end]]
            targets = token_flat[start:end]

            insert = torch.searchsorted(rows_idx, targets.unsqueeze(-1))
            insert = insert.squeeze(-1)
            in_bounds = insert < k_top
            safe_insert = insert.clamp(max=k_top - 1)
            row_arange = torch.arange(
                rows_idx.shape[0],
                device=rows_idx.device,
                dtype=torch.long,
            )
            matched_idx = rows_idx[row_arange, safe_insert]
            found = in_bounds & (matched_idx == targets)
            raw_logits = rows_vals[row_arange, safe_insert]
            flat_log_z = teacher_log_z[b_flat[start:end], pos_flat[start:end]]

            out_logp[start:end] = raw_logits / float(temperature) - flat_log_z
            out_found[start:end] = found

        logp = out_logp.reshape(original_shape)
        found = out_found.reshape(original_shape)

        if squeeze:
            logp = logp.squeeze(-1)
            found = found.squeeze(-1)
        return logp, found

    def _maybe_dump_loss(self, metrics: dict[str, Any]) -> None:
        """Append per-call raw loss values to a per-rank dump file.

        Activated by ``NRL_CSD_LOSS_DUMP_DIR``. One file per rank,
        rewritten on each call with the full record list. Records are raw
        ``loss.item()`` values from the loss-compute site — not scaled,
        aggregated, or DP-summed — for loss parity comparisons.
        """
        loss_dump_dir = self._loss_dump_dir
        if not loss_dump_dir:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # The P-KL path emits kl_loss/ce_loss/kl_loss_scale/num_valid_pairs;
        # the gold-loss path emits kl_common/l1_uncommon/num_valid_chunks;
        # the CSD path emits ce_loss_per_token/kl_loss_per_token plus the
        # per-chunk diagnostics. Record everything that's present so the
        # same dump file format serves all — downstream comparison scripts
        # read by key.
        record: dict[str, Any] = {
            "call_idx": self._loss_dump_call_idx,
            "loss": metrics["loss"],
        }
        for k in (
            "kl_loss",
            "ce_loss",
            "kl_loss_per_chunk",
            "ce_loss_per_token",
            "kl_loss_scale",
            "num_valid_pairs",
            "kl_common",
            "kl_common_per_chunk",
            "kl_partition_last_per_chunk",
            "l1_uncommon",
            "num_valid_chunks",
        ):
            if k in metrics:
                record[k] = metrics[k]
        self._loss_dump_records.append(record)
        self._loss_dump_call_idx += 1
        os.makedirs(loss_dump_dir, exist_ok=True)
        torch.save(
            self._loss_dump_records,
            os.path.join(loss_dump_dir, f"rank{rank}.pt"),
        )

    # ------------------------------------------------------------------ #
    # Loss-mode implementations
    # ------------------------------------------------------------------ #

    def _csd_fwd_table(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily load+cache teacher ``i``'s forward subtoks table.

        Forward = student->teacher sub-token chains (``subtoks`` / ``lengths``
        keys). Small CPU tensors, shared by ``_common_indices_from_subtoks`` and
        the forward branch of ``_ensure_bidir_prefix_support_index``.
        """
        cached = self._csd_fwd_subtoks_per_teacher.get(i)
        if cached is not None:
            return cached
        fwd_path = self.student_to_teacher_table_paths[i] or ""
        if not fwd_path or not os.path.exists(fwd_path):
            raise RuntimeError(
                f"teacher {i}: forward pseudo-target table required "
                f"(student_to_teacher_table_paths[{i}]); got {fwd_path!r}."
            )
        fwd = torch.load(fwd_path, map_location="cpu", weights_only=True)
        table = (fwd["subtoks"].long(), fwd["lengths"].long())
        self._csd_fwd_subtoks_per_teacher[i] = table
        return table

    def _csd_rev_table(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily load+cache teacher ``i``'s reverse subtoks table.

        Reverse = teacher->student sub-token chains, consumed by the reverse
        branch of ``_ensure_bidir_prefix_support_index``.
        """
        cached = self._csd_rev_subtoks_per_teacher.get(i)
        if cached is not None:
            return cached
        rev_path = self.teacher_to_student_table_paths[i] or ""
        if not rev_path or not os.path.exists(rev_path):
            raise RuntimeError(
                f"teacher {i}: reverse pseudo-target table required "
                f"(teacher_to_student_table_paths[{i}]); got {rev_path!r}."
            )
        rev = torch.load(rev_path, map_location="cpu", weights_only=True)
        table = (rev["subtoks"].long(), rev["lengths"].long())
        self._csd_rev_subtoks_per_teacher[i] = table
        return table

    def _get_common_indices_csd(
        self,
        device: torch.device,
        i: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached (common_student, common_teacher) for teacher ``i``.

        The exact 1-to-1 common set is sourced either from the forward subtoks
        table (``common_indices_from_subtoks=True``, the CSD-preset path, no
        projection matrix needed) or from the strict exact-token map of the
        teacher's projection matrix. Cached per ``(device, i)``.
        """
        cache_key = (device, i)
        if cache_key in self._csd_common_indices_per_device:
            return self._csd_common_indices_per_device[cache_key]
        if self.common_indices_from_subtoks:
            common_s, common_t = self._common_indices_from_subtoks(device, i)
        else:
            # CSD has no gold/xtoken modifier -> strict exact map
            # (``xtoken_loss=False``), matching every CSD preset.
            projection_path = self.projection_matrix_paths[i]
            if projection_path is None:
                raise ValueError(
                    "CSD requires a projection matrix when "
                    "common_indices_from_subtoks=False"
                )
            exact_map = build_exact_token_map(
                projection_path,
                device,
                xtoken_loss=False,
                teacher_vocab_size=self.teacher_vocab_sizes[i],
            )
            common_s = exact_map["common_student"]
            common_t = exact_map["common_teacher"]
        self._csd_common_indices_per_device[cache_key] = (common_s, common_t)
        return common_s, common_t

    def _common_indices_from_subtoks(
        self,
        device: torch.device,
        i: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive (common_student, common_teacher) from teacher ``i``'s forward subtoks table.

        A student token is exactly "common" (1-to-1 with the teacher vocab) iff
        its forward sub-token chain has length 1; the paired teacher id is
        ``subtoks_t[s, 0]``. This reads the same forward table the prefix
        support index uses (``student_to_teacher_table_paths[i]``, keys
        ``subtoks``/``lengths``), so a common-only run needs no projection
        matrix. Teacher collisions (rare for true exact matches) are deduped
        keeping the lowest student id, so ``common_teacher`` is unique. Result
        is sorted by student index, like the projection strict exact-token map.
        """
        subtoks_t, lengths_t = self._csd_fwd_table(i)
        v_teacher = int(self.teacher_vocab_sizes[i])

        common_s = (lengths_t == 1).nonzero(as_tuple=True)[0]
        common_t = subtoks_t[common_s, 0]
        in_bounds = (common_t >= 0) & (common_t < v_teacher)
        common_s = common_s[in_bounds]
        common_t = common_t[in_bounds]

        if common_t.numel() > 0:
            # Sort by student id, then keep the first (lowest-student) entry per
            # teacher id so common_teacher is unique.
            order = torch.argsort(common_s)
            common_s, common_t = common_s[order], common_t[order]
            sort_t, t_order = torch.sort(common_t, stable=True)
            first_of_t = torch.ones_like(sort_t, dtype=torch.bool)
            first_of_t[1:] = sort_t[1:] != sort_t[:-1]
            keep = torch.zeros(common_t.shape[0], dtype=torch.bool)
            keep[t_order[first_of_t]] = True
            common_s, common_t = common_s[keep], common_t[keep]
            re_order = torch.argsort(common_s)
            common_s, common_t = common_s[re_order], common_t[re_order]

        return common_s.to(device), common_t.to(device)

    def _ensure_bidir_prefix_support_index(
        self,
        device: torch.device,
        i: int,
    ) -> dict[str, Any]:
        """Lazy-build teacher ``i``'s bidirectional prefix support index.

        The cache maps ``(length, prefix_tuple) -> tuple[(s_id, t_id), ...]``
        for both the forward (``student_to_teacher_table_paths[i]``) and reverse
        (``teacher_to_student_table_paths[i]``) pseudo-target tables. Built once
        per ``(device, i)``; the table loaders raise if a path is missing.
        """
        cache_key = (device, i)
        if cache_key in self._csd_prefix_index_per_device:
            return self._csd_prefix_index_per_device[cache_key]

        # Small CPU tensors (shape [V, max_chain]); slices move to device on
        # demand.
        subtoks_t_cpu, lengths_t_cpu = self._csd_fwd_table(i)
        subtoks_s_cpu, lengths_s_cpu = self._csd_rev_table(i)

        # Build the prefix-keyed dicts.
        forward: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]] = (
            collections.defaultdict(list)
        )
        for s_id, length in enumerate(lengths_t_cpu.tolist()):
            length = int(length)
            if length <= 1 or length > subtoks_t_cpu.size(1):
                continue
            row = subtoks_t_cpu[s_id, :length]
            if bool((row < 0).any().item()):
                continue
            prefix = tuple(int(x) for x in row[: length - 1].tolist())
            final_t = int(row[length - 1].item())
            forward[(length, prefix)].append((int(s_id), final_t))

        reverse: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]] = (
            collections.defaultdict(list)
        )
        for t_id, length in enumerate(lengths_s_cpu.tolist()):
            length = int(length)
            if length <= 1 or length > subtoks_s_cpu.size(1):
                continue
            row = subtoks_s_cpu[t_id, :length]
            if bool((row < 0).any().item()):
                continue
            prefix = tuple(int(x) for x in row[: length - 1].tolist())
            final_s = int(row[length - 1].item())
            reverse[(length, prefix)].append((int(t_id), final_s))

        # Pre-dedupe. The same env override
        # gate is preserved so A/B verification stays available.
        _prededuped = os.environ.get("NRL_CSD_PREDEDUP_PREFIX_INDEX", "1") == "1"

        def _dedup_by_second(pairs_list):
            seen_second = set()
            out = []
            for p in pairs_list:
                second = p[1]
                if second in seen_second:
                    continue
                seen_second.add(second)
                out.append(p)
            return tuple(out)

        if _prededuped:
            cache: dict[str, Any] = {
                "forward": {
                    key: _dedup_by_second(value) for key, value in forward.items()
                },
                "reverse": {
                    key: _dedup_by_second(value) for key, value in reverse.items()
                },
                "_prededuped": True,
            }
        else:
            cache = {
                "forward": {key: tuple(value) for key, value in forward.items()},
                "reverse": {key: tuple(value) for key, value in reverse.items()},
                "_prededuped": False,
            }
        # Cache the table sizes used by the chunk-classification skip rules.
        cache["max_chain_t"] = int(subtoks_t_cpu.size(1))
        cache["max_chain_s"] = int(subtoks_s_cpu.size(1))
        self._csd_prefix_index_per_device[cache_key] = cache
        return cache

    @staticmethod
    def _append_rest_bucket_logp(
        logp: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Append log probability for the complement of a support set.

        ``eps`` is the clamp floor for the complement mass.
        """
        if logp.numel() == 0:
            return logp
        work = logp.float() if logp.dtype in (torch.float16, torch.bfloat16) else logp
        support_mass = work.exp().sum(dim=-1, keepdim=True)
        rest_mass = (1.0 - support_mass).clamp(min=eps)
        return torch.cat([work, rest_mass.log()], dim=-1)

    @staticmethod
    def _binary_power_bce_from_logp(
        student_logp_with_rest: torch.Tensor,
        teacher_logp_with_rest: torch.Tensor,
        tau: float = 1.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Soft-label BCE on ALM's binary realized-vs-rest distribution.

        ``student_logp_with_rest`` and ``teacher_logp_with_rest`` must have
        exactly two columns: realized-chain log-probability and rest-bucket
        log-probability. The BCE input and target follow:

            input  = p_student_realized ** (1 / tau)
            target = p_teacher_realized ** (1 / tau)

        The returned tensor has the same trailing width of 2, with the realized
        and rest BCE terms split so existing support/rest masking still applies.
        """
        if tau <= 0.0:
            raise ValueError(f"csd_alm_bce_tau must be > 0, got {tau}")
        if (
            student_logp_with_rest.shape[-1] != 2
            or teacher_logp_with_rest.shape[-1] != 2
        ):
            raise ValueError(
                "ALM BCE expects binary realized-vs-rest support; got "
                f"student shape {tuple(student_logp_with_rest.shape)} and "
                f"teacher shape {tuple(teacher_logp_with_rest.shape)}"
            )

        work_dtype = (
            torch.float32
            if student_logp_with_rest.dtype in (torch.float16, torch.bfloat16)
            else student_logp_with_rest.dtype
        )
        eps = max(float(eps), float(torch.finfo(work_dtype).eps))
        exponent = 1.0 / float(tau)
        student_real = (
            student_logp_with_rest[..., 0]
            .to(work_dtype)
            .exp()
            .clamp(min=0.0, max=1.0)
            .pow(exponent)
            .clamp(min=eps, max=1.0 - eps)
        )
        teacher_real = (
            teacher_logp_with_rest[..., 0]
            .to(work_dtype)
            .exp()
            .clamp(min=0.0, max=1.0)
            .pow(exponent)
            .detach()
        )
        realized_term = -teacher_real * student_real.log()
        rest_term = -(1.0 - teacher_real) * torch.log1p(-student_real)
        return torch.stack([realized_term, rest_term], dim=-1)

    @staticmethod
    def _topk_mask_by_score(
        candidate_mask: torch.Tensor,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Keep the row-wise top-k valid candidates by score."""
        k = int(k)
        if k <= 0:
            return torch.zeros_like(candidate_mask, dtype=torch.bool)
        candidate_mask = candidate_mask.to(torch.bool)
        if candidate_mask.shape[-1] <= k:
            return candidate_mask
        masked_scores = torch.where(
            candidate_mask,
            scores,
            torch.full_like(scores, -1.0e30),
        )
        _, keep_idx = torch.topk(masked_scores, k=k, dim=-1)
        keep_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=keep_idx, value=True)
        return keep_mask & candidate_mask

    @staticmethod
    def _lookup_cp_sharded_teacher_logits(
        teacher_logits: torch.Tensor,
        batch_indices: torch.Tensor,
        global_positions: torch.Tensor,
        token_ids: torch.Tensor,
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Fetch arbitrary dense-teacher logits without gathering ``[B,T,V]``.

        ``teacher_logits`` is the caller's contiguous CP-local sequence window
        with a fully reconstructed vocabulary.  The query tensors broadcast to
        one common shape and use global sequence positions.  Under CP, exactly
        one rank owns each position; that owner fills the queried values and a
        no-grad SUM makes the small result available on every CP replica.  This
        preserves CSD's replicated numerator while avoiding a full dense
        sequence gather for Qwen-sized vocabularies.

        Every caller must issue the same queries in the same order on every CP
        rank.  CSD constructs them only from globally gathered ids/spans and
        replicated alignment metadata.
        """
        batch_indices, global_positions, token_ids = torch.broadcast_tensors(
            batch_indices,
            global_positions,
            token_ids,
        )
        if batch_indices.numel() == 0:
            return torch.empty_like(batch_indices, dtype=teacher_logits.dtype)

        cp_size = (
            torch.distributed.get_world_size(cp_group)
            if cp_group is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(cp_group) > 1
            else 1
        )
        if cp_size == 1:
            return teacher_logits[batch_indices, global_positions, token_ids]

        batch_size, local_seq_len, vocab_size = teacher_logits.shape
        full_seq_len = local_seq_len * cp_size
        # These tensors are replicated query metadata. Validate on every rank
        # before entering the collective so malformed input cannot strand peers
        # in all_reduce after only the owning rank raises an indexing error.
        if (
            int(batch_indices.min().item()) < 0
            or int(batch_indices.max().item()) >= batch_size
        ):
            raise IndexError(
                "dense teacher batch query is out of bounds: "
                f"batch_size={batch_size}, range="
                f"[{int(batch_indices.min().item())}, "
                f"{int(batch_indices.max().item())}]"
            )
        if (
            int(global_positions.min().item()) < 0
            or int(global_positions.max().item()) >= full_seq_len
        ):
            raise IndexError(
                "dense teacher sequence query is out of bounds: "
                f"full_seq_len={full_seq_len}, range="
                f"[{int(global_positions.min().item())}, "
                f"{int(global_positions.max().item())}]"
            )
        if int(token_ids.min().item()) < 0 or int(token_ids.max().item()) >= vocab_size:
            raise IndexError(
                "dense teacher vocabulary query is out of bounds: "
                f"vocab_size={vocab_size}, range="
                f"[{int(token_ids.min().item())}, {int(token_ids.max().item())}]"
            )

        cp_rank = torch.distributed.get_rank(cp_group)
        flat_batch = batch_indices.reshape(-1)
        flat_positions = global_positions.reshape(-1)
        flat_tokens = token_ids.reshape(-1)
        owners = torch.div(flat_positions, local_seq_len, rounding_mode="floor")
        owned = owners == cp_rank
        values = torch.zeros(
            flat_batch.shape,
            dtype=teacher_logits.dtype,
            device=teacher_logits.device,
        )
        with torch.no_grad():
            if bool(owned.any().item()):
                values[owned] = teacher_logits[
                    flat_batch[owned],
                    flat_positions[owned] - cp_rank * local_seq_len,
                    flat_tokens[owned],
                ]
            torch.distributed.all_reduce(
                values,
                op=torch.distributed.ReduceOp.SUM,
                group=cp_group,
            )
        return values.view(batch_indices.shape)

    @staticmethod
    def _lookup_tp_cp_sharded_student_logits(
        student_logits: torch.Tensor,
        batch_indices: torch.Tensor,
        global_positions: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Fetch arbitrary differentiable logits from TP/CP student shards.

        ``student_logits`` has a contiguous CP-local sequence window and a
        contiguous TP-local vocabulary interval. Query tensors are globally
        replicated and broadcast to one shape. Exactly one TP/CP coordinate
        owns each value; a TP SUM with rank-local backward and then a CP SUM
        with backward SUM reproduces the former full-gather gradient semantics
        under the outer ``loss / CP`` scaling, without a rank-3 gather.
        """
        query_shape = torch.broadcast_shapes(
            batch_indices.shape,
            global_positions.shape,
            token_ids.shape,
        )
        if query_shape.numel() == 0:
            return torch.empty(
                query_shape,
                dtype=student_logits.dtype,
                device=student_logits.device,
            )

        tp_size = (
            torch.distributed.get_world_size(tp_group)
            if tp_group is not None and torch.distributed.is_initialized()
            else 1
        )
        cp_size = (
            torch.distributed.get_world_size(cp_group)
            if cp_group is not None and torch.distributed.is_initialized()
            else 1
        )
        batch_size, local_seq_len, local_vocab_size = student_logits.shape
        full_seq_len = local_seq_len * cp_size
        full_vocab_size = local_vocab_size * tp_size

        batch_min = int(batch_indices.min().item())
        batch_max = int(batch_indices.max().item())
        if batch_min < 0 or batch_max >= batch_size:
            raise IndexError(
                "student batch query is out of bounds: "
                f"batch_size={batch_size}, range=[{batch_min}, {batch_max}]"
            )
        position_min = int(global_positions.min().item())
        position_max = int(global_positions.max().item())
        if position_min < 0 or position_max >= full_seq_len:
            raise IndexError(
                "student sequence query is out of bounds: "
                f"full_seq_len={full_seq_len}, range="
                f"[{position_min}, {position_max}]"
            )
        token_min = int(token_ids.min().item())
        token_max = int(token_ids.max().item())
        if token_min < 0 or token_max >= full_vocab_size:
            raise IndexError(
                "student vocabulary query is out of bounds: "
                f"full_vocab_size={full_vocab_size}, range="
                f"[{token_min}, {token_max}]"
            )

        if tp_size == 1 and cp_size == 1:
            return student_logits[batch_indices, global_positions, token_ids]

        tp_rank = torch.distributed.get_rank(tp_group) if tp_size > 1 else 0
        cp_rank = torch.distributed.get_rank(cp_group) if cp_size > 1 else 0
        position_owners = torch.div(
            global_positions,
            local_seq_len,
            rounding_mode="floor",
        )
        token_owners = torch.div(
            token_ids,
            local_vocab_size,
            rounding_mode="floor",
        )
        local_values = student_logits[
            batch_indices,
            global_positions.remainder(local_seq_len),
            token_ids.remainder(local_vocab_size),
        ]
        # Keep the separable owner masks compact. In the dense-common path their
        # shapes are [rows, 1] and [1, vocab], so broadcasting them only inside
        # ``where`` avoids retaining full [rows, vocab] index/mask tensors in the
        # autograd graph. ``where`` also avoids propagating a non-owner NaN via
        # ``NaN * 0``.
        values = torch.where(position_owners == cp_rank, local_values, 0.0)
        values = torch.where(token_owners == tp_rank, values, 0.0)
        values = group_all_reduce_sum_with_grad(values, tp_group)
        values = group_all_reduce_sum_with_grad_backward_sum(values, cp_group)
        return values

    @staticmethod
    def _cp_sharded_teacher_target_in_topk(
        teacher_logits: torch.Tensor,
        *,
        batch_index: int,
        global_position: int,
        token_id: int,
        k: int,
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> bool:
        """Return exact target-in-top-k membership for a CP-sharded dense row."""
        batch_size, local_seq_len, vocab_size = teacher_logits.shape
        cp_size = (
            torch.distributed.get_world_size(cp_group)
            if cp_group is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(cp_group) > 1
            else 1
        )
        full_seq_len = local_seq_len * cp_size
        if not 0 <= batch_index < batch_size:
            raise IndexError(
                f"teacher top-k batch index {batch_index} outside [0, {batch_size})"
            )
        if not 0 <= global_position < full_seq_len:
            raise IndexError(
                "teacher top-k position outside the full sequence: "
                f"position={global_position}, full_seq_len={full_seq_len}"
            )
        if not 0 <= token_id < vocab_size:
            raise IndexError(
                f"teacher top-k token id {token_id} outside [0, {vocab_size})"
            )
        k = min(int(k), vocab_size)
        if k <= 0:
            return False
        if k == vocab_size:
            return True

        if cp_size == 1:
            topk_ids = torch.topk(
                teacher_logits[batch_index, global_position],
                k=k,
                dim=-1,
            ).indices
            return bool((topk_ids == token_id).any().item())

        cp_rank = torch.distributed.get_rank(cp_group)
        owner = global_position // local_seq_len
        found = torch.zeros((), dtype=torch.int32, device=teacher_logits.device)
        with torch.no_grad():
            if cp_rank == owner:
                local_position = global_position - owner * local_seq_len
                topk_ids = torch.topk(
                    teacher_logits[batch_index, local_position],
                    k=k,
                    dim=-1,
                ).indices
                found.fill_(int(bool((topk_ids == token_id).any().item())))
            torch.distributed.all_reduce(
                found,
                op=torch.distributed.ReduceOp.SUM,
                group=cp_group,
            )
        return bool(found.item())

    @staticmethod
    def _unique_bidir_pairs_cpu(
        pairs,
        swap: bool = False,
        assume_unique: bool = False,
    ) -> tuple[list[int], list[int]]:
        """CPU dedupe of ``(s_id, t_id)`` pairs.

        When ``assume_unique=True`` we skip the Python set dedupe loop and
        unpack via numpy. ``swap=True`` treats input as ``(t_id, s_id)``
        (used for the reverse prefix-index branch).
        """
        if not pairs:
            return [], []
        if assume_unique:
            arr = np.asarray(pairs, dtype=np.int64)
            if swap:
                return arr[:, 1].tolist(), arr[:, 0].tolist()
            return arr[:, 0].tolist(), arr[:, 1].tolist()
        seen_s: set[int] = set()
        seen_t: set[int] = set()
        s_list: list[int] = []
        t_list: list[int] = []
        for p in pairs:
            if swap:
                t_id = int(p[0])
                s_id = int(p[1])
            else:
                s_id = int(p[0])
                t_id = int(p[1])
            if s_id in seen_s or t_id in seen_t:
                continue
            seen_s.add(s_id)
            seen_t.add(t_id)
            s_list.append(s_id)
            t_list.append(t_id)
        return s_list, t_list

    def _partition_kl_mismatch_batched(
        self,
        chunk_records,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        student_log_z: torch.Tensor,
        teacher_log_z: torch.Tensor,
        temperature: float,
        loss_fn: str,
        teacher_sparse: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        teacher_sparse_keep_realized: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, int, int, list[int], list[bool]]:
        """Batched partition-KL over a flat list of mismatch chunk records.

        Each record is the 11-tuple
        ``(b, s_last, t_last, s_prefix_pred_pos, s_prefix_label_id,
        t_prefix_pred_pos, t_prefix_label_id, s_ids_list, t_ids_list,
        realized_s_last_id, realized_t_last_id)``.
        Returns ``(loss_sum, used_count, matches, support_sizes, valid_flags)``.
        """
        device = student_logits.device
        accum_dtype = (
            torch.float32
            if student_logits.dtype in (torch.float16, torch.bfloat16)
            else student_logits.dtype
        )
        if not chunk_records:
            zero = torch.tensor(0.0, device=device, dtype=accum_dtype)
            return zero, 0, 0, [], []

        K = len(chunk_records)
        Pmax_s = max(len(r[3]) for r in chunk_records)
        Pmax_t = max(len(r[5]) for r in chunk_records)
        Smax = max(len(r[7]) for r in chunk_records)
        # Sentinel that survives kl_div: exp(-1e30) underflows to 0 in fp32 so
        # the masked entry contributes 0 to support-mass and to the per-row KL
        # sum, while target - input stays finite (no 0*nan).
        NEG_INF = -1.0e30

        # The NumPy fast path is enabled by default. The Python-list branch
        # can be selected for parity verification through the environment.
        _use_np = os.environ.get("NRL_CSD_BATCHED_HELPER_NUMPY", "1") == "1"

        b_idx_list = [r[0] for r in chunk_records]
        s_last_list = [r[1] for r in chunk_records]
        t_last_list = [r[2] for r in chunk_records]
        s_realized_list = [r[9] if len(r) > 9 else -1 for r in chunk_records]
        t_realized_list = [r[10] if len(r) > 10 else -1 for r in chunk_records]
        support_sizes = [len(r[7]) for r in chunk_records]

        np_accum_dtype = np.float32 if accum_dtype == torch.float32 else np.float64

        s_pred_pos = None
        s_label_id = None
        s_pmask = None
        t_pred_pos = None
        t_label_id = None
        t_pmask = None

        if _use_np:
            s_ids_np = np.zeros((K, Smax), dtype=np.int64)
            t_ids_np = np.zeros((K, Smax), dtype=np.int64)
            smask_np = np.zeros((K, Smax), dtype=np_accum_dtype)
            if Pmax_s > 0:
                s_pred_pos_np = np.zeros((K, Pmax_s), dtype=np.int64)
                s_label_id_np = np.zeros((K, Pmax_s), dtype=np.int64)
                s_pmask_np = np.zeros((K, Pmax_s), dtype=np_accum_dtype)
            if Pmax_t > 0:
                t_pred_pos_np = np.zeros((K, Pmax_t), dtype=np.int64)
                t_label_id_np = np.zeros((K, Pmax_t), dtype=np.int64)
                t_pmask_np = np.zeros((K, Pmax_t), dtype=np_accum_dtype)
            for i, r in enumerate(chunk_records):
                s_pp, s_lid = r[3], r[4]
                t_pp, t_lid = r[5], r[6]
                s_list, t_list = r[7], r[8]
                ns_p = len(s_pp)
                if ns_p > 0:
                    s_pred_pos_np[i, :ns_p] = s_pp
                    s_label_id_np[i, :ns_p] = s_lid
                    s_pmask_np[i, :ns_p] = 1.0
                nt_p = len(t_pp)
                if nt_p > 0:
                    t_pred_pos_np[i, :nt_p] = t_pp
                    t_label_id_np[i, :nt_p] = t_lid
                    t_pmask_np[i, :nt_p] = 1.0
                n_sup = len(s_list)
                if n_sup > 0:
                    s_ids_np[i, :n_sup] = s_list
                    t_ids_np[i, :n_sup] = t_list
                    smask_np[i, :n_sup] = 1.0

            b_idx = torch.from_numpy(np.asarray(b_idx_list, dtype=np.int64)).to(
                device, non_blocking=True
            )
            s_last = torch.from_numpy(np.asarray(s_last_list, dtype=np.int64)).to(
                device, non_blocking=True
            )
            t_last = torch.from_numpy(np.asarray(t_last_list, dtype=np.int64)).to(
                device, non_blocking=True
            )
            s_realized_last = torch.from_numpy(
                np.asarray(s_realized_list, dtype=np.int64)
            ).to(device, non_blocking=True)
            t_realized_last = torch.from_numpy(
                np.asarray(t_realized_list, dtype=np.int64)
            ).to(device, non_blocking=True)
            s_ids = torch.from_numpy(s_ids_np).to(device, non_blocking=True)
            t_ids = torch.from_numpy(t_ids_np).to(device, non_blocking=True)
            support_mask = torch.from_numpy(smask_np).to(
                device,
                dtype=accum_dtype,
                non_blocking=True,
            )
            if Pmax_s > 0:
                s_pred_pos = torch.from_numpy(s_pred_pos_np).to(
                    device,
                    non_blocking=True,
                )
                s_label_id = torch.from_numpy(s_label_id_np).to(
                    device,
                    non_blocking=True,
                )
                s_pmask = torch.from_numpy(s_pmask_np).to(
                    device,
                    dtype=accum_dtype,
                    non_blocking=True,
                )
            if Pmax_t > 0:
                t_pred_pos = torch.from_numpy(t_pred_pos_np).to(
                    device,
                    non_blocking=True,
                )
                t_label_id = torch.from_numpy(t_label_id_np).to(
                    device,
                    non_blocking=True,
                )
                t_pmask = torch.from_numpy(t_pmask_np).to(
                    device,
                    dtype=accum_dtype,
                    non_blocking=True,
                )
        else:
            s_pred_pos_buf = [[0] * Pmax_s for _ in range(K)]
            s_label_id_buf = [[0] * Pmax_s for _ in range(K)]
            s_pmask_buf = [[0.0] * Pmax_s for _ in range(K)]
            t_pred_pos_buf = [[0] * Pmax_t for _ in range(K)]
            t_label_id_buf = [[0] * Pmax_t for _ in range(K)]
            t_pmask_buf = [[0.0] * Pmax_t for _ in range(K)]
            s_ids_buf = [[0] * Smax for _ in range(K)]
            t_ids_buf = [[0] * Smax for _ in range(K)]
            smask_buf = [[0.0] * Smax for _ in range(K)]

            for i, r in enumerate(chunk_records):
                s_pp, s_lid = r[3], r[4]
                t_pp, t_lid = r[5], r[6]
                s_list, t_list = r[7], r[8]
                for j, (p, lid) in enumerate(zip(s_pp, s_lid)):
                    s_pred_pos_buf[i][j] = p
                    s_label_id_buf[i][j] = lid
                    s_pmask_buf[i][j] = 1.0
                for j, (p, lid) in enumerate(zip(t_pp, t_lid)):
                    t_pred_pos_buf[i][j] = p
                    t_label_id_buf[i][j] = lid
                    t_pmask_buf[i][j] = 1.0
                n_sup = len(s_list)
                for j in range(n_sup):
                    s_ids_buf[i][j] = s_list[j]
                    t_ids_buf[i][j] = t_list[j]
                    smask_buf[i][j] = 1.0

            b_idx = torch.tensor(b_idx_list, device=device, dtype=torch.long)
            s_last = torch.tensor(
                s_last_list,
                device=device,
                dtype=torch.long,
            )
            t_last = torch.tensor(
                t_last_list,
                device=device,
                dtype=torch.long,
            )
            s_realized_last = torch.tensor(
                s_realized_list,
                device=device,
                dtype=torch.long,
            )
            t_realized_last = torch.tensor(
                t_realized_list,
                device=device,
                dtype=torch.long,
            )
            s_ids = torch.tensor(s_ids_buf, device=device, dtype=torch.long)
            t_ids = torch.tensor(t_ids_buf, device=device, dtype=torch.long)
            support_mask = torch.tensor(
                smask_buf,
                device=device,
                dtype=accum_dtype,
            )
            if Pmax_s > 0:
                s_pred_pos = torch.tensor(
                    s_pred_pos_buf,
                    device=device,
                    dtype=torch.long,
                )
                s_label_id = torch.tensor(
                    s_label_id_buf,
                    device=device,
                    dtype=torch.long,
                )
                s_pmask = torch.tensor(
                    s_pmask_buf,
                    device=device,
                    dtype=accum_dtype,
                )
            if Pmax_t > 0:
                t_pred_pos = torch.tensor(
                    t_pred_pos_buf,
                    device=device,
                    dtype=torch.long,
                )
                t_label_id = torch.tensor(
                    t_label_id_buf,
                    device=device,
                    dtype=torch.long,
                )
                t_pmask = torch.tensor(
                    t_pmask_buf,
                    device=device,
                    dtype=accum_dtype,
                )

        # ---- Student prefix chain (batched gather) ----
        if Pmax_s > 0:
            assert s_pred_pos is not None
            assert s_label_id is not None
            assert s_pmask is not None
            s_prefix_logits = (
                self._lookup_tp_cp_sharded_student_logits(
                    student_logits,
                    b_idx[:, None],
                    s_pred_pos,
                    s_label_id,
                    tp_group=tp_group,
                    cp_group=cp_group,
                )
                / temperature
            )
            s_prefix_scores = (
                s_prefix_logits.to(accum_dtype)
                - student_log_z[b_idx[:, None], s_pred_pos]
            )
            s_prefix_logp = (s_prefix_scores * s_pmask).sum(dim=-1)  # (K,)
        else:
            s_prefix_logp = torch.zeros(
                (K,),
                device=device,
                dtype=accum_dtype,
            )

        # ---- Teacher prefix chain (batched gather; detached) ----
        prefix_found = torch.ones((K,), device=device, dtype=torch.bool)
        if Pmax_t > 0:
            assert t_pred_pos is not None
            assert t_label_id is not None
            assert t_pmask is not None
            if teacher_sparse is not None:
                teacher_topk_logits, teacher_topk_indices = teacher_sparse
                t_prefix_scores, t_prefix_found = self._lookup_sparse_teacher_logp(
                    teacher_topk_logits,
                    teacher_topk_indices,
                    teacher_log_z,
                    b_idx,
                    t_pred_pos,
                    t_label_id,
                    temperature,
                )
                t_prefix_scores = torch.where(
                    t_prefix_found,
                    t_prefix_scores.to(accum_dtype),
                    torch.zeros_like(t_prefix_scores, dtype=accum_dtype),
                )
                prefix_found = (t_prefix_found | (t_pmask <= 0)).all(dim=-1)
            else:
                assert teacher_logits is not None
                t_prefix_logits = (
                    self._lookup_cp_sharded_teacher_logits(
                        teacher_logits,
                        b_idx[:, None],
                        t_pred_pos,
                        t_label_id,
                        cp_group,
                    )
                    / temperature
                )
                t_prefix_scores = (
                    t_prefix_logits.to(accum_dtype)
                    - teacher_log_z[b_idx[:, None], t_pred_pos]
                )
            t_prefix_logp = (t_prefix_scores * t_pmask).sum(dim=-1)
        else:
            t_prefix_logp = torch.zeros(
                (K,),
                device=device,
                dtype=accum_dtype,
            )

        # ---- Final-position support scores (batched gather) ----
        s_final_logits = (
            self._lookup_tp_cp_sharded_student_logits(
                student_logits,
                b_idx[:, None],
                s_last[:, None],
                s_ids,
                tp_group=tp_group,
                cp_group=cp_group,
            )
            / temperature
        )
        s_final_scores = s_final_logits.to(accum_dtype) - student_log_z[
            b_idx, s_last
        ].unsqueeze(-1)
        s_full_scores = s_final_scores + s_prefix_logp.unsqueeze(-1)
        s_full_scores = torch.where(
            support_mask > 0,
            s_full_scores,
            torch.full_like(s_full_scores, NEG_INF),
        )

        with torch.no_grad():
            if teacher_sparse is not None:
                teacher_topk_logits, teacher_topk_indices = teacher_sparse
                t_last_pos = t_last.unsqueeze(-1).expand_as(t_ids)
                t_final_scores, t_final_found = self._lookup_sparse_teacher_logp(
                    teacher_topk_logits,
                    teacher_topk_indices,
                    teacher_log_z,
                    b_idx,
                    t_last_pos,
                    t_ids,
                    temperature,
                )
                if teacher_sparse_keep_realized:
                    topk_width = int(teacher_topk_indices.shape[-1])
                    alt_budget = max(topk_width - 1, 0)
                    realized_pair_mask = (
                        (support_mask > 0)
                        & (s_ids == s_realized_last[:, None])
                        & (t_ids == t_realized_last[:, None])
                    )
                    alt_candidate_mask = (
                        (support_mask > 0) & t_final_found & (~realized_pair_mask)
                    )
                    alt_keep_mask = self._topk_mask_by_score(
                        alt_candidate_mask,
                        t_final_scores.to(accum_dtype),
                        alt_budget,
                    )
                    final_keep_mask = (
                        realized_pair_mask & t_final_found
                    ) | alt_keep_mask
                else:
                    final_keep_mask = (support_mask > 0) & t_final_found
                teacher_support_mask = final_keep_mask & prefix_found.unsqueeze(-1)
                t_full_scores = t_final_scores.to(
                    accum_dtype
                ) + t_prefix_logp.unsqueeze(-1)
                support_mask = support_mask * teacher_support_mask.to(accum_dtype)
            else:
                assert teacher_logits is not None
                t_final_logits = (
                    self._lookup_cp_sharded_teacher_logits(
                        teacher_logits,
                        b_idx[:, None],
                        t_last[:, None],
                        t_ids,
                        cp_group,
                    )
                    / temperature
                )
                t_final_scores = t_final_logits.to(accum_dtype) - teacher_log_z[
                    b_idx, t_last
                ].unsqueeze(-1)
                t_full_scores = t_final_scores + t_prefix_logp.unsqueeze(-1)
            t_full_scores = torch.where(
                support_mask > 0,
                t_full_scores,
                torch.full_like(t_full_scores, NEG_INF),
            )

        # Teacher sparse support pruning is computed without gradients, but the
        # student scores must be masked with autograd enabled.
        s_full_scores = torch.where(
            support_mask > 0,
            s_full_scores,
            torch.full_like(s_full_scores, NEG_INF),
        )

        # ---- Append REST bucket (per-row valid-mass complement) ----
        s_logp_with_rest = self._append_rest_bucket_logp(s_full_scores)
        t_logp_with_rest = self._append_rest_bucket_logp(t_full_scores)

        # Validity: row contributes if it has at least 1 support position.
        valid_row = support_mask.sum(dim=-1) >= 1.0  # (K,) bool

        # ---- Batched partition loss ----
        kl_per_pos = self._partition_divergence(
            s_logp_with_rest, t_logp_with_rest, loss_fn
        )
        # Mask padded support columns to zero (REST column stays).
        rest_one = torch.ones((K, 1), device=device, dtype=accum_dtype)
        full_mask = torch.cat([support_mask, rest_one], dim=-1)
        per_row_kl = (kl_per_pos * full_mask).sum(dim=-1)  # (K,)
        per_row_kl = per_row_kl * valid_row.to(per_row_kl.dtype)
        loss_sum = per_row_kl.sum()

        with torch.no_grad():
            s_arg = s_full_scores.argmax(dim=-1)
            t_arg = t_full_scores.argmax(dim=-1)
            match_row = (s_arg == t_arg) & valid_row

        stats_cpu = torch.stack([valid_row, match_row], dim=0).detach().cpu()
        valid_flags = [bool(x) for x in stats_cpu[0].tolist()]
        matches = int(stats_cpu[1].sum().item())
        used_count = int(sum(valid_flags))
        effective_support_cpu = support_mask.sum(dim=-1).detach().cpu().tolist()
        support_sizes_used = [
            int(size)
            for size, is_valid in zip(effective_support_cpu, valid_flags)
            if is_valid
        ]

        return loss_sum, used_count, matches, support_sizes_used, valid_flags

    def _compute_csd_loss(
        self,
        i: int,
        student_logits: torch.Tensor,
        teacher_full_logits: Optional[torch.Tensor],
        align: LocalizedAlignment,
        *,
        teacher_sparse_payload: Optional[SparseTeacherLogits] = None,
        teacher_vocab_size: int,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        global_valid_chunks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """CSD KD term for cross-tokenizer teacher ``i``.

        Prefix-bidir partition KL with MtoM-as-ALM + rest bucket; mismatch
        chunks are weighted by ``csd_mismatch_loss_beta``.
        ``mtom_as_alm``/``rest_bucket`` are baked in for CSD.

        Per-teacher contract (mirrors ``_compute_same_vocab_kl``):
        ``student_logits`` is this rank's CP-contiguous window,
        ``teacher_full_logits`` is the reassembled full-vocab teacher window,
        and ``align`` carries the localized student/teacher input ids, per-chunk
        global position spans, ``pair_valid`` and ``num_chunks``. The body is
        native TP/CP-correct: student and dense-teacher rank-3 logits remain
        sharded. Exact full-vocabulary log-normalizers are computed in bounded
        TP chunks and gathered across CP only after reducing to ``[B,T]``;
        arbitrary support values use bounded owner reductions. Production
        supplies ``global_valid_chunks`` from the full batch; direct callers
        retain a WORLD-reduced fallback. Collectives are no-ops at world size 1
        (single-GPU byte-exact).

        Returns ``(loss, metrics_dict)``.
        """
        cfg = self.cfg
        device = student_logits.device
        # ----- Native TP/CP sharded access -------------------------------
        # Both rank-3 tensors stay sequence-local. Student logits additionally
        # stay vocabulary-local; all arbitrary student support accesses below
        # use differentiable TP/CP owner reductions. Spans / input_ids arrive
        # global from the loss-input adapter.
        if teacher_sparse_payload is None:
            if teacher_full_logits is None:
                raise ValueError("Dense CSD transport is missing teacher logits.")
        elif teacher_full_logits is not None:
            raise ValueError(
                "CSD received both dense and sparse teacher-logit payloads."
            )
        # B is the per-sample loop bound; S is the global student-side
        # predictor-position bound used in the classify-loop range check.
        B, student_local_seq_len, _ = student_logits.shape
        student_tp_size = (
            torch.distributed.get_world_size(tp_group)
            if tp_group is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(tp_group) > 1
            else 1
        )
        student_cp_size = (
            torch.distributed.get_world_size(cp_group)
            if cp_group is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(cp_group) > 1
            else 1
        )
        S = student_local_seq_len * student_cp_size

        # Per-teacher CSD settings.
        kl_chunk_shift = bool(cfg.kl_chunk_shift)
        temperature = float(cfg.temperature)
        csd_loss_fn = str(cfg.csd_loss_fn)
        csd_last_pos_loss_fn_cfg = cfg.csd_last_pos_loss_fn
        pure_alm = bool(cfg.csd_pure_alm)
        realized_only_mismatch = bool(cfg.csd_realized_only_mismatch)
        # CSD hybrid: None means "inherit csd_loss_fn"; "kl"/"jsd" overrides
        # only the position-N-1 (last-position) partition KL.
        effective_last_pos_loss_fn = (
            csd_last_pos_loss_fn_cfg
            if csd_last_pos_loss_fn_cfg is not None
            else csd_loss_fn
        )
        mismatch_loss_beta = float(cfg.csd_mismatch_loss_beta)
        noise_filter_topk = int(cfg.csd_noise_filter_topk or 0)
        if noise_filter_topk < 0:
            raise ValueError(
                f"csd_noise_filter_topk must be >= 0, got {noise_filter_topk}"
            )
        teacher_topk_ipc_k = int(cfg.teacher_topk_ipc_k)
        teacher_topk_keep_realized = teacher_topk_ipc_k > 0 and (
            pure_alm
            or realized_only_mismatch
            or teacher_topk_ipc_k == 1
            or bool(cfg.teacher_topk_ipc_keep_realized)
        )
        # Multi-token chunks use realized-chain probabilities and a rest bucket.
        mtom_as_alm = True

        v_t = int(teacher_vocab_size)
        teacher_gt_in_topk: Optional[torch.Tensor] = None
        teacher_logits: Optional[torch.Tensor] = None
        if teacher_sparse_payload is not None:
            (
                teacher_topk_logits,
                _,
                teacher_log_z,
                teacher_gt_in_topk,
            ) = teacher_sparse_payload
            B2, T, _ = teacher_topk_logits.shape
        else:
            assert teacher_full_logits is not None
            # Slice teacher logits to the real tokenizer vocab (drop any padded
            # lm_head columns beyond len(tokenizer)) BEFORE indexing.
            if teacher_full_logits.shape[-1] > v_t:
                teacher_full_logits = teacher_full_logits[..., :v_t]
            teacher_logits = teacher_full_logits
            # V_t is implicit (common_teacher_idx_t bounds it). Dense teacher
            # logits are a contiguous CP-local sequence window, while spans and
            # input ids use global positions.
            B2, teacher_local_seq_len, _ = teacher_logits.shape
            teacher_cp_size = (
                torch.distributed.get_world_size(cp_group)
                if cp_group is not None
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(cp_group) > 1
                else 1
            )
            T = teacher_local_seq_len * teacher_cp_size
        assert B == B2, f"batch size mismatch: student={B} teacher={B2}"

        # ----- Lazy caches (per-device) ----------------------------------
        if pure_alm:
            common_student_idx_t = torch.empty(
                (0,),
                device=device,
                dtype=torch.long,
            )
            common_teacher_idx_t = torch.empty(
                (0,),
                device=device,
                dtype=torch.long,
            )
            has_common_support = True
        else:
            common_student_idx_t, common_teacher_idx_t = self._get_common_indices_csd(
                device,
                i,
            )
            has_common_support = common_student_idx_t.numel() > 0
            if not has_common_support:
                # Every 1-to-1 chunk would silently fall through to the mismatch
                # path, zeroing KL/common and making
                # top1_acc_per_chunk meaningless (it is normalized by a common
                # count of 0). That looks like a converging run, so fail loudly
                # instead: an empty common vocab means the projection matrix or
                # subtoks table does not match this (student, teacher) pair.
                source = (
                    f"student_to_teacher_table_paths[{i}]={self.student_to_teacher_table_paths[i]!r}"
                    if self.common_indices_from_subtoks
                    else f"projection_matrix_paths[{i}]={self.projection_matrix_paths[i]!r}"
                )
                raise ValueError(
                    f"Teacher {i}: the exact 1-to-1 common-vocab map is empty, so "
                    f"no chunk can take the common path. Derived from {source} "
                    f"(common_indices_from_subtoks="
                    f"{self.common_indices_from_subtoks}). Check that this table "
                    f"was built for this student/teacher tokenizer pair."
                )
        # ----- Single-shot host pull of input_ids ------------------------
        # One .detach().cpu().numpy() per tensor, then per-chunk numpy indexing
        # inside the classify/collect loop. The ids ride on the localized
        # alignment as this CP rank's contiguous window (both unshifted; the
        # classify loop applies its own per-chunk kl_chunk_shift); they are
        # CP-gathered to the full sequence to match the gathered logits and the
        # global spans (no-op at CP=1).
        assert align.student_input_ids is not None
        assert align.teacher_input_ids is not None
        input_ids_student = to_local_if_dtensor(align.student_input_ids)
        input_ids_teacher = to_local_if_dtensor(align.teacher_input_ids)
        with torch.no_grad():
            input_ids_student = allgather_cp_contiguous_tensor(
                input_ids_student, cp_group
            )
            input_ids_teacher = allgather_cp_contiguous_tensor(
                input_ids_teacher, cp_group
            )
        inp_s_np = input_ids_student.detach().cpu().numpy()
        inp_t_np = input_ids_teacher.detach().cpu().numpy()

        teacher_gt_in_topk_np: Optional[np.ndarray] = None
        teacher_filter_topk_indices_np: Optional[np.ndarray] = None
        dense_noise_filter_cp_sharded = False
        filter_k = 0
        if noise_filter_topk > 0:
            if teacher_sparse_payload is not None:
                if teacher_gt_in_topk is None:
                    raise ValueError(
                        "csd_noise_filter_topk requires sparse "
                        "teacher IPC handles with gt_in_topk metadata. Ensure "
                        "the teacher IPC path receives force_include_token_ids."
                    )
                teacher_gt_in_topk_np = (
                    teacher_gt_in_topk.detach().cpu().numpy().astype(bool)
                )
            else:
                assert teacher_logits is not None
                filter_k = min(noise_filter_topk, int(teacher_logits.shape[-1]))
                dense_noise_filter_cp_sharded = teacher_cp_size > 1
                if filter_k > 0 and not dense_noise_filter_cp_sharded:
                    with torch.no_grad():
                        teacher_filter_topk_indices_np = (
                            torch.topk(
                                teacher_logits,
                                k=filter_k,
                                dim=-1,
                            )
                            .indices.detach()
                            .cpu()
                            .numpy()
                        )

        def _teacher_targets_pass_noise_filter(
            b_idx: int,
            pred_positions: list[int],
            label_positions: list[int],
        ) -> bool:
            if noise_filter_topk <= 0:
                return True
            if len(pred_positions) != len(label_positions):
                return False
            if teacher_gt_in_topk_np is not None:
                for pred_pos in pred_positions:
                    if (
                        pred_pos < 0
                        or pred_pos >= teacher_gt_in_topk_np.shape[1]
                        or not bool(teacher_gt_in_topk_np[b_idx, pred_pos])
                    ):
                        return False
                return True
            for pred_pos, label_pos in zip(pred_positions, label_positions):
                if pred_pos < 0 or pred_pos >= T:
                    return False
                if label_pos < 0 or label_pos >= inp_t_np.shape[1]:
                    return False
                label_id = int(inp_t_np[b_idx, label_pos])
                if dense_noise_filter_cp_sharded:
                    assert teacher_logits is not None
                    found = self._cp_sharded_teacher_target_in_topk(
                        teacher_logits,
                        batch_index=b_idx,
                        global_position=pred_pos,
                        token_id=label_id,
                        k=filter_k,
                        cp_group=cp_group,
                    )
                elif teacher_filter_topk_indices_np is not None:
                    found = bool(
                        np.any(
                            teacher_filter_topk_indices_np[b_idx, pred_pos] == label_id
                        )
                    )
                else:
                    found = False
                if not found:
                    return False
            return True

        # ----- Alignment payload (padded tensors -> per-chunk records) ---
        # The collator ships padded per-chunk position spans + a validity mask
        # (dense over max_pairs); a sentinel / OOB pair keeps its slot and is
        # zeroed by ``pair_valid``. Spans / pair_valid / num_chunks ride on the
        # localized alignment.
        assert align.student_spans is not None
        assert align.teacher_spans is not None
        assert align.pair_valid is not None
        assert align.num_chunks is not None
        s_spans = to_local_if_dtensor(align.student_spans)  # [B, max_pairs, 2]
        t_spans = to_local_if_dtensor(align.teacher_spans)  # [B, max_pairs, 2]
        sample_mask = to_local_if_dtensor(align.sample_mask).bool()  # [B]
        if sample_mask.ndim != 1 or sample_mask.shape[0] != B:
            raise ValueError(
                "CSD sample_mask must have shape [B]: "
                f"expected ({B},), got {tuple(sample_mask.shape)}."
            )
        pair_valid = to_local_if_dtensor(align.pair_valid).bool()  # [B, max_pairs]
        if pair_valid.ndim != 2 or pair_valid.shape[0] != B:
            raise ValueError(
                "CSD pair_valid must have shape [B, max_pairs]: "
                f"expected first dimension {B}, got {tuple(pair_valid.shape)}."
            )
        # Gate the authoritative chunk-validity tensor before classification.
        # Every downstream numerator, count, metric, and student gradient is
        # derived from the resulting common/mismatch cohorts.
        pair_valid = pair_valid & sample_mask.unsqueeze(-1)
        num_chunks = to_local_if_dtensor(align.num_chunks)  # [B]
        # Host-side copies for the classification loop (no GPU sync per
        # chunk).
        s_spans_np = s_spans.detach().cpu().numpy()
        t_spans_np = t_spans.detach().cpu().numpy()
        pair_valid_np = pair_valid.detach().cpu().numpy().astype(bool)
        num_chunks_np = num_chunks.detach().cpu().numpy().astype(np.int64)

        # ----- Classify chunks ----------------
        common_chunks: list[tuple[int, int, int, int, int]] = []
        mismatch_chunks: list[
            tuple[int, int, int, list[int], list[int], list[int], list[int]]
        ] = []
        noise_filtered_common_chunks = 0
        noise_filtered_mismatch_chunks = 0
        T_s_label = input_ids_student.shape[1]
        T_t_label = input_ids_teacher.shape[1]
        for b in range(B):
            n = int(num_chunks_np[b])
            for k in range(n):
                if not bool(pair_valid_np[b, k]):
                    continue
                s_start = int(s_spans_np[b, k, 0])
                s_end = int(s_spans_np[b, k, 1])
                t_start = int(t_spans_np[b, k, 0])
                t_end = int(t_spans_np[b, k, 1])
                # Explicit -1 spans are invalid even when pair_valid is set.
                if s_start == -1 or t_start == -1:
                    continue
                M = s_end - s_start
                N = t_end - t_start
                if M <= 0 or N <= 0:
                    continue
                # NOTE: preserve per-chunk kl_chunk_shift.
                # Do NOT reuse nemo_rl's tensor-level global shift —
                # CSD needs the conditional per-chunk shift (chunks at
                # sequence-start are NOT shifted).
                if kl_chunk_shift and s_start > 0 and t_start > 0:
                    s_pred = list(range(s_start - 1, s_end - 1))
                    t_pred = list(range(t_start - 1, t_end - 1))
                    s_labels = list(range(s_start, s_end))
                    t_labels = list(range(t_start, t_end))
                else:
                    s_pred = list(range(s_start, s_end))
                    t_pred = list(range(t_start, t_end))
                    s_labels = s_pred
                    t_labels = t_pred

                if any(pos < 0 or pos >= S for pos in s_pred):
                    continue
                if any(pos < 0 or pos >= T for pos in t_pred):
                    continue
                if any(pos < 0 or pos >= T_s_label for pos in s_labels):
                    continue
                if any(pos < 0 or pos >= T_t_label for pos in t_labels):
                    continue

                if M == 1 and N == 1 and (has_common_support or pure_alm):
                    if not _teacher_targets_pass_noise_filter(
                        b,
                        t_pred,
                        t_labels,
                    ):
                        noise_filtered_common_chunks += 1
                        continue
                    common_chunks.append(
                        (
                            b,
                            s_pred[0],
                            t_pred[0],
                            int(inp_s_np[b, s_labels[0]]),
                            int(inp_t_np[b, t_labels[0]]),
                        )
                    )
                else:
                    if not _teacher_targets_pass_noise_filter(
                        b,
                        t_pred,
                        t_labels,
                    ):
                        noise_filtered_mismatch_chunks += 1
                        continue
                    mismatch_chunks.append(
                        (b, M, N, s_pred, t_pred, s_labels, t_labels)
                    )

        # ----- No early-out on empty chunks -----------------------------
        # A microbatch with no valid chunks (empty common + mismatch) is NOT
        # returned early: it must still reach the ``global_valid_chunks``
        # WORLD-reduce below so a rank with zero local chunks participates in
        # the collective (other DP/CP ranks may have chunks). The empty
        # common/mismatch loops leave both partition losses at zero and the
        # combine step returns a gradient-connected zero.
        accum_dtype = (
            torch.float32
            if student_logits.dtype in (torch.float16, torch.bfloat16)
            else student_logits.dtype
        )
        common_loss = torch.tensor(0.0, device=device, dtype=accum_dtype)
        mismatch_loss = torch.tensor(0.0, device=device, dtype=accum_dtype)
        common_matches = 0
        mismatch_matches = 0

        # ----- Pre-compute log_Z's per (b, predictor-position) ----------
        # Partition-KL (rest_bucket=True) needs log Z[b, s] for every
        # (b, predictor) position the chunk loops reference. Chunked along
        # the seq axis to bound peak fp32 temporary.
        def _precompute_local_log_z(_logits: torch.Tensor, _chunk: int = 256):
            _B, _S, _V = _logits.shape
            _out = torch.empty(
                (_B, _S),
                dtype=accum_dtype,
                device=_logits.device,
            )
            for _s in range(0, _S, _chunk):
                _end = min(_s + _chunk, _S)
                _chunk_t = _logits[:, _s:_end, :].to(accum_dtype) / temperature
                _out[:, _s:_end] = torch.logsumexp(_chunk_t, dim=-1)
            return _out

        student_log_z_local = chunked_vocab_parallel_logsumexp(
            student_logits,
            temperature,
            tp_group=tp_group,
            chunk_size=CSD_STUDENT_LOSS_CHUNK_SIZE,
        )
        # Only the scalar normalizer per student row crosses CP. Backward of
        # this contiguous gather SUMs the replicated CP gradient, cancelling
        # the outer loss / CP scaling exactly as the deleted rank-3 gather did.
        student_log_z = allgather_cp_contiguous_tensor(
            student_log_z_local,
            cp_group,
        )
        if teacher_sparse_payload is None:
            assert teacher_logits is not None
            with torch.no_grad():
                teacher_log_z_local = _precompute_local_log_z(teacher_logits)
                # Only the scalar normalizer per teacher row is materialized
                # across CP. The multi-GiB dense logits stay sequence-sharded.
                teacher_log_z = allgather_cp_contiguous_tensor(
                    teacher_log_z_local,
                    cp_group,
                )

        # ----- Common-vocab KL ----------------
        if common_chunks:
            sb_full = torch.tensor(
                [c[0] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )
            sp_full = torch.tensor(
                [c[1] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )
            tp_full = torch.tensor(
                [c[2] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )
            rs_full = torch.tensor(
                [c[3] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )
            rt_full = torch.tensor(
                [c[4] for c in common_chunks],
                device=device,
                dtype=torch.long,
            )

            common_loss_sum = torch.tensor(
                0.0,
                device=device,
                dtype=accum_dtype,
            )
            common_matches_sum_t = torch.zeros(
                (),
                device=device,
                dtype=torch.long,
            )

            _env_common_mb = os.environ.get("NRL_CSD_COMMON_LOOP_MB")
            common_mb = int(_env_common_mb) if _env_common_mb else len(common_chunks)
            if student_tp_size > 1 or student_cp_size > 1:
                # Student owner lookups issue TP/CP collectives. Fix the row
                # bound in code so environment drift cannot change collective
                # counts or order across ranks, regardless of teacher transport.
                common_mb = min(len(common_chunks), 64)
            elif teacher_sparse_payload is not None and not _env_common_mb:
                common_mb = min(common_mb, 64)
            teacher_to_common_student = None
            if teacher_sparse_payload is not None and not pure_alm:
                teacher_to_common_student = self._get_csd_teacher_to_common_student(
                    device,
                    i,
                    v_t,
                    common_student_idx_t,
                    common_teacher_idx_t,
                )

            for mb_start in range(0, len(common_chunks), common_mb):
                mb_end = min(mb_start + common_mb, len(common_chunks))
                sb = sb_full[mb_start:mb_end]
                sp = sp_full[mb_start:mb_end]
                tp = tp_full[mb_start:mb_end]
                rs = rs_full[mb_start:mb_end]
                rt = rt_full[mb_start:mb_end]
                support_mask = None
                common_valid_row = None
                if pure_alm:
                    student_common_logp = (
                        self._lookup_tp_cp_sharded_student_logits(
                            student_logits,
                            sb,
                            sp,
                            rs,
                            tp_group=tp_group,
                            cp_group=cp_group,
                        ).to(accum_dtype)
                        / temperature
                        - student_log_z[sb, sp]
                    ).unsqueeze(-1)
                    if teacher_sparse_payload is None:
                        assert teacher_logits is not None
                        teacher_common_logp = (
                            self._lookup_cp_sharded_teacher_logits(
                                teacher_logits,
                                sb,
                                tp,
                                rt,
                                cp_group,
                            ).to(accum_dtype)
                            / temperature
                            - teacher_log_z[sb, tp]
                        ).unsqueeze(-1)
                        common_valid_row = torch.ones(
                            (sb.shape[0],),
                            device=device,
                            dtype=torch.bool,
                        )
                    else:
                        teacher_topk_logits, teacher_topk_indices, _, _ = (
                            teacher_sparse_payload
                        )
                        t_real_logp, t_real_found = self._lookup_sparse_teacher_logp(
                            teacher_topk_logits,
                            teacher_topk_indices,
                            teacher_log_z,
                            sb,
                            tp,
                            rt,
                            temperature,
                        )
                        teacher_common_logp = t_real_logp.to(accum_dtype).unsqueeze(-1)
                        common_valid_row = t_real_found
                    support_mask = common_valid_row.unsqueeze(-1)
                    neg_inf = torch.full_like(student_common_logp, -1.0e30)
                    student_common_logp = torch.where(
                        support_mask,
                        student_common_logp,
                        neg_inf,
                    )
                    teacher_common_logp = torch.where(
                        support_mask,
                        teacher_common_logp,
                        neg_inf,
                    )
                elif teacher_sparse_payload is None:
                    assert teacher_logits is not None
                    student_common_logits = (
                        self._lookup_tp_cp_sharded_student_logits(
                            student_logits,
                            sb[:, None],
                            sp[:, None],
                            common_student_idx_t[None, :],
                            tp_group=tp_group,
                            cp_group=cp_group,
                        )
                        / temperature
                    )
                    teacher_common_logits = self._lookup_cp_sharded_teacher_logits(
                        teacher_logits,
                        sb[:, None],
                        tp[:, None],
                        common_teacher_idx_t[None, :],
                        cp_group,
                    )
                    teacher_common_logits = teacher_common_logits / temperature
                    # Append the complementary probability mass as a rest bucket.
                    s_log_z_mb = student_log_z[sb, sp].unsqueeze(-1)
                    t_log_z_mb = teacher_log_z[sb, tp].unsqueeze(-1)
                    student_common_logp = (
                        student_common_logits.to(accum_dtype) - s_log_z_mb
                    )
                    teacher_common_logp = (
                        teacher_common_logits.to(accum_dtype) - t_log_z_mb
                    )
                else:
                    assert teacher_to_common_student is not None
                    teacher_topk_logits, teacher_topk_indices, _, _ = (
                        teacher_sparse_payload
                    )
                    topk_ids = teacher_topk_indices[sb, tp]
                    topk_vals = teacher_topk_logits[sb, tp]
                    mapped_student = teacher_to_common_student[topk_ids.long()]
                    support_mask = mapped_student >= 0
                    if teacher_topk_keep_realized:
                        topk_width = int(topk_ids.shape[-1])
                        alt_budget = max(topk_width - 1, 0)
                        support_mask = (
                            support_mask
                            & (mapped_student != rs[:, None])
                            & (topk_ids.long() != rt[:, None])
                        )
                        support_mask = self._topk_mask_by_score(
                            support_mask,
                            topk_vals.to(accum_dtype),
                            alt_budget,
                        )
                    safe_student_ids = mapped_student.clamp_min(0)
                    student_common_logits = (
                        self._lookup_tp_cp_sharded_student_logits(
                            student_logits,
                            sb[:, None],
                            sp[:, None],
                            safe_student_ids,
                            tp_group=tp_group,
                            cp_group=cp_group,
                        )
                        / temperature
                    )
                    student_common_logp = student_common_logits.to(
                        accum_dtype
                    ) - student_log_z[sb, sp].unsqueeze(-1)
                    teacher_common_logp = topk_vals.to(
                        accum_dtype
                    ) / temperature - teacher_log_z[sb, tp].unsqueeze(-1)
                    neg_inf = torch.full_like(student_common_logp, -1.0e30)
                    student_common_logp = torch.where(
                        support_mask,
                        student_common_logp,
                        neg_inf,
                    )
                    teacher_common_logp = torch.where(
                        support_mask,
                        teacher_common_logp,
                        neg_inf,
                    )
                    if teacher_topk_keep_realized:
                        t_real_logp, t_real_found = self._lookup_sparse_teacher_logp(
                            teacher_topk_logits,
                            teacher_topk_indices,
                            teacher_log_z,
                            sb,
                            tp,
                            rt,
                            temperature,
                        )
                        realized_common_student = teacher_to_common_student[rt.long()]
                        realized_common_mask = t_real_found & (
                            realized_common_student == rs
                        )
                        s_real_logp = (
                            self._lookup_tp_cp_sharded_student_logits(
                                student_logits,
                                sb,
                                sp,
                                rs,
                                tp_group=tp_group,
                                cp_group=cp_group,
                            ).to(accum_dtype)
                            / temperature
                            - student_log_z[sb, sp]
                        )
                        student_common_logp = torch.cat(
                            [
                                student_common_logp,
                                torch.where(
                                    realized_common_mask,
                                    s_real_logp,
                                    torch.full_like(s_real_logp, -1.0e30),
                                ).unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        teacher_common_logp = torch.cat(
                            [
                                teacher_common_logp,
                                torch.where(
                                    realized_common_mask,
                                    t_real_logp.to(accum_dtype),
                                    torch.full_like(t_real_logp, -1.0e30),
                                ).unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        support_mask = torch.cat(
                            [
                                support_mask,
                                realized_common_mask.unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                    common_valid_row = support_mask.any(dim=-1)
                student_loss_logp = self._append_rest_bucket_logp(
                    student_common_logp,
                )
                teacher_loss_logp = self._append_rest_bucket_logp(
                    teacher_common_logp,
                )
                per_elem = self._partition_divergence(
                    student_loss_logp, teacher_loss_logp, csd_loss_fn
                )
                if support_mask is not None:
                    rest_one = torch.ones(
                        (support_mask.shape[0], 1),
                        device=device,
                        dtype=per_elem.dtype,
                    )
                    full_mask = torch.cat(
                        [support_mask.to(per_elem.dtype), rest_one],
                        dim=-1,
                    )
                    per_row = (per_elem * full_mask).sum(dim=-1)
                    per_row = per_row * common_valid_row.to(per_row.dtype)
                    common_loss_sum = common_loss_sum + per_row.sum().to(
                        accum_dtype,
                    )
                else:
                    common_loss_sum = common_loss_sum + per_elem.sum().to(
                        accum_dtype,
                    )
                with torch.no_grad():
                    row_match = student_common_logp.argmax(
                        dim=-1
                    ) == teacher_common_logp.argmax(dim=-1)
                    if common_valid_row is not None:
                        row_match = row_match & common_valid_row
                    common_matches_sum_t = common_matches_sum_t + row_match.sum().to(
                        torch.long
                    )
            common_loss = common_loss_sum / float(len(common_chunks))
            common_matches = int(common_matches_sum_t.item())

        # ----- Mismatch loss ------------------
        skipped_no_support = 0
        support_sizes: list[int] = []
        prefix_index = None
        if mismatch_chunks:
            if pure_alm or realized_only_mismatch:
                _prefix_index_prededuped = False
                max_chain_t = 0
                max_chain_s = 0
            else:
                prefix_index = self._ensure_bidir_prefix_support_index(device, i)
                _prefix_index_prededuped = bool(prefix_index.get("_prededuped", False))
                max_chain_t = int(prefix_index.get("max_chain_t", 0))
                max_chain_s = int(prefix_index.get("max_chain_s", 0))

            chunk_records: list[
                tuple[
                    int,
                    int,
                    int,
                    list[int],
                    list[int],
                    list[int],
                    list[int],
                    list[int],
                    list[int],
                    int,
                    int,
                ]
            ] = []
            for b, M, N, s_pred, t_pred, s_labels, t_labels in mismatch_chunks:
                if pure_alm or realized_only_mismatch:
                    # Realized token chains against REST.
                    s_realized_id = int(inp_s_np[b, s_labels[-1]])
                    t_realized_id = int(inp_t_np[b, t_labels[-1]])
                    s_prefix_pred = list(s_pred[:-1])
                    s_prefix_lab_ids = [int(inp_s_np[b, pos]) for pos in s_labels[:-1]]
                    t_prefix_pred = list(t_pred[:-1])
                    t_prefix_lab_ids = [int(inp_t_np[b, pos]) for pos in t_labels[:-1]]
                    chunk_records.append(
                        (
                            b,
                            int(s_pred[-1]),
                            int(t_pred[-1]),
                            s_prefix_pred,
                            s_prefix_lab_ids,
                            t_prefix_pred,
                            t_prefix_lab_ids,
                            [s_realized_id],
                            [t_realized_id],
                            s_realized_id,
                            t_realized_id,
                        )
                    )
                elif M == 1 and N > 1:
                    # 1-to-many
                    s_realized_id = int(inp_s_np[b, s_labels[0]])
                    t_realized_id = int(inp_t_np[b, t_labels[-1]])
                    if N > max_chain_t and not teacher_topk_keep_realized:
                        skipped_no_support += 1
                        continue
                    pairs = ()
                    if N <= max_chain_t:
                        assert prefix_index is not None
                        teacher_prefix = tuple(
                            int(inp_t_np[b, pos]) for pos in t_labels[: N - 1]
                        )
                        pairs = prefix_index["forward"].get(
                            (N, teacher_prefix),
                            (),
                        )
                    if not pairs and not teacher_topk_keep_realized:
                        skipped_no_support += 1
                        continue
                    s_list, t_list = self._unique_bidir_pairs_cpu(
                        pairs,
                        swap=False,
                        assume_unique=_prefix_index_prededuped,
                    )
                    if teacher_topk_keep_realized and (
                        s_realized_id,
                        t_realized_id,
                    ) not in set(zip(s_list, t_list)):
                        s_list.append(s_realized_id)
                        t_list.append(t_realized_id)
                    min_support = 1 if teacher_topk_keep_realized else 2
                    if len(s_list) < min_support:
                        skipped_no_support += 1
                        continue
                    t_prefix_pred = list(t_pred[:-1])
                    t_prefix_lab_ids = [int(inp_t_np[b, pos]) for pos in t_labels[:-1]]
                    chunk_records.append(
                        (
                            b,
                            int(s_pred[0]),
                            int(t_pred[-1]),
                            [],
                            [],
                            t_prefix_pred,
                            t_prefix_lab_ids,
                            s_list,
                            t_list,
                            s_realized_id,
                            t_realized_id,
                        )
                    )
                elif M > 1 and N == 1:
                    # Many-to-1
                    s_realized_id = int(inp_s_np[b, s_labels[-1]])
                    t_realized_id = int(inp_t_np[b, t_labels[0]])
                    if M > max_chain_s and not teacher_topk_keep_realized:
                        skipped_no_support += 1
                        continue
                    pairs = ()
                    if M <= max_chain_s:
                        assert prefix_index is not None
                        student_prefix = tuple(
                            int(inp_s_np[b, pos]) for pos in s_labels[: M - 1]
                        )
                        pairs = prefix_index["reverse"].get(
                            (M, student_prefix),
                            (),
                        )
                    if not pairs and not teacher_topk_keep_realized:
                        skipped_no_support += 1
                        continue
                    s_list, t_list = self._unique_bidir_pairs_cpu(
                        pairs,
                        swap=True,
                        assume_unique=_prefix_index_prededuped,
                    )
                    if teacher_topk_keep_realized and (
                        s_realized_id,
                        t_realized_id,
                    ) not in set(zip(s_list, t_list)):
                        s_list.append(s_realized_id)
                        t_list.append(t_realized_id)
                    min_support = 1 if teacher_topk_keep_realized else 2
                    if len(s_list) < min_support:
                        skipped_no_support += 1
                        continue
                    s_prefix_pred = list(s_pred[:-1])
                    s_prefix_lab_ids = [int(inp_s_np[b, pos]) for pos in s_labels[:-1]]
                    chunk_records.append(
                        (
                            b,
                            int(s_pred[-1]),
                            int(t_pred[0]),
                            s_prefix_pred,
                            s_prefix_lab_ids,
                            [],
                            [],
                            s_list,
                            t_list,
                            s_realized_id,
                            t_realized_id,
                        )
                    )
                else:
                    # Many-to-many chunks use the two realized token chains.
                    assert mtom_as_alm, "CSD requires realized-chain support"
                    s_realized_id = int(inp_s_np[b, s_labels[-1]])
                    t_realized_id = int(inp_t_np[b, t_labels[-1]])
                    s_list = [s_realized_id]
                    t_list = [t_realized_id]
                    s_prefix_pred = list(s_pred[:-1])
                    s_prefix_lab_ids = [int(inp_s_np[b, pos]) for pos in s_labels[:-1]]
                    t_prefix_pred = list(t_pred[:-1])
                    t_prefix_lab_ids = [int(inp_t_np[b, pos]) for pos in t_labels[:-1]]
                    chunk_records.append(
                        (
                            b,
                            int(s_pred[-1]),
                            int(t_pred[-1]),
                            s_prefix_pred,
                            s_prefix_lab_ids,
                            t_prefix_pred,
                            t_prefix_lab_ids,
                            s_list,
                            t_list,
                            s_realized_id,
                            t_realized_id,
                        )
                    )

            if chunk_records:
                (
                    loss_sum_b,
                    used_b,
                    matches_b,
                    support_sizes_b,
                    _valid_flags_b,
                ) = self._partition_kl_mismatch_batched(
                    chunk_records,
                    student_logits,
                    teacher_logits,
                    student_log_z,
                    teacher_log_z,
                    temperature,
                    loss_fn=effective_last_pos_loss_fn,
                    teacher_sparse=(
                        None
                        if teacher_sparse_payload is None
                        else (
                            teacher_sparse_payload[0],
                            teacher_sparse_payload[1],
                        )
                    ),
                    teacher_sparse_keep_realized=teacher_topk_keep_realized,
                    tp_group=tp_group,
                    cp_group=cp_group,
                )
                if used_b > 0:
                    mismatch_loss = loss_sum_b / float(used_b)
                    mismatch_matches += matches_b
                    support_sizes = support_sizes_b

        # ----- Combine + T^2 scale -----------
        effective_mismatch_count = len(support_sizes)
        effective_count = len(common_chunks) + effective_mismatch_count
        # DP×CP-global valid-chunk count, computed in-loss. Reduce over the full
        # mesh (WORLD), then divide out the TP replication: the chunk count is
        # identical on every TP rank (chunks are vocab-independent), so
        # ``WORLD-sum == tp_world × (DP×CP-sum)`` exactly, and dividing by
        # ``tp_world`` recovers the DP×CP count. This matches the CE term's
        # ``global_valid_toks`` (which carries no TP factor), so the KD keeps its
        # configured weight relative to CE at TP>1 — unlike the deleted P-KL/gold
        # WORLD-reduce, which under-weighted the KD by 1/tp. The reduce is
        # unconditional so a no-chunk rank still fires the collective (the
        # early-return was removed for exactly this). At world size 1 (and CPU
        # parity) it is a no-op, so ``global_valid_chunks`` collapses to the local
        # ``effective_count`` and the normalization is byte-exact.
        if global_valid_chunks is None:
            local_count = torch.tensor(
                float(effective_count), device=device, dtype=torch.float32
            )
            if self._dp_cp_group is not None:
                # DP x CP group: it already excludes the TP replicas (so there is
                # no TP factor to divide out) and, crucially, it lives inside a
                # single pipeline stage. A WORLD reduce here deadlocks under PP,
                # because only the last stage runs the loss and joins it.
                global_valid_chunks = group_all_reduce_sum(
                    local_count, group=self._dp_cp_group
                )
            else:
                global_valid_chunks = group_all_reduce_sum(
                    local_count, group=torch.distributed.group.WORLD
                )
                tp_world = (
                    torch.distributed.get_world_size(tp_group)
                    if tp_group is not None
                    else 1
                )
                if tp_world > 1:
                    global_valid_chunks = global_valid_chunks / float(tp_world)
        mismatch_last_loss = mismatch_loss
        mismatch_loss = mismatch_last_loss * mismatch_loss_beta
        # The mismatch objective retains its full coefficient after aggregation.
        if effective_count > 0:
            # Sum of all chunk-level KLs for this microbatch. common_loss /
            # mismatch_loss are local per-chunk averages over their
            # respective partitions; multiplying by chunk counts recovers
            # the un-normalised partition sums.
            chunk_sum = common_loss * float(len(common_chunks)) + mismatch_loss * float(
                effective_mismatch_count
            )
            # Per-chunk normalisation. ``global_valid_chunks`` is either the
            # in-loss WORLD-reduce above or a caller-supplied count (both
            # >= this rank's ``effective_count`` > 0 here, so no div-by-zero).
            # The trainer's sum-reduction across microbatches and ranks then
            # yields `total_chunk_KL × T² / global_valid_chunks` — the natural
            # per-chunk KL averaged over the global batch, matching the
            # per-token convention used by _compute_ce (each loss normalises by
            # its own natural global denominator).
            loss_total = chunk_sum / global_valid_chunks.to(chunk_sum.dtype)
        else:
            # No usable chunks. Keep gradient-connected zero.
            loss_total = (student_logits.sum() * 0.0).to(accum_dtype)
        top1_accuracy = (common_matches + mismatch_matches) / max(
            effective_count,
            1,
        )

        # Final loss × T^2.
        final_loss = loss_total * (temperature**2)

        # Metric naming convention for wandb clarity:
        #   - `loss` / `train:loss`: sum-reduced by the trainer. When
        #     global_valid_chunks is provided, final wandb value is the
        #     per-chunk KL averaged over the global batch
        #     (`total_chunk_KL × T² / global_valid_chunks`). The dispatcher
        #     additionally overwrites `loss` with the dynamic-scaled
        #     `kl_scale * kl + ce` combination — that final value mixes
        #     a per-chunk KL term and a per-token CE term (consistent
        #     gradient balance by construction of kl_scale). `train:loss`
        #     is also the checkpoint-selection metric.
        #   - `*_per_chunk`: mean-reduced by the trainer (listed in
        #     `xtoken_off_policy_distillation.MEAN_REDUCED_MB_METRICS`); final
        #     wandb value is the average per-mb local chunk-average — a clean
        #     per-chunk quantity when chunk counts are roughly uniform across
        #     microbatches.
        #   - `num_*`: sum-reduced; total count across the step.
        metrics: dict[str, Any] = {
            "loss": final_loss.item(),
            "train:loss": final_loss.item(),
            "kl_common_per_chunk": common_loss.item(),
            "kl_partition_last_per_chunk": mismatch_last_loss.item(),
            "kl_mismatch_scaled_per_chunk": mismatch_loss.item(),
            "num_common_chunks": len(common_chunks),
            "num_mismatch_chunks": effective_mismatch_count,
            "num_noise_filtered_common_chunks": noise_filtered_common_chunks,
            "num_noise_filtered_mismatch_chunks": noise_filtered_mismatch_chunks,
            "top1_acc_per_chunk": float(top1_accuracy),
            "num_valid_samples": sample_mask.sum().item(),
        }
        metrics["csd_mismatch_loss_beta"] = mismatch_loss_beta
        if noise_filter_topk > 0:
            metrics["csd_noise_filter_topk"] = noise_filter_topk
        return final_loss, metrics

    # ------------------------------------------------------------------ #
    # Owner-local evaluation (Megatron students)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detached_cp_sum(
        value: torch.Tensor, cp_group: Optional[torch.distributed.ProcessGroup]
    ) -> torch.Tensor:
        """Detached SUM of a CP-partial scalar, identical on every CP rank."""
        reported = value.detach().clone()
        if (
            cp_group is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(cp_group) > 1
        ):
            torch.distributed.all_reduce(
                reported, op=torch.distributed.ReduceOp.SUM, group=cp_group
            )
        return reported

    @staticmethod
    def _raise_if_any_rank_failed(
        error: Optional[Exception],
        *,
        device: torch.device,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> None:
        """Make every TP x CP rank raise together when one rank's setup failed.

        Reducing an ok flag with MIN over the TP group and then the CP group
        reaches every rank of the TP x CP grid, so a failure on one rank cannot
        leave the others waiting in the next collective.
        """
        ok = torch.tensor([0 if error is not None else 1], device=device)
        for group in (tp_group, cp_group):
            if (
                group is not None
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(group) > 1
            ):
                torch.distributed.all_reduce(
                    ok, op=torch.distributed.ReduceOp.MIN, group=group
                )
        if error is not None:
            raise error
        if int(ok.item()) == 0:
            raise RuntimeError(
                "Owner-local CSD evaluation failed on another tensor- or "
                "context-parallel rank; see that rank's error."
            )

    def _compute_owner_local_ce(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict[CSDLossDataDict],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """CP-partial next-token CE over this rank's own student rows.

        Each CP rank sums ``-log p(label)`` over the predictor rows it holds and
        divides by the global token count, so the CP sum of the partials is the
        full CE. Only selected log-probabilities cross TP; nothing is gathered
        across CP.
        """
        full_ids = to_local_if_dtensor(data["input_ids"])
        full_mask = to_local_if_dtensor(data["token_mask"])
        sample_mask = to_local_if_dtensor(data["sample_mask"])
        positions = cp_load_balanced_global_positions(
            int(logits.shape[1]), cp_group, device=logits.device
        )
        predictor = positions < int(full_ids.shape[1]) - 1
        local_rows = torch.arange(
            positions.numel(), device=logits.device, dtype=torch.long
        )[predictor]
        batch_size = int(full_ids.shape[0])
        batch_indices = torch.arange(
            batch_size, device=logits.device
        ).repeat_interleave(local_rows.numel())
        seq_indices = local_rows.repeat(batch_size)
        label_positions = (positions[predictor] + 1).repeat(batch_size)
        tp_rank = (
            torch.distributed.get_rank(tp_group)
            if tp_group is not None and torch.distributed.is_initialized()
            else 0
        )
        local_vocab = int(logits.shape[-1])
        logp = distributed_selected_logprobs(
            logits,
            batch_indices,
            seq_indices,
            full_ids.to(logits.device)[batch_indices, label_positions],
            vocab_start_index=tp_rank * local_vocab,
            vocab_end_index=(tp_rank + 1) * local_vocab,
            real_vocab_size=int(self.student_vocab_size),
            temperature=1.0,
            tp_group=tp_group,
            row_chunk_size=self.cfg.csd_selected_row_chunk_size,
        )
        mask = full_mask.to(logits.device)[batch_indices, label_positions].to(
            logp.dtype
        ) * sample_mask.to(logits.device)[batch_indices].to(logp.dtype)
        return -(logp * mask).sum() / global_valid_toks.to(logp.dtype)

    @staticmethod
    def _owner_local_next_token_accuracy(
        logits: torch.Tensor,
        data: BatchedDataDict[CSDLossDataDict],
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Next-token top-1 accuracy over CP-local rows (scalar collectives only)."""
        with torch.no_grad():
            full_ids = to_local_if_dtensor(data["input_ids"]).to(logits.device)
            full_mask = to_local_if_dtensor(data["token_mask"]).to(logits.device)
            sample_mask = to_local_if_dtensor(data["sample_mask"]).to(logits.device)
            positions = cp_load_balanced_global_positions(
                int(logits.shape[1]), cp_group, device=logits.device
            )
            predictor = positions < int(full_ids.shape[1]) - 1
            local_rows = torch.arange(
                positions.numel(), device=logits.device, dtype=torch.long
            )[predictor]
            label_positions = positions[predictor] + 1
            prediction = vocab_parallel_argmax(logits, tp_group=tp_group)
            mask = (
                full_mask[:, label_positions].float()
                * sample_mask.unsqueeze(-1).float()
            )
            correct = (
                (prediction[:, local_rows] == full_ids[:, label_positions]).float()
                * mask
            ).sum()
            stats = torch.stack((correct, mask.sum()))
            if (
                cp_group is not None
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(cp_group) > 1
            ):
                torch.distributed.all_reduce(stats, group=cp_group)
            return stats[0] / stats[1].clamp(min=1.0)

    def _owner_local_sum_kd(
        self,
        logits: torch.Tensor,
        student_logits_contig: Optional[torch.Tensor],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        teacher_row_payloads_by_idx: dict[int, TeacherRowPayload],
        aligns_by_idx: dict[int, LocalizedAlignment],
        kd_normalizer: torch.Tensor,
        global_valid_chunks_by_idx: Optional[dict[int, torch.Tensor]],
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Static-weight sum of CP-partial per-teacher KD terms.

        Cross-tokenizer teachers run :meth:`_compute_csd_loss_owner_local`. A
        same-tokenizer teacher keeps the contiguous-window KL, which is already
        a per-window partial over the global token count. Reported per-teacher
        values are CP sums.
        """
        device = logits.device
        if self.normalize_teacher_by_vocab:
            min_log_vocab = torch.log(
                torch.tensor(float(min(self.teacher_vocab_sizes)), device=device)
            )
        total_kd: Optional[torch.Tensor] = None
        per_metrics: dict[str, Any] = {}
        # Deterministic teacher order: each KD term issues its own collectives.
        for i in range(self.num_teachers):
            if self._teacher_is_same_vocab(i):
                if student_logits_contig is None or i not in teacher_full_logits_by_idx:
                    raise ValueError(
                        f"Same-vocab teacher {i} requires full teacher logits and "
                        "the contiguous student window."
                    )
                kd_i = same_tokenizer_topk_kl(
                    student_logits_contig,
                    teacher_full_logits_by_idx[i],
                    kd_mask=self._student_kd_mask(aligns_by_idx[i]),
                    sample_mask=aligns_by_idx[i].sample_mask,
                    num_kd_tokens=kd_normalizer,
                    vocab_size=self.student_vocab_size,
                    vocab_topk=self.vocab_topk,
                    temperature=self.temperature,
                    reverse_kl=self.reverse_kl,
                    tp_group=tp_group,
                    cp_group=cp_group,
                )
                m_i: dict[str, Any] = {
                    "kl_loss": self._detached_cp_sum(kd_i, cp_group).item()
                }
            else:
                if (
                    global_valid_chunks_by_idx is None
                    or i not in global_valid_chunks_by_idx
                ):
                    raise ValueError(
                        "Owner-local CSD needs the step-global valid-chunk count "
                        f"for teacher {i}."
                    )
                if i not in teacher_row_payloads_by_idx:
                    raise ValueError(
                        f"Cross-tokenizer teacher {i} has no IPC row payload."
                    )
                kd_i, m_i = self._compute_csd_loss_owner_local(
                    i,
                    logits,
                    teacher_row_payloads_by_idx[i],
                    aligns_by_idx[i],
                    teacher_vocab_size=self.teacher_vocab_sizes[i],
                    global_valid_chunks=global_valid_chunks_by_idx[i],
                    tp_group=tp_group,
                    cp_group=cp_group,
                )
            weight = self.teacher_weights[i]
            if self.normalize_teacher_by_vocab:
                weight = weight * float(
                    torch.log(
                        torch.tensor(float(self.teacher_vocab_sizes[i]), device=device)
                    )
                    / min_log_vocab
                )
            weighted = kd_i * weight
            total_kd = weighted if total_kd is None else total_kd + weighted
            per_metrics[f"teacher_{i}/weighted_kl"] = float(m_i["kl_loss"] * weight)
            for key in (
                "kl_loss",
                "kl_common_per_chunk",
                "kl_partition_last_per_chunk",
                "top1_acc_per_chunk",
                "num_common_chunks",
                "num_mismatch_chunks",
            ):
                if key in m_i:
                    per_metrics[f"{key}_t{i}"] = m_i[key]
            per_metrics[f"weight_t{i}"] = float(self.teacher_weights[i])
        assert total_kd is not None
        return total_kd, per_metrics

    def _partition_divergence(
        self,
        student_logp: torch.Tensor,
        teacher_logp: torch.Tensor,
        loss_fn_name: str,
    ) -> torch.Tensor:
        """Per-element partition divergence for one CSD loss setting."""
        if loss_fn_name == "bce":
            return self._binary_power_bce_from_logp(
                student_logp, teacher_logp, tau=float(self.cfg.csd_alm_bce_tau)
            )
        if loss_fn_name == "jsd":
            return _generalized_jsd(
                student_logp, teacher_logp, float(self.cfg.csd_jsd_beta)
            )
        if not self.reverse_kl:
            return torch.nn.functional.kl_div(
                student_logp, teacher_logp, reduction="none", log_target=True
            )
        return torch.nn.functional.kl_div(
            teacher_logp, student_logp, reduction="none", log_target=True
        )

    def _owner_local_mismatch_support(
        self,
        record: tuple[int, int, int, list[int], list[int], list[int], list[int], bool],
        *,
        inp_s_np: np.ndarray,
        inp_t_np: np.ndarray,
        prefix_index: Optional[dict[str, Any]],
        keep_realized: bool,
    ) -> Optional[tuple[list[int], list[int], int, int]]:
        """``(s_ids, t_ids, realized_s, realized_t)`` for one mismatch chunk.

        Mirrors the replicated CSD rules: realized-only chunks and many-to-many
        chunks use the realized pair; 1-to-many and many-to-1 chunks use the
        prefix-support alternatives, plus the realized pair when the sparse
        teacher keeps realized labels. Returns None for an unsupported chunk.
        """
        b, M, N, _s_pred, s_labels, _t_pred, t_labels, _owned = record
        realized_s = int(inp_s_np[b, s_labels[-1]])
        realized_t = int(inp_t_np[b, t_labels[-1]])
        if (
            self.cfg.csd_realized_only_mismatch
            or prefix_index is None
            or (M > 1 and N > 1)
        ):
            return [realized_s], [realized_t], realized_s, realized_t
        if M == 1:
            max_chain = int(prefix_index.get("max_chain_t", 0))
            length, side, swap = N, "forward", False
            prefix = tuple(int(inp_t_np[b, pos]) for pos in t_labels[: N - 1])
        else:
            max_chain = int(prefix_index.get("max_chain_s", 0))
            length, side, swap = M, "reverse", True
            prefix = tuple(int(inp_s_np[b, pos]) for pos in s_labels[: M - 1])
        if length > max_chain and not keep_realized:
            return None
        pairs = (
            prefix_index[side].get((length, prefix), ()) if length <= max_chain else ()
        )
        if not pairs and not keep_realized:
            return None
        s_list, t_list = self._unique_bidir_pairs_cpu(
            pairs,
            swap=swap,
            assume_unique=bool(prefix_index.get("_prededuped", False)),
        )
        if keep_realized and (realized_s, realized_t) not in set(zip(s_list, t_list)):
            s_list.append(realized_s)
            t_list.append(realized_t)
        if len(s_list) < (1 if keep_realized else 2):
            return None
        return s_list, t_list, realized_s, realized_t

    def _compute_csd_loss_owner_local(
        self,
        i: int,
        student_logits: torch.Tensor,
        payload: TeacherRowPayload,
        align: LocalizedAlignment,
        *,
        teacher_vocab_size: int,
        global_valid_chunks: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """CP-partial CSD KD term for cross-tokenizer teacher ``i``.

        The student logits stay ``[B, S_pad/CP, V/TP]`` in their load-balanced
        CP layout. Every rank classifies all chunks the same way, but only the
        rank holding a chunk's final student predictor row evaluates it; its
        teacher rows are read from the IPC records (:class:`SparseTeacherRowReader`
        or :class:`DenseTeacherRowReader`) and its student values come from
        :func:`distributed_selected_logprobs`. A mismatch chunk's student prefix
        rows may sit on other CP ranks: each rank adds its share to one
        fixed-length vector that is summed across CP with a SUM backward.

        The result is this rank's partial ``T**2 * (common + beta * mismatch)
        / global_valid_chunks``; summed over CP ranks it equals the replicated
        CSD KD term. Dense teachers follow the replicated support rules; sparse
        teachers use each request's top-k with its realized label kept.
        Returned metrics are CP sums.
        """
        cfg = self.cfg
        device = student_logits.device
        accum_dtype = torch.float32
        temperature = float(cfg.temperature)
        loss_fn_name = str(cfg.csd_loss_fn)
        last_loss_fn_name = str(cfg.csd_last_pos_loss_fn or loss_fn_name)
        mismatch_beta = float(cfg.csd_mismatch_loss_beta)
        noise_filter_topk = int(cfg.csd_noise_filter_topk)
        realized_only = bool(cfg.csd_realized_only_mismatch)
        sparse = payload.transport == "sparse"
        # Sparse owner-local transport always forces realized labels.
        keep_realized = sparse
        row_tile = int(cfg.csd_selected_row_chunk_size)
        chunk_batch = int(cfg.csd_support_chunk_batch_size)
        v_t = int(teacher_vocab_size)
        tp_rank = (
            torch.distributed.get_rank(tp_group)
            if tp_group is not None and torch.distributed.is_initialized()
            else 0
        )
        cp_size = (
            torch.distributed.get_world_size(cp_group)
            if cp_group is not None and torch.distributed.is_initialized()
            else 1
        )
        local_vocab = int(student_logits.shape[-1])
        vocab_start = tp_rank * local_vocab

        def student_logp(
            batch_rows: torch.Tensor, local_rows: torch.Tensor, token_ids: torch.Tensor
        ) -> torch.Tensor:
            return distributed_selected_logprobs(
                student_logits,
                batch_rows,
                local_rows,
                token_ids,
                vocab_start_index=vocab_start,
                vocab_end_index=vocab_start + local_vocab,
                real_vocab_size=int(self.student_vocab_size),
                temperature=temperature,
                tp_group=tp_group,
                row_chunk_size=row_tile,
            ).to(accum_dtype)

        def as_long(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, device=device, dtype=torch.long)

        # A graph-connected zero: every rank's loss must reach the student logits
        # (and the CP collectives below) even when it owns no chunk.
        connected_zero = student_logits.reshape(-1)[0].to(accum_dtype) * 0.0

        reader: Optional[SparseTeacherRowReader | DenseTeacherRowReader] = None
        try:
            setup_error: Optional[Exception] = None
            # Collective-free setup: any failure here is agreed on below, before
            # the first TP or CP collective.
            try:
                if align.student_global_positions is None:
                    raise ValueError("Owner-local CSD requires student positions.")
                assert align.student_spans is not None
                assert align.teacher_spans is not None
                assert align.pair_valid is not None
                assert align.student_input_ids is not None
                assert align.teacher_input_ids is not None
                sample_mask = to_local_if_dtensor(align.sample_mask).bool()
                pair_valid = to_local_if_dtensor(align.pair_valid).bool()
                batch_size = int(pair_valid.shape[0])
                if sample_mask.shape != (batch_size,):
                    raise ValueError(
                        "CSD sample_mask must have shape [B]: expected "
                        f"({batch_size},), got {tuple(sample_mask.shape)}."
                    )
                if len(payload.entries) != batch_size:
                    raise ValueError(
                        f"Teacher {i} IPC payload has {len(payload.entries)} "
                        f"samples for a microbatch of {batch_size}."
                    )
                pair_valid_np = (
                    (pair_valid & sample_mask.unsqueeze(-1)).cpu().numpy().astype(bool)
                )
                s_spans_np = align.student_spans.detach().cpu().numpy()
                t_spans_np = align.teacher_spans.detach().cpu().numpy()
                inp_s_np = align.student_input_ids.detach().cpu().numpy()
                inp_t_np = align.teacher_input_ids.detach().cpu().numpy()
                positions = align.student_global_positions.detach().cpu().tolist()
                local_of_global = {int(g): j for j, g in enumerate(positions)}
                student_seq_len = len(positions) * cp_size

                if sparse:
                    reader = SparseTeacherRowReader(payload.entries, device=device)
                    expected_membership = noise_filter_topk or int(
                        cfg.teacher_topk_ipc_k
                    )
                    if (
                        reader.k != int(cfg.teacher_topk_ipc_k)
                        or reader.temperature != temperature
                        or reader.real_vocab_size != v_t
                        or reader.membership_k != expected_membership
                    ):
                        raise ValueError(
                            f"Teacher {i} sparse IPC metadata (k={reader.k}, "
                            f"temperature={reader.temperature}, vocab="
                            f"{reader.real_vocab_size}, membership_k="
                            f"{reader.membership_k}) disagrees with the loss."
                        )
                else:
                    reader = DenseTeacherRowReader(payload.entries, device=device)
                    if reader.full_vocab_size < v_t:
                        raise ValueError(
                            f"Teacher {i} dense IPC vocabulary {reader.full_vocab_size} "
                            f"is narrower than the tokenizer vocabulary {v_t}."
                        )
                teacher_seq_len = reader.full_seq_len
                common_s, common_t = self._get_common_indices_csd(device, i)
                if common_s.numel() == 0:
                    raise ValueError(
                        f"Teacher {i}: the exact 1-to-1 common-vocab map is empty."
                    )

                # Classify every chunk identically on every rank; keep the
                # common chunks this rank owns and all mismatch chunks (their
                # student prefixes may live here).
                common_records: list[tuple[int, int, int, int, int]] = []
                mismatch_records: list[
                    tuple[
                        int, int, int, list[int], list[int], list[int], list[int], bool
                    ]
                ] = []
                for b, k in zip(*np.nonzero(pair_valid_np)):
                    s_start, s_end = (int(x) for x in s_spans_np[b, k])
                    t_start, t_end = (int(x) for x in t_spans_np[b, k])
                    M, N = s_end - s_start, t_end - t_start
                    if M <= 0 or N <= 0:
                        continue
                    shift = (
                        1 if cfg.kl_chunk_shift and s_start > 0 and t_start > 0 else 0
                    )
                    s_labels = list(range(s_start, s_end))
                    t_labels = list(range(t_start, t_end))
                    s_pred = [pos - shift for pos in s_labels]
                    t_pred = [pos - shift for pos in t_labels]
                    if (
                        s_pred[0] < 0
                        or s_pred[-1] >= student_seq_len
                        or t_pred[0] < 0
                        or t_pred[-1] >= teacher_seq_len
                        or s_labels[-1] >= inp_s_np.shape[1]
                        or t_labels[-1] >= inp_t_np.shape[1]
                    ):
                        continue
                    owned = s_pred[-1] in local_of_global
                    if M == 1 and N == 1:
                        if owned:
                            common_records.append(
                                (
                                    int(b),
                                    local_of_global[s_pred[0]],
                                    t_pred[0],
                                    int(inp_s_np[b, s_labels[0]]),
                                    int(inp_t_np[b, t_labels[0]]),
                                )
                            )
                    else:
                        mismatch_records.append(
                            (int(b), M, N, s_pred, s_labels, t_pred, t_labels, owned)
                        )

                prefix_index = None
                if not realized_only and any(r[-1] for r in mismatch_records):
                    prefix_index = self._ensure_bidir_prefix_support_index(device, i)
                owned_mismatch: list[
                    tuple[int, tuple[list[int], list[int], int, int]]
                ] = []
                for record_idx, record in enumerate(mismatch_records):
                    if not record[-1]:
                        continue
                    support = self._owner_local_mismatch_support(
                        record,
                        inp_s_np=inp_s_np,
                        inp_t_np=inp_t_np,
                        prefix_index=prefix_index,
                        keep_realized=keep_realized,
                    )
                    if support is not None:
                        owned_mismatch.append((record_idx, support))

                # Sparse teacher rows are small (K values per request), so read
                # every one this rank needs up front; this also validates the
                # forced labels before any collective.
                sparse_row_of: dict[tuple[int, int, int], int] = {}
                sparse_rows: Optional[tuple[torch.Tensor, ...]] = None
                if sparse:
                    assert isinstance(reader, SparseTeacherRowReader)
                    for b, _sl, t_pos, _rs, rt in common_records:
                        sparse_row_of.setdefault((b, t_pos, rt), len(sparse_row_of))
                    for record_idx, _support in owned_mismatch:
                        b, _M, _N, _sp, _sl, t_pred, t_labels, _ow = mismatch_records[
                            record_idx
                        ]
                        for t_pos, t_lab in zip(t_pred, t_labels):
                            key = (b, t_pos, int(inp_t_np[b, t_lab]))
                            sparse_row_of.setdefault(key, len(sparse_row_of))
                    if sparse_row_of:
                        keys = list(sparse_row_of)
                        tiles = [
                            reader.gather_rows(
                                as_long(
                                    [key[0] for key in keys[start : start + row_tile]]
                                ),
                                as_long(
                                    [key[1] for key in keys[start : start + row_tile]]
                                ),
                                as_long(
                                    [key[2] for key in keys[start : start + row_tile]]
                                ),
                            )
                            for start in range(0, len(keys), row_tile)
                        ]
                        sparse_rows = tuple(
                            torch.cat([tile[field] for tile in tiles])
                            for field in range(4)
                        )
            except (
                AssertionError,
                IndexError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                setup_error = exc
            self._raise_if_any_rank_failed(
                setup_error, device=device, tp_group=tp_group, cp_group=cp_group
            )

            def sparse_label_logp(
                keys: list[tuple[int, int, int]],
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Teacher log-prob of each request's own label, and its membership."""
                assert sparse_rows is not None
                rows = as_long([sparse_row_of[key] for key in keys])
                values, ids, log_z, in_topk = (field[rows] for field in sparse_rows)
                match = ids.long() == as_long([key[2] for key in keys]).unsqueeze(-1)
                raw = torch.where(match, values, torch.full_like(values, -torch.inf))
                return raw.max(dim=-1).values / temperature - log_z, in_topk

            def dense_rows(
                batch_rows: list[int], teacher_rows: list[int]
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Real-vocabulary teacher logits ``[n, V_t]`` and their log Z."""
                assert isinstance(reader, DenseTeacherRowReader)
                rows = reader.gather_rows(as_long(batch_rows), as_long(teacher_rows))
                rows = rows[:, :v_t]
                return rows, torch.logsumexp(rows / temperature, dim=-1)

            def dense_label_in_topk(
                rows: torch.Tensor, labels: torch.Tensor
            ) -> torch.Tensor:
                filter_k = min(noise_filter_topk, int(rows.shape[-1]))
                topk_ids = torch.topk(rows, k=filter_k, dim=-1).indices
                return (topk_ids == labels.unsqueeze(-1)).any(dim=-1)

            # ----- Common chunks owned by this rank -----------------------
            common_sum = connected_zero
            common_count = 0
            common_matches = 0
            noise_filtered_common = 0
            teacher_to_common_student = (
                self._get_csd_teacher_to_common_student(
                    device, i, v_t, common_s, common_t
                )
                if sparse
                else None
            )
            for start in range(0, len(common_records), chunk_batch):
                batch = common_records[start : start + chunk_batch]
                cb = as_long([r[0] for r in batch])
                cs = as_long([r[1] for r in batch])
                crs = as_long([r[3] for r in batch])
                crt = as_long([r[4] for r in batch])
                if sparse:
                    assert (
                        sparse_rows is not None
                        and teacher_to_common_student is not None
                    )
                    rows = as_long([sparse_row_of[(r[0], r[2], r[4])] for r in batch])
                    topk_logits, topk_ids, teacher_log_z, in_topk = (
                        field[rows] for field in sparse_rows
                    )
                    mapped = teacher_to_common_student[topk_ids.long()]
                    # One of the K columns is reserved for the realized pair:
                    # keep the best K-1 other common alternatives by score and
                    # append the realized pair when it maps 1-to-1.
                    support = self._topk_mask_by_score(
                        (mapped >= 0)
                        & (mapped != crs.unsqueeze(-1))
                        & (topk_ids.long() != crt.unsqueeze(-1)),
                        topk_logits.to(accum_dtype),
                        max(int(topk_ids.shape[-1]) - 1, 0),
                    )
                    realized_raw = (
                        torch.where(
                            topk_ids.long() == crt.unsqueeze(-1),
                            topk_logits,
                            torch.full_like(topk_logits, -torch.inf),
                        )
                        .max(dim=-1)
                        .values
                    )
                    realized_common = teacher_to_common_student[crt] == crs
                    queried = student_logp(
                        cb,
                        cs,
                        torch.cat([mapped.clamp_min(0), crs.unsqueeze(-1)], dim=-1),
                    )
                    teacher_logp = torch.cat(
                        [
                            topk_logits.to(accum_dtype) / temperature
                            - teacher_log_z.unsqueeze(-1),
                            (realized_raw / temperature - teacher_log_z).unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    support = torch.cat(
                        [support, realized_common.unsqueeze(-1)], dim=-1
                    )
                    s_logp = torch.where(support, queried, -1.0e30)
                    t_logp = torch.where(support, teacher_logp, -1.0e30)
                    valid = support.any(dim=-1)
                    noise_ok = (
                        in_topk if noise_filter_topk > 0 else torch.ones_like(valid)
                    )
                else:
                    rows, teacher_log_z = dense_rows(
                        [r[0] for r in batch], [r[2] for r in batch]
                    )
                    support = None
                    s_logp = student_logp(
                        cb, cs, common_s.unsqueeze(0).expand(len(batch), -1)
                    )
                    t_logp = rows[:, common_t] / temperature - teacher_log_z.unsqueeze(
                        -1
                    )
                    valid = torch.ones((len(batch),), device=device, dtype=torch.bool)
                    noise_ok = (
                        dense_label_in_topk(rows, crt)
                        if noise_filter_topk > 0
                        else torch.ones_like(valid)
                    )
                per_elem = self._partition_divergence(
                    self._append_rest_bucket_logp(s_logp),
                    self._append_rest_bucket_logp(t_logp),
                    loss_fn_name,
                )
                if support is not None:
                    per_elem = per_elem * torch.cat(
                        [support.to(per_elem.dtype), torch.ones_like(per_elem[:, :1])],
                        dim=-1,
                    )
                keep = valid & noise_ok
                common_sum = (
                    common_sum + (per_elem.sum(dim=-1) * keep.to(accum_dtype)).sum()
                )
                common_count += int(noise_ok.sum().item())
                noise_filtered_common += int((~noise_ok).sum().item())
                with torch.no_grad():
                    common_matches += int(
                        ((s_logp.argmax(dim=-1) == t_logp.argmax(dim=-1)) & keep)
                        .sum()
                        .item()
                    )

            # ----- Mismatch chunks ------------------------------------------
            mismatch_sum = connected_zero
            mismatch_count = 0
            mismatch_matches = 0
            noise_filtered_mismatch = 0
            if mismatch_records:
                # Each rank adds the realized student prefix log-probs of the
                # rows it holds; the CP SUM gives every owner its complete prefix
                # (realized-only chunks: the complete chain).
                term_chunks: list[int] = []
                term_b: list[int] = []
                term_rows: list[int] = []
                term_labels: list[int] = []
                for record_idx, (b, _M, _N, s_pred, s_labels, *_rest) in enumerate(
                    mismatch_records
                ):
                    terms = (
                        zip(s_pred, s_labels)
                        if realized_only
                        else zip(s_pred[:-1], s_labels[:-1])
                    )
                    for pred_pos, label_pos in terms:
                        local_row = local_of_global.get(pred_pos)
                        if local_row is not None:
                            term_chunks.append(record_idx)
                            term_b.append(b)
                            term_rows.append(local_row)
                            term_labels.append(int(inp_s_np[b, label_pos]))
                # Keep the vector in every rank's graph: a rank may own no final
                # rows yet hold prefix rows, and must join the backward SUM.
                local_terms = connected_zero + torch.zeros(
                    (len(mismatch_records),), device=device, dtype=accum_dtype
                )
                if term_b:
                    local_terms = local_terms.index_add(
                        0,
                        as_long(term_chunks),
                        student_logp(
                            as_long(term_b), as_long(term_rows), as_long(term_labels)
                        ),
                    )
                shared_terms = group_all_reduce_sum_with_grad_backward_sum(
                    local_terms, cp_group
                )
                mismatch_sum = mismatch_sum + shared_terms.sum() * 0.0

                for start in range(0, len(owned_mismatch), chunk_batch):
                    batch = owned_mismatch[start : start + chunk_batch]
                    records = [mismatch_records[idx] for idx, _ in batch]
                    n = len(batch)
                    student_prefix = shared_terms[as_long([idx for idx, _ in batch])]
                    # Teacher log-probs of every realized label in each chunk:
                    # the final row's label is needed only for noise filtering
                    # and realized-only chains.
                    requests = [
                        (row, (r[0], t_pos, int(inp_t_np[r[0], t_lab])), j == r[2] - 1)
                        for row, r in enumerate(records)
                        for j, (t_pos, t_lab) in enumerate(zip(r[5], r[6]))
                    ]
                    owner = as_long([req[0] for req in requests])
                    is_final = torch.tensor(
                        [req[2] for req in requests], device=device, dtype=torch.bool
                    )
                    if sparse:
                        label_logp, label_ok = sparse_label_logp(
                            [req[1] for req in requests]
                        )
                    else:
                        label_logp_parts, label_ok_parts = [], []
                        for tile in range(0, len(requests), row_tile):
                            part = requests[tile : tile + row_tile]
                            rows, log_z = dense_rows(
                                [req[1][0] for req in part], [req[1][1] for req in part]
                            )
                            labels = as_long([req[1][2] for req in part])
                            label_logp_parts.append(
                                rows.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                                / temperature
                                - log_z
                            )
                            label_ok_parts.append(
                                dense_label_in_topk(rows, labels)
                                if noise_filter_topk > 0
                                else torch.ones_like(labels, dtype=torch.bool)
                            )
                        label_logp = torch.cat(label_logp_parts)
                        label_ok = torch.cat(label_ok_parts)
                    chain_or_prefix_terms = (
                        label_logp
                        if realized_only
                        else torch.where(is_final, 0.0, label_logp.to(accum_dtype))
                    )
                    teacher_prefix = torch.zeros(
                        (n,), device=device, dtype=accum_dtype
                    ).index_add(0, owner, chain_or_prefix_terms.to(accum_dtype))
                    noise_ok = (
                        torch.ones((n,), device=device, dtype=torch.int32)
                        .index_reduce(0, owner, label_ok.to(torch.int32), "amin")
                        .bool()
                        if noise_filter_topk > 0
                        else torch.ones((n,), device=device, dtype=torch.bool)
                    )

                    if realized_only:
                        s_scores = student_prefix.unsqueeze(-1)
                        t_scores = teacher_prefix.unsqueeze(-1)
                        support = torch.ones((n, 1), device=device, dtype=torch.bool)
                    else:
                        width = max(len(sup[0]) for _, sup in batch)
                        s_ids = torch.zeros((n, width), device=device, dtype=torch.long)
                        t_ids = torch.zeros_like(s_ids)
                        candidate = torch.zeros(
                            (n, width), device=device, dtype=torch.bool
                        )
                        realized_pair = torch.zeros_like(candidate)
                        for row, (_, (s_list, t_list, rs, rt)) in enumerate(batch):
                            count = len(s_list)
                            s_ids[row, :count] = as_long(s_list)
                            t_ids[row, :count] = as_long(t_list)
                            candidate[row, :count] = True
                            realized_pair[row, :count] = (s_ids[row, :count] == rs) & (
                                t_ids[row, :count] == rt
                            )
                        final_b = [r[0] for r in records]
                        final_t = [r[5][-1] for r in records]
                        if sparse:
                            assert sparse_rows is not None
                            final_rows = as_long(
                                [
                                    sparse_row_of[
                                        (b, t_pos, int(inp_t_np[b, r[6][-1]]))
                                    ]
                                    for b, t_pos, r in zip(final_b, final_t, records)
                                ]
                            )
                            topk_logits, topk_ids, final_log_z, _ = (
                                field[final_rows] for field in sparse_rows
                            )
                            slot_match = t_ids.unsqueeze(
                                -1
                            ) == topk_ids.long().unsqueeze(1)
                            found = slot_match.any(dim=-1)
                            t_raw = (
                                torch.where(
                                    slot_match,
                                    topk_logits.unsqueeze(1),
                                    torch.full_like(
                                        topk_logits.unsqueeze(1), -torch.inf
                                    ),
                                )
                                .max(dim=-1)
                                .values
                            )
                            t_final = t_raw.to(
                                accum_dtype
                            ) / temperature - final_log_z.unsqueeze(-1)
                            alternatives = self._topk_mask_by_score(
                                candidate & found & ~realized_pair,
                                torch.where(found, t_final, -1.0e30),
                                max(int(topk_ids.shape[-1]) - 1, 0),
                            )
                            support = (realized_pair & found) | alternatives
                        else:
                            rows, final_log_z = dense_rows(final_b, final_t)
                            t_final = rows.gather(
                                -1, t_ids
                            ) / temperature - final_log_z.unsqueeze(-1)
                            support = candidate
                        final_s = as_long([local_of_global[r[3][-1]] for r in records])
                        s_scores = student_logp(as_long(final_b), final_s, s_ids) + (
                            student_prefix.unsqueeze(-1)
                        )
                        t_scores = t_final + teacher_prefix.unsqueeze(-1)
                    s_scores = torch.where(support, s_scores, -1.0e30)
                    t_scores = torch.where(support, t_scores.to(accum_dtype), -1.0e30)
                    per_elem = self._partition_divergence(
                        self._append_rest_bucket_logp(s_scores),
                        self._append_rest_bucket_logp(t_scores),
                        last_loss_fn_name,
                    )
                    per_elem = per_elem * torch.cat(
                        [support.to(per_elem.dtype), torch.ones_like(per_elem[:, :1])],
                        dim=-1,
                    )
                    keep = support.any(dim=-1) & noise_ok
                    mismatch_sum = (
                        mismatch_sum
                        + (per_elem.sum(dim=-1) * keep.to(accum_dtype)).sum()
                    )
                    mismatch_count += int(keep.sum().item())
                    noise_filtered_mismatch += int((~noise_ok).sum().item())
                    with torch.no_grad():
                        mismatch_matches += int(
                            (
                                (s_scores.argmax(dim=-1) == t_scores.argmax(dim=-1))
                                & keep
                            )
                            .sum()
                            .item()
                        )
        finally:
            if reader is not None:
                reader.close()

        local_chunk_sum = common_sum + mismatch_beta * mismatch_sum
        final_loss = (
            local_chunk_sum
            / global_valid_chunks.to(device=device, dtype=accum_dtype).clamp(min=1.0)
            * (temperature**2)
        )
        stats = torch.tensor(
            [
                float(common_sum.detach()),
                float(mismatch_sum.detach()),
                float(final_loss.detach()),
                common_count,
                mismatch_count,
                common_matches,
                mismatch_matches,
                noise_filtered_common,
                noise_filtered_mismatch,
            ],
            device=device,
            dtype=torch.float64,
        )
        if cp_size > 1:
            torch.distributed.all_reduce(stats, group=cp_group)
        (
            common_total,
            mismatch_total,
            kd_total,
            common_total_count,
            mismatch_total_count,
            common_total_matches,
            mismatch_total_matches,
            noise_common_total,
            noise_mismatch_total,
        ) = stats.tolist()
        mismatch_mean = mismatch_total / max(mismatch_total_count, 1.0)
        metrics: dict[str, Any] = {
            "loss": kd_total,
            "train:loss": kd_total,
            "kl_loss": kd_total,
            "kl_common_per_chunk": common_total / max(common_total_count, 1.0),
            "kl_partition_last_per_chunk": mismatch_mean,
            "kl_mismatch_scaled_per_chunk": mismatch_mean * mismatch_beta,
            "num_common_chunks": int(common_total_count),
            "num_mismatch_chunks": int(mismatch_total_count),
            "num_noise_filtered_common_chunks": int(noise_common_total),
            "num_noise_filtered_mismatch_chunks": int(noise_mismatch_total),
            "top1_acc_per_chunk": (common_total_matches + mismatch_total_matches)
            / max(common_total_count + mismatch_total_count, 1.0),
            "csd_mismatch_loss_beta": mismatch_beta,
        }
        if noise_filter_topk > 0:
            metrics["csd_noise_filter_topk"] = noise_filter_topk
        return final_loss, metrics
