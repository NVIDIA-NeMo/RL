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
from typing import Any, NotRequired, Optional, TypedDict

import torch

from nemo_rl.algorithms.cross_tokenizer.loss_utils import (
    LocalizedAlignment,
    build_exact_token_map,
    ce_label_mask,
    chunk_average_log_probs,
    get_sparse_projection_matrix,
    next_token_accuracy,
    project_student_to_teacher_vocab,
    same_tokenizer_full_vocab_kl,
    same_tokenizer_topk_kl,
    select_teacher_topk_indices,
    slice_sparse_projection_cols,
    student_next_token_ce,
    valid_chunk_mask,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    cp_shift_next,
    group_all_reduce_sum,
    group_all_reduce_sum_with_grad,
    to_local_if_dtensor,
    vocab_parallel_full_log_softmax,
    vocab_parallel_log_softmax,
)
from nemo_rl.utils.tensor_ops import masked_mean

# =====================================================================
# Cross-tokenizer distillation
# =====================================================================


class CrossTokenizerDistillationLossConfig(TypedDict):
    """Config for cross-tokenizer distillation loss.

    Attributes:
        projection_matrix_paths: Per-teacher list of filesystem paths to the
            .pt projection file (``None`` marks a same-tokenizer teacher: direct
            KL, no projection). Each .pt holds either the dense top-k projection
            (dict with 'indices' and 'likelihoods' tensors of shape
            [V_student, top_k]) or the sparse multi-token format
            (dict[(student_id, teacher_id)] -> count), loaded lazily on first
            call by each worker process. Runtime-injected by
            ``cross_tokenizer_distillation.setup`` from
            ``teachers[i].aligner.projection_matrix_path``; not a user loss_fn
            key in YAML.
        gold_loss: If True, switch to the gold-loss formulation: split the
            vocab into an exact-token-mapped *common* set (KL) and an
            *uncommon* set (sorted L1).
        xtoken_loss: Modifier inside the gold-loss path. If True, relaxes
            the exact-map threshold to ``>= 0.6`` (vs ``== 1.0``) and adds
            a collision-replacement rule so multi-token projections can
            still contribute exact maps. Requires ``gold_loss=True``.
        temperature: Softmax temperature applied symmetrically to student
            and teacher logits before KL.
        vocab_topk: Microbatch-global top-k size. Used by the P-KL path
            (``gold_loss=False``) and by every same-tokenizer teacher, whatever
            ``gold_loss`` is, except the all-same-tokenizer ``averaged_logits``
            path, which uses the full vocab. Computed inside the loss fn from
            full teacher logits.
        uncommon_topk: Cap on the L1 uncommon-tail sort in the gold path.
            Defaults to 8192. Inert when ``gold_loss=False``.
        reverse_kl: If True, compute KL(student || teacher) instead of
            KL(teacher || student).
        exact_token_match_only: If True, only aligned pairs flagged as
            'is_correct' contribute to KL; mismatched pairs are masked out.
            Used by the P-KL path only.
        kl_loss_weight: Scalar multiplier on the distillation (KD) term in
            fixed-weight mode (``dynamic_loss_scaling=False``). Applies to
            both the P-KL and gold-loss paths.
        ce_loss_scale: Scalar multiplier on the next-token CE term in
            fixed-weight mode. Applies to both the P-KL and gold-loss paths.
        dynamic_loss_scaling: If True, rescale the KD term each step so its
            detached magnitude matches CE, then add CE; ``kl_loss_weight`` /
            ``ce_loss_scale`` are ignored in this mode. Applies to both the
            P-KL and gold-loss paths.
        student_vocab_size: Full student tokenizer vocab size, used to size
            the projection matrix's student-side (V_s) axis. Runtime-injected
            by ``cross_tokenizer_distillation.setup`` from ``len(student_tokenizer)``;
            not a user knob in YAML. Sizing V_s from the configured tokenizer
            vocab (rather than ``max(observed student_id) + 1`` from the
            sparse projection file) keeps V_s in lockstep with
            ``logits.shape[-1]`` when the file's highest student ids happen
            to be absent.
        teacher_vocab_sizes: Per-teacher list of full teacher tokenizer vocab
            sizes, used to size each projection matrix's teacher-side (V_t) axis.
            Runtime-injected symmetrically to ``student_vocab_size`` from each
            ``len(teacher_tokenizer)``; not a user loss_fn key in YAML.
    """

    gold_loss: bool
    xtoken_loss: bool
    temperature: float
    vocab_topk: int
    uncommon_topk: int
    reverse_kl: bool
    exact_token_match_only: bool
    kl_loss_weight: float
    ce_loss_scale: float
    dynamic_loss_scaling: bool
    # Multi-teacher aggregation (user loss_fn knobs). gold_loss/xtoken_loss above
    # are the global defaults; teachers[i].gold_loss / .xtoken_loss can override
    # them per teacher in kd_loss_mode="sum".
    kd_loss_mode: str  # "sum" | "averaged_logits" | "select_teacher"
    normalize_teacher_by_vocab: (
        bool  # sum-mode only: scale each teacher's KD by log(V_t_i)/log(min_j V_t_j)
    )
    alpha: float  # softmax temperature on dynamic teacher-weight scores (sum_weights_metric)
    sum_weights_metric: NotRequired[
        Optional[str]
    ]  # "ce" | "entropy" | "max_prob"; None => static teacher_weights. sum-mode only.
    # Runtime-injected by cross_tokenizer_distillation.setup (parallel
    # per-teacher lists + the student vocab size); not user loss_fn keys.
    student_vocab_size: NotRequired[int]
    teacher_vocab_sizes: NotRequired[list[int]]
    projection_matrix_paths: NotRequired[list[Optional[str]]]
    teacher_weights: NotRequired[list[float]]
    teacher_gold_loss: NotRequired[list[Optional[bool]]]
    teacher_xtoken_loss: NotRequired[list[Optional[bool]]]


class CrossTokenizerDistillationLossDataDict(TypedDict):
    """Student-side keys are fixed; teacher-side keys are teacher-indexed.

    Only the student keys below are static. Each teacher ``i`` contributes a
    dynamic set of keys produced by ``CrossTokenizerCollator`` / the trainer and
    so cannot be enumerated here:

    - Every teacher: ``teacher_{i}_full_logits_ipc`` — List[B] of CUDA IPC handle
      dicts (``payload_ipc`` + ``buf_idx``/``sample_index_in_buf`` + TP/CP shard
      metadata) from ``Policy.get_full_logits_ipc``.
      ``rebuild_teacher_full_logits_from_ipc`` (in ``prepare_loss_input``)
      P2P-reads and reassembles full-vocab teacher logits, routing across
      heterogeneous teacher/student TP/CP.
    - Cross-tokenizer teacher only: ``teacher_{i}_input_ids`` /
      ``teacher_{i}_token_mask`` ``[B, T_t]`` and ``alignment_{i}_*``
      (``pair_valid`` / ``pair_is_correct`` ``[B, max_pairs]``;
      ``student_chunk_id`` ``[B, T_s]``; ``teacher_chunk_id`` ``[B, T_t]``).
    - Same-tokenizer teacher: no ``teacher_{i}_input_ids`` / ``alignment_{i}_*``;
      it reuses the student tokenization (identity 1:1 aligned).
    """

    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class CrossTokenizerDistillationLossFn(LossFunction):
    """Cross-tokenizer distillation loss.

    Mode is selected by the ``(gold_loss, xtoken_loss)`` flags:

    - ``(False, False)`` -> P-KL: full-vocab projection KL (student logits
      mapped through the projection matrix M) plus a standard next-token
      student CE term, combined as ``kl_loss_weight * kl + ce_loss_scale * ce``
      — or, when ``dynamic_loss_scaling`` is set, with the KL term rescaled
      each step to match the detached CE magnitude.
    - ``(True, False)`` -> gold-loss: KL on the exact-mapped *common* partition
      plus a sorted-L1 term on the *uncommon* tail
      (``kd = (kl_common + l1_uncommon) * T**2``), combined with a next-token
      student CE term the same way as the P-KL path —
      ``kl_loss_weight * kd + ce_loss_scale * ce``, or, when
      ``dynamic_loss_scaling`` is set, with the KD term rescaled each step to
      match the detached CE magnitude.
    - ``(True, True)`` -> gold-loss with the xtoken modifier: same objective,
      but the exact-map threshold is relaxed (``>= 0.6`` instead of ``== 1.0``)
      and a collision-replacement rule lets multi-token projections still
      contribute exact maps.

    ``(False, True)`` is rejected in ``__init__``: xtoken_loss is a modifier
    inside the gold path and is undefined for P-KL.

    Multi-teacher: ``setup`` injects per-teacher metadata (projection paths,
    weights, vocab sizes, per-teacher gold/xtoken overrides). The per-teacher KD
    terms are aggregated by ``kd_loss_mode`` (``sum`` / ``averaged_logits`` /
    ``select_teacher``) and combined with a single student CE term. A teacher
    with a ``None`` projection path is a *same-tokenizer* teacher: projection and
    alignment are skipped and its KD term is a direct top-k per-position KL on the
    shared vocab (top-k selected student-side over the reassembled full-vocab
    teacher logits). The single-teacher path is just ``num_teachers == 1``.

    Inputs (via ``LossInputType.DISTILLATION_CROSS_TOKENIZER``):
        logits: ``[B, T_s, V_s]`` raw student logits from the worker forward.
        student_logits_contig: CP-relaid contiguous student logits shared by
            every teacher's KD term.
        teacher_full_logits_by_idx: ``dict[int, [B, T, V_t]]`` full-vocab teacher
            logits per teacher, rebuilt from the CUDA IPC handles by
            ``prepare_loss_input`` (see
            :func:`nemo_rl.algorithms.cross_tokenizer.loss_utils.rebuild_teacher_full_logits_from_ipc`).
        aligns_by_idx: ``dict[int, LocalizedAlignment]`` per teacher (cross-tok:
            localized chunk alignment; same-tok: thin, student fields only).

    Inputs (via ``data: BatchedDataDict``):
        See :class:`CrossTokenizerDistillationLossDataDict`.

    Returns:
        ``(loss, metrics)``. Aggregate metrics: ``loss``, ``kl_loss`` (the
        aggregated KD term), ``ce_loss``, ``kl_loss_scale``, ``accuracy``,
        ``num_valid_samples``. Per-teacher metrics are suffixed ``_t{i}`` (e.g.
        ``kl_loss_t0``, ``proj_accuracy_t0``, ``weight_t0``); ``select_teacher``
        additionally reports ``selected_teacher``.
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION_CROSS_TOKENIZER

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        if cfg["xtoken_loss"] and not cfg["gold_loss"]:
            raise ValueError(
                "xtoken_loss=True requires gold_loss=True; xtoken_loss is "
                "a modifier inside the gold path (relaxes the exact-map "
                "threshold and adds collision resolution) and is undefined "
                "in the P-KL path."
            )
        # Dynamic teacher weighting (sum_weights_metric) and normalize_teacher_by_vocab
        # are only applied in kd_loss_mode="sum"; reject the combo instead of
        # silently ignoring them under the other modes.
        if cfg.get("sum_weights_metric") is not None and cfg["kd_loss_mode"] != "sum":
            raise ValueError(
                f"sum_weights_metric={cfg['sum_weights_metric']!r} is only applied "
                f"in kd_loss_mode='sum'; it is ignored by '{cfg['kd_loss_mode']}'. "
                "Unset one of them."
            )
        if cfg.get("normalize_teacher_by_vocab") and cfg["kd_loss_mode"] != "sum":
            raise ValueError(
                "normalize_teacher_by_vocab is only applied in kd_loss_mode='sum'; "
                f"it is ignored by '{cfg['kd_loss_mode']}'. Unset one of them."
            )
        # averaged_logits forms a convex combination of teacher logits
        # (weight_i / sum(weights)); a zero weight-sum (all zeros, or a signed
        # set cancelling to 0) makes that division undefined. Reject it here
        # rather than fail with a deep ZeroDivisionError mid-step.
        _weights = cfg.get("teacher_weights")
        if (
            cfg["kd_loss_mode"] == "averaged_logits"
            and _weights is not None
            and sum(_weights) == 0
        ):
            raise ValueError(
                "teacher_weights must not sum to zero in "
                "kd_loss_mode='averaged_logits' (they form a convex combination "
                f"of teacher logits); got teacher_weights={list(_weights)}."
            )
        # Global loss knobs (shared across all teachers).
        self.gold_loss = cfg["gold_loss"]
        self.xtoken_loss = cfg["xtoken_loss"]
        self.temperature = cfg["temperature"]
        self.vocab_topk = cfg["vocab_topk"]
        self.uncommon_topk = cfg["uncommon_topk"]
        self.reverse_kl = cfg["reverse_kl"]
        self.exact_token_match_only = cfg["exact_token_match_only"]
        self.kl_loss_weight = cfg["kl_loss_weight"]
        self.ce_loss_scale = cfg["ce_loss_scale"]
        self.dynamic_loss_scaling = cfg["dynamic_loss_scaling"]
        self.student_vocab_size = cfg["student_vocab_size"]
        # Multi-teacher aggregation knobs.
        self.kd_loss_mode = cfg["kd_loss_mode"]
        self.normalize_teacher_by_vocab = cfg["normalize_teacher_by_vocab"]
        self.alpha = cfg["alpha"]
        # sum_weights_metric is NotRequired -> None means static teacher_weights.
        self.sum_weights_metric = cfg.get("sum_weights_metric")
        # Per-teacher metadata: parallel lists (one entry per ``teachers[i]``),
        # injected by ``cross_tokenizer_distillation.setup``. Every teacher ships
        # full-vocab logits (the loss derives the top-k subset student-side), so
        # there is no per-teacher ``send_full_logits`` flag.
        self.projection_matrix_paths = list(cfg["projection_matrix_paths"])
        self.teacher_vocab_sizes = list(cfg["teacher_vocab_sizes"])
        self.teacher_weights = list(cfg["teacher_weights"])
        self.teacher_gold_loss = list(cfg["teacher_gold_loss"])
        self.teacher_xtoken_loss = list(cfg["teacher_xtoken_loss"])
        # Every per-teacher list must have the same length (one entry per
        # teacher); a mismatch would otherwise surface as a deep IndexError
        # mid-training instead of a clear error here.
        per_teacher_lens = {
            "projection_matrix_paths": len(self.projection_matrix_paths),
            "teacher_vocab_sizes": len(self.teacher_vocab_sizes),
            "teacher_weights": len(self.teacher_weights),
            "teacher_gold_loss": len(self.teacher_gold_loss),
            "teacher_xtoken_loss": len(self.teacher_xtoken_loss),
        }
        if len(set(per_teacher_lens.values())) != 1:
            raise ValueError(
                f"per-teacher lists must be equal length, got {per_teacher_lens}"
            )
        self.num_teachers = len(self.projection_matrix_paths)
        # The materialized projection matrix and the derived exact-map
        # partition both live in process-local caches in
        # ``x_token.loss_utils`` (see ``get_sparse_projection_matrix``,
        # ``get_topk_projection``, ``build_exact_token_map``), not on
        # this instance. That keeps the driver-side ``loss_fn`` free of
        # any large CUDA tensors and lets multiple loss instances on
        # the same worker share one load.

    def _teacher_is_same_vocab(self, i: int) -> bool:
        """A teacher is same-vocab (direct KL, no projection) iff its path is None."""
        return self.projection_matrix_paths[i] is None

    def __call__(
        self,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        logits: torch.Tensor,
        student_logits_contig: torch.Tensor,
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        *,
        student_next_token_logprobs: Optional[torch.Tensor] = None,
        student_next_token_mask: Optional[torch.Tensor] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the (multi-teacher) cross-tokenizer distillation loss.

        Per-teacher KD terms are aggregated per ``kd_loss_mode`` and combined
        with a single student next-token CE term — dynamic-scaled when
        ``dynamic_loss_scaling`` is set (KD rescaled to match the detached CE
        magnitude, ``kl_loss_weight`` / ``ce_loss_scale`` ignored), else
        fixed-weighted. The single-teacher path is just ``num_teachers == 1``.

        ``student_logits_contig`` (CP-relaid) and the per-teacher ``aligns_by_idx``
        / ``teacher_full_logits_by_idx`` are precomputed in ``prepare_loss_input``;
        the Automodel CP path also supplies its sequence-local CE inputs.
        """
        ce_loss = self._compute_ce(
            logits,
            data,
            global_valid_toks,
            student_next_token_logprobs=student_next_token_logprobs,
            student_next_token_mask=student_next_token_mask,
            cp_group=cp_group,
        )

        if self.kd_loss_mode == "sum":
            total_kd, per_teacher_metrics = self._sum_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        elif self.kd_loss_mode == "averaged_logits":
            total_kd, per_teacher_metrics = self._averaged_logits_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        elif self.kd_loss_mode == "select_teacher":
            total_kd, per_teacher_metrics = self._select_teacher_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        else:
            raise ValueError(f"Unknown kd_loss_mode: {self.kd_loss_mode!r}")

        # Combine the aggregated KD term with the single student CE term.
        if self.dynamic_loss_scaling:
            # loss = sg(ce/kd) * kd + ce; user kl_loss_weight / ce_loss_scale
            # are intentionally ignored in this branch.
            kd_detached = total_kd.detach().abs()
            ce_detached = ce_loss.detach().abs()
            kl_scale = torch.where(
                kd_detached > 0,
                ce_detached / kd_detached,
                torch.ones_like(kd_detached),
            )
            loss = kl_scale * total_kd + ce_loss
        else:
            kl_scale = torch.tensor(1.0, device=total_kd.device, dtype=total_kd.dtype)
            loss = self.kl_loss_weight * total_kd + self.ce_loss_scale * ce_loss

        # Next-token accuracy on the student side (quick per-step signal), masked
        # to valid tokens. Computed once on the student from the shared CP-relaid
        # fields (carried on every teacher's align); the CP-aware shift pairs
        # predictors with the right labels under load-balanced sharding.
        align0 = aligns_by_idx[0]
        accuracy = next_token_accuracy(
            student_logits_contig,
            input_ids=align0.student_input_ids,
            token_mask=align0.student_token_mask,
            sample_mask=data["sample_mask"],
            tp_group=tp_group,
            cp_group=cp_group,
        )

        metrics: dict[str, Any] = {
            "loss": loss.item(),
            # Aggregate KD term (kept under ``kl_loss`` so existing trainer
            # metric handling continues to work); per-teacher terms are suffixed
            # ``_t{i}``.
            "kl_loss": total_kd.item(),
            "ce_loss": ce_loss.item(),
            "kl_loss_scale": kl_scale.item(),
            "accuracy": accuracy.item(),
            "num_valid_samples": data["input_ids"].shape[0],
        }
        metrics.update(per_teacher_metrics)
        return loss, metrics

    # ------------------------------------------------------------------ #
    # Multi-teacher aggregation
    # ------------------------------------------------------------------ #
    def _resolve_gold_xtoken(self, i: int, use_per_teacher: bool) -> tuple[bool, bool]:
        """Effective ``(gold_loss, xtoken_loss)`` for teacher ``i``.

        Per-teacher overrides are honored only when ``use_per_teacher`` is set
        (``sum`` mode); ``select_teacher`` / ``averaged_logits`` use the global
        flags. A ``None`` override falls back to the global value.
        """
        if not use_per_teacher:
            return self.gold_loss, self.xtoken_loss
        gold = (
            self.teacher_gold_loss[i]
            if self.teacher_gold_loss[i] is not None
            else self.gold_loss
        )
        xtoken = (
            self.teacher_xtoken_loss[i]
            if self.teacher_xtoken_loss[i] is not None
            else self.xtoken_loss
        )
        return gold, xtoken

    def _compute_teacher_kd(
        self,
        i: int,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        use_per_teacher_flags: bool,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """KD term for teacher ``i`` plus its (unsuffixed) metrics.

        Dispatches on tokenizer kind: same-vocab -> direct top-k per-position KL;
        cross-tokenizer -> P-KL or gold path using teacher ``i``'s projection and
        its localized alignment. Both consume the shared CP-relaid student logits
        and route TP/CP through main's parameterized loss-mode helpers.
        """
        if self._teacher_is_same_vocab(i):
            return self._compute_same_vocab_kl(
                i,
                student_logits_contig,
                teacher_full_logits_by_idx[i],
                aligns_by_idx[i],
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )

        teacher_logits = teacher_full_logits_by_idx[i]
        align = aligns_by_idx[i]
        proj_path = self.projection_matrix_paths[i]
        v_t = self.teacher_vocab_sizes[i]
        gold, xtoken = self._resolve_gold_xtoken(i, use_per_teacher_flags)
        if xtoken and not gold:
            raise ValueError(f"teacher {i}: xtoken_loss=True requires gold_loss=True.")

        if gold:
            kd, kl_common, l1_uncommon, num_valid_chunks, top1 = self._compute_gold(
                student_logits_contig,
                teacher_logits,
                align,
                projection_matrix_path=proj_path,
                teacher_vocab_size=v_t,
                xtoken_loss=xtoken,
                tp_group=tp_group,
                cp_group=cp_group,
            )
            return kd, {
                "kl_loss": kd.item(),
                "kl_common": kl_common.item(),
                "l1_uncommon": l1_uncommon.item(),
                "proj_accuracy": top1.item(),
                "num_valid_chunks": int(num_valid_chunks.item()),
            }

        kl, num_valid_pairs, proj_acc = self._compute_p_kl(
            student_logits_contig,
            teacher_logits,
            align,
            projection_matrix_path=proj_path,
            teacher_vocab_size=v_t,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        return kl, {
            "kl_loss": kl.item(),
            "proj_accuracy": proj_acc.item(),
            "num_valid_pairs": int(num_valid_pairs.item()),
        }

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
        position alignment), so no projection / no chunk-averaging. The reduction
        matches CE: masked next-token mean normalized by ``global_valid_toks``,
        scaled by ``T**2``.
        """
        assert align.student_token_mask is not None
        kd = same_tokenizer_topk_kl(
            student_logits_contig,
            teacher_full_logits,
            kd_mask=align.student_token_mask,
            sample_mask=align.sample_mask,
            num_kd_tokens=global_valid_toks,
            vocab_size=self.student_vocab_size,
            vocab_topk=self.vocab_topk,
            temperature=self.temperature,
            reverse_kl=self.reverse_kl,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        return kd, {"kl_loss": kd.item()}

    def _sum_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Weighted sum: ``total_kd = Σ_i weight_i · KD_i``.

        Weights are static (config ``weight``) or dynamic (``sum_weights_metric``).
        When ``normalize_teacher_by_vocab`` is set, each teacher's KD is
        additionally scaled by ``log(V_t_i) / log(min_j V_t_j)``.
        """
        device = student_logits_contig.device
        if self.sum_weights_metric is not None:
            weights = self._compute_dynamic_weights(
                data, teacher_full_logits_by_idx, aligns_by_idx, cp_group=cp_group
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
                use_per_teacher_flags=True,
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
            for k, v in m_i.items():
                per_metrics[f"{k}_t{i}"] = v
            per_metrics[f"weight_t{i}"] = float(weights[i].item())
        assert total_kd is not None
        return total_kd, per_metrics

    def _averaged_logits_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
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
        same_tokenizer = all(p is None for p in self.projection_matrix_paths)
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
                    use_per_teacher_flags=False,
                    tp_group=tp_group,
                    cp_group=cp_group,
                )
                w = self.teacher_weights[i]
                weighted = kd_i * w
                total_kd = weighted if total_kd is None else total_kd + weighted
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
        assert align.student_token_mask is not None
        kd = same_tokenizer_full_vocab_kl(
            student_logits_contig,
            avg,
            kd_mask=align.student_token_mask,
            sample_mask=align.sample_mask,
            num_kd_tokens=global_valid_toks,
            vocab_size=self.student_vocab_size,
            temperature=self.temperature,
            reverse_kl=self.reverse_kl,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        return kd, {"kl_loss": kd.item()}

    def _dp_global_masked_mean(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked mean of ``values`` over the *process-global* valid count.

        Teacher selection / dynamic weighting must be identical on every rank: a
        rank-local mean lets ranks pick a different teacher / different weights,
        and the per-teacher KD's collectives then see divergent participation
        (deadlock when one rank's choice fires a collective another's does not).
        All-reduce the masked sum and the mask count over the full group so every
        rank gets the same score (mirrors ``_compute_p_kl``'s WORLD-reduced
        denominator). The result is detached (it gates selection / weighting and
        is not back-propagated).
        """
        num = group_all_reduce_sum(
            (values * mask).sum(), group=torch.distributed.group.WORLD
        )
        den = group_all_reduce_sum(
            mask.sum(), group=torch.distributed.group.WORLD
        ).clamp(min=1.0)
        return num / den

    def _select_teacher_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Use only the teacher with the lowest next-token CE on its own tokens."""
        with torch.no_grad():
            ces: list[float] = []
            for i in range(self.num_teachers):
                t_logits, next_ids, next_mask, _ = self._teacher_score_inputs(
                    i, teacher_full_logits_by_idx, aligns_by_idx, cp_group=cp_group
                )
                ce = self._teacher_next_token_ce(
                    t_logits,
                    next_ids=next_ids,
                    next_mask=next_mask,
                    sample_mask=data["sample_mask"],
                )
                ces.append(ce.item())
            best = int(min(range(self.num_teachers), key=lambda j: ces[j]))

        kd, m = self._compute_teacher_kd(
            best,
            student_logits_contig,
            data,
            teacher_full_logits_by_idx,
            aligns_by_idx,
            global_valid_toks,
            use_per_teacher_flags=False,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        per_metrics: dict[str, Any] = {f"{k}_t{best}": v for k, v in m.items()}
        per_metrics["selected_teacher"] = best
        return kd, per_metrics

    def _teacher_score_inputs(
        self,
        i: int,
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        *,
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, next_ids, next_mask, token_mask)`` for teacher ``i``'s scores.

        All four cover this CP rank's contiguous window of the tokenization the
        score is computed on: the shared student tokens for a same-vocab teacher,
        teacher ``i``'s own otherwise. ``next_ids`` / ``next_mask`` hold each
        position's next token: the window's last position takes the first token
        of the next CP rank's window, and the sequence's last position has none
        (``next_mask`` is 0). Every teacher ships full logits, so the full
        distribution is always available.
        """
        align = aligns_by_idx[i]
        if self._teacher_is_same_vocab(i):
            ids = to_local_if_dtensor(align.student_input_ids)
            token_mask = to_local_if_dtensor(align.student_token_mask)
        else:
            if align.teacher_input_ids is None or align.teacher_token_mask is None:
                raise ValueError(
                    f"localized alignment for teacher {i} is missing the teacher "
                    "input ids or token mask"
                )
            ids = to_local_if_dtensor(align.teacher_input_ids)
            token_mask = to_local_if_dtensor(align.teacher_token_mask)
        token_mask = token_mask.float()
        next_ids = cp_shift_next(ids, cp_group, fill=0)
        next_mask = cp_shift_next(token_mask, cp_group, fill=0.0)
        return teacher_full_logits_by_idx[i], next_ids, next_mask, token_mask

    def _teacher_next_token_ce(
        self,
        t_logits: torch.Tensor,
        *,
        next_ids: torch.Tensor,
        next_mask: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher's next-token CE on its own tokens, averaged over every rank's window."""
        ce_pos = torch.nn.functional.cross_entropy(
            t_logits.reshape(-1, t_logits.shape[-1]).float(),
            next_ids.reshape(-1),
            reduction="none",
        )
        samp = to_local_if_dtensor(sample_mask).unsqueeze(-1).float()
        mask = (next_mask * samp).reshape(-1)
        return self._dp_global_masked_mean(ce_pos, mask)

    def _compute_dynamic_weights(
        self,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        *,
        cp_group: Optional[torch.distributed.ProcessGroup],
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
            t_logits, next_ids, next_mask, token_mask = self._teacher_score_inputs(
                i, teacher_full_logits_by_idx, aligns_by_idx, cp_group=cp_group
            )
            score = self._teacher_weight_score(
                t_logits,
                next_ids=next_ids,
                next_mask=next_mask,
                token_mask=token_mask,
                sample_mask=data["sample_mask"],
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
        *,
        next_ids: torch.Tensor,
        next_mask: torch.Tensor,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar weight-metric score for one teacher (higher = more trusted).

        Padded positions and masked-out samples are excluded, so long-padded
        batches don't let near-uniform padding logits dominate the score.
        """
        if self.sum_weights_metric == "ce":
            return -self._teacher_next_token_ce(
                t_logits,
                next_ids=next_ids,
                next_mask=next_mask,
                sample_mask=sample_mask,
            )
        samp = to_local_if_dtensor(sample_mask).unsqueeze(-1).float()
        mask = token_mask * samp
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
    def _compute_p_kl(
        self,
        student_logits: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        *,
        projection_matrix_path: Optional[str],
        teacher_vocab_size: int,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """P-KL: chunk-averaged KL over a microbatch-global top-k teacher subset.

        ``student_logits`` (CP-relaid to contiguous) and ``align`` (localized,
        with next-token-shifted chunk ids) are precomputed in
        ``prepare_loss_input``.

        Steps:

        1. Use the full teacher logits materialized by ``prepare_loss_input``.
        2. Compute one ``global_top_indices [k]`` per microbatch from the
           teacher's importance: ``max`` over flat ``(B*T_t)``, ``topk``
           over ``V_t``. Same vocab subset across every sample/position —
           keeps chunk-averaged KL well-defined.
        3. Restrict ``M`` to those ``k`` teacher columns and project the
           student probs through it, so only the ``k`` columns are ever
           produced. Each teacher column of ``M.t() @ p`` is an independent
           contraction over the student axis, so slicing before the matmul
           is value-preserving.
        4. Slice the teacher logits to the same ``k`` columns.
        5. Build per-token chunk masks from ``alignment_*_chunk_id`` and
           chunk-average via ``bmm`` (shared helper).
        6. Renormalize student chunk distributions inside the top-k subset
           (avg-then-renormalize, log).
        7. Forward (or reverse) KL between chunk distributions.
        """
        T = self.temperature
        device = student_logits.device
        eps = 1e-10

        # Vocab-sharded (TP) students keep the shard with globally-correct
        # normalization; the result is projected onto the teacher vocab next.
        student_log_probs = vocab_parallel_log_softmax(
            student_logits, T, tp_group=tp_group
        )
        student_probs = student_log_probs.exp()  # [B, T_s_local, V_s_local]

        sparse_projection = get_sparse_projection_matrix(
            projection_matrix_path,
            device,
            student_vocab_size=self.student_vocab_size,
            teacher_vocab_size=teacher_vocab_size,
        )  # [V_s, V_t] sparse COO, fp32
        # The projection itself is deferred until the top-k columns are known
        # (see below); its teacher-vocab width is available from the matrix.
        full_teacher_vocab_size = sparse_projection.size(1)

        # `teacher_full_logits` [B, T_t, V_t_model] is materialized by
        # `prepare_loss_input` (rebuilt from the IPC handles). Same transport
        # as the gold path consumes; here we additionally compute a
        # microbatch-global top-k inline.
        # HF models commonly pad lm_head out_features beyond len(tokenizer)
        # for embedding/FFN alignment (e.g. Qwen3: tokenizer 151669,
        # lm_head 151936). The projection matrix is sized to the real
        # tokenizer vocab (the ``teacher_vocab_size`` arg); the padded
        # columns aren't real tokens and the projection has no entries
        # there. Slice to the projection's V_t to keep the projected
        # student probs and the teacher logits on the same vocab axis.
        if teacher_full_logits.shape[-1] > full_teacher_vocab_size:
            teacher_full_logits = teacher_full_logits[..., :full_teacher_vocab_size]

        # Chunk ids (localized + next-token-shifted) come from `prepare_loss_input`.
        student_chunk_id = align.student_chunk_id
        teacher_chunk_id = align.teacher_chunk_id
        pair_valid = align.pair_valid  # [B, max_pairs]
        if self.exact_token_match_only:
            pair_valid = pair_valid & align.pair_is_correct
        max_chunks = pair_valid.shape[1]

        # One microbatch-global top-k teacher subset (CP-reduced so every rank
        # agrees on the same vocab columns), shared across all samples/positions.
        vocab_topk = min(self.vocab_topk, full_teacher_vocab_size)
        global_top_indices = select_teacher_topk_indices(
            teacher_full_logits, vocab_topk, cp_group=cp_group
        )  # [k]

        # Slice both sides to the shared [k] columns. On the student side the
        # slice is folded into the projection matrix rather than applied to its
        # output: every teacher column of ``M.t() @ p`` is an independent
        # contraction over the student axis, so restricting M's columns first is
        # value-preserving, and the only renormalization here is within the [k]
        # subset (below), never over V_t. Projecting all V_t columns and then
        # discarding all but k built -- and all-reduced, under TP -- a
        # [B, T_s, V_t] fp32 tensor to read k of its columns.
        if vocab_topk < full_teacher_vocab_size:
            sparse_projection = slice_sparse_projection_cols(
                sparse_projection, global_top_indices
            )
        projected_topk = project_student_to_teacher_vocab(
            student_probs, sparse_projection, tp_group=tp_group
        )  # [B, T_s_local, k]
        teacher_topk_logits = teacher_full_logits[
            ..., global_top_indices
        ]  # [B, T_t, k]
        target_log_probs = torch.log_softmax(
            teacher_topk_logits / T, dim=-1
        )  # [B, T_t, k] (renormalized within the [k] subset).

        # Chunk-average both sides via the shared helper.
        proj_chunks, proj_sizes = chunk_average_log_probs(
            projected_topk, student_chunk_id, max_chunks, cp_group=cp_group
        )
        tgt_log_chunks, tgt_sizes = chunk_average_log_probs(
            target_log_probs, teacher_chunk_id, max_chunks, cp_group=cp_group
        )

        # Renormalize the projected chunk distribution within the top-k
        # subset, then take log. Teacher side is already log-probs (avg of
        # log_softmaxes; not a true log of mean).
        proj_chunks = proj_chunks / (proj_chunks.sum(dim=-1, keepdim=True) + eps)
        proj_log_chunks = (proj_chunks + eps).log()

        chunk_mask = valid_chunk_mask(proj_sizes, tgt_sizes, pair_valid)
        # Compute the DP-global valid-chunk count BEFORE the early
        # return so the collective fires on every rank (a local-only
        # `chunk_mask.any()` check would deadlock when one rank skips
        # and others do not). The reduction at the bottom uses this
        # global count so the KL is normalized by
        # `sum(global_valid_chunks)` rather than a per-rank mean —
        # mirrors the `global_valid_toks` convention used by CE.
        sample_mask_bool = align.sample_mask.bool()
        valid_bool = chunk_mask & sample_mask_bool.unsqueeze(-1)
        global_valid_chunks = group_all_reduce_sum(
            valid_bool.sum().to(torch.float32), group=torch.distributed.group.WORLD
        )
        if global_valid_chunks.item() == 0:
            zero = torch.zeros((), device=device, dtype=proj_log_chunks.dtype)
            return (
                zero,
                torch.zeros((), device=device, dtype=torch.long),
                zero.detach(),
            )

        # Projection top-1 accuracy: per-chunk argmax of the student-side
        # projected distribution vs the teacher's argmax over the same
        # top-k subset.
        with torch.no_grad():
            proj_top1 = proj_chunks.argmax(dim=-1)  # [B, C]
            tgt_top1 = torch.exp(tgt_log_chunks).argmax(dim=-1)  # [B, C]
            proj_matches = (proj_top1 == tgt_top1) & chunk_mask
            proj_acc = proj_matches.sum().float() / chunk_mask.sum().float().clamp(
                min=1.0
            )

        # KL between chunk-averaged distributions.
        if self.reverse_kl:
            # KL(student || teacher)
            per_chunk_kl = torch.nn.functional.kl_div(
                tgt_log_chunks, proj_log_chunks, reduction="none", log_target=True
            ).sum(dim=-1)
        else:
            # Forward KL(teacher || student)
            per_chunk_kl = torch.nn.functional.kl_div(
                proj_log_chunks, tgt_log_chunks, reduction="none", log_target=True
            ).sum(dim=-1)

        sample_mask = align.sample_mask.to(per_chunk_kl.dtype)  # [B]
        valid = chunk_mask.to(per_chunk_kl.dtype) * sample_mask.unsqueeze(-1)
        denom = global_valid_chunks.to(per_chunk_kl.dtype).clamp(min=1.0)
        kl_loss = (per_chunk_kl * valid).sum() / denom * (T * T)

        return kl_loss, valid.sum().detach(), proj_acc.detach()

    def _compute_gold(
        self,
        student_logits: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        *,
        projection_matrix_path: Optional[str],
        teacher_vocab_size: int,
        xtoken_loss: bool,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gold-loss path: KL on common (exact-mapped) vocab + L1 on uncommon.

        ``student_logits`` (CP-relaid to contiguous) and ``align`` (localized,
        with next-token-shifted chunk ids) are precomputed in
        ``prepare_loss_input``.

        1. Lazy-build the exact-token map (cached per device).
        2. Use the full teacher logits materialized by ``prepare_loss_input``.
        3. ``log_softmax`` on full vocab both sides; chunk-average via the
           shared helper using the precomputed next-token-shifted chunk ids.
        4. Slice each chunk-averaged tensor to ``common_*`` indices and
           compute (forward or reverse) KL, reduced as
           ``sum / valid_chunk.sum()`` where ``valid_chunk`` is the
           geometric chunk mask AND'd with ``sample_mask`` (mirrors the
           P-KL path).
        5. Slice to ``uncommon_*`` indices, ``.exp()`` to probs, sort/topk
           descending (capped at ``self.uncommon_topk``), truncate to
           ``min(student_len, teacher_len)``, L1 with ``reduction="none"``
           summed over vocab and meaned across valid chunks.
        6. Combine: ``loss = (kl_common + l1_uncommon) * T**2``.
        7. Top-1 accuracy on the common slice over valid chunks.

        Returns ``(loss, kl_common, l1_uncommon, num_valid_chunks, top1_acc)``.
        Components other than ``loss`` are detached.
        """
        T = self.temperature
        device = student_logits.device

        exact_map = build_exact_token_map(
            projection_matrix_path,
            device,
            xtoken_loss=xtoken_loss,
            teacher_vocab_size=teacher_vocab_size,
        )
        common_s = exact_map["common_student"]
        common_t = exact_map["common_teacher"]
        uncommon_s = exact_map["uncommon_student"]
        uncommon_t = exact_map["uncommon_teacher"]
        v_teacher = teacher_vocab_size

        # `teacher_full_logits` [B, T_t, V_t_model] is materialized by
        # `prepare_loss_input` (rebuilt from the IPC handles).
        # Drop any padded lm_head vocab beyond the real tokenizer vocab —
        # the exact-token map's t-axis is bounded by `teacher_vocab_size`,
        # so chunked teacher log-probs must use the same axis. See the
        # matching note in `_compute_p_kl` for why the model vocab can
        # exceed `len(tokenizer)`.
        if teacher_full_logits.shape[-1] > v_teacher:
            teacher_full_logits = teacher_full_logits[..., :v_teacher]

        # common_s / uncommon_s are arbitrary V_s indices, so the gold path needs
        # full-vocab student log-probs (TP-sharded students are all-gathered).
        student_log_probs = vocab_parallel_full_log_softmax(
            student_logits, T, tp_group=tp_group
        )
        # teacher_full_logits is already vocab-full (consumer-side routing
        # P2P-concat'd across TP siblings); local log_softmax is correct.
        teacher_log_probs = torch.log_softmax(
            teacher_full_logits / T, dim=-1
        )  # [B, T_t_local, V_t]

        # Chunk ids (localized + next-token-shifted) come from `prepare_loss_input`.
        student_chunk_id = align.student_chunk_id
        teacher_chunk_id = align.teacher_chunk_id
        pair_valid = align.pair_valid
        max_chunks = pair_valid.shape[1]

        student_chunks, s_sizes = chunk_average_log_probs(
            student_log_probs, student_chunk_id, max_chunks, cp_group=cp_group
        )
        teacher_chunks, t_sizes = chunk_average_log_probs(
            teacher_log_probs, teacher_chunk_id, max_chunks, cp_group=cp_group
        )

        chunk_mask = valid_chunk_mask(s_sizes, t_sizes, pair_valid)
        # Match the P-KL path: a chunk only contributes if its alignment is
        # geometrically valid AND its sample isn't masked out by sample_mask.
        sample_mask = align.sample_mask  # [B]
        valid_chunk = chunk_mask & sample_mask.bool().unsqueeze(-1)
        zero_dtype = student_log_probs.dtype
        # Compute the DP-global valid-chunk count BEFORE any potentially
        # divergent early return so the collective fires on every rank;
        # both `kl_common` and `l1_uncommon` use this as their denom so
        # the loss is normalized by `sum(global_valid_chunks)`, not a
        # per-rank mean.
        global_valid_chunks = group_all_reduce_sum(
            valid_chunk.sum().to(torch.float32), group=torch.distributed.group.WORLD
        )
        if global_valid_chunks.item() == 0:
            zero = torch.zeros((), device=device, dtype=zero_dtype)
            return (
                zero,
                zero.detach(),
                zero.detach(),
                torch.zeros((), device=device, dtype=torch.long),
                zero.detach(),
            )

        # ---------------------- KL on common ----------------------
        if common_s.numel() > 0:
            student_common = student_chunks[:, :, common_s]  # [B, C, N_common]
            teacher_common = teacher_chunks[:, :, common_t]  # [B, C, N_common]
            if self.reverse_kl:
                kl_per_elem = torch.nn.functional.kl_div(
                    teacher_common,
                    student_common,
                    reduction="none",
                    log_target=True,
                )
            else:
                kl_per_elem = torch.nn.functional.kl_div(
                    student_common,
                    teacher_common,
                    reduction="none",
                    log_target=True,
                )
            kl_per_chunk = kl_per_elem.sum(dim=-1) * valid_chunk  # [B, C]
            kl_common = kl_per_chunk.sum() / global_valid_chunks.to(
                kl_per_chunk.dtype
            ).clamp(min=1.0)
        else:
            kl_common = torch.zeros(
                (), device=device, dtype=zero_dtype, requires_grad=True
            )
            student_common = None
            teacher_common = None

        # -------------------- L1 on uncommon ----------------------
        uncommon_topk = self.uncommon_topk
        if uncommon_s.numel() > 0 or uncommon_t.numel() > 0:
            student_unc = student_chunks[:, :, uncommon_s][
                valid_chunk
            ]  # [N_valid, N_u_s]
            teacher_unc = teacher_chunks[:, :, uncommon_t][
                valid_chunk
            ]  # [N_valid, N_u_t]
            n_valid = student_unc.shape[0]
            max_uncommon = min(
                student_unc.shape[-1],
                teacher_unc.shape[-1],
                uncommon_topk,
            )
            if n_valid > 0 and max_uncommon > 0:
                student_unc_probs = student_unc.exp()
                teacher_unc_probs = teacher_unc.exp()
                if student_unc_probs.shape[-1] > max_uncommon:
                    student_sorted = torch.topk(
                        student_unc_probs, k=max_uncommon, dim=-1, largest=True
                    ).values
                else:
                    student_sorted = student_unc_probs.sort(
                        dim=-1, descending=True
                    ).values
                if teacher_unc_probs.shape[-1] > max_uncommon:
                    teacher_sorted = torch.topk(
                        teacher_unc_probs, k=max_uncommon, dim=-1, largest=True
                    ).values
                else:
                    teacher_sorted = teacher_unc_probs.sort(
                        dim=-1, descending=True
                    ).values
                min_len = min(student_sorted.shape[-1], teacher_sorted.shape[-1])
                student_sorted = student_sorted[:, :min_len]
                teacher_sorted = teacher_sorted[:, :min_len]
                l1_per_chunk = torch.nn.functional.l1_loss(
                    student_sorted, teacher_sorted, reduction="none"
                ).sum(dim=-1)
                l1_uncommon = l1_per_chunk.sum() / global_valid_chunks.to(
                    l1_per_chunk.dtype
                ).clamp(min=1.0)
            else:
                l1_uncommon = torch.zeros(
                    (), device=device, dtype=zero_dtype, requires_grad=True
                )
        else:
            l1_uncommon = torch.zeros(
                (), device=device, dtype=zero_dtype, requires_grad=True
            )

        # -------------------- Top-1 accuracy ----------------------
        with torch.no_grad():
            if student_common is not None:
                s_common_valid = student_common[valid_chunk]
                t_common_valid = teacher_common[valid_chunk]
                matches = (
                    (s_common_valid.argmax(dim=-1) == t_common_valid.argmax(dim=-1))
                    .sum()
                    .float()
                )
                top1_acc = matches / valid_chunk.sum().float().clamp(min=1.0)
            else:
                top1_acc = torch.zeros((), device=device, dtype=zero_dtype)

        loss = (kl_common + l1_uncommon) * (T * T)
        return (
            loss,
            kl_common.detach(),
            l1_uncommon.detach(),
            valid_chunk.sum().detach(),
            top1_acc.detach(),
        )

    def _compute_ce(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        global_valid_toks: torch.Tensor,
        *,
        student_next_token_logprobs: Optional[torch.Tensor] = None,
        student_next_token_mask: Optional[torch.Tensor] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Next-token CE on the student side (TP/CP handled by the helpers)."""
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

        per_token_ce = student_next_token_ce(logits, input_ids=data["input_ids"])
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
