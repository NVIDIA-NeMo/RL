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
"""Shared base class of the cross-tokenizer distillation losses."""

import abc
from typing import Any, Optional

import torch

from nemo_rl.algorithms.cross_tokenizer.loss_utils import (
    LocalizedAlignment,
    ce_label_mask,
    same_tokenizer_full_vocab_kl,
    same_tokenizer_topk_kl,
    student_next_token_ce,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    cp_shift_next,
    group_all_reduce_sum,
    group_all_reduce_sum_with_grad,
    to_local_if_dtensor,
)
from nemo_rl.utils.tensor_ops import masked_mean


class CrossTokenizerDistillationLossFn(LossFunction):
    """Shared multi-teacher logic of the cross-tokenizer distillation losses.

    Each teacher contributes a KD term from the subclass hook
    :meth:`_compute_teacher_kd`; this class aggregates the terms by
    ``kd_loss_mode`` (``sum`` / ``averaged_logits`` / ``select_teacher``),
    computes static or dynamic (``sum_weights_metric``) teacher weights, scores
    teachers on their own tokens, and provides the student next-token CE term
    and the direct KL of a teacher that shares the student's tokenizer. Whether
    a teacher shares the tokenizer is the subclass hook
    :meth:`_teacher_is_same_vocab`.

    Subclasses set the attributes the shared methods read: ``num_teachers``,
    ``teacher_weights``, ``teacher_vocab_sizes``, ``alpha``,
    ``sum_weights_metric``, ``normalize_teacher_by_vocab``,
    ``student_vocab_size``, ``vocab_topk``, ``temperature`` and ``reverse_kl``.

    Per-teacher metrics are suffixed ``_t{i}`` (for example ``kl_loss_t0`` and
    ``weight_t0``); every aggregation mode also reports each teacher's share of
    the aggregated KD term as ``teacher_{i}/weighted_kl``, and
    ``select_teacher`` reports ``selected_teacher``.
    """

    # Group that _dp_global_masked_mean reduces over. Declared at class scope so
    # instances built without __init__ have it; None reduces over WORLD.
    _dp_cp_group: Optional[torch.distributed.ProcessGroup] = None

    @abc.abstractmethod
    def _teacher_is_same_vocab(self, i: int) -> bool:
        """Whether teacher ``i`` shares the student's tokenizer (direct KL, no alignment)."""

    @abc.abstractmethod
    def _compute_teacher_kd(
        self,
        i: int,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[Any],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        use_per_teacher_flags: bool,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """KD term for teacher ``i`` plus its (unsuffixed) metrics.

        ``use_per_teacher_flags`` is True in ``sum`` mode only: per-teacher loss
        overrides apply there, and the other modes use the global settings.
        """

    @staticmethod
    def _student_kd_mask(align: LocalizedAlignment) -> torch.Tensor:
        """Return the semantic KD mask, with legacy text-batch fallback."""
        mask = align.student_kd_token_mask
        if mask is None:
            mask = align.student_token_mask
        if mask is None:
            raise ValueError("localized alignment is missing a student KD token mask")
        return to_local_if_dtensor(mask)

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
        return kd, {"kl_loss": kd.item()}

    def _sum_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[Any],
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

        temp_weight: Optional[torch.Tensor] = None
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
            if temp_weight is not None:
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
        data: BatchedDataDict[Any],
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
                    use_per_teacher_flags=False,
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
        data: BatchedDataDict[Any],
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
        for i in range(self.num_teachers):
            per_metrics[f"teacher_{i}/weighted_kl"] = 0.0
        per_metrics[f"teacher_{best}/weighted_kl"] = float(kd.item())
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
            assert align.student_input_ids is not None
            ids = to_local_if_dtensor(align.student_input_ids)
            token_mask = self._student_kd_mask(align)
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
        data: BatchedDataDict[Any],
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
        temp_weight: Optional[torch.Tensor] = None
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
            if temp_weight is not None:
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

    def _compute_ce(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict[Any],
        global_valid_toks: torch.Tensor,
        *,
        student_next_token_logprobs: Optional[torch.Tensor] = None,
        student_next_token_mask: Optional[torch.Tensor] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Next-token CE on the student side (TP/CP handled by the helpers).

        The Automodel CP path supplies sequence-local target log-probabilities,
        avoiding full-tensor gathers. DTensor logits otherwise carry their own
        device mesh, so the vocab-parallel log-prob helper resolves TP/CP from
        it.
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
