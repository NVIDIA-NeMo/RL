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

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from nemo_rl.algorithms.loss.draft import DraftLossStats
from nemo_rl.models.megatron.draft.eagle_ttt import (
    EagleTTTAttentionPlan,
    EagleTTTResourceLedger,
    EagleTTTResourceLimitError,
    EagleTTTSequenceLayout,
    EagleTTTStoragePlan,
    reset_eagle_ttt_attention_state,
)

__all__ = [
    "EaglePassLoss",
    "EaglePassRunner",
    "EagleTTTSession",
    "EagleTTTOutput",
    "EagleTTTProvider",
    "EagleTTTResourceLedger",
    "EagleTTTResourceLimitError",
    "ProjectedEaglePassLoss",
]


class EagleTTTSession(Protocol):
    """Stateful session over a structured trunk-plus-branch attention backend."""

    def begin(
        self,
        *,
        layout: EagleTTTSequenceLayout,
        storage_plan: EagleTTTStoragePlan,
        excluded_tensors: tuple[Tensor, ...],
        resource_ledger: EagleTTTResourceLedger,
        packed_seq_params: object | None = None,
    ) -> None:
        """Arm one invocation before any pass executes."""

    def __call__(
        self,
        *,
        embeddings: Tensor,
        hidden_states: Tensor,
        plan: EagleTTTAttentionPlan,
        rope_positions: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    def reset(self) -> None:
        """Release all per-invocation cache and hook state."""


EaglePassRunner = EagleTTTSession


class EaglePassLoss(Protocol):
    """Project and reduce one pass without returning its vocabulary logits."""

    def __call__(
        self,
        *,
        hidden_states: Tensor,
        plan: EagleTTTAttentionPlan,
    ) -> DraftLossStats: ...


@dataclass(frozen=True, slots=True)
class EagleTTTOutput:
    """Additive pass statistics and the detached state for the next invocation."""

    stats: DraftLossStats
    final_branch_state: Tensor
    plans: tuple[EagleTTTAttentionPlan, ...]


class ProjectedLoss(Protocol):
    """PR6 projected/recompute loss seam, kept dependency-only in this PR."""

    def __call__(
        self,
        *,
        student_hidden: Tensor,
        output_weight: Tensor,
        teacher_logits: Tensor,
        teacher_row_indices: Tensor,
        mask: Tensor,
        token_chunk_size: int,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats: ...


@dataclass(frozen=True, slots=True)
class ProjectedEaglePassLoss:
    """Connect aligned EAGLE rows to PR6 without materializing pass logits."""

    projected_loss: ProjectedLoss
    output_weight: Tensor
    teacher_logits: Tensor
    valid_mask: Tensor
    token_chunk_size: int
    tp_group: torch.distributed.ProcessGroup | None

    def __post_init__(self) -> None:
        if self.teacher_logits.ndim != 3:
            raise ValueError("teacher_logits must have [batch, sequence, vocab] shape")
        if self.valid_mask.shape != self.teacher_logits.shape[:2]:
            raise ValueError("valid_mask must match teacher batch and sequence axes")
        if self.output_weight.ndim != 2:
            raise ValueError("output_weight must have [vocab, hidden] shape")
        if self.output_weight.shape[0] != self.teacher_logits.shape[-1]:
            raise ValueError("output_weight and teacher_logits vocab sizes must match")
        if self.token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")

    def __call__(
        self,
        *,
        hidden_states: Tensor,
        plan: EagleTTTAttentionPlan,
    ) -> DraftLossStats:
        batch_size, sequence_length = self.teacher_logits.shape[:2]
        expected_rows = sequence_length - plan.teacher_offset
        if hidden_states.shape != (
            expected_rows,
            batch_size,
            self.output_weight.shape[1],
        ):
            raise ValueError(
                "aligned hidden_states must have [sequence - offset, batch, hidden] "
                f"shape, got {hidden_states.shape}"
            )
        positions = torch.arange(
            plan.teacher_offset,
            sequence_length,
            device=hidden_states.device,
        )
        batch_offsets = (
            torch.arange(batch_size, device=hidden_states.device) * sequence_length
        )[:, None]
        teacher_row_indices = batch_offsets + positions[None, :]
        return self.projected_loss(
            student_hidden=hidden_states.transpose(0, 1),
            output_weight=self.output_weight,
            teacher_logits=self.teacher_logits,
            teacher_row_indices=teacher_row_indices,
            mask=self.valid_mask[:, plan.teacher_slice],
            token_chunk_size=self.token_chunk_size,
            tp_group=self.tp_group,
        )


class EagleTTTProvider:
    """Bounded method provider over a structured EAGLE pass runner.

    The provider deliberately does not call ModelOpt's dense multi-step mask
    helper. A model adapter must supply ``EaglePassRunner`` using the structured
    attention backend, while a PR6-style projected loss callable consumes each
    pass immediately. This keeps both square masks and retained pass logits out
    of the provider contract.
    """

    def __init__(
        self,
        *,
        max_passes: int,
        activation_budget_bytes: int,
        layer_count: int,
        kv_heads: int,
        head_dim: int,
        rope_dim: int,
    ) -> None:
        if max_passes < 1:
            raise ValueError(f"max_passes must be positive, got {max_passes}")
        if activation_budget_bytes < 1:
            raise ValueError(
                "activation_budget_bytes must be positive, "
                f"got {activation_budget_bytes}"
            )
        self.max_passes = max_passes
        self.activation_budget_bytes = activation_budget_bytes
        for name, value in {
            "layer_count": layer_count,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "rope_dim": rope_dim,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        self.layer_count = layer_count
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim

    def _storage_plan(
        self,
        *,
        target_trunk_states: Tensor,
        pass_count: int,
    ) -> EagleTTTStoragePlan:
        if target_trunk_states.ndim != 3:
            raise ValueError(
                "target_trunk_states must have [sequence, batch, hidden] shape, "
                f"got {target_trunk_states.shape}"
            )
        sequence_length, batch_size, hidden_size = target_trunk_states.shape
        return EagleTTTStoragePlan(
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            sequence_length=sequence_length,
            head_dim=self.head_dim,
            layer_count=self.layer_count,
            hidden_size=hidden_size,
            rope_dim=self.rope_dim,
            dtype=target_trunk_states.dtype,
            pass_count=pass_count,
            max_passes=self.max_passes,
            activation_budget_bytes=self.activation_budget_bytes,
        )

    def forward_loss_stats(
        self,
        *,
        pass_runner: EaglePassRunner,
        pass_loss: EaglePassLoss,
        target_trunk_states: Tensor,
        input_embeds: Tensor,
        pass_count: int,
        pass_weights: Tensor,
        sequence_layout: EagleTTTSequenceLayout | None = None,
        packed_seq_params: object | None = None,
    ) -> EagleTTTOutput:
        """Run bounded passes and consume each projected loss before the next pass."""
        storage = self._storage_plan(
            target_trunk_states=target_trunk_states,
            pass_count=pass_count,
        )
        if input_embeds.shape != target_trunk_states.shape:
            raise ValueError(
                "input_embeds must match target_trunk_states, "
                f"got {input_embeds.shape} and {target_trunk_states.shape}"
            )
        if (
            input_embeds.device != target_trunk_states.device
            or input_embeds.dtype != target_trunk_states.dtype
        ):
            raise ValueError(
                "input_embeds and target_trunk_states must share device and dtype"
            )
        if pass_weights.shape != (pass_count,):
            raise ValueError(
                f"pass_weights must have shape ({pass_count},), got {pass_weights.shape}"
            )
        if pass_weights.device != target_trunk_states.device:
            raise ValueError("pass_weights and target_trunk_states must share a device")
        if storage.sequence_length <= pass_count:
            raise ValueError(
                "sequence length must exceed pass_count so every pass has a target"
            )
        if sequence_layout is None:
            sequence_layout = EagleTTTSequenceLayout.unpacked(
                batch_size=storage.batch_size,
                sequence_length=storage.sequence_length,
                device=target_trunk_states.device,
            )
        elif sequence_layout.valid_tokens.shape != (
            storage.batch_size,
            storage.sequence_length,
        ):
            raise ValueError(
                "sequence_layout must match target batch and sequence axes, "
                f"got {sequence_layout.valid_tokens.shape}"
            )
        elif sequence_layout.valid_tokens.device != target_trunk_states.device:
            raise ValueError("sequence_layout and target states must share a device")

        current_hidden = target_trunk_states
        numerators: list[Tensor] = []
        counts: list[Tensor] = []
        plans: list[EagleTTTAttentionPlan] = []
        rope_positions = torch.arange(
            storage.sequence_length,
            dtype=torch.int64,
            device=current_hidden.device,
        )
        resource_ledger = EagleTTTResourceLedger(
            limit_bytes=self.activation_budget_bytes
        )
        resource_ledger.exclude((target_trunk_states, input_embeds, pass_weights))
        resource_ledger.track_owned(
            sequence_layout.valid_tokens,
            category="layout",
        )
        resource_ledger.track_owned(
            sequence_layout.document_ids,
            category="layout",
        )
        resource_ledger.track_owned(rope_positions, category="rope")
        try:
            pass_runner.begin(
                layout=sequence_layout,
                storage_plan=storage,
                excluded_tensors=(target_trunk_states, input_embeds, pass_weights),
                resource_ledger=resource_ledger,
                packed_seq_params=packed_seq_params,
            )
            with resource_ledger.saved_tensors():
                for pass_index in range(pass_count):
                    plan = EagleTTTAttentionPlan(
                        pass_index=pass_index,
                        pass_count=pass_count,
                        max_passes=self.max_passes,
                        sequence_length=storage.sequence_length,
                    )
                    hidden_states, next_hidden_states = pass_runner(
                        embeddings=input_embeds,
                        hidden_states=current_hidden,
                        plan=plan,
                        rope_positions=rope_positions,
                    )
                    self._validate_pass_output(
                        hidden_states=hidden_states,
                        next_hidden_states=next_hidden_states,
                        target_trunk_states=target_trunk_states,
                    )
                    pass_stats = pass_loss(
                        hidden_states=hidden_states[plan.student_slice],
                        plan=plan,
                    )
                    if pass_stats.numerators.shape != (
                        1,
                    ) or pass_stats.counts.shape != (1,):
                        raise ValueError(
                            "pass_loss must return exactly one additive bin per pass"
                        )
                    if not (
                        pass_stats.numerators.device
                        == pass_stats.counts.device
                        == target_trunk_states.device
                    ):
                        raise ValueError(
                            "pass loss statistics must share the model device"
                        )
                    numerators.append(pass_stats.numerators)
                    counts.append(pass_stats.counts)
                    plans.append(plan)

                    shifted_branch = torch.cat(
                        (
                            torch.zeros_like(next_hidden_states[:1]),
                            next_hidden_states[:-1],
                        ),
                        dim=0,
                    )
                    # Pinned ModelOpt EAGLE-3 captures the next-pass input with
                    # clone().detach(); enforce that contract for every runner.
                    current_hidden = shifted_branch.detach()
        finally:
            try:
                pass_runner.reset()
            finally:
                try:
                    resource_ledger.reset()
                finally:
                    reset_eagle_ttt_attention_state()

        return EagleTTTOutput(
            stats=DraftLossStats(
                numerators=torch.cat(numerators),
                counts=torch.cat(counts),
                weights=pass_weights.detach().to(dtype=torch.float32),
            ),
            final_branch_state=current_hidden,
            plans=tuple(plans),
        )

    @staticmethod
    def _validate_pass_output(
        *,
        hidden_states: Tensor,
        next_hidden_states: Tensor,
        target_trunk_states: Tensor,
    ) -> None:
        for name, tensor in (
            ("hidden", hidden_states),
            ("branch", next_hidden_states),
        ):
            if tensor.shape != target_trunk_states.shape:
                raise ValueError(
                    f"EAGLE pass {name} output must match target trunk shape, "
                    f"got {tensor.shape} and {target_trunk_states.shape}"
                )
            if (
                tensor.device != target_trunk_states.device
                or tensor.dtype != target_trunk_states.dtype
            ):
                raise ValueError(
                    f"EAGLE pass {name} output must share target device and dtype"
                )
