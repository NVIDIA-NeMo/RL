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

import math
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

import torch
import torch.distributed
from megatron.core.transformer.multi_token_prediction import roll_tensor

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.utils import masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import DistributedCrossEntropy

if TYPE_CHECKING:
    from nemo_rl.models.draft.hidden_capture import CapturedStates

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class SequencePackingLossWrapper:
    def __init__(
        self,
        loss_fn: LossFunction,
        prepare_fn: Callable[Any, Any],
        cu_seqlens_q: Tensor,
        cu_seqlens_q_padded: Optional[Tensor] = None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """Wrap a loss function to handle sequence packing.

        Args:
            loss_fn: Loss function.
            prepare_fn: Prepare function.
            cu_seqlens_q: Unpadded cu seqlens q.
            cu_seqlens_q_padded: Padded cu seqlens q.
            vocab_parallel_rank: Vocab parallel rank.
            vocab_parallel_group: Vocab parallel group.
            context_parallel_group: Context parallel group.

            vocab_parallel_rank, vocab_parallel_group, context_parallel_group are only used for megatron policy worker.

        Returns:
            Sequence packing loss wrapper.
        """
        self.loss_fn = loss_fn
        self.prepare_fn = prepare_fn
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_q_padded = cu_seqlens_q_padded
        self.vocab_parallel_rank = vocab_parallel_rank
        self.vocab_parallel_group = vocab_parallel_group
        self.context_parallel_group = context_parallel_group

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor | None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Wraps a loss function to handle sequence packing by doing one sequence at a time to avoid excessive padding."""
        unpadded_cu_seqlens = self.cu_seqlens_q
        unpadded_seq_lengths = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
        if self.cu_seqlens_q_padded is not None:
            padded_cu_seqlens = self.cu_seqlens_q_padded
            padded_seq_lengths = (
                self.cu_seqlens_q_padded[1:] - self.cu_seqlens_q_padded[:-1]
            )
        else:
            padded_cu_seqlens = unpadded_cu_seqlens
            padded_seq_lengths = unpadded_seq_lengths
        seq_starts = padded_cu_seqlens[:-1]
        seq_ends = padded_cu_seqlens[1:]

        loss_accum = 0
        metrics_accum = {}
        for seq_idx in range(len(seq_starts)):
            seq_start = seq_starts[seq_idx].item()
            seq_end = seq_ends[seq_idx].item()

            # get sequence and unpad all 'data' tensors. The data dict is a BatchedDataDict of unpacked tensors
            seq_data = data.slice(seq_idx, seq_idx + 1)
            unpadded_seq_data = {}
            for k, v in seq_data.items():
                if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[1] > 1:
                    unpadded_seq_data[k] = v[:, : unpadded_seq_lengths[seq_idx]]
                else:
                    unpadded_seq_data[k] = v

            # get next_token_logits
            cp_size = (
                1
                if self.context_parallel_group is None
                else torch.distributed.get_world_size(self.context_parallel_group)
            )
            logit_start = seq_start // cp_size
            logit_end = (seq_start + padded_seq_lengths[seq_idx]) // cp_size
            logit_length = logit_end - logit_start
            next_token_logits_slice = next_token_logits.narrow(
                1, logit_start, logit_length
            )

            # prepare data for loss function
            loss_input = self.prepare_fn(
                logits=next_token_logits_slice,
                data=unpadded_seq_data,
                loss_fn=self.loss_fn,
                vocab_parallel_rank=self.vocab_parallel_rank,
                vocab_parallel_group=self.vocab_parallel_group,
                context_parallel_group=self.context_parallel_group,
            )

            # call loss function
            loss, metrics = self.loss_fn(
                data=unpadded_seq_data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                **loss_input,
            )

            # aggregate loss and metrics
            loss_accum += loss
            for k, v in metrics.items():
                if k not in metrics_accum:
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        metrics_accum[k] = float("inf")
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        metrics_accum[k] = float("-inf")
                    else:
                        metrics_accum[k] = 0

                val = v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v

                # Skip inf/-inf sentinel values (from sequences with no valid tokens)
                if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                    if not math.isinf(val):
                        metrics_accum[k] = min(metrics_accum[k], val)
                elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                    if not math.isinf(val):
                        metrics_accum[k] = max(metrics_accum[k], val)
                else:
                    metrics_accum[k] += val

        return loss_accum, metrics_accum


class DraftLossWrapper:
    """Combine policy loss with draft soft cross-entropy loss."""

    def __init__(
        self,
        loss_fn: Callable[..., tuple[torch.Tensor, dict[str, Any]]],
        draft_model: torch.nn.Module,
        lm_head_weight: torch.Tensor,
        captured_states: "CapturedStates | None",
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[Any] = None,
        loss_weight: float = 1.0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.loss_fn = loss_fn
        self.draft_model = draft_model
        self.lm_head_weight = lm_head_weight
        self.captured_states = captured_states
        self.attention_mask = attention_mask
        self.packed_seq_params = packed_seq_params
        self.loss_weight = loss_weight
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_q_padded = (
            cu_seqlens_q_padded if cu_seqlens_q_padded is not None else cu_seqlens_q
        )
        self.vocab_parallel_group = vocab_parallel_group
        self.context_parallel_group = context_parallel_group

        if lm_head_weight.requires_grad:
            raise ValueError(
                "DraftLossWrapper requires a detached LM head weight tensor."
            )

    def _compute_draft_logits(self) -> torch.Tensor:
        if (
            self.captured_states is None
            or self.captured_states.hidden_states is None
            or self.captured_states.inputs_embeds is None
        ):
            raise ValueError(
                "DraftLossWrapper requires captured hidden states and input embeddings."
            )

        shifted_input_embeds, _ = roll_tensor(
            self.captured_states.inputs_embeds,
            shifts=-1,
            dims=0,
            cp_group=self.context_parallel_group,
            packed_seq_params=self.packed_seq_params,
        )
        return self.draft_model(
            hidden_states=self.captured_states.hidden_states,
            input_embeds=shifted_input_embeds,
            attention_mask=self.attention_mask,
            lm_head_weight=self.lm_head_weight,
        )

    def _build_packed_mask(
        self,
        data: BatchedDataDict[Any],
        kl_seq_len: int,
        context_parallel_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        assert self.cu_seqlens_q is not None
        assert self.cu_seqlens_q_padded is not None

        cp_size = (
            1
            if context_parallel_group is None
            else torch.distributed.get_world_size(context_parallel_group)
        )
        device = data["token_mask"].device
        num_sequences = len(self.cu_seqlens_q) - 1

        packed_length = kl_seq_len + 1
        packed_mask = torch.zeros(1, packed_length, device=device)

        for sequence_idx in range(num_sequences):
            padded_start = self.cu_seqlens_q_padded[sequence_idx].item() // cp_size
            unpadded_length = (
                self.cu_seqlens_q[sequence_idx + 1].item()
                - self.cu_seqlens_q[sequence_idx].item()
            ) // cp_size
            seq_mask = data["token_mask"][sequence_idx, :unpadded_length].to(
                packed_mask.dtype
            )
            if "sample_mask" in data:
                seq_mask = seq_mask * data["sample_mask"][sequence_idx]
            packed_mask[0, padded_start : padded_start + unpadded_length] = seq_mask

        mask = packed_mask[:, 1:]

        # Remove sequence-boundary transitions in packed layout.
        for sequence_idx in range(1, num_sequences + 1):
            boundary = self.cu_seqlens_q_padded[sequence_idx].item() // cp_size
            if 0 < boundary <= kl_seq_len:
                mask[0, boundary - 1] = 0

        return mask

    def _build_mask(
        self,
        data: BatchedDataDict[Any],
        kl_seq_len: int,
    ) -> torch.Tensor:
        if self.cu_seqlens_q is not None:
            return self._build_packed_mask(
                data=data,
                kl_seq_len=kl_seq_len,
                context_parallel_group=self.context_parallel_group,
            )

        token_mask = roll_tensor(data["token_mask"], shifts=-1, dims=1)[0][
            :, :kl_seq_len
        ]
        if "sample_mask" in data:
            return token_mask * data["sample_mask"].unsqueeze(-1)
        return token_mask

    def _compute_per_token_draft_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.vocab_parallel_group is not None:
            # Soft cross entropy matches the forward-KL student gradient.
            return DistributedCrossEntropy.apply(
                student_logits,
                teacher_logits,
                self.vocab_parallel_group,
                False,
            )

        teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
        student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
        return -(teacher_probs * student_log_probs).sum(dim=-1)

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if global_valid_toks is None:
            raise ValueError("global_valid_toks is required for DraftLossWrapper.")

        policy_loss, metrics = self.loss_fn(
            next_token_logits,
            data,
            global_valid_seqs,
            global_valid_toks,
            **kwargs,
        )
        draft_logits = self._compute_draft_logits()

        teacher_logits = next_token_logits.detach().to(torch.float32)
        student_logits = draft_logits.to(torch.float32)

        # Align teacher logits with the draft model's next-token prediction target.
        teacher_logits, _ = roll_tensor(
            teacher_logits,
            shifts=-1,
            dims=1,
            cp_group=self.context_parallel_group,
            packed_seq_params=self.packed_seq_params,
        )

        kl_seq_len = teacher_logits.shape[1]
        student_logits = student_logits[:, :kl_seq_len, :]
        mask = self._build_mask(data, kl_seq_len)

        effective_seq_len = min(
            int(teacher_logits.shape[1]),
            int(mask.shape[1]),
        )
        teacher_logits = teacher_logits[:, :effective_seq_len, :]
        student_logits = student_logits[:, :effective_seq_len, :]
        mask = mask[:, :effective_seq_len]

        per_token_draft_loss = self._compute_per_token_draft_loss(
            teacher_logits,
            student_logits,
        )
        draft_loss = masked_mean(
            per_token_draft_loss,
            mask,
            global_normalization_factor=global_valid_toks,
        )
        combined_loss = policy_loss + self.loss_weight * draft_loss
        metrics["draft_loss"] = float(draft_loss.detach().item())
        return combined_loss, metrics


def wrap_loss_fn_with_input_preparation(
    next_token_logits: Tensor,
    data: BatchedDataDict[Any],
    global_valid_seqs: Tensor | None,
    global_valid_toks: Tensor | None,
    loss_fn: LossFunction,
    prepare_fn: Callable[Any, Any],
    vocab_parallel_rank: Optional[int] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Wraps a loss function to handle input preparation for megatron policy worker."""
    # prepare loss input
    loss_input = prepare_fn(
        logits=next_token_logits,
        data=data,
        loss_fn=loss_fn,
        vocab_parallel_rank=vocab_parallel_rank,
        vocab_parallel_group=vocab_parallel_group,
        context_parallel_group=context_parallel_group,
    )

    # call loss function
    loss, loss_metrics = loss_fn(
        data=data,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
        **loss_input,
    )

    return loss, loss_metrics
