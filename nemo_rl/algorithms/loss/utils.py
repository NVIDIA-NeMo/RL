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
from typing import Any, Callable, Optional, TypeVar

import torch
import torch.distributed

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

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
            loss_fn_args = self.prepare_fn(
                logits=next_token_logits_slice,
                data_dict=unpadded_seq_data,
                vocab_parallel_rank=self.vocab_parallel_rank,
                vocab_parallel_group=self.vocab_parallel_group,
                context_parallel_group=self.context_parallel_group,
            )

            loss, metrics = self.loss_fn(
                *loss_fn_args,
                unpadded_seq_data,
                global_valid_seqs,
                global_valid_toks,
            )
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
