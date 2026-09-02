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
"""Differentiable SDMC training loop for GDPO."""

from contextlib import nullcontext
from typing import Any, Callable, Iterator, Optional

import torch

from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
from nemo_rl.models.automodel.train import LossPostProcessor


def gdpo_forward_backward(
    *,
    data_iterator: Iterator[ProcessedMicrobatch],
    post_processing_fn: LossPostProcessor,
    elbo_scorer: Callable[[ProcessedMicrobatch], torch.Tensor],
    forward_only: bool,
    global_valid_seqs: torch.Tensor,
    global_valid_toks: torch.Tensor,
    sequence_dim: int,
    dp_size: int,
    cp_size: int,
    num_global_batches: int,
    train_context_fn: Optional[Callable[[ProcessedInputs], Any]],
    num_valid_microbatches: Optional[int],
    on_microbatch_start: Optional[Callable[[int], None]],
) -> list[tuple[Any, dict[str, Any]]]:
    """Accumulate the SDMC ELBO and backpropagate the GDPO loss.

    Unlike an autoregressive training step, one GDPO likelihood evaluation
    performs several model forwards over corrupted views. The scorer owns those
    forwards and returns position-aligned ELBO contributions with autograd
    history spanning every quadrature point.
    """
    results = []
    for mb_idx, processed_mb in enumerate(data_iterator):
        if on_microbatch_start is not None:
            on_microbatch_start(mb_idx)

        processed_inputs = processed_mb.processed_inputs
        ctx = (
            train_context_fn(processed_inputs)
            if train_context_fn is not None
            else nullcontext()
        )
        with ctx:
            elbo_logprobs = elbo_scorer(processed_mb)
            result, metrics = post_processing_fn(
                logits=elbo_logprobs,
                data_dict=processed_mb.data_dict,
                processed_inputs=processed_inputs,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                sequence_dim=sequence_dim,
            )

            is_dummy = (
                num_valid_microbatches is not None and mb_idx >= num_valid_microbatches
            )
            if is_dummy:
                result = result * 0
            else:
                for key in metrics:
                    if "_min" not in key and "_max" not in key:
                        metrics[key] /= num_global_batches

            if not forward_only:
                loss = result * dp_size * cp_size
                loss.backward()

        results.append((result, metrics))

    return results
