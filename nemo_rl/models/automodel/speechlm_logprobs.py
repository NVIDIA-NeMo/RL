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

"""Response-only projection and audio/text alignment for native SpeechLM."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def response_logprobs(
    hidden_states: torch.Tensor,
    native_targets: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    chunk_size: int = 256,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Project sampled response positions and restore canonical text-token layout.

    Native targets are shifted by Speech's audio-placeholder expansion. Their
    nonignored IDs must match the shifted text response IDs, in order, per row.
    Projection is checkpointed during training so backward recomputes a bounded
    chunk instead of retaining an audio/context-by-vocabulary logits tensor.
    The frozen head must already be a full local tensor (gathered once by its
    owning model when FSDP uses DTensor storage).
    """
    if chunk_size < 1 or temperature <= 0:
        raise ValueError("Projection chunk size and temperature must be positive")
    if hidden_states.shape[:2] != native_targets.shape:
        raise ValueError("Native targets and hidden-state positions differ")
    if (
        input_ids.shape != response_mask.shape
        or input_ids.shape[0] != native_targets.shape[0]
    ):
        raise ValueError("Text response mask and native batch dimensions differ")
    native_mask = native_targets != -100
    text_mask = response_mask[:, 1:].bool()
    text_targets = input_ids[:, 1:]
    for native_row, text_row, native_valid, text_valid in zip(
        native_targets, text_targets, native_mask, text_mask
    ):
        if not torch.equal(native_row[native_valid], text_row[text_valid]):
            raise ValueError(
                "Expanded audio targets do not match canonical response token IDs"
            )
    targets = native_targets[native_mask]
    if targets.numel() and (
        torch.any(targets < 0) or torch.any(targets >= weight.shape[0])
    ):
        raise ValueError(
            "Sampled response token is outside the native output vocabulary"
        )
    selected = hidden_states[native_mask]
    values = []

    def project(
        states: torch.Tensor,
        labels: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = (
            F.linear(states.to(head_weight.dtype), head_weight, head_bias).float()
            / temperature
        )
        return -F.cross_entropy(logits, labels, reduction="none")

    for start in range(0, len(targets), chunk_size):
        args = (
            selected[start : start + chunk_size],
            targets[start : start + chunk_size],
            weight,
            bias,
        )
        values.append(
            checkpoint(project, *args, use_reentrant=False)
            if torch.is_grad_enabled()
            else project(*args)
        )
    result = hidden_states.new_zeros(text_mask.shape, dtype=torch.float32)
    if values:
        result = result.masked_scatter(text_mask, torch.cat(values))
    else:
        # Keep a zero-gradient graph for all-empty response microbatches.
        result = result + hidden_states.sum() * 0
    return result
