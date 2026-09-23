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
"""Block-diffusion input preparation and same-position Megatron processors.

The ordinary Megatron forward owns model execution and temperature scaling.
Both processors share selected-token extraction; the loss processor delegates
normalization and microbatch scaling to NeMo-RL's LossPostProcessor.
"""

from dataclasses import replace
from typing import Any, Callable

import torch
from megatron.core import parallel_state

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import get_next_token_logprobs_from_logits
from nemo_rl.models.megatron.data import ProcessedMicrobatch
from nemo_rl.models.megatron.train import LogprobsPostProcessor, LossPostProcessor


def prepare_diffusion_microbatch(
    microbatch: ProcessedMicrobatch, *, mask_token_id: int
) -> ProcessedMicrobatch:
    """Prepare fixed-width [noisy | clean] inputs for the native block mask.

    The research config restricts this layout to unpacked PP=CP=1 training.
    Keep clean targets and loss metadata in data_dict, independent of inputs.
    """
    if microbatch.packed_seq_params is not None:
        raise ValueError("Diffusion processors require unpacked microbatches")
    data = microbatch.data_dict
    clean = data["input_ids"]
    noisy = clean.masked_fill(data["masked_indices"], mask_token_id)
    inputs = torch.cat([noisy, clean], dim=1)
    return replace(
        microbatch,
        input_ids=inputs,
        input_ids_cp_sharded=inputs,
        position_ids=data["position_ids"].repeat(1, 2),
        attention_mask=None,
        original_seq_length=inputs.shape[1],
    )


def selected_diffusion_logprobs(
    logits: torch.Tensor, data: BatchedDataDict[Any]
) -> torch.Tensor:
    """Extract selected same-position scores from temperature-scaled logits.

    Ignore the clean half. Scores retain the clean sequence shape, with zeros
    outside the selection, so training and reference scoring use one layout.
    """
    targets = data["target_ids"]
    selected = data["token_mask"].bool()
    noisy_logits = logits[:, : targets.shape[1]]
    if not selected.any():
        # Empty views must still participate in Megatron's backward schedule.
        return noisy_logits[..., 0].float() * 0
    values = get_next_token_logprobs_from_logits(
        input_ids=targets[selected].unsqueeze(0),
        next_token_logits=noisy_logits[selected].float().unsqueeze(0),
        vocab_parallel_rank=parallel_state.get_tensor_model_parallel_rank(),
        vocab_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        shift_labels=False,
    ).squeeze(0)
    return values.new_zeros(targets.shape).masked_scatter(selected, values)


def _prepared_logprobs(
    logits: torch.Tensor, data: BatchedDataDict[Any], **kwargs: Any
) -> tuple[dict[str, Any], BatchedDataDict[Any]]:
    return {
        "next_token_logprobs": logits,
        "shift_labels": False,
    }, data


class DiffusionLossPostProcessor(LossPostProcessor):
    """Use shared diffusion scores with upstream loss and microbatch scaling."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, prepare_fn=_prepared_logprobs, **kwargs)

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Any = None,
        global_valid_seqs: torch.Tensor | None = None,
        global_valid_toks: torch.Tensor | None = None,
    ) -> Callable:
        if packed_seq_params is not None:
            raise ValueError("Diffusion processors require unpacked microbatches")
        process_loss = super().__call__(
            data_dict=data_dict,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )

        def process(logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
            return process_loss(selected_diffusion_logprobs(logits, data_dict))

        return process


class DiffusionLogprobsPostProcessor(LogprobsPostProcessor):
    """Use the same diffusion scores for policy and frozen-reference scoring."""

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        input_ids: torch.Tensor,
        cu_seqlens_padded: torch.Tensor | None,
        original_seq_length: int,
    ) -> Callable:
        if self.use_fused_linear_logprobs or cu_seqlens_padded is not None:
            raise ValueError("Diffusion scoring requires unpacked vocabulary logits")

        def process(
            logits: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            values = selected_diffusion_logprobs(logits, data_dict)
            return values.new_zeros(()), {"logprobs": values}

        return process
