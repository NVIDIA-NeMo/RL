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

"""Turn a worker's local batch into pipeline schedule calls.

One schedule call (an execution batch) runs ``microbatches_per_step`` (M)
microbatches of ``micro_batch_size`` samples. The planning unit is whatever data
one call hands over -- a global batch or a logprob request -- so nothing here
assumes it equals ``gbs / dp`` or that DP ranks hold the same amount.
"""

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from nemo_automodel.components.distributed.context_parallel import (
    ContextParallelSharder,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    ProcessedMicrobatch,
    make_processed_microbatch_iterator,
)
from nemo_rl.models.automodel.train import (
    _build_model_batch,
    shard_model_batch_for_cp,
)

# Masks that make a padding sample contribute nothing to losses or metrics.
_DUMMY_SAMPLE_MASK_KEYS = ("sample_mask", "token_mask")


@dataclass
class ExecutionBatch:
    """The inputs of one pipeline schedule call.

    ``sample_ids`` is passed to the schedule as the loss target. The schedule
    splits it along dim 0 exactly like the model inputs, so the last stage can
    tell which microbatch each loss call belongs to no matter how often or in
    which order the schedule calls the loss.
    """

    # The call's microbatches in schedule order. The first
    # ``num_real_microbatches`` hold at least one real sample; the rest are
    # fully masked padding.
    microbatches: list[ProcessedMicrobatch]
    num_real_microbatches: int
    # Position of every row in the planned data, real rows first. Shape [E] on
    # CPU, where E = micro_batch_size * microbatches_per_step.
    sample_ids: torch.Tensor
    # Model-facing inputs for the whole call. Tensors are [E, ...], the
    # row-wise concatenation of each microbatch's model batch; the schedule
    # splits them back into the same microbatches. Under context parallelism
    # ``cp_sharder`` has prepared them, and entering ``model_context_factory()``
    # may shard them along the sequence in place.
    model_inputs: dict[str, Any]
    # The context the schedule call runs in: Automodel's context-parallel
    # attention context, or ``nullcontext`` without context parallelism.
    model_context_factory: Callable[[], AbstractContextManager[Any]]
    # Maps a microbatch's token tensors between the canonical sequence layout
    # and this rank's context-parallel shard; None without context
    # parallelism. One sharder serves every microbatch of the call because the
    # layout depends only on the sequence position.
    cp_sharder: Optional[ContextParallelSharder]


def _check_batch_major_inputs(model: nn.Module, model_batch: dict[str, Any]) -> None:
    """Reject model inputs whose batch dimension is not dim 0.

    Execution batches concatenate microbatch inputs along dim 0 and the schedule
    splits them back along dim 0. A model whose
    ``get_pipeline_kwargs_chunk_dims`` hook places the batch on another axis for
    these inputs (for example multimodal M-RoPE position IDs of shape
    ``[3, B, S]``) would be split wrongly, so it fails here. Text-only inputs of
    such models (the hook returns no exceptions) are batch-major and run.

    Args:
        model: A local model part.
        model_batch: One microbatch's model-facing inputs, or a whole call's
            once a context-parallel hook has prepared them.
    """
    chunk_dims_hook = getattr(model, "get_pipeline_kwargs_chunk_dims", None)
    if chunk_dims_hook is None:
        return
    non_batch_major = chunk_dims_hook(model_batch)
    if non_batch_major:
        raise NotImplementedError(
            f"{type(model).__name__} splits pipeline inputs "
            f"{sorted(non_batch_major)} along a non-batch dimension, which NeMo "
            "RL's pipeline execution batches do not support yet. Set "
            "policy.automodel_cfg.pipeline_parallel_size=1 for these inputs."
        )


def plan_execution_batches(
    data: BatchedDataDict[Any],
    *,
    model: nn.Module,
    micro_batch_size: int,
    microbatches_per_step: int,
    dp_group: torch.distributed.ProcessGroup,
    tokenizer: AutoTokenizer,
    cfg: dict[str, Any],
    allow_flash_attn_args: bool,
    device_mesh: DeviceMesh,
    cp_size: int,
    padding_token_id: int,
) -> list[ExecutionBatch]:
    """Split ``data`` into pipeline execution batches.

    The local data becomes ``ceil(size / (micro_batch_size * M))`` execution
    batches. The tail, and any extra execution batches needed to match the DP
    rank with the most, are filled with copies of the first sample whose
    ``sample_mask`` and ``token_mask`` are zero. Every DP rank therefore issues
    the same number of schedule calls and FSDP collectives. All execution
    batches keep ``data``'s sequence length, so one training update never
    re-initializes the pipeline stages (which would drop accumulated gradients).
    Under context parallelism each execution batch is sharded once for all of
    its microbatches, as Automodel's own pipeline trainer does.

    Args:
        data: The rows to plan, already on the training device.
        model: A local model part, used to build its model-facing batch.
        micro_batch_size: Samples per pipeline microbatch.
        microbatches_per_step: Microbatches per schedule call (M).
        dp_group: Group whose ranks must issue the same number of calls.
        tokenizer: Tokenizer used to process microbatches.
        cfg: Policy config used to process microbatches.
        allow_flash_attn_args: Whether the model accepts flash-attention kwargs.
        device_mesh: Device mesh with the ``cp`` axis.
        cp_size: Context-parallel size.
        padding_token_id: Token used to pad ``input_ids`` to the
            context-parallel granularity.

    Returns:
        The execution batches in order. Real rows keep their order, so row
        ``i`` of ``data`` has sample ID ``i``.
    """
    num_samples = data.size
    if num_samples == 0:
        raise ValueError(
            "Pipeline-parallel batch planning needs at least one sample per "
            "data-parallel rank to build padding samples from."
        )
    samples_per_call = micro_batch_size * microbatches_per_step
    num_calls = torch.tensor([math.ceil(num_samples / samples_per_call)], device="cuda")
    torch.distributed.all_reduce(
        num_calls, op=torch.distributed.ReduceOp.MAX, group=dp_group
    )
    num_calls = int(num_calls.item())

    num_padding = num_calls * samples_per_call - num_samples
    if num_padding > 0:
        padded = data.select_indices(list(range(num_samples)) + [0] * num_padding)
        for key in _DUMMY_SAMPLE_MASK_KEYS:
            if key in padded:
                padded[key][num_samples:] = 0
    else:
        padded = data

    execution_batches = []
    for call_idx in range(num_calls):
        start = call_idx * samples_per_call
        call_data = padded.slice(start, start + samples_per_call)
        microbatches = list(
            make_processed_microbatch_iterator(
                call_data.make_microbatch_iterator(micro_batch_size),
                tokenizer,
                cfg,
            )
        )
        model_batches = [
            _build_model_batch(
                model,
                mb.processed_inputs,
                is_reward_model=False,
                allow_flash_attn_args=allow_flash_attn_args,
                clone_model_tensors=False,
            )
            for mb in microbatches
        ]
        _check_batch_major_inputs(model, model_batches[0])
        model_inputs = {
            key: (
                torch.cat([batch[key] for batch in model_batches])
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in model_batches[0].items()
        }
        if cp_size > 1:
            # ``torch.cat`` gave the inputs their own storage, so sharding them
            # in place cannot reach the canonical microbatch data.
            cp_sharder, model_context_factory, model_inputs = shard_model_batch_for_cp(
                model,
                model_inputs,
                device_mesh=device_mesh,
                padding_token_id=padding_token_id,
                num_chunks=microbatches_per_step,
            )
            # The model's CP hook may have rewritten inputs.
            _check_batch_major_inputs(model, model_inputs)
        else:
            cp_sharder, model_context_factory = None, nullcontext
        num_real_samples = min(max(num_samples - start, 0), samples_per_call)
        execution_batches.append(
            ExecutionBatch(
                microbatches=microbatches,
                num_real_microbatches=math.ceil(num_real_samples / micro_batch_size),
                sample_ids=torch.arange(start, start + samples_per_call),
                model_inputs=model_inputs,
                model_context_factory=model_context_factory,
                cp_sharder=cp_sharder,
            )
        )
    return execution_batches
