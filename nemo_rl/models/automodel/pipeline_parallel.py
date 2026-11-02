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

"""Pipeline-parallel execution for the Automodel policy worker.

Automodel's ``AutoPipeline`` splits the model into stages and builds the
PyTorch schedule, which runs the forward and backward passes and the
inter-stage communication. This module turns a worker's local batch into
execution batches and feeds them through that schedule.

One schedule call (an execution batch) runs ``microbatches_per_step`` (M)
microbatches of ``micro_batch_size`` samples. The planning unit is whatever data
one call hands over -- a global batch or a logprob request -- so nothing here
assumes it equals ``gbs / dp`` or that DP ranks hold the same amount. The RL
loss runs on the last stage inside the schedule's loss callback,
``PipelineOutputAdapter``, which delegates to the same post-processors as the
non-pipelined path.
"""

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import torch
from nemo_automodel.components.distributed.context_parallel import (
    ContextParallelSharder,
)
from nemo_automodel.components.distributed.pipelining.autopipeline import (
    AutoPipeline,
)
from nemo_automodel.components.distributed.pipelining.functional import (
    reset_pp_stage_shapes,
)
from nemo_automodel.components.distributed.tensor_utils import to_local_if_dtensor
from nemo_automodel.components.training.utils import (
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    ProcessedMicrobatch,
    make_processed_microbatch_iterator,
)
from nemo_rl.models.automodel.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    _build_model_batch,
    post_process_logits,
    scale_metrics_for_global_batches,
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


# "train" returns the loss for the schedule's backward pass, "eval" records the
# loss and metrics without a backward pass, "logprobs" records token logprobs.
PipelineCallMode = Literal["train", "eval", "logprobs"]


@dataclass
class PipelineCallContext:
    """Per-call state for ``PipelineOutputAdapter``, passed via ``loss_kwargs``."""

    execution_batch: ExecutionBatch
    mode: PipelineCallMode
    post_processing_fn: LossPostProcessor | LogprobsPostProcessor
    sampling_params: Optional[TrainingSamplingParams]
    sequence_dim: int
    # Loss normalization recorded when the update was opened. The adapter only
    # reads them. None in logprobs mode.
    global_valid_seqs: Optional[torch.Tensor]
    global_valid_toks: Optional[torch.Tensor]
    # Factor on the returned training loss: dp_size * cp_size /
    # cp_gradient_fanout, the same backward scale as automodel_forward_backward.
    loss_scale: float
    # Results by microbatch index within the execution batch: (loss, metrics) in
    # train/eval mode, token logprobs [B, S] in logprobs mode.
    results: dict[int, Any]

    def microbatch_index(self, target: torch.Tensor) -> int:
        """Map the sample IDs the schedule passed as ``target`` to a microbatch.

        Args:
            target: One microbatch's slice of ``ExecutionBatch.sample_ids``.

        Returns:
            Index of that microbatch in ``execution_batch.microbatches``.
        """
        sample_ids = target.tolist()
        num_microbatches = len(self.execution_batch.microbatches)
        micro_batch_size = len(sample_ids)
        offset = sample_ids[0] - int(self.execution_batch.sample_ids[0])
        mb_index = offset // micro_batch_size
        if (
            offset % micro_batch_size != 0
            or not 0 <= mb_index < num_microbatches
            or sample_ids
            != list(range(sample_ids[0], sample_ids[0] + micro_batch_size))
        ):
            raise RuntimeError(
                f"Pipeline loss received sample IDs {sample_ids}, which are not one "
                f"microbatch of this schedule call (sample IDs "
                f"{self.execution_batch.sample_ids.tolist()})."
            )
        return mb_index


def _free_logits_storage(logits: torch.Tensor) -> None:
    """Free the last stage's logits once a forward-only loss call has used them.

    A forward-only schedule call keeps every microbatch's output in the stage's
    forward cache until the next call; without this the last stage would hold
    the fp32 logits of all microbatches at once.
    """
    to_local_if_dtensor(logits).untyped_storage().resize_(0)


class PipelineOutputAdapter:
    """The pipeline schedule's loss callback (``PipelineConfig.loss_fn``).

    The schedule calls it on the last stage once per microbatch as
    ``loss_fn(logits, target, **loss_kwargs)``, where ``target`` holds the
    microbatch's sample IDs and ``loss_kwargs`` is
    ``{"context": PipelineCallContext}``. It is stateless: everything it needs
    and everything it produces lives in the context. Results are written by
    microbatch index, so a repeated call for the same microbatch (for example
    during dynamic stage-metadata inference) overwrites instead of
    double-counting.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        context: PipelineCallContext,
    ) -> torch.Tensor:
        """Compute the RL loss or token logprobs of one microbatch.

        Args:
            logits: The last stage's output for the microbatch, [B, S, V].
            target: The microbatch's sample IDs, [B].
            context: State of the current schedule call.

        Returns:
            The scaled loss to backpropagate in train mode; a value the schedule
            only records otherwise.
        """
        mb_index = context.microbatch_index(target)
        result, metrics = post_process_logits(
            logits,
            context.post_processing_fn,
            context.execution_batch.microbatches[mb_index],
            cp_sharder=context.execution_batch.cp_sharder,
            global_valid_seqs=context.global_valid_seqs,
            global_valid_toks=context.global_valid_toks,
            sampling_params=context.sampling_params,
            sequence_dim=context.sequence_dim,
        )
        if context.mode == "logprobs":
            context.results[mb_index] = result
            _free_logits_storage(logits)
            return result.new_zeros(())

        if mb_index >= context.execution_batch.num_real_microbatches:
            # Padding microbatch: run the backward pass for the schedule's
            # sake, but contribute nothing.
            result = result * 0
        context.results[mb_index] = (result.detach(), metrics)
        if context.mode == "eval":
            _free_logits_storage(logits)
            return result.detach()
        return result * context.loss_scale


def _run_schedule_call(
    pipeline: AutoPipeline,
    execution_batch: ExecutionBatch,
    context: PipelineCallContext,
    *,
    forward_only: bool,
) -> None:
    """Run one execution batch through the pipeline schedule.

    Calls the PyTorch schedule directly instead of ``AutoPipeline.step`` so that
    ``return_outputs`` and ``loss_kwargs`` reach the schedule rather than the
    model inputs, and so that forward-only calls can use ``schedule.eval``.
    """
    info = pipeline.info
    model_inputs = dict(execution_batch.model_inputs)
    input_ids = model_inputs.pop("input_ids")
    args = (input_ids,) if info.has_first_stage else ()
    target = execution_batch.sample_ids if info.has_last_stage else None
    run = info.schedule.eval if forward_only else info.schedule.step
    with execution_batch.model_context_factory():
        # Context parallelism may shard the inputs in place on entering the
        # context, so the stage shapes follow the length inside it. Models
        # that shard inside their own forward keep full-length inputs here and
        # describe their local activations through their stage-metadata hook.
        pipeline.update_seq_len(input_ids.shape[1])
        run(
            *args,
            target=target,
            losses=None,
            return_outputs=False,
            loss_kwargs={"context": context},
            **model_inputs,
        )
    if forward_only:
        # Forward-only calls never pop the forward cache; release it now.
        for stage in info.stages:
            stage.clear_runtime_states()


def pipeline_forward_backward(
    pipeline: AutoPipeline,
    execution_batches: list[ExecutionBatch],
    post_processing_fn: LossPostProcessor,
    *,
    autocast_context_factory: Callable[[], AbstractContextManager[Any]],
    forward_only: bool,
    global_valid_seqs: torch.Tensor,
    global_valid_toks: torch.Tensor,
    sampling_params: Optional[TrainingSamplingParams],
    sequence_dim: int,
    dp_size: int,
    cp_size: int,
    num_global_batches: int,
) -> list[tuple[torch.Tensor, dict[str, Any]]]:
    """Pipeline-parallel counterpart of ``automodel_forward_backward``.

    Runs every execution batch of one optimizer update through the schedule;
    gradients accumulate across calls and no optimizer step happens here.

    Args:
        pipeline: The AutoPipeline that drives this rank's stage.
        execution_batches: The update's planned schedule calls.
        post_processing_fn: The loss post-processor.
        autocast_context_factory: Worker-owned precision context factory.
        forward_only: Compute losses and metrics without a backward pass.
        global_valid_seqs: Global valid sequence count for loss normalization.
        global_valid_toks: Global valid token count for loss normalization.
        sampling_params: Sampling parameters (top-k, top-p, temperature).
        sequence_dim: Sequence dimension.
        dp_size: Data parallel size.
        cp_size: Context parallel size.
        num_global_batches: Number of global batches (for metric scaling).

    Returns:
        ``(loss, metrics)`` per microbatch in order, including padding
        microbatches (zero loss, unscaled metrics), on the rank that owns the
        last stage; an empty list on other ranks.
    """
    loss_scale = dp_size * cp_size / post_processing_fn.cp_gradient_fanout
    if not forward_only:
        # Automodel defers MoE expert-gradient sync to the final backward of
        # the update, as its own pipeline trainer does.
        prepare_for_grad_accumulation(pipeline.parts, pp_enabled=True)
    results: list[tuple[torch.Tensor, dict[str, Any]]] = []
    for call_idx, execution_batch in enumerate(execution_batches):
        if not forward_only and call_idx == len(execution_batches) - 1:
            prepare_for_final_backward(pipeline.parts, pp_enabled=True)
        context = PipelineCallContext(
            execution_batch=execution_batch,
            mode="eval" if forward_only else "train",
            post_processing_fn=post_processing_fn,
            sampling_params=sampling_params,
            sequence_dim=sequence_dim,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            loss_scale=loss_scale,
            results={},
        )
        with autocast_context_factory():
            _run_schedule_call(
                pipeline, execution_batch, context, forward_only=forward_only
            )
        if not forward_only and call_idx == 0:
            prepare_after_first_microbatch()

        if not pipeline.info.has_last_stage:
            continue
        for mb_index in range(len(execution_batch.microbatches)):
            loss, metrics = context.results[mb_index]
            if mb_index < execution_batch.num_real_microbatches:
                scale_metrics_for_global_batches(metrics, num_global_batches)
            results.append((loss, metrics))
    return results


def pipeline_forward_logprobs(
    pipeline: AutoPipeline,
    execution_batches: list[ExecutionBatch],
    post_processing_fn: LogprobsPostProcessor,
    *,
    autocast_context_factory: Callable[[], AbstractContextManager[Any]],
    sampling_params: Optional[TrainingSamplingParams],
    sequence_dim: int,
) -> list[torch.Tensor]:
    """Compute token logprobs through the pipeline schedule, forward only.

    Returns:
        Token logprobs [B, S] of every microbatch holding real samples, in order,
        on the rank that owns the last stage; an empty list on other ranks.
    """
    results: list[torch.Tensor] = []
    for execution_batch in execution_batches:
        context = PipelineCallContext(
            execution_batch=execution_batch,
            mode="logprobs",
            post_processing_fn=post_processing_fn,
            sampling_params=sampling_params,
            sequence_dim=sequence_dim,
            global_valid_seqs=None,
            global_valid_toks=None,
            loss_scale=1.0,
            results={},
        )
        with autocast_context_factory():
            _run_schedule_call(pipeline, execution_batch, context, forward_only=True)
        if pipeline.info.has_last_stage:
            results.extend(
                context.results[mb_index]
                for mb_index in range(execution_batch.num_real_microbatches)
            )
    return results


def pipeline_optimizer_part_ids(
    pipeline: Optional[AutoPipeline],
) -> Optional[list[int]]:
    """Global stage index of this rank's pipeline stage, or None without PP.

    Optimizer checkpoints namespace their state by it, as Automodel's own
    pipeline trainer does, so stages cannot write overlapping keys.
    """
    if pipeline is None:
        return None
    return [stage.stage_index for stage in pipeline.info.stages]


def _last_stage_global_rank(pipeline: AutoPipeline) -> int:
    """Global rank of the PP rank that owns the last pipeline stage."""
    pp_group = pipeline.pp_mesh.get_group()
    return torch.distributed.get_global_rank(
        pp_group, torch.distributed.get_world_size(pp_group) - 1
    )


def _materialize_for_object_broadcast(value: Any) -> Any:
    """Move tensors in a metrics payload to host values before pickling."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu()
    if isinstance(value, dict):
        return {k: _materialize_for_object_broadcast(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_materialize_for_object_broadcast(v) for v in value)
    return value


def broadcast_object_from_last_stage(pipeline: AutoPipeline, obj: Any) -> Any:
    """Broadcast a picklable payload from the last pipeline stage to the PP group.

    Tensors are converted to host values first, on every rank, so all PP ranks
    return identical payloads.

    Args:
        pipeline: The AutoPipeline that drives this rank's stage.
        obj: The payload on the last stage; ignored on other ranks.

    Returns:
        The last stage's payload with tensors materialized on the host.
    """
    payload = [
        _materialize_for_object_broadcast(obj) if pipeline.info.has_last_stage else None
    ]
    torch.distributed.broadcast_object_list(
        payload,
        src=_last_stage_global_rank(pipeline),
        group=pipeline.pp_mesh.get_group(),
    )
    return payload[0]


def broadcast_tensor_from_last_stage(
    pipeline: AutoPipeline,
    tensor: Optional[torch.Tensor],
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a CUDA tensor of known shape from the last stage to the PP group.

    Args:
        pipeline: The AutoPipeline that drives this rank's stage.
        tensor: The tensor on the last stage; ignored on other ranks.
        shape: The tensor's shape, known on every rank.
        dtype: The tensor's dtype.

    Returns:
        The last stage's tensor on every PP rank.
    """
    if pipeline.info.has_last_stage:
        buffer = tensor.to(dtype).contiguous()
        if tuple(buffer.shape) != shape:
            raise RuntimeError(
                f"Broadcast tensor has shape {tuple(buffer.shape)}, expected {shape}."
            )
    else:
        buffer = torch.empty(shape, dtype=dtype, device="cuda")
    torch.distributed.broadcast(
        buffer,
        src=_last_stage_global_rank(pipeline),
        group=pipeline.pp_mesh.get_group(),
    )
    return buffer


def release_pipeline_runtime_state(pipeline: AutoPipeline) -> None:
    """Free pipeline buffers before the GPU is handed to the rollout engine.

    Drops each stage's forward cache and its per-microbatch activation and
    gradient receive buffers, and marks the stages uninitialized so the next
    schedule call reallocates them. Gradients must already be zero: the
    re-initialization clears them.
    """
    info = pipeline.info
    for stage in info.stages:
        stage.clear_runtime_states()
    reset_pp_stage_shapes(
        info.schedule,
        info.stages,
        pipeline.parts[0].config,
        pipeline.pp_microbatch_size,
        pipeline.pp_seq_len,
        tensor_dtype=pipeline.dtype,
    )
    # The stages now describe pp_seq_len; keep AutoPipeline's record in sync so
    # the next update_seq_len() re-describes them for the next batch's length.
    pipeline.update_seq_len(pipeline.pp_seq_len)
