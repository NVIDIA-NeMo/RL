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

"""Pipeline-parallel batching, execution, and weight export for Automodel."""

import inspect
import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Optional

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

_DUMMY_SAMPLE_MASK_KEYS = ("sample_mask", "token_mask")


@dataclass
class ExecutionBatch:
    """Inputs for one pipeline schedule call.

    The schedule splits CPU ``sample_ids`` with the model inputs along dim 0,
    allowing the loss callback to identify each microbatch. Real samples
    precede padding samples.
    """

    microbatches: list[ProcessedMicrobatch]
    num_real_microbatches: int
    sample_ids: torch.Tensor
    model_inputs: dict[str, Any]
    model_context_factory: Callable[[], AbstractContextManager[Any]]
    cp_sharder: Optional[ContextParallelSharder]


def _forward_names_padding_mask(model: nn.Module) -> bool:
    """Check whether forward explicitly accepts padding_mask.

    Models with only **kwargs may forward it to layers that do not support it.
    """
    try:
        parameters = inspect.signature(type(model).forward).parameters
    except (TypeError, ValueError):
        return False
    return "padding_mask" in parameters


def _check_batch_major_inputs(model: nn.Module, model_batch: dict[str, Any]) -> None:
    """Reject inputs that the schedule cannot split along batch dimension 0."""
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
    """Pad and split data into fixed-size pipeline schedule calls.

    Match the maximum call count across DP ranks to keep FSDP collectives aligned.
    Padding rows have zero loss masks and, where supported, a router padding mask.
    Keep sequence lengths fixed across calls: resizing stages clears gradients.
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

    mark_padding_rows = _forward_names_padding_mask(model)
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
        if mark_padding_rows:
            # Exclude padding rows from MoE routing as well as the loss.
            input_ids = model_inputs["input_ids"]
            row_ids = torch.arange(
                start, start + samples_per_call, device=input_ids.device
            )
            model_inputs["padding_mask"] = (
                (row_ids >= num_samples).unsqueeze(1).expand_as(input_ids).contiguous()
            )
        if cp_size > 1:
            # Concatenation isolates loss-side data from in-place CP sharding.
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


PipelineCallMode = Literal["train", "eval", "logprobs"]


@dataclass
class PipelineCallContext:
    """Per-call state for ``PipelineOutputAdapter``, passed via ``loss_kwargs``."""

    execution_batch: ExecutionBatch
    mode: PipelineCallMode
    post_processing_fn: LossPostProcessor | LogprobsPostProcessor
    sampling_params: Optional[TrainingSamplingParams]
    sequence_dim: int
    global_valid_seqs: Optional[torch.Tensor]
    global_valid_toks: Optional[torch.Tensor]
    loss_scale: float
    # Microbatch index -> (loss, metrics) or token logprobs [B, S].
    results: dict[int, Any]

    def microbatch_index(self, target: torch.Tensor) -> int:
        """Map a schedule target slice to its microbatch index."""
        offset = int(target[0]) - int(self.execution_batch.sample_ids[0])
        return offset // target.numel()


def _free_logits_storage(logits: torch.Tensor) -> None:
    """Release logits retained in the forward-only schedule's cache.

    Otherwise the last stage holds every microbatch's fp32 logits until the
    next schedule call.
    """
    to_local_if_dtensor(logits).untyped_storage().resize_(0)


class PipelineOutputAdapter:
    """Compute last-stage losses or logprobs using the shared post-processors.

    Results are keyed by microbatch index so repeated callbacks overwrite
    rather than double-count results.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        context: PipelineCallContext,
    ) -> torch.Tensor:
        """Record results and return the scaled loss for backward in train mode."""
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
            # Padding must still participate in the schedule's backward pass.
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
    """Call the schedule directly to support eval, loss_kwargs, and return_outputs."""
    info = pipeline.info
    model_inputs = dict(execution_batch.model_inputs)
    input_ids = model_inputs.pop("input_ids")
    args = (input_ids,) if info.has_first_stage else ()
    target = execution_batch.sample_ids if info.has_last_stage else None
    run = info.schedule.eval if forward_only else info.schedule.step
    with execution_batch.model_context_factory():
        # CP may change the sequence length on context entry.
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
    """Accumulate gradients over one update's execution batches.

    Returns (loss, metrics) per microbatch, including padding, on the last stage;
    other stages return an empty list. The caller owns the optimizer step.
    """
    loss_scale = dp_size * cp_size / post_processing_fn.cp_gradient_fanout
    if not forward_only:
        # Defer MoE expert-gradient synchronization until the final backward.
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
    """Global stage indices used to namespace optimizer checkpoints, or None without PP."""
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
    """Broadcast the last stage's payload, converting tensors to host values."""
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
    """Broadcast the last stage's CUDA tensor with shape and dtype known on every rank."""
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
    """Release stage caches and communication buffers before rollout.

    Gradients must already be zero: stage re-initialization clears them.
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
    # Keep AutoPipeline's cached length consistent with the reset stage shapes.
    pipeline.update_seq_len(pipeline.pp_seq_len)


LocalParamsFn = Callable[[], Iterator[tuple[str, torch.Tensor]]]


@dataclass(frozen=True)
class _ManifestEntry:
    """One tensor of the full-model stream."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    owner_pp_rank: int


class PipelineWeightExporter:
    """Stream full-model weights on every PP rank for CUDA IPC and NCCL refit.

    Broadcast each stage's tensors in PP rank order, preserving Hugging Face
    names and refit dtypes from the local parameter stream.
    """

    def __init__(self, pipeline: AutoPipeline, *, local_params: LocalParamsFn):
        """Initialize with a factory for this stage's refit parameter stream."""
        self._local_params = local_params
        self._pp_group = pipeline.pp_mesh.get_group()
        self._manifest: Optional[list[_ManifestEntry]] = None

    def _build_manifest(self) -> list[_ManifestEntry]:
        """List every stage's tensors, in PP rank order."""
        local_entries = [
            (name, tuple(tensor.shape), tensor.dtype)
            for name, tensor in self._local_params()
        ]
        gathered: list[Any] = [None] * torch.distributed.get_world_size(self._pp_group)
        torch.distributed.all_gather_object(
            gathered, local_entries, group=self._pp_group
        )

        manifest = []
        producing_rank: dict[str, int] = {}
        for owner_pp_rank, entries in enumerate(gathered):
            for name, shape, dtype in entries:
                if name in producing_rank:
                    raise RuntimeError(
                        f"Weight {name!r} is produced by the pipeline stages of "
                        f"PP ranks {producing_rank[name]} and {owner_pp_rank}."
                    )
                producing_rank[name] = owner_pp_rank
                manifest.append(_ManifestEntry(name, shape, dtype, owner_pp_rank))
        return manifest

    def manifest(self) -> list[_ManifestEntry]:
        """The full model's tensors in stream order; built once, collectively."""
        if self._manifest is None:
            self._manifest = self._build_manifest()
        return self._manifest

    def refit_info(self) -> dict[str, tuple[torch.Size, torch.dtype]]:
        """Shape and dtype of every streamed tensor, as ``prepare_refit_info``."""
        return {
            entry.name: (torch.Size(entry.shape), entry.dtype)
            for entry in self.manifest()
        }

    def iter_params(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield full-model (name, tensor) pairs in manifest order.

        Every PP and DP rank must consume the whole stream to complete collectives.
        """
        manifest = self.manifest()
        pp_rank = torch.distributed.get_rank(self._pp_group)
        local_stream = self._local_params()
        for entry in manifest:
            if entry.owner_pp_rank == pp_rank:
                name, tensor = next(local_stream)
                if name != entry.name or tuple(tensor.shape) != entry.shape:
                    raise RuntimeError(
                        f"Pipeline weight stream produced {name!r} "
                        f"{tuple(tensor.shape)} where the manifest expects "
                        f"{entry.name!r} {entry.shape}."
                    )
            else:
                tensor = torch.empty(entry.shape, dtype=entry.dtype, device="cuda")
            torch.distributed.broadcast(
                tensor,
                src=torch.distributed.get_global_rank(
                    self._pp_group, entry.owner_pp_rank
                ),
                group=self._pp_group,
            )
            yield entry.name, tensor
