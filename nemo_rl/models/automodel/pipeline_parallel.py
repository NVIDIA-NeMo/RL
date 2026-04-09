# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Pipeline parallelism utilities for automodel-based training.

Contains PP-specific classes and functions:
- Broadcast helpers for PP stage communication
- PPLossAdapter: stateful loss wrapper for PP schedules
- PPLogprobsCapturer / PPTopkCapturer: logit capture for eval
- pp_forward_backward: PP schedule step/eval wrapper
- reset_pp_stage_shapes_for_thd: THD stage shape management
- prepare_pp_seqpack_batch: pack sequences for PP training step
- pad_batch_for_pp: pad batch to pp_batch_size for PP schedule
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nemo_rl.models.automodel.data import THDBatch

import torch
from torch import nn

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss import SequencePackingLossWrapper, prepare_loss_input
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


# ---------------------------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------------------------


def broadcast_tensors_from_last_pp_stage(
    tensors: dict[str, Optional[torch.Tensor]],
    pp_mesh: Any,
    has_last_stage: bool,
) -> dict[str, torch.Tensor]:
    """Broadcast tensors from last PP stage to all stages.

    Adapted from nemo_rl/models/megatron/pipeline_parallel.py for device-mesh
    based PP (no Megatron parallel state).

    Args:
        tensors: Dict mapping names to tensors. On last stage, tensors are
            populated; on other stages they may be None.
        pp_mesh: Pipeline parallel DeviceMesh.
        has_last_stage: Whether this rank owns the last pipeline stage.

    Returns:
        Dict with the same keys, all tensors populated on every rank.
    """
    pp_group = pp_mesh.get_group()
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    last_global_rank = pp_ranks[-1]

    result = {}
    for name, tensor in tensors.items():
        # Broadcast metadata (shape, dtype)
        if has_last_stage:
            assert tensor is not None, f"Last stage must provide tensor '{name}'"
            meta = [list(tensor.shape), str(tensor.dtype)]
        else:
            meta = [None, None]
        meta_list = [meta]
        torch.distributed.broadcast_object_list(
            meta_list, src=last_global_rank, group=pp_group
        )
        shape, dtype_str = meta_list[0]

        if not has_last_stage:
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            tensor = torch.empty(shape, dtype=dtype, device="cuda")

        torch.distributed.broadcast(tensor, src=last_global_rank, group=pp_group)
        result[name] = tensor

    return result


def broadcast_loss_metrics_from_last_pp_stage(
    metrics: Optional[list[dict[str, Any]]],
    pp_mesh: Any,
    has_last_stage: bool,
) -> list[dict[str, Any]]:
    """Broadcast loss metrics from last PP stage to all stages.

    Args:
        metrics: List of metric dicts (populated on last stage, None elsewhere).
        pp_mesh: Pipeline parallel DeviceMesh.
        has_last_stage: Whether this rank owns the last pipeline stage.

    Returns:
        List of metric dicts on all ranks.
    """
    pp_group = pp_mesh.get_group()
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    last_global_rank = pp_ranks[-1]

    obj = [metrics]
    torch.distributed.broadcast_object_list(obj, src=last_global_rank, group=pp_group)
    return obj[0]


# ---------------------------------------------------------------------------
# Stateful loss adapters
# ---------------------------------------------------------------------------


class PPLossAdapter:
    """Stateful loss adapter for pipeline parallel schedules.

    Bridges NeMo RL's LossFunction interface with the PP schedule's
    ``loss_fn(output, target)`` contract by pre-chunking RL data across
    microbatches.

    The PP schedule calls ``__call__(output, target)`` once per microbatch.
    This adapter indexes into pre-chunked RL tensors to compute the loss for
    each microbatch.

    **Critical**: The loss is scaled by ``dp_size * cp_size`` to cancel FSDP's
    automatic gradient averaging, matching the non-PP path.
    """

    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: Any,
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        dp_size: int,
        enable_seq_packing: bool = False,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ):
        self._loss_fn = loss_fn
        self._cfg = cfg
        self._device_mesh = device_mesh
        self._cp_mesh = cp_mesh
        self._tp_mesh = tp_mesh
        self._cp_size = cp_size
        self._dp_size = dp_size
        self._enable_seq_packing = enable_seq_packing
        self._sampling_params = sampling_params

        self._microbatches: list[dict[str, Any]] = []
        self._cu_seqlens_list: list[Optional[torch.Tensor]] = []
        self._cu_seqlens_padded_list: list[Optional[torch.Tensor]] = []
        self._call_idx: int = 0
        self._all_metrics: list[dict[str, Any]] = []
        self._global_valid_seqs: Optional[torch.Tensor] = None
        self._global_valid_toks: Optional[torch.Tensor] = None
        self._num_global_batches: int = 1
        self._context_parallel_group: Any = None

    def set_microbatches(
        self,
        data_dict: Any,
        n_microbatches: int,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        num_global_batches: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_padded: Optional[torch.Tensor] = None,
        context_parallel_group: Any = None,
    ) -> None:
        """Pre-chunk RL tensors along batch dim into n_microbatches."""
        self._call_idx = 0
        self._all_metrics = []
        self._global_valid_seqs = global_valid_seqs
        self._global_valid_toks = global_valid_toks
        self._num_global_batches = num_global_batches

        self._microbatches = []
        for i in range(n_microbatches):
            mb: dict[str, Any] = {}
            for key in data_dict:
                val = data_dict[key]
                if torch.is_tensor(val) and val.shape[0] > 0:
                    chunks = torch.tensor_split(val, n_microbatches, dim=0)
                    mb[key] = chunks[i]
                else:
                    mb[key] = val
            self._microbatches.append(mb)

        # Set per-microbatch cu_seqlens for sequence packing loss.
        self._context_parallel_group = context_parallel_group

        def _split_cu_seqlens(cu):
            if cu is None:
                return [None] * n_microbatches
            if isinstance(cu, list):
                return cu
            if cu.ndim == 1:
                return [cu] * n_microbatches
            return [cu[i] for i in range(n_microbatches)]

        self._cu_seqlens_list = _split_cu_seqlens(cu_seqlens)
        self._cu_seqlens_padded_list = _split_cu_seqlens(cu_seqlens_padded)

    def __call__(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        """Called by PP schedule per microbatch."""
        logits = getattr(output, "logits", output)
        # THD format: [total_tokens, vocab] → [1, total_tokens, vocab]
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        mb_data = self._microbatches[self._call_idx]
        cu_seqlens = self._cu_seqlens_list[self._call_idx] if self._cu_seqlens_list else None
        cu_seqlens_padded = self._cu_seqlens_padded_list[self._call_idx] if self._cu_seqlens_padded_list else cu_seqlens
        self._call_idx += 1

        if cu_seqlens is not None:
            if not isinstance(mb_data, BatchedDataDict):
                mb_data = BatchedDataDict(mb_data)

            prepare_loss_input_wrapped = partial(
                prepare_loss_input, sampling_params=self._sampling_params
            )
            loss_fn_wrapped = SequencePackingLossWrapper(
                loss_fn=self._loss_fn,
                prepare_fn=prepare_loss_input_wrapped,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens,
                context_parallel_group=self._context_parallel_group,
            )
            loss, loss_metrics = loss_fn_wrapped(
                logits,
                mb_data,
                self._global_valid_seqs,
                self._global_valid_toks,
            )
        else:
            # Standard (non-packed) loss path
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            curr_logprobs = (
                log_probs[:, :-1]
                .gather(dim=-1, index=target[:, 1:].unsqueeze(-1).clamp(min=0))
                .squeeze(-1)
            )

            if self._loss_fn.input_type == LossInputType.LOGPROB:
                loss_input = {"next_token_logprobs": curr_logprobs}
            else:
                loss_input = {"logits": logits}

            loss, loss_metrics = self._loss_fn(
                data=mb_data,
                global_valid_seqs=self._global_valid_seqs,
                global_valid_toks=self._global_valid_toks,
                **loss_input,
            )

        # Scale metrics for aggregation
        for k in loss_metrics:
            if "_min" not in k and "_max" not in k:
                loss_metrics[k] /= self._num_global_batches

        self._all_metrics.append(loss_metrics)

        # Scale loss to cancel FSDP's automatic gradient averaging
        return loss * self._dp_size * self._cp_size

    def reset(self) -> None:
        """Reset state for the next forward-backward call."""
        self._call_idx = 0
        self._all_metrics = []
        self._cu_seqlens_list = []
        self._cu_seqlens_padded_list = []


# ---------------------------------------------------------------------------
# Logit/logprob capturers for eval
# ---------------------------------------------------------------------------


class PPLogprobsCapturer:
    """Pseudo-loss that captures logits from each PP microbatch.

    Used with ``schedule.eval()`` to collect logits on the last pipeline
    stage for logprob computation.
    """

    def __init__(self):
        self.captured_logits: list[torch.Tensor] = []

    def __call__(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        logits = getattr(output, "logits", output)
        self.captured_logits.append(logits.detach())
        return torch.tensor(0.0, device="cuda")

    def reset(self) -> None:
        self.captured_logits = []


class PPTopkCapturer:
    """Pseudo-loss that captures logits for top-k computation on last stage."""

    def __init__(self):
        self.captured_logits: list[torch.Tensor] = []

    def __call__(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        logits = getattr(output, "logits", output)
        self.captured_logits.append(logits.detach())
        return torch.tensor(0.0, device="cuda")

    def reset(self) -> None:
        self.captured_logits = []


# ---------------------------------------------------------------------------
# PP forward/backward
# ---------------------------------------------------------------------------


def pp_forward_backward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_adapter: PPLossAdapter,
    *,
    forward_only: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Execute forward (and optionally backward) using the PP schedule.

    The PP schedule internally handles microbatch splitting, stage-to-stage
    communication, and gradient accumulation.

    Args:
        model: AutoPipeline model with .info and .parts attributes.
        batch: Dict with at least ``input_ids`` and ``labels``.
            May also contain ``flash_attn_kwargs``, ``position_ids``, etc.
            for sequence packing support. Extra keys are passed as kwargs
            to schedule.step/eval following automodel's train_ft.py pattern.
        loss_adapter: PPLossAdapter (already configured via set_microbatches).
        forward_only: If True, use schedule.eval() instead of schedule.step().

    Returns:
        Tuple of (total_loss, list_of_metric_dicts).
        total_loss is the summed loss on last stage, 0.0 on other stages.
    """
    schedule = model.info.schedule
    has_first = model.info.has_first_stage
    has_last = model.info.has_last_stage

    input_ids = batch.pop("input_ids")
    targets = batch.pop("labels", None) if has_last else None
    losses: Optional[list[torch.Tensor]] = [] if has_last else None

    # Inject the loss adapter into the schedule
    schedule._loss_fn = loss_adapter

    # Build args: first stage receives input_ids, others don't
    args = (input_ids,) if has_first else ()

    # Pass remaining batch keys (flash_attn_kwargs, position_ids, etc.)
    # as kwargs to the schedule, following automodel's train_ft.py pattern.
    # Filter out None values and empty dicts to avoid PP chunking errors.
    batch_kwargs = {
        k: v
        for k, v in batch.items()
        if v is not None and not (isinstance(v, dict) and len(v) == 0)
    }

    if forward_only:
        schedule.eval(*args, target=targets, losses=losses, **batch_kwargs)
    else:
        schedule.step(*args, target=targets, losses=losses, **batch_kwargs)

    if has_last and losses:
        total_loss = torch.sum(torch.stack(losses))
    else:
        total_loss = torch.tensor(0.0, device="cuda")

    return total_loss, loss_adapter._all_metrics


# ---------------------------------------------------------------------------
# THD stage shape management
# ---------------------------------------------------------------------------


def reset_pp_stage_shapes_for_thd(model: Any, tokens_per_chunk: int) -> None:
    """Reset PP stage shapes for THD format (packed sequences).

    THD format produces [1, T, dim] outputs instead of [batch, seq, dim].
    Must be called before each schedule.step() when sequence lengths change.
    """
    from nemo_automodel.components.distributed.pipelining.functional import (
        _get_hidden_and_vocab_size,
    )

    schedule = model.info.schedule
    stages = model.info.stages
    model_config = model.parts[0].config
    hidden_size, vocab_size = _get_hidden_and_vocab_size(model_config)

    schedule._stages_forward_initialized = False
    if hasattr(schedule, "_stages_backward_initialized"):
        schedule._stages_backward_initialized = False

    for stage in stages:
        try:
            model_dtype = next(stage.submod.parameters()).dtype
        except StopIteration:
            model_dtype = torch.bfloat16

        if stage.is_first:
            stage.inputs_meta = (
                torch.empty(1, tokens_per_chunk, device="meta", dtype=torch.long),
            )
        else:
            stage.inputs_meta = (
                torch.empty(1, tokens_per_chunk, hidden_size, device="meta", dtype=model_dtype),
            )

        has_lm_head = hasattr(stage.submod, "lm_head") and stage.submod.lm_head is not None
        out_dim = vocab_size if has_lm_head else hidden_size
        stage._outputs_meta = (
            torch.empty(1, tokens_per_chunk, out_dim, device="meta", dtype=model_dtype),
        )


# ---------------------------------------------------------------------------
# PP seqpack batch preparation
# ---------------------------------------------------------------------------


def prepare_pp_seqpack_batch(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    accum_data: dict[str, Any],
    pp_batch_size: int,
    n_microbatches: int,
    tokenizer_eos_id: int,
    train_mb_tokens: int,
    cp_size: int = 1,
    cp_mesh: Any = None,
    device_mesh: Any = None,
    token_mask: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], dict[str, Any], "THDBatch"]:
    """Pack sequences into THD format for a PP training step.

    Handles dummy-padding when the batch has fewer than pp_batch_size sequences.
    Returns the PP batch dict (for schedule.step), the updated accum_data
    (with dummy sample_mask=0), and the THDBatch for loss metadata.
    """
    from nemo_rl.models.automodel.data import pack_for_thd

    actual_seqs = input_ids.shape[0]
    pp_mbs = pp_batch_size // n_microbatches

    if actual_seqs < pp_batch_size:
        pad_count = pp_batch_size - actual_seqs
        input_ids = torch.cat([
            input_ids,
            torch.zeros(pad_count, input_ids.shape[1],
                        dtype=input_ids.dtype, device=input_ids.device),
        ], dim=0)
        input_lengths = torch.cat([
            input_lengths,
            torch.ones(pad_count, dtype=input_lengths.dtype,
                       device=input_lengths.device),
        ])
        for key in accum_data:
            val = accum_data[key]
            if torch.is_tensor(val) and val.ndim >= 1 and val.shape[0] == actual_seqs:
                pad_shape = (pad_count,) + val.shape[1:]
                accum_data[key] = torch.cat(
                    [val, torch.zeros(pad_shape, dtype=val.dtype, device=val.device)]
                )
        if "sample_mask" in accum_data:
            accum_data["sample_mask"][actual_seqs:] = 0

    if token_mask is not None:
        if token_mask.shape[0] < len(input_lengths):
            pad_rows = len(input_lengths) - token_mask.shape[0]
            token_mask = torch.cat([
                token_mask,
                torch.zeros(pad_rows, token_mask.shape[1],
                            dtype=token_mask.dtype, device=token_mask.device),
            ], dim=0)
        token_mask = token_mask[:len(input_lengths)]

    thd_batch = pack_for_thd(
        input_ids=input_ids,
        input_lengths=input_lengths,
        packed_sequence_size=[pp_mbs] * n_microbatches,
        padding_value=tokenizer_eos_id,
        min_seq_len=train_mb_tokens,
        num_chunks=n_microbatches,
        cp_size=cp_size,
        cp_mesh=cp_mesh,
        device_mesh=device_mesh,
        token_mask=token_mask,
    )
    pp_batch = thd_batch.to_model_kwargs(device=input_ids.device)
    return pp_batch, accum_data, thd_batch


def pad_batch_for_pp(
    input_ids: torch.Tensor,
    pp_batch_size: int,
) -> tuple[torch.Tensor, int]:
    """Pad input_ids to pp_batch_size with zero rows. Returns (padded_ids, actual_seqs)."""
    actual_seqs = input_ids.shape[0]
    if actual_seqs < pp_batch_size:
        pad_count = pp_batch_size - actual_seqs
        input_ids = torch.cat([
            input_ids,
            torch.zeros(pad_count, input_ids.shape[1],
                        dtype=input_ids.dtype, device=input_ids.device),
        ], dim=0)
    return input_ids, actual_seqs
