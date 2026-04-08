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
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType


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
        self._call_idx: int = 0
        self._all_metrics: list[dict[str, Any]] = []
        self._global_valid_seqs: Optional[torch.Tensor] = None
        self._global_valid_toks: Optional[torch.Tensor] = None
        self._num_global_batches: int = 1

    def set_microbatches(
        self,
        data_dict: Any,
        n_microbatches: int,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        num_global_batches: int = 1,
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

    def __call__(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        """Called by PP schedule per microbatch."""
        logits = getattr(output, "logits", output)
        mb_data = self._microbatches[self._call_idx]
        self._call_idx += 1

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

    if forward_only:
        schedule.eval(*args, target=targets, losses=losses)
    else:
        schedule.step(*args, target=targets, losses=losses)

    if has_last and losses:
        total_loss = torch.sum(torch.stack(losses))
    else:
        total_loss = torch.tensor(0.0, device="cuda")

    return total_loss, loss_adapter._all_metrics
