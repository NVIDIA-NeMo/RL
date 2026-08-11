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

from typing import Any, Optional

import torch
import torch.distributed as dist
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
)
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return (
        ((value + multiple - 1) // multiple * multiple)
        if value % multiple != 0
        else value
    )


def broadcast_tensor(
    tensor: torch.Tensor | None, src_rank: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Broadcasts a tensor from src_rank to all ranks in the group using broadcast_object_list for metadata.

    Handles the case where the input tensor might be None on non-source ranks.
    If the input tensor is provided on non-source ranks, it must have the
    correct shape and dtype matching the tensor on the source rank.

    Args:
        tensor: The tensor to broadcast on the source rank. Can be None on
                non-source ranks (will be created with correct shape/dtype).
                If not None on non-source ranks, it's used as the buffer
                for the broadcast and must match the source tensor's metadata.
        src_rank (int): The global rank of the source process.
        group: The process group for communication.

    Returns:
        torch.Tensor: The broadcasted tensor. On non-source ranks, this will
                      be the tensor received from the source.

    Raises:
        ValueError: If the tensor is None on the source rank, or if a tensor
                    provided on a non-source rank has mismatched shape/dtype/device.
        TypeError: If broadcasting metadata fails (e.g., due to pickling issues).
    """
    rank = dist.get_rank()
    # Assume operations happen on the default CUDA device for the rank
    # TODO: Consider making device explicit if needed, e.g., derive from tensor on src
    device = torch.cuda.current_device()

    # 1. Broadcast metadata (shape and dtype) using broadcast_object_list
    if rank == src_rank:
        if tensor is None:
            raise ValueError(f"Rank {rank} is source ({src_rank}) but tensor is None.")
        # Package metadata into a list containing shape and dtype
        metadata = [tensor.shape, tensor.dtype]
        object_list = [metadata]
    else:
        # Placeholder for receiving the object on non-source ranks
        object_list = [None]

    # Broadcast the list containing the metadata object
    # This relies on the underlying distributed backend supporting object serialization (pickle)
    try:
        dist.broadcast_object_list(object_list, src=src_rank, group=group)
    except Exception as e:
        # Catch potential issues with pickling or backend support
        raise TypeError(
            f"Failed to broadcast tensor metadata using broadcast_object_list: {e}"
        ) from e

    # All ranks now have the metadata in object_list[0]
    received_shape, received_dtype = object_list[0]

    # 2. Prepare tensor buffer on non-source ranks
    if rank != src_rank:
        if tensor is None:
            # Create tensor if it wasn't provided by the caller
            tensor = torch.empty(received_shape, dtype=received_dtype, device=device)
        else:
            # Validate the tensor provided by the caller on the non-source rank
            if tensor.shape != received_shape:
                raise ValueError(
                    f"Rank {rank}: Provided tensor has shape {tensor.shape}, "
                    f"but source rank {src_rank} is broadcasting shape {received_shape}."
                )
            if tensor.dtype != received_dtype:
                raise ValueError(
                    f"Rank {rank}: Provided tensor has dtype {tensor.dtype}, "
                    f"but source rank {src_rank} is broadcasting dtype {received_dtype}."
                )
            # Ensure the provided tensor is on the correct device
            # Compare torch.device objects directly for accuracy
            if tensor.device != torch.device(device):
                raise ValueError(
                    f"Rank {rank}: Provided tensor is on device {tensor.device}, "
                    f"but expected broadcast device is {device}."
                )

    # 3. Broadcast the actual tensor data
    # The tensor object (either original on src, newly created, or validated user-provided on non-src)
    # must exist on all ranks before calling broadcast.
    # `dist.broadcast` operates in-place on the provided tensor object.
    dist.broadcast(tensor, src=src_rank, group=group)

    return tensor


#: Mapping from a Megatron ``moe_router_load_balancing_type`` value to the aux-loss
#: name the router records for it. Mirrors ``MoETopKRouter.is_aux_loss_enabled``, which
#: treats these three types as first-class and drives one all_reduce per recorded name.
_AUX_LOSS_TRACK_NAMES: dict[str, str] = {
    "aux_loss": "load_balancing_loss",
    "seq_aux_loss": "seq_load_balancing_loss",
    "global_aux_loss": "global_load_balancing_loss",
}


def get_aux_loss_track_names(model_config: Any) -> list[str]:
    """Returns the aux-loss tracker names the router records for a model config.

    Megatron's router only records an aux loss when its balancing type is configured
    *and* the matching coefficient is non-zero (``MoETopKRouter.get_aux_loss_coeff``
    returns 0.0 otherwise, and ``_apply_aux_loss`` returns early). Deriving the names
    the same way keeps the pre-initialization in ``get_moe_metrics`` aligned with what
    the router actually tracks, so no permanently-zero metric is reported for models
    that have load balancing disabled (e.g. ``moe_router_load_balancing_type: "none"``).

    ``moe_router_load_balancing_type`` may be a single string or a list, in which case
    ``moe_aux_loss_coeff`` is a list of the same length (validated by Megatron's
    ``TransformerConfig``), so more than one aux loss can be live at once.

    Args:
        model_config: Megatron ``TransformerConfig`` (or any object exposing
            ``moe_router_load_balancing_type`` / ``moe_aux_loss_coeff``).

    Returns:
        list[str]: Aux-loss tracker names to pre-initialize, in the order Megatron
        records them. Empty when no aux loss is enabled.
    """
    routing_type = getattr(model_config, "moe_router_load_balancing_type", None)
    aux_loss_coeff = getattr(model_config, "moe_aux_loss_coeff", 0.0)

    routing_types: list[Any] = []
    coeffs: list[Any] = []
    if isinstance(routing_type, str):
        routing_types = [routing_type]
        # A single balancing type pairs with a scalar coefficient.
        coeffs = [aux_loss_coeff]
    elif isinstance(routing_type, (list, tuple)):
        routing_types = list(routing_type)
        if isinstance(aux_loss_coeff, (list, tuple)):
            coeffs = list(aux_loss_coeff)
        else:
            # Defensive: Megatron validates that the lists have matching lengths, but
            # tolerate a scalar coefficient rather than raising while collecting metrics.
            coeffs = [aux_loss_coeff] * len(routing_types)

    track_names: list[str] = []
    for index, single_routing_type in enumerate(routing_types):
        if not isinstance(single_routing_type, str):
            continue
        name = _AUX_LOSS_TRACK_NAMES.get(single_routing_type)
        if name is None:
            continue
        coeff = coeffs[index] if index < len(coeffs) else 0.0
        # Only a real number can enable the loss; anything else (None, or a stand-in
        # object from a partially-populated config) means "not configured".
        if isinstance(coeff, (int, float)) and coeff > 0 and name not in track_names:
            track_names.append(name)

    # z_loss is recorded independently of the load balancing type, gated only on its
    # own coefficient being set (see MoETopKRouter.apply_z_loss).
    z_loss_coeff = getattr(model_config, "moe_z_loss_coeff", None)
    if isinstance(z_loss_coeff, (int, float)):
        track_names.append("z_loss")

    return track_names


def get_moe_metrics(
    loss_scale: float,
    total_loss_dict: Optional[dict] = None,
    per_layer_logging: bool = False,
    num_layers: Optional[int] = None,
    mtp_num_layers: Optional[int] = None,
    track_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Returns Mixture of Experts (MoE) auxiliary-loss metrics.

    This function reduces MoE auxiliary losses across ranks, aggregates them, and
    returns a dictionary of metrics.

    Args:
        loss_scale: Scale factor to apply to each auxiliary loss (e.g., 1/num_microbatches).
        total_loss_dict: If provided, accumulate means into this dict (by name).
        per_layer_logging: If True, include per-layer values in the returned dict.
        num_layers: Total number of transformer layers. When provided together with a
            non-empty ``track_names``, the aux-loss tracker is pre-initialized on every
            rank before the reduction (see Note). Defaults to None, which disables
            pre-initialization.
        mtp_num_layers: Extra layers contributed by Multi-Token Prediction, added to
            ``num_layers`` to size the pre-initialized tensor, matching the size the
            router uses when recording. Defaults to None (treated as 0).
        track_names: Aux-loss names to pre-initialize; must mirror what the router
            records for the configured ``moe_router_load_balancing_type``, so callers
            should derive it via ``get_aux_loss_track_names(model_config)``. Defaults to
            None, which disables pre-initialization.

    Returns:
        dict[str, Any]: A flat dict of aggregated metrics. For each aux loss name,
        the mean value is returned under the same key (e.g., "load_balancing_loss").
        If per_layer_logging is True, per-layer values are returned under keys of the
        form "moe/{name}_layer_{i}".

    Note:
        num_layers/mtp_num_layers/track_names pre-initialize the aux-loss tracker so
        every pipeline-parallel rank participates in the collective all_reduce below
        with an equally-sized tensor, preventing a hang when some PP rank did not
        record an aux loss this step (e.g. a stage with no MoE layer, or an MTP MoE
        layer that lives only on the last stage).
    """
    # Pre-initialize the aux-loss tracker so every PP rank has the same set of
    # named, equally-sized tensors BEFORE the collective all_reduce below.
    #
    # reduce_aux_losses_tracker_across_ranks() all_reduces over the pipeline-parallel
    # group for each name present in the *local* tracker. Tracker entries are created
    # lazily (MoEMetricsTracker.record only allocates torch.zeros(num_layers) the first
    # time a rank records a loss), and _sync_metrics skips names it does not have. If
    # any PP rank did not record an aux loss this step, it skips the all_reduce for that
    # name while other PP ranks perform it -> the collective mismatches participants and
    # hangs with no traceback.
    #
    # Mirror Megatron's own report(force_initialize=True) guard, which calls
    # ensure_initialized() for the same reason. Sizing must be
    # (num_layers + mtp_num_layers) to match what the router passes to record().
    if num_layers is not None and track_names:
        tracker_num_layers = num_layers + (mtp_num_layers or 0)
        # Use the live tracker rather than get_moe_layer_wise_logging_tracker(), which is
        # a deprecated shim returning a fresh dict copy -- writes to it are discarded.
        mcore_tracker = get_moe_metrics_tracker()
        for name in track_names:
            mcore_tracker.ensure_initialized(name, tracker_num_layers)

    reduce_aux_losses_tracker_across_ranks()
    tracker = get_moe_layer_wise_logging_tracker()

    metrics: dict[str, Any] = {}
    if len(tracker) > 0:
        aux_losses = {k: v["values"].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            # Megatron-LM aggregates aux losses across layers and normalizes by number of MoE layers
            num_tracked_layers = int(loss_list.numel()) if loss_list.numel() > 0 else 1
            aggregated_value = loss_list.sum() / num_tracked_layers
            metrics[name] = float(aggregated_value.item())
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = aggregated_value
                else:
                    total_loss_dict[name] += aggregated_value

            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    metrics[f"moe/{name}_layer_{i}"] = float(loss)

    clear_aux_losses_tracker()
    return metrics


def get_mtp_metrics(loss_scale: float = 1.0) -> dict[str, Any]:
    """Returns Multi-Token Prediction (MTP) loss and acceptance rate metrics.

    This function reduces MTP metrics across ranks and returns a dictionary of metrics.

    Args:
        loss_scale: Scale factor applied to each MTP layer's loss (e.g., 1/num_microbatches).
            ``MTPLossLoggingHelper`` accumulates the per-microbatch loss across microbatches
            without dividing, so callers must pass 1/num_microbatches to recover the mean
            (mirroring ``get_moe_metrics``). Acceptance rate is a ratio of counts and is not
            scaled. Defaults to 1.0.

    Returns:
        dict[str, Any]: A flat dict of metrics. Each MTP layer's loss is returned
        under the key "mtp_{i}_loss" and acceptance rate under "mtp_{i}_acceptance_rate"
        where i is 1-indexed (matching Megatron-LM).
    """
    MTPLossLoggingHelper.reduce_metrics_in_tracker()
    tracker = MTPLossLoggingHelper.tracker

    metrics: dict[str, Any] = {}
    if "loss_values" in tracker:
        mtp_losses = tracker["loss_values"].float() * loss_scale
        mtp_corrects = tracker.get("correct_values", torch.zeros_like(mtp_losses))
        mtp_totals = tracker.get("total_values", torch.ones_like(mtp_losses))
        mtp_num_layers = mtp_losses.shape[0]

        # Log per-layer losses and acceptance rates
        for i in range(mtp_num_layers):
            metrics[f"mtp_{i + 1}_loss"] = float(mtp_losses[i].item())
            # Compute acceptance rate as percentage
            acceptance_rate = (mtp_corrects[i] / mtp_totals[i].clamp(min=1)) * 100.0
            metrics[f"mtp_{i + 1}_acceptance_rate"] = float(acceptance_rate.item())

        MTPLossLoggingHelper.clean_metrics_in_tracker()
    return metrics
