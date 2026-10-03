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
"""Rollout Routing Replay (R3) for the automodel (DTensor v2) policy backend.

vLLM records the top-k experts it routed each token to during rollout
(``routed_experts``, shape ``[B, S, layer slots, topk]``). Small numerical
differences between vLLM and the training forward flip near-tied routing decisions,
and the flips compound across layers and positions until a token's logprob can differ
by tens of nats. Replaying the rollout selection through Automodel's per-gate
``RouterReplay`` hooks makes the policy score each token through the experts that
generated it. Only the discrete selection is replayed; routing weights are still
computed from the live router, so gradients keep flowing into it.
"""

import types
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.moe.router_replay import RouterReplay
from torch import nn

from nemo_rl.models.generation.interfaces import ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL
from nemo_rl.models.megatron.router_replay import router_replay_enabled
from nemo_rl.models.policy import PolicyConfig


@dataclass
class ReplayGate:
    """A policy MoE gate whose expert selection nemo-rl can replay."""

    layer: Optional[int]  # index i of the ``layers.<i>`` decoder block owning the gate
    num_experts: Optional[int]
    handle: RouterReplay


def validate_automodel_router_replay_config(config: PolicyConfig) -> None:
    """Reject setups where R3 cannot get routes or gate tokens would not align with them."""
    if not router_replay_enabled(config):
        return
    generation = config.get("generation")
    if generation is None or generation["backend"] != "vllm":
        raise ValueError(
            "policy.router_replay.enabled requires vLLM generation, which returns the "
            "rollout routed_experts."
        )
    dtensor_cfg = config["dtensor_cfg"]
    packing = config.get("sequence_packing")
    unsupported = {
        "sequence_packing": bool(packing and packing["enabled"]),
        "context_parallel_size > 1": dtensor_cfg["context_parallel_size"] > 1,
        "sequence_parallel": dtensor_cfg["sequence_parallel"],
    }
    enabled = [name for name, on in unsupported.items() if on]
    if enabled:
        raise NotImplementedError(
            "policy.router_replay.enabled with the automodel backend does not "
            f"support {enabled} yet."
        )


def require_automodel_moe_model(
    automodel_kwargs: dict[str, Any], model_config: Any
) -> None:
    """Reject models that load through HF, whose MoE gates have no replay hooks.

    Call after ``force_hf`` is settled; otherwise HF rejects the injected
    ``moe_overrides`` with an opaque ``TypeError``.
    """
    architectures = getattr(model_config, "architectures", None) or []
    if (
        automodel_kwargs.get("force_hf")
        or not architectures
        or architectures[0] not in ModelRegistry.model_arch_name_to_cls
    ):
        raise ValueError(
            "policy.router_replay.enabled needs an Automodel MoE implementation with "
            f"replay hooks, but this model loads through HF (architectures={architectures}, "
            f"force_hf={bool(automodel_kwargs.get('force_hf'))})."
        )


def enable_routing_replay_in_automodel_kwargs(automodel_kwargs: dict[str, Any]) -> None:
    """Build every Automodel MoE gate with a ``RouterReplay`` handle."""
    overrides = automodel_kwargs.get("moe_overrides") or {}
    automodel_kwargs["moe_overrides"] = {**overrides, "enable_routing_replay": True}


def _decoder_layer_index(module_name: str) -> Optional[int]:
    """Index ``i`` of the ``layers.<i>`` block that owns ``module_name``, if any."""
    parts = module_name.split(".")
    for prev, part in zip(parts, parts[1:]):
        if prev == "layers" and part.isdigit():
            return int(part)
    return None


def _replaying_apply(handle: RouterReplay, indices: torch.Tensor) -> torch.Tensor:
    """Swap in the replay target; tokens marked keep-live retain the gate's own selection."""
    replay = handle._nrl_replay
    if replay is None:
        return RouterReplay.apply(handle, indices)
    target, keep_live = replay
    if target.shape != indices.shape:
        raise ValueError(
            f"Replay indices shape {tuple(target.shape)} does not match the gate selection "
            f"shape {tuple(indices.shape)}; routed_experts must cover the same tokens and topk."
        )
    return torch.where(
        keep_live.to(indices.device),
        indices,
        target.to(device=indices.device, dtype=indices.dtype),
    )


def router_replay_gates(model: nn.Module) -> list[ReplayGate]:
    """The policy's MoE gates in layer order, each set up for nemo-rl replay.

    Replay is installed on these instances only (Automodel's own ``RouterReplay`` API is
    untouched). MTP gates are excluded: vLLM never routes them.
    """
    gates = []
    for name, module in model.named_modules():
        handle = getattr(module, "router_replay", None)
        if handle is None or any(part.startswith("mtp") for part in name.split(".")):
            continue
        handle._nrl_replay = None
        handle.apply = types.MethodType(_replaying_apply, handle)
        gates.append(
            ReplayGate(
                layer=_decoder_layer_index(name),
                num_experts=getattr(module, "n_experts", None),
                handle=handle,
            )
        )
    return gates


def _payload_layers(gates: list[ReplayGate], num_payload_layers: int) -> list[int]:
    """Which ``routed_experts`` layer slot each gate replays.

    vLLM may emit one slot per MoE layer, or one per decoder layer (hybrid models such as
    NemotronH also count Mamba/attention layers); the latter is indexed by decoder layer.
    """
    if num_payload_layers == len(gates):
        return list(range(len(gates)))
    layers = [gate.layer for gate in gates]
    if None in layers or max(layers) >= num_payload_layers:
        raise ValueError(
            f"routed_experts has {num_payload_layers} layer slots, which matches neither the "
            f"{len(gates)} MoE gates nor their decoder-layer indices {layers}."
        )
    return layers


def microbatch_routed_experts(data_dict: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """The microbatch's ``routed_experts`` (token-aligned with ``input_ids``), padding unreplayed.

    Batching right-pads ``routed_experts`` with 0, i.e. ``[0, 0, ..., 0]`` rows that repeat
    one expert; positions past each sequence's ``input_lengths`` get the missing-route
    sentinel so they keep the live selection instead.
    """
    routed_experts = data_dict.get("routed_experts")
    if routed_experts is None:
        raise ValueError(
            "policy.router_replay.enabled but the batch carries no routed_experts; "
            "vLLM must run with enable_return_routed_experts (set by configure_vllm_for_router_replay)."
        )
    if routed_experts.dim() != 4 or tuple(routed_experts.shape[:2]) != tuple(
        input_ids.shape
    ):
        raise ValueError(
            f"routed_experts {tuple(routed_experts.shape)} must be [batch, seq, layer slots, topk] "
            f"aligned with input_ids {tuple(input_ids.shape)}."
        )
    lengths = data_dict["input_lengths"].to(routed_experts.device)
    positions = torch.arange(routed_experts.shape[1], device=routed_experts.device)
    padding = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    return routed_experts.masked_fill(
        padding[:, :, None, None], ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL
    )


def _validated_targets(
    gates: list[ReplayGate], routed_experts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-gate replay targets ``[tokens, gates, topk]`` and keep-live mask ``[tokens, gates, 1]``.

    Only all-sentinel rows mean "no route captured" and keep the live selection. Any other
    invalid row (partially negative, out of range, or repeating an expert) is corruption.
    """
    batch, seq, num_payload_layers, topk = routed_experts.shape
    slots = _payload_layers(gates, num_payload_layers)
    device = "cuda" if torch.cuda.is_available() else routed_experts.device
    targets = (
        routed_experts[:, :, slots].reshape(batch * seq, len(gates), topk).to(device)
    )
    keep_live = (targets == ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL).all(
        dim=-1, keepdim=True
    )
    unbounded = torch.iinfo(torch.int64).max
    num_experts = torch.tensor(
        [unbounded if g.num_experts is None else g.num_experts for g in gates],
        device=device,
    ).view(1, -1, 1)
    sorted_targets = torch.sort(targets, dim=-1)[0]
    repeats = sorted_targets[..., 1:] == sorted_targets[..., :-1]
    invalid = ~keep_live.squeeze(-1) & (
        ((targets < 0) | (targets >= num_experts)).any(dim=-1) | repeats.any(dim=-1)
    )
    if invalid.any():
        row, gate_idx = (int(i) for i in invalid.nonzero()[0])
        gate = gates[gate_idx]
        raise ValueError(
            f"Corrupt routed_experts for the MoE gate in decoder layer {gate.layer}: token "
            f"{row % seq} of microbatch sample {row // seq} has route "
            f"{targets[row, gate_idx].tolist()}. Experts must be distinct ids in "
            f"[0, {gate.num_experts}); only an all-{ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL} row "
            "marks a missing route."
        )
    return targets, keep_live


@contextmanager
def replay_routes(
    gates: list[ReplayGate], routed_experts: torch.Tensor
) -> Iterator[None]:
    """Replay ``routed_experts`` ([B, S, layer slots, topk], aligned with input_ids) on ``gates``.

    Wrap both the forward and the backward of a microbatch so activation-checkpoint
    recomputation replays the same selection.
    """
    targets, keep_live = _validated_targets(gates, routed_experts)
    try:
        for i, gate in enumerate(gates):
            gate.handle._nrl_replay = (targets[:, i], keep_live[:, i])
        yield
    finally:
        for gate in gates:
            gate.handle._nrl_replay = None


def microbatch_replay_context(
    gates: Optional[list[ReplayGate]], processed_mb: Any
) -> AbstractContextManager[None]:
    """Replay the microbatch's rollout routes on ``gates``; a no-op when ``gates`` is None."""
    if gates is None:
        return nullcontext()
    routed_experts = microbatch_routed_experts(
        processed_mb.data_dict, processed_mb.processed_inputs.input_ids
    )
    return replay_routes(gates, routed_experts)
