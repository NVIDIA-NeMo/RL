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

"""Replay rollout expert choices for unpacked AutoModel microbatches (TP1).

Replay changes discrete expert IDs only. Gate scores and weights remain live,
and the context must enclose backward for activation-checkpoint recomputation.
"""

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch
from nemo_automodel.components.moe.layers import Gate
from nemo_automodel.components.moe.router_replay import RouterReplay, RouterReplayMode
from torch import nn

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.context_parallel import (
        ContextParallelSharder,
    )

    from nemo_rl.models.automodel.data import ProcessedMicrobatch
    from nemo_rl.models.policy import PolicyConfig


def _replay_gates(model: nn.Module) -> list[tuple[int, Gate]]:
    """Bind model-owned replay handles to global decoder layer IDs.

    Hash-routed layers have no generic Gate and are deliberately excluded.
    """
    gates = []
    for name, module in model.named_modules():
        if isinstance(module, Gate) and module.router_replay is not None:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match is None:
                raise ValueError(f"Cannot map replay gate to a decoder layer: {name}")
            gates.append((int(match.group(1)), module))
    if len({layer for layer, _ in gates}) != len(gates):
        raise ValueError("Router replay requires one gate per decoder layer")
    return gates


def configure_router_replay(model: nn.Module, config: "PolicyConfig") -> None:
    """Enable existing per-gate replay handles without changing parameters.

    Supports torch Gate routing, unpacked token batches, and actor TP1.
    DeepSeek V4 CP uses the model-owned token sharder.
    EP is supported: replay happens before the expert dispatch/all-gather.
    """
    replay_config = config.get("router_replay")
    if replay_config is None or not replay_config["enabled"]:
        return
    if config["generation"]["backend"] != "vllm":
        raise ValueError("AutoModel router replay requires vLLM generation")
    if config["sequence_packing"]["enabled"]:
        raise ValueError(
            "AutoModel router replay does not yet support sequence packing"
        )
    dtensor = config["dtensor_cfg"]
    if dtensor.get("tensor_parallel_size") != 1:
        raise ValueError("AutoModel router replay currently requires actor TP1")
    if (
        dtensor.get("context_parallel_size", 1) > 1
        and getattr(model.config, "model_type", None) != "deepseek_v4"
    ):
        raise ValueError(
            "AutoModel router replay CP currently supports DeepSeek V4 only"
        )

    gates = [module for module in model.modules() if isinstance(module, Gate)]
    if not gates or any(gate.use_routing_core for gate in gates):
        raise ValueError("AutoModel router replay requires torch Gate routing")
    for gate in gates:
        if gate.router_replay is None:
            gate.router_replay = RouterReplay()
    _replay_gates(model)


@contextmanager
def router_replay_context(
    model: nn.Module,
    processed_mb: "ProcessedMicrobatch",
    *,
    cp_sharder: "ContextParallelSharder | None" = None,
) -> Iterator[None]:
    """Install this microbatch's [B,S,L,K] rollout IDs until backward finishes.

    IDs refer to the input token at S, not its next-token logprob. Require
    complete, unique, in-range IDs at every causal ancestor of a loss token.
    Final sampled tokens, environment suffixes and padding need no route when
    they cannot affect a scored token; use valid dummy IDs there. Never mutate
    input tensors. Restore each model-owned handle even if forward fails.
    """
    bindings = _replay_gates(model)
    gates = [gate for _, gate in bindings]
    if not gates:
        yield
        return
    data = processed_mb.data_dict
    if "routed_experts" not in data:
        raise ValueError("AutoModel replay enabled but routed_experts is missing")
    ids = processed_mb.processed_inputs.input_ids
    routes = data["routed_experts"]
    if (
        routes.ndim != 4
        or routes.shape[:2] != ids.shape
        or routes.shape[2] != max(layer for layer, _ in bindings) + 1
    ):
        raise ValueError(
            f"Replay shape {tuple(routes.shape)} does not match tokens "
            f"{tuple(ids.shape)} and decoder layer IDs {[layer for layer, _ in bindings]}"
        )
    if routes.is_floating_point() or routes.dtype == torch.bool:
        raise ValueError("routed_experts must contain integer expert IDs")
    positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    loss_mask = data["token_mask"].to(ids.device).bool()
    last_scored = torch.where(loss_mask, positions, -1).amax(dim=1)
    required = positions < last_scored[:, None]
    # Validate causal coverage globally, including tokens on earlier CP ranks.
    # Reuse the forward's layout for both route IDs and padding validity.
    local_required = (
        cp_sharder.shard_token_tensor(required, fill=0)
        if cp_sharder is not None
        else required
    )
    targets = []
    for layer, gate in bindings:
        target = routes[:, :, layer].to(device=ids.device, dtype=torch.long)
        if target.shape[-1] != gate.topk:
            raise ValueError(f"Layer {layer}: replay top-k does not match gate")
        valid = ((target >= 0) & (target < gate.n_experts)).all(-1)
        ordered = target.sort(-1).values
        valid &= (ordered[..., 1:] != ordered[..., :-1]).all(-1)
        if not bool((valid | ~required).all()):
            raise ValueError(
                f"Layer {layer}: missing, duplicate or out-of-range routes on required tokens"
            )
        if cp_sharder is not None:
            target = cp_sharder.shard_token_tensor(target, fill=0)
        dummy = torch.arange(gate.topk, device=ids.device)
        target = torch.where(local_required[..., None], target, dummy)
        targets.append(target.reshape(-1, gate.topk).contiguous())
    previous = [(g.router_replay.mode, g.router_replay.target_indices) for g in gates]
    calls_before = [g.router_replay.replay_calls for g in gates]
    try:
        for gate, target in zip(gates, targets):
            gate.router_replay.set_target(target)
            gate.router_replay.mode = RouterReplayMode.REPLAY
        yield
        if any(
            g.router_replay.replay_calls == count
            for g, count in zip(gates, calls_before)
        ):
            raise RuntimeError("A gate did not consume its rollout replay target")
    finally:
        for gate, (mode, target) in zip(gates, previous):
            gate.router_replay.mode = mode
            gate.router_replay.target_indices = target
