"""Replay rollout expert choices for unpacked AutoModel microbatches (TP1/CP1).

Replay changes discrete expert IDs only. Gate scores and weights remain live,
and the context must enclose backward for activation-checkpoint recomputation.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch
from torch import nn

if TYPE_CHECKING:
    from nemo_rl.models.automodel.data import ProcessedMicrobatch
    from nemo_rl.models.policy import PolicyConfig


def configure_router_replay(model: nn.Module, config: "PolicyConfig") -> None:
    """Enable existing per-gate replay handles without changing parameters.

    Supports torch Gate routing, unpacked token batches, and actor TP1/CP1.
    EP is supported: replay happens before the expert dispatch/all-gather.
    """
    replay_config = config.get("router_replay")
    if replay_config is None or not replay_config["enabled"]:
        return
    if config["generation"]["backend"] != "vllm":
        raise ValueError("AutoModel router replay requires vLLM generation")
    if config["sequence_packing"]["enabled"]:
        raise ValueError("AutoModel router replay does not yet support sequence packing")
    dtensor = config["dtensor_cfg"]
    if dtensor["tensor_parallel_size"] != 1 or dtensor["context_parallel_size"] != 1:
        raise ValueError("AutoModel router replay currently requires actor TP1/CP1")
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.moe.router_replay import RouterReplay

    gates = [module for module in model.modules() if isinstance(module, Gate)]
    if not gates or any(gate.use_routing_core for gate in gates):
        raise ValueError("AutoModel router replay requires torch Gate routing")
    for gate in gates:
        if gate.router_replay is None:
            gate.router_replay = RouterReplay()


@contextmanager
def router_replay_context(
    model: nn.Module, processed_mb: "ProcessedMicrobatch"
) -> Iterator[None]:
    """Install this microbatch's [B,S,L,K] rollout IDs until backward finishes.

    IDs refer to the input token at S, not its next-token logprob. Require
    complete, unique, in-range IDs at every causal ancestor of a loss token.
    Final sampled tokens, environment suffixes and padding need no route when
    they cannot affect a scored token; use valid dummy IDs there. Never mutate
    input tensors. Restore each model-owned handle even if forward fails.
    """
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.moe.router_replay import RouterReplayMode

    gates = [
        m for m in model.modules()
        if isinstance(m, Gate) and m.router_replay is not None
    ]
    if not gates:
        yield
        return
    data = processed_mb.data_dict
    if "routed_experts" not in data:
        raise ValueError("AutoModel replay enabled but routed_experts is missing")
    ids = processed_mb.processed_inputs.input_ids
    routes = data["routed_experts"]
    if routes.ndim != 4 or routes.shape[:2] != ids.shape or routes.shape[2] != len(gates):
        raise ValueError(
            f"Replay shape {tuple(routes.shape)} does not match tokens "
            f"{tuple(ids.shape)} and {len(gates)} layers"
        )
    if routes.is_floating_point() or routes.dtype == torch.bool:
        raise ValueError("routed_experts must contain integer expert IDs")
    positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    loss_mask = data["token_mask"].to(ids.device).bool()
    last_scored = torch.where(loss_mask, positions, -1).amax(dim=1)
    required = positions < last_scored[:, None]
    targets = []
    for layer, gate in enumerate(gates):
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
        dummy = torch.arange(gate.topk, device=ids.device)
        target = torch.where(required[..., None], target, dummy)
        targets.append(target.reshape(-1, gate.topk).contiguous())
    previous = [(g.router_replay.mode, g.router_replay.target_indices) for g in gates]
    calls_before = [g.router_replay.replay_calls for g in gates]
    try:
        for gate, target in zip(gates, targets):
            gate.router_replay.set_target(target)
            gate.router_replay.mode = RouterReplayMode.REPLAY
        yield
        if any(g.router_replay.replay_calls == count for g, count in zip(gates, calls_before)):
            raise RuntimeError("A gate did not consume its rollout replay target")
    finally:
        for gate, (mode, target) in zip(gates, previous):
            gate.router_replay.mode = mode
            gate.router_replay.target_indices = target
