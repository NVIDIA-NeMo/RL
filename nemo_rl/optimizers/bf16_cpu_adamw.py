# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU AdamW with BF16 moment storage and bounded FP32 update workspaces."""

from collections import defaultdict
from collections.abc import Callable, Iterable
import math
from typing import Any

import torch
from torch.distributed.tensor import DTensor


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class BF16CPUAdamW(torch.optim.Optimizer):
    """Update FP32 CPU parameters using BF16 moments and FP32 chunk arithmetic.

    DTensor parameters keep their distributed state layout; computation touches
    only local shards, after the caller has reduced/scaled/clipped gradients.
    Each moment is rounded once after its FP32 update. The parameter update uses
    the unrounded FP32 moments. Parameters and moments remain on CPU. Optional CUDA gradient shards are
    consumed through a bounded CPU staging buffer. AMSGrad and sparse gradients
    are unsupported.
    All options are explicit so experiment YAML owns their defaults.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        chunk_numel: int,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
            chunk_numel=chunk_numel,
        )
        self.last_streamed_chunk_bytes = 0
        self._validate_group(defaults)
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)
            for param in group["params"]:
                self._validate_tensor(_local(param))

    @staticmethod
    def _validate_group(group: dict[str, Any]) -> None:
        for key in ("lr", "eps", "weight_decay"):
            value = group[key]
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
        if len(group["betas"]) != 2 or not all(0 <= b < 1 for b in group["betas"]):
            raise ValueError("betas must contain two values in [0, 1)")
        if type(group["chunk_numel"]) is not int or group["chunk_numel"] <= 0:
            raise ValueError("chunk_numel must be a positive integer")

    @staticmethod
    def _validate_tensor(tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("BF16CPUAdamW requires FP32 CPU parameters and gradients")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError("BF16CPUAdamW requires dense contiguous local shards")

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None,
        *,
        gradient_shards: dict[int, torch.Tensor] | None = None,
    ) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if gradient_shards is not None:
            params = [p for group in self.param_groups for p in group["params"]]
            if not set(gradient_shards).issubset({id(p) for p in params}):
                raise ValueError("GPU gradient is not owned by this optimizer")
            if any(p.grad is not None for p in params):
                raise ValueError(
                    "Cannot mix attached CPU gradients and streamed gradients"
                )
        self.last_streamed_chunk_bytes = 0
        for group in self.param_groups:
            staging = None
            if gradient_shards is not None:
                staging = torch.empty(
                    group["chunk_numel"], dtype=torch.float32, device="cpu"
                )
                self.last_streamed_chunk_bytes = max(
                    self.last_streamed_chunk_bytes, staging.numel() * 4
                )
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                supplied_grad = (
                    param.grad
                    if gradient_shards is None
                    else gradient_shards.get(id(param))
                )
                if supplied_grad is None:
                    continue
                local_param, local_grad = _local(param), _local(supplied_grad)
                self._validate_tensor(local_param)
                if gradient_shards is None:
                    self._validate_tensor(local_grad)
                elif (
                    local_grad.device.type != "cuda"
                    or local_grad.dtype != torch.float32
                    or not local_grad.is_contiguous()
                ):
                    raise ValueError(
                        "Streaming requires contiguous FP32 CUDA gradient shards"
                    )
                if local_param.shape != local_grad.shape:
                    raise ValueError("Parameter and gradient local shapes differ")
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.bfloat16)
                state["step"] += 1
                step = state["step"]
                p, grad = local_param.view(-1), local_grad.view(-1)
                stored_m = _local(state["exp_avg"]).view(-1)
                stored_v = _local(state["exp_avg_sq"]).view(-1)
                step_size = group["lr"] / (1 - beta1**step)
                variance_correction = math.sqrt(1 - beta2**step)
                for start in range(0, p.numel(), group["chunk_numel"]):
                    stop = min(start + group["chunk_numel"], p.numel())
                    if staging is None:
                        g = grad[start:stop]
                    else:
                        g = staging[: stop - start]
                        g.copy_(grad[start:stop], non_blocking=False)
                    m = stored_m[start:stop].float()
                    v = stored_v[start:stop].float()
                    m.lerp_(g, 1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    denom = v.sqrt().div_(variance_correction).add_(group["eps"])
                    p[start:stop].mul_(1 - group["lr"] * group["weight_decay"])
                    p[start:stop].addcdiv_(m, denom, value=-step_size)
                    stored_m[start:stop].copy_(m)
                    stored_v[start:stop].copy_(v)
        return loss

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["bf16_cpu_adamw_version"] = 1
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore BF16 state without Optimizer's FP32 parameter-dtype cast.

        This is same-layout restore; distributed checkpoint resharding is not
        provided by this method. Incompatible optimizer states fail explicitly.
        """
        if state_dict.get("bf16_cpu_adamw_version") != 1:
            raise ValueError("Expected a BF16CPUAdamW version 1 checkpoint")
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("Optimizer parameter group count differs")
        new_groups, new_state = [], defaultdict(dict)
        for saved, current in zip(saved_groups, self.param_groups, strict=True):
            self._validate_group(saved)
            if len(saved["params"]) != len(current["params"]):
                raise ValueError("Optimizer parameter group size differs")
            new_groups.append({**saved, "params": current["params"]})
            for saved_id, param in zip(saved["params"], current["params"], strict=True):
                if saved_id not in state_dict["state"]:
                    continue
                state = state_dict["state"][saved_id]
                if type(state["step"]) is not int or state["step"] < 0:
                    raise ValueError("Invalid optimizer step")
                restored = {"step": state["step"]}
                for key in ("exp_avg", "exp_avg_sq"):
                    tensor = state[key]
                    if tensor.dtype != torch.bfloat16 or tensor.shape != param.shape:
                        raise ValueError(
                            "Moment dtype or shape differs from BF16 state layout"
                        )
                    if isinstance(tensor, DTensor) != isinstance(param, DTensor):
                        raise ValueError("Moment DTensor layout differs")
                    if isinstance(param, DTensor) and (
                        tensor.device_mesh != param.device_mesh
                        or tensor.placements != param.placements
                    ):
                        raise ValueError("Moment mesh or placements differ")
                    if (
                        _local(tensor).shape != _local(param).shape
                        or not _local(tensor).is_contiguous()
                    ):
                        raise ValueError("Moment local layout differs")
                    restored[key] = tensor.to("cpu")
                new_state[param] = restored
        self.__setstate__({"state": new_state, "param_groups": new_groups})
