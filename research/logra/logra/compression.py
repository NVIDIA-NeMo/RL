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

"""Direct linear sketches and transient-gradient sketches for fused MoE weights."""

import hashlib
import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Shard
from torch.utils.hooks import RemovableHandle

from logra.config import LoGRAConfig


def projection_seed(base_seed: int, update: int, name: str) -> int:
    """Generate a stable per-layer seed independently of module wrappers."""
    wrappers = {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"}
    name = ".".join(part for part in name.split(".") if part not in wrappers)
    digest = int.from_bytes(
        hashlib.blake2b(name.encode(), digest_size=8).digest(), "little"
    )
    return (base_seed * 1_000_003 + update * 7_919 + digest) & ((1 << 62) - 1)


def make_projection(
    rank: int,
    width: int,
    seed: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    distribution: str,
) -> Tensor:
    """Draw on CPU so the same seed describes the same matrix on every device."""
    if rank <= 0 or width <= 0:
        raise ValueError("Projection dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed & ((1 << 62) - 1))
    if distribution == "rademacher":
        bits = torch.randint(
            0, 2, (rank, width), generator=generator, dtype=torch.uint8
        )
        result = bits.to(dtype).mul_(2).sub_(1).mul_(1 / math.sqrt(rank))
    elif distribution == "gaussian":
        result = torch.randn(rank, width, generator=generator, dtype=torch.float64)
        result = result.mul_(1 / math.sqrt(rank)).to(dtype)
    else:
        raise ValueError(f"Unsupported projection distribution: {distribution}")
    return result.to(device)


class _AccumulateSketch(torch.autograd.Function):
    """Keep projected activations in autograd-managed storage, freed after backward."""

    @staticmethod
    # PyTorch autograd dispatches this fixed signature through Function.apply.
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any, output: Tensor, projected: Tensor, sketch: Tensor
    ) -> Tensor:
        ctx.save_for_backward(projected)
        # Autograd supplies a dynamic context for per-call non-differentiable state.
        ctx.sketch = sketch  # pyrefly: ignore [implicitly-defined-attribute]
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None, None]:
        (gradient,) = grad_outputs
        (projected,) = ctx.saved_tensors
        sketch = ctx.sketch
        with torch.no_grad():
            contribution = gradient.reshape(-1, gradient.shape[-1]).T @ projected
            sketch.add_(contribution.float())
        return gradient, None, None


class _ProjectWeightGradient(torch.autograd.Function):
    """Project a fused expert gradient before it reaches the stored parameter.

    Unlike the linear hook, this path materializes one transient dense gradient.
    It keeps the backend's expert dispatch and matrix multiplication unchanged.
    """

    @staticmethod
    # Autograd dispatches a fixed signature and supplies a dynamic context.
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any, weight: Tensor, state: "SketchState", row_offset: int
    ) -> Tensor:
        ctx.state = state  # pyrefly: ignore [implicitly-defined-attribute]
        ctx.row_offset = row_offset  # pyrefly: ignore [implicitly-defined-attribute]
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[None, None, None]:
        (gradient,) = grad_outputs
        state = ctx.state
        rows = gradient.reshape(-1, gradient.shape[-1])
        # Bound the FP32 conversion workspace, including large expert banks.
        with torch.no_grad():
            for start in range(0, rows.shape[0], 256):
                stop = min(start + 256, rows.shape[0])
                state.sketch[ctx.row_offset + start : ctx.row_offset + stop].add_(
                    rows[start:stop].float() @ state.projection.T
                )
        return None, None, None


@dataclass
class SketchState:
    """Per-weight sketch; model parameter names and checkpoint keys stay unchanged."""

    name: str
    module: nn.Module
    weight_name: str
    projection: Tensor
    sketch: Tensor
    handles: list[RemovableHandle]
    swapped_weight: Tensor | None

    @property
    def weight(self) -> nn.Parameter:
        weight = self.module._parameters[self.weight_name]
        if weight is None:
            raise RuntimeError(f"Selected parameter disappeared: {self.name}")
        return weight

    def forward_hook(
        self, module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
    ) -> Tensor | None:
        if not torch.is_grad_enabled() or not output.requires_grad:
            return None
        with torch.no_grad():
            activation = inputs[0]
            projected = (
                activation.reshape(-1, activation.shape[-1])
                @ self.projection.to(activation.dtype).T
            )
        return _AccumulateSketch.apply(output, projected, self.sketch)

    def expert_pre_hook(self, module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        if not torch.is_grad_enabled():
            return
        if self.swapped_weight is not None:
            raise RuntimeError(f"Reentrant expert forward is unsupported: {self.name}")
        weight = self.weight
        local, offset = local_weight_rows(weight)
        # FSDP's pre-forward hook has already unsharded this module. EP may still
        # expose a DTensor containing only this rank's experts.
        local_shape = (
            weight.to_local().shape if isinstance(weight, DTensor) else weight.shape
        )
        projected = _ProjectWeightGradient.apply(
            local.reshape(local_shape).detach().requires_grad_(True), self, offset
        )
        if isinstance(weight, DTensor):
            projected = DTensor.from_local(
                projected,
                weight.device_mesh,
                weight.placements,
                run_check=False,
                shape=weight.shape,
                stride=weight.stride(),
            )
        self.swapped_weight = weight
        # This temporary non-leaf tensor must retain its autograd projection.
        cast(dict[str, Tensor | None], module._parameters)[self.weight_name] = projected

    def expert_post_hook(
        self, module: nn.Module, inputs: tuple[Tensor, ...], output: Any
    ) -> None:
        if self.swapped_weight is not None:
            cast(dict[str, Tensor | None], module._parameters)[self.weight_name] = (
                self.swapped_weight
            )
            self.swapped_weight = None


def install_sketches(model: nn.Module, config: LoGRAConfig) -> list[SketchState]:
    """Select explicit linear modules and fused expert parameters.

    Linear layers accumulate sketches directly. Fused experts retain their
    existing forward implementation and project the transient backward gradient.
    Hooks run after FSDP unsharding and restore original parameters even when
    checkpoint recomputation stops early. TP and partial expert-row sharding are
    rejected by ``local_weight_rows``.
    """
    targets: list[tuple[str, nn.Module, str, bool]] = []
    for name, module in model.named_modules():
        if config.target_prefixes and not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in config.target_prefixes
        ):
            continue
        if name.rsplit(".", 1)[-1] in config.target_modules:
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"LoGRA requires nn.Linear targets, got {name}: {type(module)}"
                )
            targets.append((name, module, "weight", False))
        for parameter_name in config.target_parameters:
            parameter = module._parameters.get(parameter_name)
            if parameter is None:
                continue
            if parameter.ndim != 3:
                raise ValueError(
                    f"Expected a 3-D fused expert weight: {name}.{parameter_name}"
                )
            targets.append((f"{name}.{parameter_name}", module, parameter_name, True))
    if not targets:
        raise ValueError("LoGRA targets matched no modules or parameters")
    # Validate all targets before mutating the model.
    for _, module, weight_name, _ in targets:
        weight = module._parameters[weight_name]
        if weight is None:
            raise ValueError(f"Selected parameter is empty: {weight_name}")
        local_weight_rows(weight)
    states = []
    for name, module, weight_name, fused in targets:
        weight = module._parameters[weight_name]
        assert weight is not None  # Validated above before installing any hooks.
        projection = make_projection(
            config.rank,
            weight.shape[-1],
            projection_seed(config.seed, 0, name),
            device=weight.device,
            dtype=torch.float32,
            distribution=config.distribution,
        )
        state = SketchState(
            name,
            module,
            weight_name,
            projection,
            torch.zeros(
                math.prod(weight.shape[:-1]),
                config.rank,
                device=weight.device,
                dtype=torch.float32,
            ),
            [],
            None,
        )
        weight.requires_grad_(False)
        if fused:
            state.handles = [
                module.register_forward_pre_hook(state.expert_pre_hook),
                module.register_forward_hook(state.expert_post_hook, always_call=True),
            ]
        else:
            state.handles = [module.register_forward_hook(state.forward_hook)]
        states.append(state)
    return states


def local_weight_rows(weight: Tensor) -> tuple[Tensor, int]:
    """Return the local weight and global row offset, rejecting other sharding."""
    if not isinstance(weight, DTensor):
        return weight.reshape(-1, weight.shape[-1]) if weight.ndim > 2 else weight, 0
    if any(isinstance(p, Shard) and p.dim != 0 for p in weight.placements):
        raise ValueError("LoGRA supports FSDP row sharding, not tensor parallelism")
    # PyTorch's helper accounts for uneven and empty FSDP shards.
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _, offset = compute_local_shape_and_global_offset(
        weight.shape, weight.device_mesh, weight.placements
    )
    local = weight.to_local()
    if weight.ndim > 2:
        return local.reshape(-1, weight.shape[-1]), offset[0] * math.prod(
            weight.shape[1:-1]
        )
    return local, offset[0]
