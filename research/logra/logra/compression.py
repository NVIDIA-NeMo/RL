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

"""Direct gradient sketches for dense linear layers; no full weight gradient."""

import hashlib
import math
from dataclasses import dataclass
from typing import Any

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


@dataclass
class SketchState:
    """Per-layer state stored outside autograd and explicitly reduced by the trainer."""

    name: str
    module: nn.Linear
    projection: Tensor
    sketch: Tensor
    handle: RemovableHandle | None

    def forward_hook(
        self, module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
    ) -> Tensor | None:
        if not torch.is_grad_enabled() or not output.requires_grad:
            return
        # Autograd owns projected activation storage. Checkpoint replay must
        # recreate the same saved tensors; only the original graph runs backward.
        with torch.no_grad():
            activation = inputs[0]
            projected = (
                activation.reshape(-1, activation.shape[-1])
                @ self.projection.to(activation.dtype).T
            )

        return _AccumulateSketch.apply(output, projected, self.sketch)


def install_sketches(model: nn.Module, config: LoGRAConfig) -> list[SketchState]:
    """Install sketches on explicit linear targets, leaving the model state_dict clean.

    Supported initial scope is dense Hugging Face linear layers and non-reentrant
    activation checkpointing. Call before freezing unrelated parameters. A frozen
    embedding output must be made differentiable by the training integration.
    """
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in config.target_modules
    ]
    if not selected:
        raise ValueError("LoGRA target_modules matched no modules")
    for name, module in selected:
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"LoGRA requires nn.Linear targets, got {name}: {type(module)}"
            )
    states = []
    for name, module in selected:
        weight = module.weight
        projection = make_projection(
            config.rank,
            module.in_features,
            projection_seed(config.seed, 0, name),
            device=weight.device,
            dtype=torch.float32,
            distribution=config.distribution,
        )
        state = SketchState(
            name,
            module,
            projection,
            torch.zeros(
                module.out_features,
                config.rank,
                device=weight.device,
                dtype=torch.float32,
            ),
            None,
        )
        weight.requires_grad_(False)
        state.handle = module.register_forward_hook(state.forward_hook)
        states.append(state)
    return states


def local_weight_rows(weight: Tensor) -> tuple[Tensor, int]:
    """Return the local weight and global row offset, rejecting other sharding."""
    if not isinstance(weight, DTensor):
        return weight, 0
    if any(isinstance(p, Shard) and p.dim != 0 for p in weight.placements):
        raise ValueError("LoGRA supports FSDP row sharding, not tensor parallelism")
    # PyTorch's helper accounts for uneven and empty FSDP shards.
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _, offset = compute_local_shape_and_global_offset(
        weight.shape, weight.device_mesh, weight.placements
    )
    return weight.to_local(), offset[0]
