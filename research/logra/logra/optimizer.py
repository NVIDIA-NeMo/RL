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

"""Optimize sketches while delegating ordinary parameters to native AdamW."""

from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn

from logra.compression import (
    install_sketches,
    local_weight_rows,
    make_projection,
    projection_seed,
)
from logra.config import LoGRAConfig
from logra.row_adam import row_adam_direction


class LoGRAOptimizer(torch.optim.Optimizer):
    """RowAdam or SGD on sketches, with a shared learning-rate schedule.

    Construct before the native optimizer's first step. Its target parameters
    are removed before Adam can allocate full-size moment buffers.
    """

    def __init__(
        self, model: nn.Module, native: torch.optim.Optimizer, config: LoGRAConfig
    ):
        if type(native) is not torch.optim.AdamW or native.state:
            raise ValueError(
                "LoGRA currently requires a fresh native torch.optim.AdamW"
            )
        if len(native.param_groups) != 1:
            raise ValueError(
                "Initial LoGRA support requires one native parameter group"
            )
        self.config = config
        self.layers = install_sketches(model, config)
        targets = {id(layer.module.weight) for layer in self.layers}
        group = native.param_groups[0]
        if config.freeze_non_target:
            for parameter in model.parameters():
                if id(parameter) not in targets:
                    parameter.requires_grad_(False)
        group["params"] = [
            p for p in group["params"] if id(p) not in targets and p.requires_grad
        ]
        self.native = native
        self.anchor_handle = (
            cast(Any, model).get_input_embeddings().register_forward_hook(self._anchor)
            if hasattr(model, "get_input_embeddings")
            else None
        )
        self.update_count = 0
        self.ready = False
        super().__init__(
            [layer.module.weight for layer in self.layers],
            dict(lr=group["lr"], weight_decay=group["weight_decay"]),
        )
        for layer in self.layers:
            if config.optimizer == "row_adam":
                self.state[layer.module.weight]["row_second_moment"] = torch.zeros(
                    layer.module.out_features,
                    1,
                    device=layer.sketch.device,
                    dtype=torch.float32,
                )
        # The scheduler sees both groups; native AdamW retains its own parameter list.
        self.param_groups.extend(native.param_groups)

    @staticmethod
    def _anchor(module, inputs, output):
        if torch.is_grad_enabled():
            output.requires_grad_(True)
        return output

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        self.native.zero_grad(set_to_none=set_to_none)
        for layer in self.layers:
            layer.sketch.zero_()
        self.ready = False

    @torch.no_grad()
    def synchronize_and_clip(self, process_group, max_norm):
        """Match FSDP averaging, then clip the reconstructed gradient globally."""
        if self.ready:
            raise RuntimeError("Sketches have already been synchronized")
        world = dist.get_world_size(process_group) if dist.is_initialized() else 1
        norm2 = torch.zeros(
            (), device=self.layers[0].sketch.device, dtype=torch.float64
        )
        for layer in self.layers:
            if world > 1:
                dist.all_reduce(layer.sketch, group=process_group)
                layer.sketch.div_(world)
            local, offset = local_weight_rows(layer.module.weight)
            # Bounded workspace rather than a full reconstructed gradient.
            for start in range(0, local.shape[0], 256):
                rows = layer.sketch[
                    offset + start : offset + min(start + 256, local.shape[0])
                ]
                reconstructed = rows @ layer.projection
                norm2.add_(reconstructed.double().square().sum())
        for group in self.native.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    grad, _ = local_weight_rows(parameter.grad)
                    norm2.add_(grad.double().square().sum())
        if world > 1:
            dist.all_reduce(norm2, group=process_group)
        norm = norm2.sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite LoGRA gradient")
        if max_norm is not None:
            scale = (max_norm / (norm + 1e-6)).clamp(max=1).float()
            for layer in self.layers:
                layer.sketch.mul_(scale)
            for group in self.native.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
        self.ready = True
        return norm.item()

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        if closure is not None or not self.ready:
            raise RuntimeError(
                "Call synchronize_and_clip before step; closures are unsupported"
            )
        group = self.param_groups[0]
        for layer in self.layers:
            direction = layer.sketch
            if self.config.optimizer == "row_adam":
                moment = self.state[layer.module.weight]["row_second_moment"]
                direction = row_adam_direction(
                    direction,
                    moment,
                    step=self.update_count + 1,
                    beta2=self.config.beta2,
                    epsilon=self.config.epsilon,
                )
            local, offset = local_weight_rows(layer.module.weight)
            local.mul_(1 - group["lr"] * group["weight_decay"])
            for start in range(0, local.shape[0], 256):
                stop = min(start + 256, local.shape[0])
                local[start:stop].add_(
                    direction[offset + start : offset + stop] @ layer.projection,
                    alpha=-group["lr"],
                )
        self.native.step()
        self.update_count += 1
        self.ready = False
        self.refresh_projection()

    def refresh_projection(self):
        update = self.update_count if self.config.refresh else 0
        for layer in self.layers:
            layer.projection.copy_(
                make_projection(
                    self.config.rank,
                    layer.module.in_features,
                    projection_seed(self.config.seed, update, layer.name),
                    device=layer.projection.device,
                    dtype=layer.projection.dtype,
                    distribution=self.config.distribution,
                )
            )

    def state_dict(self) -> dict[str, Any]:
        return dict(
            sketch_optimizer=super().state_dict(),
            native=self.native.state_dict(),
            update_count=self.update_count,
            config=self.config.model_dump(),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = state_dict
        if state["config"] != self.config.model_dump():
            raise ValueError("LoGRA checkpoint configuration differs")
        super().load_state_dict(state["sketch_optimizer"])
        self.native.load_state_dict(state["native"])
        self.param_groups[1:] = self.native.param_groups
        self.update_count = state["update_count"]
        self.refresh_projection()
