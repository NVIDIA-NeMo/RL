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
    fill_projection_,
    install_sketches,
    local_weight_rows,
    projection_seed,
    sketch_buffer,
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
        self.projections_verified = False
        # Every sketch is a row block of this buffer: one reduction, one scale,
        # one zero per step regardless of the number of target layers.
        self.sketches = sketch_buffer(self.layers)
        # Shard geometry is fixed for the model's lifetime; resolve it once.
        self.local_rows = [
            local_weight_rows(layer.module.weight)[1] for layer in self.layers
        ]
        super().__init__(
            [layer.module.weight for layer in self.layers],
            dict(lr=group["lr"], weight_decay=group["weight_decay"]),
        )
        self.row_second_moment: torch.Tensor | None = None
        if config.optimizer == "row_adam":
            # One moment per sketch row, laid out like the sketch buffer so the
            # whole RowAdam update is a handful of row-wise kernels.
            self.row_second_moment = torch.zeros(
                self.sketches.shape[0], 1, device=self.sketches.device
            )
            self._link_row_moments()
        # The scheduler sees both groups; native AdamW retains its own parameter list.
        self.param_groups.extend(native.param_groups)

    def _link_row_moments(self) -> None:
        """Expose each layer's slice of the flat moments through the optimizer state."""
        assert self.row_second_moment is not None
        row = 0
        for layer in self.layers:
            rows = layer.module.out_features
            self.state[layer.module.weight]["row_second_moment"] = (
                self.row_second_moment[row : row + rows]
            )
            row += rows

    @staticmethod
    def _anchor(module, inputs, output):
        if torch.is_grad_enabled():
            output.requires_grad_(True)
        return output

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        self.native.zero_grad(set_to_none=set_to_none)
        self.sketches.zero_()
        self.ready = False

    @torch.no_grad()
    def synchronize_and_clip(self, process_group, max_norm):
        """Match FSDP averaging, then clip the reconstructed gradient globally."""
        if self.ready:
            raise RuntimeError("Sketches have already been synchronized")
        world = dist.get_world_size(process_group) if dist.is_initialized() else 1
        if world > 1 and not self.projections_verified:
            self._verify_shared_projections(process_group)
        if world > 1:
            dist.all_reduce(self.sketches, group=process_group)
            self.sketches.div_(world)
        # ||S A||^2 = <S (A A^T), S>: a rank x rank Gram matrix per layer instead
        # of materialising the reconstructed gradient row block by row block.
        partial = []
        for layer, offset in zip(self.layers, self.local_rows):
            local, _ = local_weight_rows(layer.module.weight)
            rows = layer.sketch[offset : offset + local.shape[0]]
            gram = layer.projection @ layer.projection.T
            partial.append(((rows @ gram) * rows).sum())
        native_grads = [
            local_weight_rows(parameter.grad)[0]
            for group in self.native.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if native_grads:
            partial.extend(torch._foreach_norm(native_grads))
            partial[len(partial) - len(native_grads) :] = [
                n.square() for n in partial[len(partial) - len(native_grads) :]
            ]
        norm2 = torch.stack(partial).sum()
        if world > 1:
            dist.all_reduce(norm2, group=process_group)
        norm = norm2.sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite LoGRA gradient")
        if max_norm is not None:
            scale = (max_norm / (norm + 1e-6)).clamp(max=1)
            self.sketches.mul_(scale)
            if native_grads:
                torch._foreach_mul_(native_grads, scale)
        self.ready = True
        return norm.item()

    @torch.no_grad()
    def _verify_shared_projections(self, process_group):
        """Fail loudly if ranks drew different projections from the shared seed.

        Projections are drawn on each device from the same seed, which yields
        identical matrices on identical hardware. A heterogeneous data-parallel
        group would otherwise average sketches taken against different bases
        and silently corrupt every update.
        """
        fingerprint = torch.stack(
            [layer.projection.double().sum() for layer in self.layers]
        )
        lowest, highest = fingerprint.clone(), fingerprint.clone()
        dist.all_reduce(lowest, op=dist.ReduceOp.MIN, group=process_group)
        dist.all_reduce(highest, op=dist.ReduceOp.MAX, group=process_group)
        if not torch.equal(lowest, highest):
            raise RuntimeError(
                "LoGRA projections differ across data-parallel ranks; the ranks "
                "must run identical hardware so a shared seed draws one matrix"
            )
        self.projections_verified = True

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        if closure is not None or not self.ready:
            raise RuntimeError(
                "Call synchronize_and_clip before step; closures are unsupported"
            )
        group = self.param_groups[0]
        if self.row_second_moment is not None:
            # The sketch buffer is not needed after this step, so the direction
            # overwrites it: one row-wise pass over every layer at once.
            row_adam_direction(
                self.sketches,
                self.row_second_moment,
                step=self.update_count + 1,
                beta2=self.config.beta2,
                epsilon=self.config.epsilon,
                inplace=True,
            )
        decay = 1 - group["lr"] * group["weight_decay"]
        for layer, offset in zip(self.layers, self.local_rows):
            local, _ = local_weight_rows(layer.module.weight)
            # W <- decay * W - lr * D_local @ A, written straight into the shard.
            local.addmm_(
                layer.sketch[offset : offset + local.shape[0]],
                layer.projection,
                beta=decay,
                alpha=-group["lr"],
            )
        self.native.step()
        self.update_count += 1
        self.ready = False
        self.refresh_projection()

    def refresh_projection(self):
        update = self.update_count if self.config.refresh else 0
        for layer in self.layers:
            fill_projection_(
                layer.projection,
                projection_seed(self.config.seed, update, layer.name),
                distribution=self.config.distribution,
            )
            layer.projection_cast = None

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
        if self.row_second_moment is not None:
            # torch.optim replaces state tensors with copies; move the restored
            # moments back into the flat buffer and re-expose the views.
            for layer in self.layers:
                restored = self.state[layer.module.weight]["row_second_moment"]
                row = sum(
                    other.module.out_features
                    for other in self.layers[: self.layers.index(layer)]
                )
                self.row_second_moment[row : row + restored.shape[0]].copy_(restored)
            self._link_row_moments()
        self.native.load_state_dict(state["native"])
        self.param_groups[1:] = self.native.param_groups
        self.update_count = state["update_count"]
        self.refresh_projection()
