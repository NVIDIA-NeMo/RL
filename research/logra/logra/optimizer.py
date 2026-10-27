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

from collections.abc import Iterator
from contextlib import contextmanager
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
        targets = {id(layer.weight) for layer in self.layers}
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
        self.local_rows = [local_weight_rows(layer.weight)[1] for layer in self.layers]
        super().__init__(
            [layer.weight for layer in self.layers],
            dict(lr=group["lr"], weight_decay=group["weight_decay"]),
        )
        self.row_second_moment: torch.Tensor | None = None
        if config.optimizer == "row_adam":
            # One moment per sketch row, laid out like the sketch buffer so the
            # whole RowAdam update is a handful of row-wise kernels.
            self.row_second_moment = torch.zeros(
                self.sketches.shape[0],
                1,
                device=self.sketches.device,
                dtype=torch.float32,
            )
            self._link_row_moments()
        for layer in self.layers:
            # Small CUA steps must accumulate in FP32 before rounding to BF16.
            if layer.weight.dtype != torch.float32:
                local, _ = local_weight_rows(layer.weight)
                self.state[layer.weight]["master_weight"] = local.float().clone()
        # The scheduler sees both groups; native AdamW retains its own parameter list.
        self.param_groups.extend(native.param_groups)

    def _link_row_moments(self) -> None:
        """Expose each layer's slice of the flat moments through the optimizer state."""
        moments = self.row_second_moment
        assert moments is not None
        row = 0
        for layer in self.layers:
            rows = layer.sketch.shape[0]
            self.state[layer.weight]["row_second_moment"] = moments[row : row + rows]
            row += rows

    @torch.no_grad()
    def move_row_moments(self, device: torch.device | str) -> None:
        """Move the flat moments and re-expose their per-layer views.

        Generic optimizer offload replaces each state tensor with an independent
        copy, which would detach the exposed views from the buffer RowAdam updates.
        """
        if self.row_second_moment is not None:
            self.row_second_moment = self.row_second_moment.to(device)
            self._link_row_moments()

    @staticmethod
    def _anchor(module, inputs, output):
        if torch.is_grad_enabled():
            output.requires_grad_(True)
            # Multimodal models may replace image positions in-place. A frozen
            # embedding needs a non-leaf output so that replacement remains legal.
            if output.is_leaf:
                output = output.clone()
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
        for layer, offset in zip(self.layers, self.local_rows, strict=True):
            local, _ = local_weight_rows(layer.weight)
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
            # Accumulate in FP32 even for BF16 gradients.
            norms = torch._foreach_norm(native_grads, 2, dtype=torch.float32)
            partial.extend(norm.square() for norm in norms)
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
    def step(self, closure=None, *, scale: float = 1.0) -> Any:
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
        alpha = -group["lr"] * scale
        for layer, offset in zip(self.layers, self.local_rows, strict=True):
            local, _ = local_weight_rows(layer.weight)
            master = self.state[layer.weight].get("master_weight")
            # W <- decay * W - lr * scale * D_local @ A in one GEMM. Low-precision
            # targets update their FP32 master, then round into the shard.
            destination = local if master is None else master
            destination.addmm_(
                layer.sketch[offset : offset + local.shape[0]],
                layer.projection,
                beta=decay,
                alpha=alpha,
            )
            if master is not None:
                local.copy_(master)
        self.native.step()
        self.update_count += 1
        self.ready = False
        self.refresh_projection()

    @contextmanager
    def trial_step(self, *, scale: float) -> Iterator[None]:
        """Temporarily render a scaled SGD proposal without changing optimizer state.

        BF16 weights restore from their unchanged FP32 masters. FP32 targets use
        a temporary backup. The context also restores weights if the probe fails.
        """
        if not self.ready or self.config.optimizer != "sgd":
            raise ValueError("A trial step requires synchronized SGD sketches")
        group = self.param_groups[0]
        if group["weight_decay"] or any(g["params"] for g in self.native.param_groups):
            raise ValueError(
                "A trial step requires zero decay and frozen non-target weights"
            )
        originals = []
        with torch.no_grad():
            try:
                for layer, offset in zip(self.layers, self.local_rows, strict=True):
                    local, _ = local_weight_rows(layer.weight)
                    master = self.state[layer.weight].get("master_weight")
                    original = local.clone() if master is None else master
                    originals.append((layer, original))
                    rows = layer.sketch[offset : offset + local.shape[0]]
                    alpha = -group["lr"] * scale
                    if master is None:
                        # The backup is separate storage, so write straight into
                        # the shard without a temporary.
                        torch.addmm(
                            original, rows, layer.projection, alpha=alpha, out=local
                        )
                    else:
                        # One FP32 temporary per layer, rounded into the shard.
                        local.copy_(
                            torch.addmm(original, rows, layer.projection, alpha=alpha)
                        )
                yield
            finally:
                for layer, original in originals:
                    local, _ = local_weight_rows(layer.weight)
                    local.copy_(original)

    @torch.no_grad()
    def reset_master_weights(self) -> None:
        """Rebase masters after loading model weights without optimizer state."""
        for layer in self.layers:
            master = self.state[layer.weight].get("master_weight")
            if master is not None:
                local, _ = local_weight_rows(layer.weight)
                master.copy_(local)

    def refresh_projection(self):
        update = self.update_count if self.config.refresh else 0
        for layer in self.layers:
            fill_projection_(
                layer.projection,
                projection_seed(self.config.seed, update, layer.name),
                distribution=self.config.distribution,
            )
            layer.refresh_cast()

    def state_dict(self) -> dict[str, Any]:
        return dict(
            sketch_optimizer=super().state_dict(),
            native=self.native.state_dict(),
            update_count=self.update_count,
            config=self.config.model_dump(),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = state_dict
        if LoGRAConfig.model_validate(state["config"]) != self.config:
            raise ValueError("LoGRA checkpoint configuration differs")
        # torch Optimizer.load_state_dict casts floating states to parameter dtype.
        # Restore our FP32 masters/moments explicitly so BF16 resume is lossless.
        saved_optimizer = state["sketch_optimizer"]
        super().load_state_dict({**saved_optimizer, "state": {}})
        saved_ids = saved_optimizer["param_groups"][0]["params"]
        for layer, saved_id in zip(self.layers, saved_ids, strict=True):
            self.state[layer.weight] = {
                name: value.to(
                    device=layer.weight.device, dtype=torch.float32, copy=True
                )
                for name, value in saved_optimizer["state"].get(saved_id, {}).items()
            }
        moments = self.row_second_moment
        if moments is not None:
            # Copy the restored per-layer moments back into the flat buffer that
            # RowAdam updates, then re-expose the views. Masters stay per layer.
            row = 0
            for layer in self.layers:
                rows = layer.sketch.shape[0]
                restored = self.state[layer.weight].get("row_second_moment")
                if restored is None:
                    moments[row : row + rows].zero_()
                else:
                    moments[row : row + rows].copy_(restored)
                row += rows
            self._link_row_moments()
        self.native.load_state_dict(state["native"])
        self.param_groups[1:] = self.native.param_groups
        self.update_count = state["update_count"]
        self.refresh_projection()
