# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading already-resharded BF16 experts into active checkpoint storage."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

ExpertProjection = Literal["gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class LocalExpertBinding:
    target_name: str
    projection: ExpertProjection
    local_shape: tuple[int, int, int]


class LocalBf16ExpertReload:
    """One update's local loads beneath vLLM's native online wrapper.

    Construct before native reload initialization; load after initialization;
    require all components before finalization, then verify saved runtime
    storage. This object never initializes or finalizes the model itself.
    """

    def __init__(
        self, model: torch.nn.Module, bindings: Mapping[str, LocalExpertBinding]
    ) -> None:
        self._model = model
        self._bindings = dict(bindings)
        self._loaded: set[str] = set()
        self._bridged_ids: set[int] = set()
        self._failure: BaseException | None = None
        targets = {
            (binding.target_name, binding.projection) for binding in bindings.values()
        }
        if len(targets) != len(bindings):
            raise ValueError("Duplicate local expert destination projections")
        owners: dict[str, dict[ExpertProjection, LocalExpertBinding]] = {}
        for binding in bindings.values():
            owner_name, _ = binding.target_name.rsplit(".", 1)
            owner_bindings = owners.setdefault(owner_name, {})
            if binding.projection in owner_bindings:
                raise ValueError("Repeated projection in one expert module")
            owner_bindings[binding.projection] = binding
        for owner_name, components in owners.items():
            owner = model.get_submodule(owner_name)
            gated = getattr(getattr(owner, "moe_config", None), "is_act_and_mul", None)
            if not isinstance(gated, bool):
                raise ValueError(
                    "Expert module does not declare gated checkpoint layout"
                )
            required = {"up_proj", "down_proj"}
            if gated:
                required.add("gate_proj")
            if set(components) != required:
                raise ValueError(
                    f"Incomplete expert projection plan for {owner_name!r}"
                )
            up, down = components["up_proj"], components["down_proj"]
            experts, intermediate, hidden = up.local_shape
            if min(up.local_shape) <= 0 or down.local_shape != (
                experts,
                hidden,
                intermediate,
            ):
                raise ValueError("Expert projection shapes disagree in the refit plan")
            if gated and (
                components["gate_proj"].local_shape != up.local_shape
                or components["gate_proj"].target_name != up.target_name
            ):
                raise ValueError(
                    "Gate and up must describe the same fused checkpoint target"
                )
            if up.target_name == down.target_name:
                raise ValueError("Up and down require different checkpoint targets")
        self._runtime = {
            binding.target_name: model.get_parameter(binding.target_name)
            for binding in bindings.values()
        }
        self._storage = {
            name: (param.data_ptr(), tuple(param.shape), param.dtype)
            for name, param in self._runtime.items()
        }
        if any(param.dtype != torch.bfloat16 for param in self._runtime.values()):
            raise ValueError(
                "Local BF16 expert reload requires BF16 runtime parameters"
            )

    def _require_usable(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "Local BF16 expert reload is unusable after failure"
            ) from self._failure

    def load(self, logical_name: str, weight: torch.Tensor) -> None:
        self._require_usable()
        try:
            if logical_name in self._loaded:
                raise ValueError(f"Duplicate local expert component {logical_name!r}")
            binding = self._bindings[logical_name]
            if tuple(weight.shape) != binding.local_shape:
                raise ValueError(
                    f"Local expert shape differs from refit plan for {logical_name!r}"
                )
            # Initialization replaces runtime parameters with checkpoint tensors.
            target = self._model.get_parameter(binding.target_name)
            if target is self._runtime[binding.target_name]:
                raise RuntimeError(
                    "Local expert load requires active checkpoint storage"
                )
            owner_name, parameter_name = binding.target_name.rsplit(".", 1)
            owner = self._model.get_submodule(owner_name)
            gated = getattr(getattr(owner, "moe_config", None), "is_act_and_mul", None)
            if not isinstance(gated, bool):
                raise ValueError(
                    "Expert module does not declare gated checkpoint layout"
                )
            if id(target) not in self._bridged_ids:
                self._install_bridge(owner, parameter_name, target, gated)
                self._bridged_ids.add(id(target))
            # Clone outside the counted loader; vLLM may defer the real copy.
            owned_weight = weight.detach().clone()
            target.weight_loader(target, owned_weight, projection=binding.projection)
            self._loaded.add(logical_name)
        except BaseException as error:
            self._failure = error
            raise

    @staticmethod
    def _install_bridge(
        owner: torch.nn.Module, parameter_name: str, target: torch.Tensor, gated: bool
    ) -> None:
        from vllm.model_executor.model_loader.reload.layerwise import (
            make_online_process_loader,
        )

        wrapped = getattr(target, "weight_loader", None)
        if (
            not callable(wrapped)
            or getattr(wrapped, "__name__", None) != "online_process_loader"
        ):
            raise RuntimeError("Local expert checkpoint parameter has no online loader")

        def local_loader(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            *,
            projection: ExpertProjection,
        ) -> None:
            copy_local_bf16_expert(
                param, loaded_weight, projection=projection, gated=gated
            )

        target.weight_loader = local_loader
        try:
            target.weight_loader = make_online_process_loader(owner, parameter_name)
        except BaseException:
            target.weight_loader = wrapped
            raise

    def require_complete(self) -> None:
        self._require_usable()
        missing = self._bindings.keys() - self._loaded
        if missing:
            error = RuntimeError(
                f"Missing local expert refit components: {sorted(missing)!r}"
            )
            self._failure = error
            raise error

    def verify_runtime_storage(self) -> None:
        self.require_complete()
        for name, original in self._runtime.items():
            current = self._model.get_parameter(name)
            if (
                current is not original
                or (current.data_ptr(), tuple(current.shape), current.dtype)
                != self._storage[name]
            ):
                error = RuntimeError(
                    f"Refit changed runtime expert storage for {name!r}"
                )
                self._failure = error
                raise error


def copy_local_bf16_expert(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    projection: ExpertProjection,
    gated: bool,
) -> None:
    """Copy one grouped local projection; leave other projections untouched.

    Both tensors use checkpoint orientation, never TRTLLM packed orientation.
    Reshard has already selected TP and EP regions. Zeroing and one payload
    copy are deliberately inside this loader so both run on real replay after
    vLLM's meta accounting pass, without counting padding as extra copies.
    """
    if param.dtype != torch.bfloat16 or loaded_weight.dtype != torch.bfloat16:
        raise ValueError("Local BF16 expert loading requires BF16 tensors")
    if param.ndim != 3 or loaded_weight.ndim != 3:
        raise ValueError("Local BF16 expert loading requires 3D checkpoint tensors")
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError(f"Unsupported expert projection: {projection!r}")
    if projection == "gate_proj" and not gated:
        raise ValueError("A non-gated expert has no gate projection")
    if param.shape[0] != loaded_weight.shape[0]:
        raise ValueError("Local expert count does not match checkpoint storage")
    destination = param.data
    if gated and projection != "down_proj":
        if param.shape[1] % 2:
            raise ValueError("Gated w13 checkpoint width must have two equal halves")
        half = param.shape[1] // 2
        start = half if projection == "up_proj" else 0
        destination = destination[:, start : start + half, :]
    if any(size <= 0 for size in loaded_weight.shape):
        raise ValueError("Local expert payload dimensions must be positive")
    if any(
        received > available
        for received, available in zip(
            loaded_weight.shape, destination.shape, strict=True
        )
    ):
        raise ValueError(
            f"Local expert payload {tuple(loaded_weight.shape)} exceeds "
            f"checkpoint block {tuple(destination.shape)}"
        )
    destination.zero_()
    destination[:, : loaded_weight.shape[1], : loaded_weight.shape[2]].copy_(
        loaded_weight
    )
