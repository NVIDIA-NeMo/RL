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

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_FLEXTRON_MLP_LAYER_TYPES = frozenset(("E",))
_FLEXTRON_EMB_LAYER_TYPES = frozenset(("M", "E", "*"))
_FLEXTRON_HIDDEN_MODULE_CLASS_NAMES = frozenset(
    ("MambaMixer", "MoELayer", "SelfAttention", "TEGroupedMLP")
)
_FLEXTRON_HIDDEN_MODULE_NAMES = frozenset(
    ("attention", "mamba_mixer", "mixer", "mlp", "self_attention")
)


@dataclass(frozen=True)
class _LayerRouteIndex:
    emb_idx: int | None
    mlp_idx: int | None


class FrozenFlextronRouter:
    """Apply deterministic Flextron masks without using a trainable router."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        model_cfg: Any,
    ) -> None:
        self.model = model
        self.model_cfg = model_cfg
        self.flex_routers = list(getattr(model_cfg, "flex_routers", []))
        self.flextron_sampling_rates = list(
            getattr(model_cfg, "flextron_sampling_rates", [])
        )
        self.active_router_id: int | None = None
        self._handles: list[RemovableHandle] = []
        self._mask_cache: dict[
            tuple[torch.device, torch.dtype, int, int], torch.Tensor
        ] = {}
        self._route_index_by_global_layer = self._build_route_index_by_global_layer()
        self._model_with_decoder = self._unwrap_model_with_decoder(model)

        if self.enabled:
            self._attach_hooks()

    @property
    def enabled(self) -> bool:
        return bool(self.flex_routers and self.flextron_sampling_rates)

    @property
    def has_nested_sampling(self) -> bool:
        return any(rate > 0 for rate in self.flextron_sampling_rates[1:])

    def sample_router_ids(
        self, *, batch_size: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Sample route ids from configured `[base, *routers]` probabilities."""
        if not self.enabled:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        rates = torch.tensor(
            self.flextron_sampling_rates, dtype=torch.float32, device=device
        )
        probs = rates / rates.sum()
        return torch.multinomial(probs, num_samples=batch_size, replacement=True)

    def get_router_ids(
        self, data: BatchedDataDict[Any], *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Return explicit route ids from data or sample new route ids."""
        if "flex_router_ids" in data:
            return data["flex_router_ids"].to(device=device, dtype=torch.long)
        return self.sample_router_ids(batch_size=data.size, device=device)

    def grouped_indices(self, router_ids: torch.Tensor) -> dict[int, list[int]]:
        """Group batch indices by route id."""
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, router_id in enumerate(router_ids.detach().cpu().tolist()):
            self._validate_router_id(router_id)
            groups[int(router_id)].append(idx)
        return dict(groups)

    @contextmanager
    def use_router(self, router_id: int | None) -> Iterator[None]:
        """Temporarily activate one deterministic route for model forwards."""
        if router_id is not None:
            self._validate_router_id(router_id)
        previous_router_id = self.active_router_id
        self.active_router_id = router_id
        try:
            yield
        finally:
            self.active_router_id = previous_router_id

    def clear_router(self) -> None:
        self.active_router_id = None

    def detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _validate_router_id(self, router_id: int) -> None:
        if router_id < 0 or router_id > len(self.flex_routers):
            raise ValueError(
                f"Flextron router id must be in [0, {len(self.flex_routers)}]; "
                f"got {router_id}."
            )

    def _build_route_index_by_global_layer(self) -> list[_LayerRouteIndex]:
        pattern = self._main_layer_pattern()
        route_indices = []
        emb_idx = 0
        mlp_idx = 0
        for layer_type in pattern:
            current_emb_idx = None
            current_mlp_idx = None
            if layer_type in _FLEXTRON_EMB_LAYER_TYPES:
                current_emb_idx = emb_idx
                emb_idx += 1
            if layer_type in _FLEXTRON_MLP_LAYER_TYPES:
                current_mlp_idx = mlp_idx
                mlp_idx += 1
            route_indices.append(
                _LayerRouteIndex(emb_idx=current_emb_idx, mlp_idx=current_mlp_idx)
            )
        return route_indices

    def _main_layer_pattern(self) -> str:
        pattern = getattr(self.model_cfg, "hybrid_layer_pattern", None)
        if not pattern:
            return ""
        return pattern.split("/", maxsplit=1)[0].replace("|", "")

    def _unwrap_model_with_decoder(
        self, model: torch.nn.Module
    ) -> torch.nn.Module | None:
        module = model
        visited = set()
        while id(module) not in visited:
            visited.add(id(module))
            if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
                return module
            for attr_name in ("module", "language_model", "thinker"):
                child = getattr(module, attr_name, None)
                if child is not None and child is not module:
                    module = child
                    break
            else:
                return None
        return None

    def _attach_hooks(self) -> None:
        if self._model_with_decoder is None:
            return

        for local_idx, layer in enumerate(self._model_with_decoder.decoder.layers):
            global_idx = self._global_layer_idx(layer, local_idx)
            if global_idx >= len(self._route_index_by_global_layer):
                continue

            for hidden_module in self._iter_hidden_mask_modules(layer):
                self._handles.append(
                    hidden_module.register_forward_pre_hook(
                        self._make_hidden_pre_hook(global_idx)
                    )
                )
                self._handles.append(
                    hidden_module.register_forward_hook(
                        self._make_hidden_post_hook(global_idx)
                    )
                )

            for linear_fc1 in self._iter_linear_fc1_modules(layer):
                self._handles.append(
                    linear_fc1.register_forward_hook(
                        self._make_mlp_post_hook(global_idx)
                    )
                )

        final_norm = getattr(self._model_with_decoder.decoder, "final_norm", None)
        if final_norm is not None:
            self._handles.append(
                final_norm.register_forward_pre_hook(self._final_norm_pre_hook)
            )
            self._handles.append(
                final_norm.register_forward_hook(self._final_norm_post_hook)
            )

    def _iter_hidden_mask_modules(
        self, layer: torch.nn.Module
    ) -> Iterator[torch.nn.Module]:
        yielded_ids = set()
        yielded_ids.add(id(layer))
        yield layer

        for module_name, module in layer.named_modules():
            if module_name == "":
                continue
            if id(module) in yielded_ids:
                continue
            class_name = module.__class__.__name__
            leaf_name = module_name.rsplit(".", maxsplit=1)[-1]
            if (
                class_name in _FLEXTRON_HIDDEN_MODULE_CLASS_NAMES
                or leaf_name in _FLEXTRON_HIDDEN_MODULE_NAMES
            ):
                yielded_ids.add(id(module))
                yield module

    def _iter_linear_fc1_modules(
        self, layer: torch.nn.Module
    ) -> Iterator[torch.nn.Module]:
        yielded_ids = set()
        for module in layer.modules():
            linear_fc1 = getattr(module, "linear_fc1", None)
            if linear_fc1 is None or id(linear_fc1) in yielded_ids:
                continue
            yielded_ids.add(id(linear_fc1))
            yield linear_fc1

    def _global_layer_idx(self, layer: torch.nn.Module, local_idx: int) -> int:
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1
        return local_idx

    def _make_hidden_pre_hook(self, global_layer_idx: int):
        def hook(module, inputs):
            emb_int = self._active_emb_int(global_layer_idx)
            if emb_int is None or not inputs:
                return inputs
            return self._mask_first_tensor(inputs, emb_int)

        return hook

    def _make_hidden_post_hook(self, global_layer_idx: int):
        def hook(module, inputs, output):
            emb_int = self._active_emb_int(global_layer_idx)
            if emb_int is None:
                return output
            return self._mask_output(output, emb_int)

        return hook

    def _make_mlp_post_hook(self, global_layer_idx: int):
        def hook(module, inputs, output):
            mlp_int = self._active_mlp_int(global_layer_idx)
            if mlp_int is None:
                return output
            return self._mask_output(
                output, mlp_int, full_dim=self.model_cfg.ffn_hidden_size
            )

        return hook

    def _final_norm_pre_hook(self, module, inputs):
        emb_int = self._final_norm_emb_int()
        if emb_int is None or not inputs:
            return inputs
        return self._mask_first_tensor(inputs, emb_int)

    def _final_norm_post_hook(self, module, inputs, output):
        emb_int = self._final_norm_emb_int()
        if emb_int is None:
            return output
        return self._mask_output(output, emb_int)

    def _active_route(self) -> dict[str, list[int]] | None:
        if self.active_router_id is None or self.active_router_id == 0:
            return None
        return self.flex_routers[self.active_router_id - 1]

    def _active_emb_int(self, global_layer_idx: int) -> int | None:
        route = self._active_route()
        if route is None:
            return None
        route_index = self._route_index_by_global_layer[global_layer_idx]
        if route_index.emb_idx is None:
            return None
        return route["emb_int_list"][route_index.emb_idx]

    def _active_mlp_int(self, global_layer_idx: int) -> int | None:
        route = self._active_route()
        if route is None:
            return None
        route_index = self._route_index_by_global_layer[global_layer_idx]
        if route_index.mlp_idx is None:
            return None
        return route["mlp_int_list"][route_index.mlp_idx]

    def _final_norm_emb_int(self) -> int | None:
        route = self._active_route()
        if route is None:
            return None
        emb_values = route["emb_int_list"]
        if not emb_values:
            return None
        return min(emb_values)

    def _mask_first_tensor(
        self, inputs: tuple[Any, ...], keep_dim: int
    ) -> tuple[Any, ...]:
        first = inputs[0]
        if not torch.is_tensor(first):
            return inputs
        return (self._mask_tensor(first, keep_dim), *inputs[1:])

    def _mask_output(
        self, output: Any, keep_dim: int, full_dim: int | None = None
    ) -> Any:
        if torch.is_tensor(output):
            return self._mask_tensor(output, keep_dim, full_dim=full_dim)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (
                self._mask_tensor(output[0], keep_dim, full_dim=full_dim),
                *output[1:],
            )
        return output

    def _mask_tensor(
        self, tensor: torch.Tensor, keep_dim: int, full_dim: int | None = None
    ) -> torch.Tensor:
        local_dim = tensor.shape[-1]
        if full_dim is None:
            full_dim = local_dim
        local_keep_dim = self._local_keep_dim(
            keep_dim=keep_dim,
            full_dim=full_dim,
            local_dim=local_dim,
        )
        if local_keep_dim >= local_dim:
            return tensor
        if local_keep_dim <= 0:
            return torch.zeros_like(tensor)

        mask = self._get_mask(
            device=tensor.device,
            dtype=tensor.dtype,
            local_dim=local_dim,
            local_keep_dim=local_keep_dim,
        )
        return tensor * mask.view(*([1] * (tensor.ndim - 1)), local_dim)

    def _local_keep_dim(self, *, keep_dim: int, full_dim: int, local_dim: int) -> int:
        if local_dim == full_dim:
            return keep_dim

        rank = self._parallel_rank(full_dim=full_dim, local_dim=local_dim)
        local_start = rank * local_dim
        local_end = local_start + local_dim
        return max(0, min(keep_dim, local_end) - local_start)

    def _parallel_rank(self, *, full_dim: int, local_dim: int) -> int:
        inferred_world_size = max(1, full_dim // local_dim)
        if inferred_world_size == 1:
            return 0

        try:
            from megatron.core import parallel_state

            if hasattr(parallel_state, "get_expert_tensor_parallel_rank"):
                rank = parallel_state.get_expert_tensor_parallel_rank()
            else:
                rank = parallel_state.get_tensor_model_parallel_rank()
        except (AssertionError, ImportError, RuntimeError):
            rank = 0

        return min(rank, inferred_world_size - 1)

    def _get_mask(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        local_dim: int,
        local_keep_dim: int,
    ) -> torch.Tensor:
        key = (device, dtype, local_dim, local_keep_dim)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.zeros(local_dim, dtype=dtype, device=device)
            mask[:local_keep_dim] = 1
            self._mask_cache[key] = mask
        return mask
