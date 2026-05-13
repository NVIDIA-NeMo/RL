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
from nemo_rl.models.megatron.flextron_hooks import moe_hooks, attention_hooks, mamba_hooks

# layer types used to expand `int_lists` to generalize to heterogeneous routing
_FLEXTRON_MLP_LAYER_TYPES = frozenset(("E",))
_FLEXTRON_EMB_LAYER_TYPES = frozenset(("M", "E", "*"))

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
        # normalize
        self.flextron_sampling_rates = self._normalize_sampling_rates(self.flextron_sampling_rates)

        self.active_router_id: int | None = None
        self._handles: list[RemovableHandle] = []
        self._mask_cache: dict[
            tuple[torch.device, torch.dtype, int, int], torch.Tensor
        ] = {}
        self._zero_tensor_cache: dict[
            tuple[torch.device, torch.dtype, tuple[int, ...]], torch.Tensor
        ] = {}
        self._route_index_by_global_layer = self._build_route_index_by_global_layer()
        self._model_with_decoder = self._unwrap_model_with_decoder(model)

        if self.enabled:
            self._attach_hooks()

    def _normalize_sampling_rates(self, sampling_rates: list[float]) -> list[float]:
        """Normalize sampling rates to sum to 1."""
        if not sampling_rates:
            return []
        total = sum(sampling_rates)
        return [rate / total for rate in sampling_rates]

    @property
    def enabled(self) -> bool:
        return bool(self.flex_routers)

    def resolve_router_ids(
        self,
        probs: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Map per-sample probabilities in [0, 1) to deterministic route ids.

        Performs inverse-CDF lookup over `flextron_sampling_rates` so collation
        can attach a route-agnostic uniform sample to every datum while this
        router owns the partitioning. Because the probabilities flow through the
        batch identically on every rank, no cross-rank broadcast is required.
        """
        if not self.enabled:
            return torch.zeros(probs.numel(), dtype=torch.long, device=device)

        cdf = torch.cumsum(
            torch.tensor(
                self.flextron_sampling_rates,
                dtype=torch.float32,
                device=probs.device,
            ),
            dim=0,
        )
        router_ids = torch.searchsorted(
            cdf, probs.to(dtype=torch.float32), right=True
        )
        router_ids = router_ids.clamp(max=len(self.flex_routers)).to(dtype=torch.long)
        if device is not None and router_ids.device != torch.device(device):
            router_ids = router_ids.to(device=device)
        return router_ids

    def get_router_ids(
        self, data: BatchedDataDict[Any], *, device: torch.device | str | None = None, offset_router_id: int = 0
    ) -> torch.Tensor:
        """Return per-sample route ids from `flex_router_probs`, `flex_router_ids`, or default to base route (id 0) if neither is present."""
        if "flex_router_probs" in data:
            return self.resolve_router_ids(data["flex_router_probs"], device=device)
        if "flex_router_ids" in data:
            return data["flex_router_ids"].to(device=device, dtype=torch.long)
        # default to route (id offset_router_id) if neither is present (during evals?)
        return torch.zeros(data.size, dtype=torch.long, device=device) + offset_router_id

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

        # print_attached_modules = (
        #     not self._distributed_is_initialized()
        #     or torch.distributed.get_rank() == 0
        # )
        # TODO: @rohitrango, change this later to verify the code modifies the correct modules.
        print_attached_modules = False

        hybrid_layer_pattern = self._main_layer_pattern()

        for local_idx, layer in enumerate(self._model_with_decoder.decoder.layers):
            global_idx = self._global_layer_idx(layer, local_idx)
            if global_idx >= len(hybrid_layer_pattern):
                continue

            # layer type: M = mamba, E = expert, * = transformer
            layer_type = hybrid_layer_pattern[global_idx]

            if layer_type == 'M':
                mamba_hooks.attach_mamba_hooks(layer, global_idx, self)
            elif layer_type == 'E':
                moe_hooks.attach_moe_hooks(layer, global_idx, self)
            elif layer_type == '*':
                attention_hooks.attach_attention_hooks(layer, global_idx, self)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}")


        # attach final norm hooks
        final_norm = getattr(self._model_with_decoder.decoder, "final_norm", None)
        if final_norm is not None:
            if print_attached_modules:
                print(
                    "Flextron attaching final norm pre/post hooks to "
                    "decoder.final_norm "
                    f"(class={final_norm.__class__.__qualname__})"
                )
            self._handles.append(
                final_norm.register_forward_pre_hook(self._final_norm_pre_hook)
            )
            self._handles.append(
                final_norm.register_forward_hook(self._final_norm_post_hook)
            )

        if print_attached_modules:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    input(
                        "[rank 0] Flextron hooks attached; press Enter to continue..."
                    )
                torch.distributed.barrier()
            else:
                input("Flextron hooks attached; press Enter to continue...")


    def _global_layer_idx(self, layer: torch.nn.Module, local_idx: int) -> int:
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1
        return local_idx

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

    def _final_norm_pre_hook(
        self, module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        emb_int = self._final_norm_emb_int()
        if emb_int is None or not inputs:
            return inputs

        layernorm_epsilon = getattr(self.model_cfg, "layernorm_epsilon", None)
        if layernorm_epsilon is not None and hasattr(module, "eps"):
            emb_effective_per = emb_int / self.model_cfg.hidden_size
            module.eps = layernorm_epsilon * emb_effective_per
        return self._mask_first_tensor(inputs, emb_int)

    def _final_norm_post_hook(
        self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> Any:
        del inputs
        emb_int = self._final_norm_emb_int()
        if emb_int is None:
            return output

        layernorm_epsilon = getattr(self.model_cfg, "layernorm_epsilon", None)
        if layernorm_epsilon is not None and hasattr(module, "eps"):
            module.eps = layernorm_epsilon
        emb_effective_per = emb_int / self.model_cfg.hidden_size
        return self._scale_output(output, emb_effective_per**0.5)

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

    def _scale_output(self, output: Any, scale: float) -> Any:
        if torch.is_tensor(output):
            return output * scale
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (output[0] * scale, *output[1:])
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

            # TODO: @rohitrango, check if this is correct for MoE TP versus MLP/attn TP
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
            if not torch.is_inference_mode_enabled():
                self._mask_cache[key] = mask
        return mask
