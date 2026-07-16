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

from typing import cast

import torch

from nemo_rl.models.generation.vllm.refit_layout import (
    VllmExpertParamLayout,
    VllmWeightLayout,
    parse_hf_expert_weight,
)


class VllmRefitLoaderMixin:
    """Load full or destination-local HF weights into vLLM storage."""

    _nrl_named_parameters: dict[str, torch.nn.Parameter]

    def _is_sharded_refit_weight(self, name: str, tensor: torch.Tensor) -> bool:
        param_names = self._sharded_refit_param_names(name)
        if not param_names:
            return False

        state_dict_info = getattr(self, "state_dict_info", None)
        if state_dict_info is None or name not in state_dict_info:
            return False
        full_shape, _dtype = state_dict_info[name]
        return torch.Size(full_shape) != tensor.shape

    def _sharded_refit_param_names(self, name: str) -> list[str]:
        expert_weight = parse_hf_expert_weight(name)
        return [] if expert_weight is None else [expert_weight.parameter_name]

    def _get_named_parameters(self) -> dict[str, torch.nn.Parameter]:
        params = getattr(self, "_nrl_named_parameters", None)
        if params is None:
            params = dict(self.model_runner.model.named_parameters())
            self._nrl_named_parameters = params
        return params

    def _checkpoint_engine_weight_layout(self) -> VllmWeightLayout:
        from vllm.model_executor.models.utils import get_pp_missing_layer_names

        expert_params: dict[str, VllmExpertParamLayout] = {}
        for name, param in self._get_named_parameters().items():
            if not name.endswith((".w13_weight", ".w2_weight")):
                continue

            if bool(getattr(param, "is_transposed", False)):
                raise ValueError(
                    "Sharded NIXL expert refit requires canonical expert-weight "
                    f"orientation, but {name} is transposed."
                )

            weight_loader = getattr(param, "weight_loader", None)
            owner = getattr(weight_loader, "__self__", None)
            if owner is None:
                raise RuntimeError(
                    f"Could not inspect the vLLM expert weight loader for {name}."
                )

            self._validate_expert_storage(name, param)

            quant_method = getattr(owner, "base_quant_method", None)
            backend = getattr(quant_method, "unquantized_backend", None)
            backend_name = getattr(backend, "name", None)
            if getattr(owner, "quant_config", None) is not None or backend_name not in {
                "TRITON",
                "BATCHED_TRITON",
            }:
                raise ValueError(
                    "Sharded NIXL expert refit requires canonical unquantized Triton "
                    f"expert weights, but {name} uses "
                    f"{type(quant_method).__name__}/{backend_name}. Set "
                    "policy.generation.vllm_kwargs.moe_backend=triton."
                )

            use_ep = bool(getattr(owner, "use_ep", False))
            local_expert_ids: list[int] | None = None
            if use_ep:
                if bool(getattr(owner, "enable_eplb", False)):
                    raise RuntimeError(
                        "Sharded refit does not support dynamic vLLM expert load "
                        "balancing because ownership can change after metadata "
                        "exchange."
                    )
                expert_map = getattr(owner, "_expert_map", None)
                if expert_map is None:
                    raise RuntimeError(
                        f"vLLM reports EP for {name} without an expert ownership map."
                    )
                logical_num_experts = int(
                    cast(
                        int,
                        getattr(owner, "logical_num_experts", expert_map.numel()),
                    )
                )
                global_num_experts = int(
                    cast(
                        int,
                        getattr(owner, "global_num_experts", logical_num_experts),
                    )
                )
                if global_num_experts != logical_num_experts:
                    raise RuntimeError(
                        "Sharded refit does not support redundant vLLM experts."
                    )
                local_expert_ids = [
                    expert_id
                    for expert_id, local_id in enumerate(
                        expert_map.detach().cpu().tolist()[:logical_num_experts]
                    )
                    if int(local_id) >= 0
                ]

            expert_params[name] = {
                "tp_rank": int(getattr(owner, "tp_rank", 0)),
                "tp_size": int(getattr(owner, "tp_size", 1)),
                "local_expert_ids": local_expert_ids,
            }

        return {
            "expert_params": expert_params,
            "missing_weight_prefixes": get_pp_missing_layer_names(
                self.model_runner.model
            ),
        }

    @staticmethod
    def _validate_expert_storage(param_name: str, param: torch.nn.Parameter) -> None:
        """Reject incompatible vLLM storage before advertising shards."""
        is_w13 = param_name.endswith(".w13_weight")
        if param.ndim != 3 or (is_w13 and param.shape[1] % 2 != 0):
            raise RuntimeError(
                f"Unsupported vLLM expert storage for {param_name}: "
                f"shape={tuple(param.shape)}."
            )

    def _local_expert_id(self, param: torch.nn.Parameter, expert_id: int) -> int:
        weight_loader = getattr(param, "weight_loader", None)
        owner = getattr(weight_loader, "__self__", None)
        mapper = getattr(owner, "_map_global_expert_id_to_local_expert_id", None)
        if mapper is None:
            return expert_id
        return int(mapper(expert_id))

    def _load_destination_local_expert_group(
        self,
        param_name: str,
        param: torch.nn.Parameter,
        shard_id: str,
        items: list[tuple[int, torch.Tensor]],
    ) -> None:
        """Copy destination-local experts into canonical vLLM storage."""
        if shard_id not in {"w1", "w2", "w3"}:
            raise ValueError(f"Unexpected sharded expert shard_id: {shard_id}")

        target = param.data
        if shard_id in {"w1", "w3"}:
            shard_size = target.shape[1] // 2
            shard_offset = 0 if shard_id == "w1" else shard_size
            target = target.narrow(1, shard_offset, shard_size)

        def copy_to_target(destination: torch.Tensor, source: torch.Tensor) -> None:
            for dim, size in enumerate(source.shape):
                destination = destination.narrow(dim, 0, size)
            destination.copy_(source)

        sorted_items = sorted(items, key=lambda item: item[0])
        expert_ids = [expert_id for expert_id, _tensor in sorted_items]
        contiguous_ids = list(range(expert_ids[0], expert_ids[0] + len(expert_ids)))
        with torch.no_grad():
            if expert_ids == contiguous_ids:
                expert_data = target.narrow(0, expert_ids[0], len(expert_ids))
                loaded_weight = torch.stack(
                    [tensor for _expert_id, tensor in sorted_items]
                )
                copy_to_target(expert_data, loaded_weight)
            else:
                for local_expert_id, loaded_weight in sorted_items:
                    copy_to_target(target[local_expert_id], loaded_weight)

    def _load_sharded_expert_weight_groups(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        params = self._get_named_parameters()
        groups: dict[tuple[str, str], list[tuple[int, torch.Tensor]]] = {}
        remaining_weights: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            expert_weight = parse_hf_expert_weight(name)
            if expert_weight is None:
                remaining_weights.append((name, tensor))
                continue

            mapped_name = expert_weight.parameter_name
            param = params.get(mapped_name)
            if param is None:
                remaining_weights.append((name, tensor))
                continue

            owner = getattr(getattr(param, "weight_loader", None), "__self__", None)
            if not self._is_sharded_refit_weight(name, tensor) and not bool(
                getattr(owner, "use_ep", False)
            ):
                remaining_weights.append((name, tensor))
                continue

            if owner is None:
                raise RuntimeError(
                    "Could not resolve the vLLM expert weight loader for "
                    f"{mapped_name}."
                )

            if param.data.ndim != 3 or tensor.ndim != 2:
                raise ValueError(
                    f"Sharded expert {name} requires a 3-D vLLM parameter and "
                    f"2-D source tensor, got {param.data.ndim}-D and {tensor.ndim}-D."
                )

            full_shape, _dtype = self.state_dict_info[name]
            expected_shape = list(full_shape)
            tp_size = int(getattr(owner, "tp_size", 1))
            shard_dim = expert_weight.tp_shard_dim
            expected_shape[shard_dim] //= tp_size
            if tensor.shape != torch.Size(expected_shape):
                raise ValueError(
                    f"Received sharded expert {name} with shape "
                    f"{tuple(tensor.shape)}, expected {tuple(expected_shape)} from "
                    f"full shape {tuple(full_shape)} and TP size {tp_size}."
                )

            local_expert_id = self._local_expert_id(param, expert_weight.expert_id)
            if local_expert_id == -1:
                continue

            groups.setdefault((mapped_name, expert_weight.shard_id), []).append(
                (local_expert_id, tensor)
            )

        for (mapped_name, shard_id), items in groups.items():
            self._load_destination_local_expert_group(
                mapped_name, params[mapped_name], shard_id, items
            )

        return remaining_weights

    def _use_sharded_expert_refit(self) -> bool:
        checkpoint_engine = getattr(self, "checkpoint_engine", None)
        return bool(getattr(checkpoint_engine, "shard_expert_weights", False))

    def _load_full_hf_weights(
        self, policy_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Reference path for complete HF weights and the sharded fallback."""
        self.model_runner.model.load_weights(weights=policy_weights)

    def _load_hf_weights(self, policy_weights: list[tuple[str, torch.Tensor]]) -> None:
        if self._use_sharded_expert_refit():
            self._load_sharded_expert_weights(policy_weights)
            return

        from nemo_rl.models.generation.vllm.quantization import fp8

        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
            return

        self._load_full_hf_weights(policy_weights)

    def _load_sharded_expert_weights(
        self, policy_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        from nemo_rl.models.generation.vllm.quantization import fp8

        if fp8.is_fp8_model(self.model_runner.vllm_config):
            raise ValueError(
                "Sharded NIXL expert refit is not supported for FP8 vLLM models."
            )

        remaining_weights = self._load_sharded_expert_weight_groups(policy_weights)
        if remaining_weights:
            self._load_full_hf_weights(remaining_weights)
