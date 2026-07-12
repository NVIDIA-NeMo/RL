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

"""Direct sparse-delta placement and application for vLLM workers."""

import io
import os
import re
import time
from dataclasses import dataclass
from typing import Any, cast

import torch

from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.nsys import wrap_with_nvtx_name

_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class _SparseDeltaTargetPlan:
    target: torch.Tensor | None
    source_shape: tuple[int, ...] = ()
    source_strides: tuple[int, ...] = ()
    target_strides: tuple[int, ...] = ()
    target_offset: int = 0
    shard_dim: int | None = None
    shard_start: int = 0
    shard_size: int = 0
    segment_shards: tuple[tuple[int, int, int], ...] = ()
    log_delta_transform: bool = False
    identity: bool = False


# Sentinel used when the additive apply path bypasses plan computation.
_ADDITIVE_SENTINEL_PLAN = _SparseDeltaTargetPlan(target=None)


class VllmSparseDeltaApplier:
    """Own sparse placement state without extending the normal refit path."""

    def __init__(
        self,
        model_runner: Any,
        device: torch.device,
        *,
        rank: int = 0,
    ) -> None:
        self.model_runner = model_runner
        self._cuda_device_index = device.index
        self.rank = rank
        self._direct_sparse_delta_targets: dict[str, torch.Tensor] | None = None
        self._direct_sparse_delta_plan_cache: dict[
            str, _SparseDeltaTargetPlan | None
        ] = {}
        self._direct_sparse_delta_verification: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._direct_sparse_delta_verification_candidates = 0

        # --- R5: one-shot param dump for E2E correctness comparison ---
        self._dump_vllm_params_path: str | None = os.environ.get(
            "NRL_REFIT_DUMP_VLLM_PARAMS", ""
        ) or None
        self._dump_vllm_params_done: bool = False

        # --- M2: env-flag for additive apply path ---
        _mode = os.environ.get("NRL_REFIT_SPARSE_APPLY_MODE", "plan").strip().lower()
        if _mode not in ("plan", "additive", "allowlist"):
            raise ValueError(
                f"NRL_REFIT_SPARSE_APPLY_MODE must be 'plan', 'additive', or"
                f" 'allowlist'; got {_mode!r}"
            )
        self._sparse_apply_mode: str = _mode
        _allowlist_pat = os.environ.get("NRL_REFIT_SPARSE_APPLY_ALLOWLIST", "")
        self._sparse_apply_allowlist: re.Pattern[str] | None = (
            re.compile(_allowlist_pat) if _allowlist_pat else None
        )

    @staticmethod
    def _is_plain_linear_name(target_name: str) -> bool:
        """Return True if *target_name* falls through to the plain-linear (shard) path.

        Mirrors the dispatch order in ``_direct_sparse_delta_target_plan``:
        mamba / QKV / expert / gate_up names are NOT plain-linear.  Everything
        else (``ColumnParallelLinear``, ``RowParallelLinear``, embed, lm_head …)
        is treated as plain-linear and is safe to apply via the additive load path
        for tp_size=1.
        """
        if ".mixer." in target_name:
            return False
        if any(f".{x}_proj." in target_name for x in ("q", "k", "v")):
            return False
        if _EXPERT_WEIGHT_RE.match(target_name):
            return False
        if any(f".{x}_proj." in target_name for x in ("gate", "up")):
            return False
        return True

    def _additive_apply_mode(self, target_name: str) -> bool:
        """Return True if the additive path should be used for *target_name*."""
        mode = self._sparse_apply_mode
        if mode == "plan":
            return False
        if mode == "additive":
            return self._is_plain_linear_name(target_name)
        # mode == "allowlist"
        pat = self._sparse_apply_allowlist
        return pat is not None and pat.search(target_name) is not None

    def _apply_sparse_weight_deltas(
        self,
        payload_tensors: tuple[torch.Tensor, torch.Tensor],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Apply sparse deltas directly after validating every target plan."""
        architectures = self.model_runner.vllm_config.model_config.architectures
        # Delay the vLLM-dependent FP8 helper until a payload is applied.
        from nemo_rl.models.generation.vllm.quantization import fp8

        if {"GptOssForCausalLM", "Gemma3ForConditionalGeneration"} & set(
            architectures
        ) or fp8.is_fp8_model(self.model_runner.vllm_config):
            raise RuntimeError(
                "Direct sparse delta refit does not support transformed or FP8 weights."
            )

        if self._direct_sparse_delta_targets is None:
            model = self.model_runner.model
            self._direct_sparse_delta_targets = dict(model.named_parameters()) | dict(
                model.named_buffers()
            )
        targets = self._direct_sparse_delta_targets
        raw_locations, raw_values = payload_tensors
        plan_cache = self._direct_sparse_delta_plan_cache
        if plan_cache is None:
            plan_cache = self._direct_sparse_delta_plan_cache = {}
        # Lazy import: only pay the cost when additive mode is active.
        _use_additive_mode = self._sparse_apply_mode != "plan"
        if _use_additive_mode:
            from nemo_rl.models.generation.vllm.vllm_sparse_delta_additive import (
                apply_sparse_delta_via_additive_load,
            )
            _mapper = getattr(self.model_runner.model, "hf_to_vllm_mapper", None)

        plans: list[tuple[dict[str, Any], _SparseDeltaTargetPlan, str, bool]] = []
        for item in metadata:
            name = str(item["name"])
            # Compute target_name (same mapping as _direct_sparse_delta_target_plan).
            if _use_additive_mode:
                _mapped = (
                    cast(Any, _mapper)._map_name(name) if _mapper is not None else name
                )
                if _mapped is None or _mapped.startswith("draft."):
                    # Mirrors the early-out in _direct_sparse_delta_target_plan;
                    # fall through to plan path for these names.
                    _target_name = name
                    _use_additive = False
                else:
                    _target_name = _mapped
                    _use_additive = self._additive_apply_mode(_target_name)
            else:
                _target_name = name
                _use_additive = False

            if not _use_additive:
                if name not in plan_cache:
                    plan_cache[name] = self._direct_sparse_delta_target_plan(
                        item, targets
                    )
                plan = plan_cache[name]
                if plan is None:
                    raise RuntimeError(
                        f"No direct sparse delta target plan for {item['name']!r}."
                    )
            else:
                # Additive path: use a sentinel plan so the apply loop can
                # `continue` past plan-specific logic.
                plan = _ADDITIVE_SENTINEL_PLAN
            plans.append((item, plan, _target_name, _use_additive))

        with torch.no_grad():
            for item, plan, _target_name, _use_additive in plans:
                target = plan.target
                verification_locations = item.get("verification_locations", [])
                self._direct_sparse_delta_verification_candidates += len(
                    verification_locations
                )
                if _use_additive:
                    # Additive path: decode sparse payload and delegate to
                    # apply_sparse_delta_via_additive_load.
                    value_start = int(item["value_start"])
                    value_end = int(item["value_end"])
                    sparse_indices = sparse_codec.sparse_locations_for_item(
                        item, raw_locations, device="cpu"
                    )
                    sparse_values = raw_values[value_start:value_end]
                    apply_sparse_delta_via_additive_load(
                        _target_name,
                        sparse_indices,
                        sparse_values,
                        tuple(item["shape"]),
                        targets[_target_name].dtype,
                        self.model_runner.model,
                        targets[_target_name].device,
                    )
                    continue
                if target is None:
                    continue

                if verification_locations and not plan.log_delta_transform:
                    sample_locations, sample_deltas = (
                        self._local_sparse_delta_update_inputs(
                            torch.tensor(verification_locations, device=target.device),
                            torch.tensor(
                                item["verification_deltas"],
                                device=target.device,
                                dtype=target.dtype,
                            ),
                            plan,
                        )
                    )
                    if sample_locations.numel():
                        before = target.data.view(-1).index_select(0, sample_locations)
                        expected_delta = (
                            before + sample_deltas
                        ).float() - before.float()
                        verification = self._direct_sparse_delta_verification
                        if verification is None:
                            verification = self._direct_sparse_delta_verification = []
                        verification.append(
                            (
                                target,
                                sample_locations,
                                before.float(),
                                expected_delta,
                            )
                        )

                value_start = int(item["value_start"])
                value_end = int(item["value_end"])
                values = raw_values[value_start:value_end].to(
                    device=target.device,
                    dtype=target.dtype,
                    non_blocking=True,
                )
                if plan.identity and item["index_encoding"] == "range":
                    range_start = int(item["range_start"])
                    range_count = value_end - value_start
                    target.data.view(-1).narrow(0, range_start, range_count).add_(
                        values
                    )
                else:
                    locations = sparse_codec.sparse_locations_for_item(
                        item,
                        raw_locations,
                        device=target.device,
                    )
                    locations, values = self._local_sparse_delta_update_inputs(
                        locations,
                        values,
                        plan,
                    )
                    if locations.numel():
                        target_flat = target.data.view(-1)
                        if plan.log_delta_transform:
                            current = target_flat.index_select(0, locations)
                            updated = current * values.float().exp().to(
                                dtype=current.dtype
                            )
                            target_flat.index_copy_(0, locations, updated)
                        else:
                            target_flat.index_add_(0, locations, values)

    def _direct_sparse_delta_module(
        self,
        target: torch.Tensor,
        module_name: str,
    ) -> Any:
        loader = getattr(target, "weight_loader", None)
        return getattr(
            loader, "__self__", None
        ) or self.model_runner.model.get_submodule(module_name)

    def _direct_sparse_delta_target_plan(
        self,
        item: dict[str, Any],
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        name = str(item["name"])
        if name.startswith("mtp."):
            return _SparseDeltaTargetPlan(target=None)
        mapper = getattr(self.model_runner.model, "hf_to_vllm_mapper", None)
        target_name = cast(Any, mapper)._map_name(name) if mapper is not None else name
        if target_name is None or target_name.startswith("draft."):
            return None
        if ".mixer." in target_name:
            mamba_plan = self._direct_sparse_delta_mamba2_plan(
                item, target_name, targets
            )
            if mamba_plan is not None:
                return mamba_plan
            if ".mixer.conv1d." in target_name or ".mixer.in_proj." in target_name:
                return None
        if any(f".{candidate}_proj." in target_name for candidate in ("q", "k", "v")):
            return self._direct_sparse_delta_qkv_plan(item, target_name, targets)
        if _EXPERT_WEIGHT_RE.match(target_name):
            return self._direct_sparse_delta_expert_plan(item, target_name, targets)
        if any(f".{candidate}_proj." in target_name for candidate in ("gate", "up")):
            merged_plan = self._direct_sparse_delta_merged_column_plan(
                item, target_name, targets
            )
            if merged_plan is not None:
                return merged_plan

        target = targets.get(target_name)
        if target is None:
            return None
        source_shape = tuple(item["shape"])
        target_shape = tuple(target.shape)
        if target_shape == source_shape:
            return self._make_sparse_delta_target_plan(target, source_shape)
        return self._direct_sparse_delta_shard_plan(item, target)

    def _direct_sparse_delta_qkv_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        shard_id = next(x for x in "qkv" if f".{x}_proj." in target_name)
        packed_name = target_name.replace(f".{shard_id}_proj.", ".qkv_proj.", 1)
        target = targets.get(packed_name)
        if target is None:
            return None
        output_dim = int(cast(Any, target).output_dim) % target.ndim
        module = self._direct_sparse_delta_module(target, packed_name.rsplit(".", 1)[0])
        shard_offset = int(module._get_shard_offset_mapping(shard_id))
        shard_size = int(module._get_shard_size_mapping(shard_id))
        shard_rank = int(module.tp_rank)
        if shard_id != "q":
            shard_rank //= int(module.num_kv_head_replicas)

        source_shape = tuple(item["shape"])
        shard_start = shard_rank * shard_size
        if source_shape[output_dim] < shard_start:
            return _SparseDeltaTargetPlan(target=None)
        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            shard_dim=output_dim,
            shard_start=shard_start,
            shard_size=min(shard_size, source_shape[output_dim] - shard_start),
            target_offset=shard_offset * target.stride(output_dim),
        )

    def _direct_sparse_delta_merged_column_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        projection = next(
            candidate
            for candidate in ("gate", "up")
            if f".{candidate}_proj." in target_name
        )
        shard_id = 0 if projection == "gate" else 1
        packed_name = target_name.replace(f".{projection}_proj.", ".gate_up_proj.", 1)
        target = targets.get(packed_name)
        output_dim = getattr(target, "output_dim", None)
        if target is None or not isinstance(output_dim, int):
            return None

        output_dim %= target.ndim
        module = self._direct_sparse_delta_module(target, packed_name.rsplit(".", 1)[0])
        output_sizes = tuple(int(size) for size in module.output_sizes)
        tp_size = int(module.tp_size)
        source_shape = tuple(item["shape"])
        if (
            shard_id >= len(output_sizes)
            or tp_size < 1
            or output_sizes[shard_id] % tp_size
            or output_dim >= len(source_shape)
            or source_shape[output_dim] != output_sizes[shard_id]
        ):
            return None

        shard_size = output_sizes[shard_id] // tp_size
        target_start = sum(output_sizes[:shard_id]) // tp_size
        if target.shape[output_dim] < target_start + shard_size:
            return None
        shard_start = int(module.tp_rank) * shard_size
        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            shard_dim=output_dim,
            shard_start=shard_start,
            shard_size=shard_size,
            target_offset=target_start * target.stride(output_dim),
        )

    def _direct_sparse_delta_mamba2_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        target = targets.get(target_name)
        if target is None:
            return None

        if target_name.endswith(".A"):
            source_shape = tuple(item["shape"])
            if tuple(target.shape) == source_shape:
                return self._make_sparse_delta_target_plan(
                    target,
                    source_shape=source_shape,
                    log_delta_transform=True,
                )
            return self._direct_sparse_delta_shard_plan(
                item,
                target,
                log_delta_transform=True,
            )

        if not (".mixer.conv1d." in target_name or ".mixer.in_proj." in target_name):
            return None

        mixer_name = target_name.split(".mixer.", 1)[0] + ".mixer"
        attrs = cast(Any, self.model_runner.model.get_submodule(mixer_name))
        tp_size = int(attrs.tp_size)
        if tp_size <= 1:
            return None
        intermediate_size = int(attrs.intermediate_size)
        groups_ssm_state_size = int(attrs.groups_ssm_state_size)
        num_heads = int(attrs.num_heads)
        source_shape = tuple(item["shape"])
        fixed_size = intermediate_size
        if ".mixer.in_proj." in target_name:
            fixed_size += intermediate_size + num_heads
        group_size, remainder = divmod(source_shape[0] - fixed_size, 2)
        extra_group_size = groups_ssm_state_size - group_size
        if remainder or group_size <= 0 or extra_group_size < 0:
            return None
        tp_rank = int(
            getattr(target, "tp_rank", int(getattr(self, "rank", 0)) % tp_size)
        )
        intermediate = (intermediate_size, 0, False)
        group = (groups_ssm_state_size, extra_group_size, extra_group_size > 0)
        segment_specs = (
            (intermediate, group, group)
            if ".mixer.conv1d." in target_name
            else (intermediate, intermediate, group, group, (num_heads, 0, False))
        )

        target_shape = tuple(target.shape)
        source_to_target_dims = tuple(range(len(source_shape)))
        if len(target_shape) == len(source_shape) + 1 and target_shape[1] == 1:
            source_to_target_dims = (0, *range(2, len(target_shape)))
        elif len(target_shape) != len(source_shape):
            return None
        segment_shards: list[tuple[int, int, int]] = []
        target_start = 0
        source_start = 0
        for full_dim, extra, duplicate_groups in segment_specs:
            shard_size = full_dim // tp_size
            rank = 0 if duplicate_groups else tp_rank
            source_dim = full_dim - extra
            source_local_start = source_start + rank * shard_size
            take = min(shard_size, source_dim - rank * shard_size)
            if take > 0:
                segment_shards.append((source_local_start, target_start, take))
            target_start += shard_size
            source_start += source_dim
        if source_shape[0] != source_start or target_shape[0] != target_start:
            return None

        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            source_to_target_dims=source_to_target_dims,
            shard_dim=0,
            segment_shards=tuple(segment_shards),
        )

    def _direct_sparse_delta_expert_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        match = cast(re.Match[str], _EXPERT_WEIGHT_RE.match(target_name))

        prefix = match.group("prefix")
        global_expert_id = int(match.group("expert"))
        proj = match.group("proj")
        packed_weight, shard_id = {
            "gate_proj": ("w13_weight", "w1"),
            "up_proj": ("w13_weight", "w3"),
            "down_proj": ("w2_weight", "w2"),
        }[proj]
        packed_name = f"{prefix}.{packed_weight}"

        target = targets.get(packed_name)
        if target is None:
            return None
        module_attrs = self._direct_sparse_delta_module(
            target, packed_name.rsplit(".", 1)[0]
        )
        if shard_id == "w3" and not module_attrs.moe_config.is_act_and_mul:
            shard_id = "w1"
        local_expert_id = int(
            module_attrs._map_global_expert_id_to_local_expert_id(global_expert_id)
        )
        if local_expert_id < 0:
            return _SparseDeltaTargetPlan(target=None)

        source_shape = tuple(item["shape"])
        target_shape = tuple(target.shape)
        shard_dim = 1 if shard_id == "w2" else 0
        if local_expert_id >= target_shape[0]:
            return None
        target_shard_dim = shard_dim + 1

        shard_size = target_shape[target_shard_dim]
        if shard_id in ("w1", "w3") and module_attrs.moe_config.is_act_and_mul:
            shard_size //= 2
        target_shard_offset = shard_size if shard_id == "w3" else 0
        if target_shape[target_shard_dim] < target_shard_offset + shard_size:
            return None
        tp_rank = int(module_attrs.tp_rank)
        shard_start = tp_rank * shard_size
        if source_shape[shard_dim] < shard_start:
            return _SparseDeltaTargetPlan(target=None)

        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            source_to_target_dims=tuple(dim + 1 for dim in range(len(source_shape))),
            target_offset=(
                local_expert_id * target.stride(0)
                + target_shard_offset * target.stride(target_shard_dim)
            ),
            shard_dim=shard_dim,
            shard_start=shard_start,
            shard_size=min(shard_size, source_shape[shard_dim] - shard_start),
        )

    def _make_sparse_delta_target_plan(
        self,
        target: torch.Tensor,
        source_shape: tuple[int, ...],
        *,
        source_to_target_dims: tuple[int, ...] | None = None,
        target_offset: int = 0,
        shard_dim: int | None = None,
        shard_start: int = 0,
        shard_size: int = 0,
        segment_shards: tuple[tuple[int, int, int], ...] = (),
        log_delta_transform: bool = False,
    ) -> _SparseDeltaTargetPlan | None:
        if source_to_target_dims is None:
            source_to_target_dims = tuple(range(len(source_shape)))
        target_shape = tuple(target.shape)
        ignored_dim = (
            shard_dim if shard_dim is not None else 0 if segment_shards else -1
        )
        if len(source_to_target_dims) != len(source_shape) or any(
            target_dim >= len(target_shape)
            or (
                source_dim != ignored_dim
                and source_shape[source_dim] != target_shape[target_dim]
            )
            for source_dim, target_dim in enumerate(source_to_target_dims)
        ):
            return None
        identity = (
            shard_dim is None
            and target_offset == 0
            and not segment_shards
            and not log_delta_transform
            and source_to_target_dims == tuple(range(len(source_shape)))
            and source_shape == target_shape
        )
        return _SparseDeltaTargetPlan(
            target=target,
            source_shape=source_shape,
            source_strides=torch.empty(source_shape, device="meta").stride(),
            target_strides=tuple(
                target.stride(target_dim) for target_dim in source_to_target_dims
            ),
            target_offset=target_offset,
            shard_dim=shard_dim,
            shard_start=shard_start,
            shard_size=shard_size,
            segment_shards=segment_shards,
            log_delta_transform=log_delta_transform,
            identity=identity,
        )

    def _direct_sparse_delta_shard_plan(
        self,
        item: dict[str, Any],
        target: torch.Tensor,
        *,
        log_delta_transform: bool = False,
    ) -> _SparseDeltaTargetPlan | None:
        source_shape = tuple(item["shape"])
        target_shape = tuple(target.shape)
        if len(source_shape) != len(target_shape):
            return None

        candidate_dims = list(
            dict.fromkeys(
                dim % len(source_shape)
                for attr in ("output_dim", "input_dim")
                if isinstance(dim := getattr(target, attr, None), int)
            )
        )
        if not candidate_dims:
            candidate_dims = [
                dim
                for dim, (source_dim, target_dim) in enumerate(
                    zip(source_shape, target_shape, strict=True)
                )
                if source_dim != target_dim
            ]
            if len(candidate_dims) != 1:
                return None

        for shard_dim in candidate_dims:
            shard_size = target_shape[shard_dim]
            tp_size = int(getattr(target, "tp_size", 1))
            if tp_size <= 1:
                if shard_size <= 0 or source_shape[shard_dim] % shard_size:
                    continue
                tp_size = source_shape[shard_dim] // shard_size
                if tp_size <= 1:
                    continue
            if source_shape[shard_dim] > shard_size * tp_size:
                continue
            tp_rank = int(
                getattr(target, "tp_rank", int(getattr(self, "rank", 0)) % tp_size)
            )
            plan = self._make_sparse_delta_target_plan(
                target=target,
                source_shape=source_shape,
                shard_dim=shard_dim,
                shard_start=tp_rank * shard_size,
                shard_size=shard_size,
                log_delta_transform=log_delta_transform,
            )
            if plan is not None:
                return plan
        return None

    def _local_sparse_delta_update_inputs(
        self,
        locations: torch.Tensor,
        values: torch.Tensor,
        plan: _SparseDeltaTargetPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if plan.identity:
            return locations, values

        source_shape = plan.source_shape
        source_strides = plan.source_strides
        target_strides = plan.target_strides
        shard_dim = plan.shard_dim

        if source_strides == target_strides:
            if shard_dim is None:
                return locations + plan.target_offset, values
            if shard_dim == 0:
                shard_stride = source_strides[0]
                shard_coords = torch.div(locations, shard_stride, rounding_mode="floor")
                if plan.segment_shards:
                    mapped_locations = locations + plan.target_offset
                    keep = torch.zeros_like(locations, dtype=torch.bool)
                    for source_start, target_start, take in plan.segment_shards:
                        segment = (shard_coords >= source_start) & (
                            shard_coords < source_start + take
                        )
                        mapped_locations[segment] += (
                            target_start - source_start
                        ) * shard_stride
                        keep |= segment
                    return mapped_locations[keep], values[keep]
                shard_end = min(
                    plan.shard_start + plan.shard_size, source_shape[shard_dim]
                )
                keep = (shard_coords >= plan.shard_start) & (shard_coords < shard_end)
                return (
                    locations[keep]
                    + plan.target_offset
                    - plan.shard_start * shard_stride,
                    values[keep],
                )

        selected_locations = locations
        selected_values = values

        if shard_dim is not None:
            shard_coords = torch.div(
                locations,
                source_strides[shard_dim],
                rounding_mode="floor",
            ).remainder(source_shape[shard_dim])
            shard_end = min(
                plan.shard_start + plan.shard_size,
                source_shape[shard_dim],
            )
            keep = (shard_coords >= plan.shard_start) & (shard_coords < shard_end)
            selected_locations = locations[keep]
            selected_values = values[keep]
            if selected_locations.numel() == 0:
                return selected_locations, selected_values

        local_locations = torch.full_like(selected_locations, plan.target_offset)
        for dim, (source_stride, target_stride) in enumerate(
            zip(source_strides, target_strides, strict=True)
        ):
            coord = torch.div(
                selected_locations,
                source_stride,
                rounding_mode="floor",
            ).remainder(source_shape[dim])
            if dim == plan.shard_dim:
                coord = coord - plan.shard_start
            local_locations.add_(coord * target_stride)
        return local_locations, selected_values

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_serialized_sparse_payload"
    )
    def update_weights_from_serialized_sparse_payload(
        self,
        serialized_payload: bytes,
    ) -> dict[str, Any]:
        """Apply one serialized sparse-delta payload."""
        return self._load_and_apply_sparse_payload(io.BytesIO(serialized_payload))

    def _load_and_apply_sparse_payload(
        self,
        source: str | io.BytesIO,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        payload = cast(
            sparse_codec.TensorPayload,
            torch.load(
                source,
                map_location="cpu",
                weights_only=True,
            ),
        )
        deserialize_s = time.perf_counter() - started
        result = self._apply_sparse_request(payload)
        result["receiver_deserialize_s"] = deserialize_s
        result["receiver_total_s"] = time.perf_counter() - started
        return result

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_sparse_payload_files"
    )
    def update_weights_from_sparse_payload_files(
        self,
        *payload_paths: str,
    ) -> dict[str, Any]:
        """Apply sparse payloads in FIFO order."""
        started = time.perf_counter()
        deserialize_s = 0.0
        sparse_apply_s = 0.0
        for path in payload_paths:
            result = self._load_and_apply_sparse_payload(path)
            deserialize_s += float(result["receiver_deserialize_s"])
            sparse_apply_s += float(result["receiver_sparse_apply_s"])
        return {
            "ok": True,
            "receiver_deserialize_s": deserialize_s,
            "receiver_sparse_apply_s": sparse_apply_s,
            "receiver_total_s": time.perf_counter() - started,
        }

    def _apply_sparse_request(
        self,
        payload: sparse_codec.TensorPayload,
    ) -> dict[str, Any]:
        locations, values, metadata = payload

        sparse_started = time.perf_counter()
        self._apply_sparse_weight_deltas((locations, values), metadata)
        sparse_apply_s = time.perf_counter() - sparse_started

        return {
            "ok": True,
            "receiver_sparse_apply_s": sparse_apply_s,
        }

    def synchronize_device(self) -> None:
        """Synchronize this vLLM worker's CUDA device after deferred refit applies."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self._cuda_device_index)

    def finish_sparse_delta_refit(self) -> dict[str, Any]:
        """Synchronize and compare bounded producer samples with applied weights."""
        self.synchronize_device()

        # R5 one-shot dump: write all vLLM-worker params AFTER full refit
        # completes (all chunks applied + CUDA sync) for E2E correctness comparison.
        # Use an exclusive flag file to ensure exactly one writer across all
        # vLLM worker processes (TP=1 means all have identical weights).
        dump_path = self._dump_vllm_params_path
        if dump_path and not self._dump_vllm_params_done:
            flag_path = dump_path + ".writing"
            try:
                fd = os.open(flag_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                won_race = True
            except FileExistsError:
                won_race = False
            if won_race:
                self._dump_vllm_params_done = True
                snapshot = {
                    name: param.data.detach().cpu().clone()
                    for name, param in self.model_runner.model.named_parameters()
                }
                torch.save(snapshot, dump_path)
                os.unlink(flag_path)
                print(
                    f"[R5] Saved vLLM param snapshot ({len(snapshot)} tensors)"
                    f" -> {dump_path}",
                    flush=True,
                )

        verification = self._direct_sparse_delta_verification or []
        candidates = self._direct_sparse_delta_verification_candidates
        self._direct_sparse_delta_verification = []
        self._direct_sparse_delta_verification_candidates = 0
        if not verification:
            return {
                "ok": True,
                "verification_candidates": candidates,
                "verification_samples": 0,
                "verification_exact_mismatches": 0,
                "verification_mismatches": 0,
                "verification_abs_sum": 0.0,
                "verification_max_abs": 0.0,
            }

        with torch.no_grad():
            actual_delta = torch.cat(
                [
                    target.data.view(-1).index_select(0, locations).float() - before
                    for target, locations, before, _ in verification
                ]
            )
            expected_delta = torch.cat([expected for _, _, _, expected in verification])
            difference = (actual_delta - expected_delta).abs()
            exact_mismatches = actual_delta.ne(expected_delta)
            mismatches = ~torch.isclose(
                actual_delta, expected_delta, rtol=1e-6, atol=1e-8
            )
            stats = torch.stack(
                (
                    difference.sum(),
                    difference.max(),
                    exact_mismatches.sum().float(),
                    mismatches.sum().float(),
                )
            ).cpu()
        return {
            "ok": True,
            "verification_candidates": candidates,
            "verification_samples": actual_delta.numel(),
            "verification_exact_mismatches": int(stats[2]),
            "verification_mismatches": int(stats[3]),
            "verification_abs_sum": float(stats[0]),
            "verification_max_abs": float(stats[1]),
        }
