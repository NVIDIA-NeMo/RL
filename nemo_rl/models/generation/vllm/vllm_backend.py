# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import contextlib
import gc
import os
import re
import traceback
import types
from typing import Any

import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
    unpack_ipc_refit_payload_entry,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


def fix_gemma3_vision_weight_name(key: str) -> str:
    """Re-insert the `vision_model` segment into Gemma3 vision-tower weights.

    When performing refit, the vision-tower weight paths are flattened. This unflattens them.
    """
    return re.sub(
        r"vision_tower\.(?!vision_model\.)", "vision_tower.vision_model.", key
    )


_KIMI_K25_ARCHITECTURES = {"KimiK25ForConditionalGeneration"}

_KIMI_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts)\."
    r"(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_packed|weight_scale|weight_shape)$"
)


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "false").lower() in {"1", "true", "yes", "on"}


def _zmq_timeout_ms() -> int:
    raw_value = os.environ.get("NRL_ZMQ_TIMEOUT_MS")
    if raw_value is None:
        return 120000
    try:
        timeout_ms = int(raw_value)
    except ValueError:
        return 120000
    return max(timeout_ms, 0)


def _is_kimi_k25_architecture(architectures: Any) -> bool:
    return any(
        str(architecture) in _KIMI_K25_ARCHITECTURES
        for architecture in architectures or []
    )


def _source_shape_tuple(
    source_shape: torch.Tensor | None,
    source_packed: torch.Tensor,
) -> tuple[int, int]:
    if source_shape is not None:
        values = source_shape.detach().int().cpu().tolist()
        if len(values) >= 2:
            return int(values[0]), int(values[1])
    return int(source_packed.shape[0]), int(source_packed.shape[1] * 8)


def _source_tp_shard_for_kimi_expert(
    *,
    proj: str,
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Tensor | None,
    tp_rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    out_features, in_features = _source_shape_tuple(shape, packed)
    tp_size = max(1, tp_size)
    tp_rank = max(0, min(tp_rank, tp_size - 1))

    if proj in {"gate_proj", "up_proj"}:
        packed_rows = packed.shape[0] // tp_size
        scale_rows = scale.shape[0] // tp_size
        row_start = tp_rank * packed_rows
        scale_row_start = tp_rank * scale_rows
        return (
            packed[row_start : row_start + packed_rows].contiguous(),
            scale[scale_row_start : scale_row_start + scale_rows].contiguous(),
            (out_features // tp_size, in_features),
        )

    packed_cols = packed.shape[1] // tp_size
    scale_cols = scale.shape[1] // tp_size
    col_start = tp_rank * packed_cols
    scale_col_start = tp_rank * scale_cols
    return (
        packed[:, col_start : col_start + packed_cols].contiguous(),
        scale[:, scale_col_start : scale_col_start + scale_cols].contiguous(),
        (out_features, in_features // tp_size),
    )


def _interleave_tp_chunks(
    first: torch.Tensor,
    second: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(
            "Expected 2D Kimi expert scale tensors for TP interleave, "
            f"got {tuple(first.shape)} and {tuple(second.shape)}"
        )
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            "Expected matching Kimi expert scale rows for TP interleave, "
            f"got {tuple(first.shape)} and {tuple(second.shape)}"
        )
    tp_size = max(1, tp_size)
    if tp_size == 1:
        return torch.cat([first, second], dim=1).contiguous()
    if first.shape[1] % tp_size != 0 or second.shape[1] % tp_size != 0:
        raise ValueError(
            "Kimi expert scale columns are not divisible by TP="
            f"{tp_size}: {tuple(first.shape)} and {tuple(second.shape)}"
        )

    first_chunk = first.shape[1] // tp_size
    second_chunk = second.shape[1] // tp_size
    chunks = []
    for rank in range(tp_size):
        chunks.append(
            torch.cat(
                (
                    first[:, rank * first_chunk : (rank + 1) * first_chunk],
                    second[:, rank * second_chunk : (rank + 1) * second_chunk],
                ),
                dim=1,
            )
        )
    return torch.cat(chunks, dim=1).contiguous()


def _vllm_marlin_moe_module() -> Any:
    return __import__(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe",
        fromlist=["marlin_moe_permute_scales"],
    )


def _vllm_gptq_marlin_moe_repack_op(module: Any) -> Any:
    candidates = [module]
    module_ops = getattr(module, "ops", None)
    if module_ops is not None:
        candidates.append(module_ops)

    for module_name in (
        "vllm._custom_ops",
        "vllm.model_executor.layers.fused_moe.fused_marlin_moe",
        "vllm.model_executor.layers.quantization.utils.marlin_utils",
    ):
        try:
            candidates.append(
                __import__(module_name, fromlist=["gptq_marlin_moe_repack"])
            )
        except Exception:
            continue

    for candidate in candidates:
        op = getattr(candidate, "gptq_marlin_moe_repack", None)
        if op is not None:
            return op

    candidate_names = ", ".join(
        getattr(candidate, "__name__", type(candidate).__name__)
        for candidate in candidates
    )
    raise RuntimeError(
        "Could not resolve vLLM gptq_marlin_moe_repack op from candidates: "
        f"{candidate_names}"
    )


def _vllm_marlin_moe_permute_scales_op(module: Any) -> Any:
    op = getattr(module, "marlin_moe_permute_scales", None)
    if op is not None:
        return op

    for module_name in (
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_wna16_marlin",
        "vllm.model_executor.layers.fused_moe.fused_marlin_moe",
    ):
        try:
            candidate = __import__(module_name, fromlist=["marlin_moe_permute_scales"])
        except Exception:
            continue
        op = getattr(candidate, "marlin_moe_permute_scales", None)
        if op is not None:
            return op

    raise RuntimeError("Could not resolve vLLM marlin_moe_permute_scales op")


def _repack_kimi_expert_for_vllm_marlin(
    preprocessed_packed: torch.Tensor,
    target: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if preprocessed_packed.ndim != 2:
        raise ValueError(
            f"Expected 2D preprocessed packed tensor for {name}, "
            f"got {tuple(preprocessed_packed.shape)}"
        )
    module = _vllm_marlin_moe_module()
    device = target.device
    packed = preprocessed_packed.to(device=device, dtype=torch.int32).unsqueeze(0)
    g_idx_sort_indices = torch.empty((1, 0), dtype=torch.int32, device=device)
    repack_op = _vllm_gptq_marlin_moe_repack_op(module)
    repacked = repack_op(
        packed,
        g_idx_sort_indices,
        packed.shape[1] * 8,
        packed.shape[2],
        4,
        is_a_8bit=False,
    )[0].contiguous()
    if tuple(repacked.shape) != tuple(target.shape):
        raise ValueError(
            f"Kimi expert refit packed shape mismatch for {name}: "
            f"candidate={tuple(repacked.shape)} target={tuple(target.shape)}"
        )
    return repacked


def _permute_kimi_expert_scales_for_vllm_marlin(
    scale: torch.Tensor,
    target: torch.Tensor,
    proj: str,
    name: str,
) -> torch.Tensor:
    if scale.ndim != 2:
        raise ValueError(f"Expected 2D scale tensor for {name}, got {tuple(scale.shape)}")
    module = _vllm_marlin_moe_module()
    scales = scale.to(device=target.device, dtype=target.dtype).unsqueeze(0)
    if proj == "w13_fused_gate_up":
        size_k = int(scales.shape[2] * 2)
    elif proj == "down_proj":
        size_k = int(scales.shape[1] * 32)
    else:
        raise ValueError(f"Unsupported Kimi expert scale projection {proj} for {name}")
    permute_scales = _vllm_marlin_moe_permute_scales_op(module)
    permuted = permute_scales(
        s=scales,
        size_k=size_k,
        size_n=int(scales.shape[2]),
        group_size=32,
        is_a_8bit=False,
    )[0].contiguous()
    if tuple(permuted.shape) != tuple(target.shape):
        raise ValueError(
            f"Kimi expert refit scale shape mismatch for {name}: "
            f"candidate={tuple(permuted.shape)} target={tuple(target.shape)}"
        )
    return permuted


def _get_vllm_tp_rank_and_size(vllm_config: Any | None = None) -> tuple[int, int]:
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except Exception:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() % tp_size, tp_size
        return 0, tp_size


def _copy_refit_tensor_(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if tuple(dst.shape) != tuple(src.shape):
        raise RuntimeError(
            f"Kimi refit shape mismatch for {name}: "
            f"target={tuple(dst.shape)} source={tuple(src.shape)}"
        )
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _copy_kimi_expert_weight_shape_if_present(
    target_tensors: dict[str, torch.Tensor],
    name: str,
    local_expert: int,
    shape: tuple[int, int],
) -> int:
    target = target_tensors.get(name)
    if target is None:
        return 0
    target_slice = target[local_expert] if target.ndim > 1 else target
    source = torch.tensor(shape, device=target_slice.device, dtype=target_slice.dtype)
    _copy_refit_tensor_(target_slice, source, f"{name}[{local_expert}]")
    return 1


def _merge_kimi_expert_proj_maps(
    cached: dict[str, dict[str, torch.Tensor | None]] | None,
    current: dict[str, dict[str, torch.Tensor | None]] | None,
) -> dict[str, dict[str, torch.Tensor | None]]:
    merged: dict[str, dict[str, torch.Tensor | None]] = {}
    for proj_map in (cached or {}, current or {}):
        for proj, kind_map in proj_map.items():
            merged.setdefault(proj, {}).update(kind_map)
    return merged


def _cache_kimi_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().contiguous()


def _kimi_proj_map_has_required_tensor(
    proj_map: dict[str, dict[str, torch.Tensor | None]],
) -> bool:
    return any(
        kind in {"weight_packed", "weight_scale"}
        for kind_map in proj_map.values()
        for kind in kind_map
    )


def _kimi_expert_refit_missing_required(
    proj_map: dict[str, dict[str, torch.Tensor | None]],
) -> list[str]:
    return [
        proj
        for proj in ("gate_proj", "up_proj", "down_proj")
        if proj not in proj_map
        or proj_map[proj].get("weight_packed") is None
        or proj_map[proj].get("weight_scale") is None
    ]


def _finalize_kimi_expert_refit(model: torch.nn.Module) -> None:
    cache = getattr(model, "_kimi_expert_refit_cache", None) or {}
    stale_shape_only = [
        group_key
        for group_key, proj_map in cache.items()
        if not _kimi_proj_map_has_required_tensor(proj_map)
    ]
    for group_key in stale_shape_only:
        cache.pop(group_key, None)

    if cache:
        samples = []
        for (prefix, expert_id), proj_map in list(cache.items())[:5]:
            missing = ",".join(_kimi_expert_refit_missing_required(proj_map))
            present = ",".join(
                f"{proj}:{'/'.join(sorted(kind_map))}"
                for proj, kind_map in sorted(proj_map.items())
            )
            samples.append(
                f"{prefix}.{expert_id} missing=[{missing}] present=[{present}]"
            )
        raise RuntimeError(
            "Kimi expert refit ended with incomplete local expert groups: "
            f"count={len(cache)} samples={samples}"
        )

    updated = int(getattr(model, "_kimi_expert_refit_projection_updates", 0))
    if updated:
        local_experts = int(getattr(model, "_kimi_expert_refit_local_experts", 0))
        groups_seen = int(getattr(model, "_kimi_expert_refit_groups_seen", 0))
        shape_updates = int(getattr(model, "_kimi_expert_refit_shape_updates", 0))
        print(
            "Kimi expert refit complete: "
            f"groups_seen={groups_seen} local_experts={local_experts} "
            f"projection_updates={updated} shape_updates={shape_updates} "
            f"stale_shape_only={len(stale_shape_only)}",
            flush=True,
        )


def _apply_kimi_expert_refit_in_place(
    weights: list[tuple[str, torch.Tensor]],
    model: torch.nn.Module,
) -> tuple[list[tuple[str, torch.Tensor]], int]:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    target_tensors = dict(buffers)
    target_tensors.update(params)

    grouped: dict[tuple[str, int], dict[str, dict[str, torch.Tensor | None]]] = {}
    kept: list[tuple[str, torch.Tensor]] = []
    for key, tensor in weights:
        match = _KIMI_EXPERT_WEIGHT_RE.match(key)
        if match is None:
            kept.append((key, tensor))
            continue
        group_key = (match.group("prefix"), int(match.group("expert")))
        grouped.setdefault(group_key, {}).setdefault(match.group("proj"), {})[
            match.group("kind")
        ] = tensor

    if not grouped:
        return kept, 0

    setattr(
        model,
        "_kimi_expert_refit_groups_seen",
        int(getattr(model, "_kimi_expert_refit_groups_seen", 0)) + len(grouped),
    )

    cache = getattr(model, "_kimi_expert_refit_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_kimi_expert_refit_cache", cache)

    updated = 0
    shape_updates = 0
    local_experts = 0
    processed_group_keys: set[tuple[str, int]] = set()
    current_or_cached_group_keys = set(cache) | set(grouped)
    with torch.no_grad():
        for prefix, expert_id in list(current_or_cached_group_keys):
            group_key = (prefix, expert_id)
            proj_map = _merge_kimi_expert_proj_maps(
                cache.get(group_key),
                grouped.get(group_key),
            )
            expert_map = buffers.get(prefix + "._expert_map")
            if expert_map is None or expert_id >= expert_map.numel():
                cache.pop(group_key, None)
                continue
            local_expert = int(expert_map[expert_id].item())
            if local_expert < 0:
                cache.pop(group_key, None)
                continue

            missing = _kimi_expert_refit_missing_required(proj_map)
            if missing:
                continue

            gate = proj_map["gate_proj"]
            up = proj_map["up_proj"]
            down = proj_map["down_proj"]
            gate_packed = gate["weight_packed"]
            gate_scale = gate["weight_scale"]
            up_packed = up["weight_packed"]
            up_scale = up["weight_scale"]
            down_packed = down["weight_packed"]
            down_scale = down["weight_scale"]
            if not (
                isinstance(gate_packed, torch.Tensor)
                and isinstance(gate_scale, torch.Tensor)
                and isinstance(up_packed, torch.Tensor)
                and isinstance(up_scale, torch.Tensor)
                and isinstance(down_packed, torch.Tensor)
                and isinstance(down_scale, torch.Tensor)
            ):
                raise RuntimeError(f"Invalid Kimi expert tensor group for {prefix}.{expert_id}")

            gate_shape = gate.get("weight_shape")
            up_shape = up.get("weight_shape")
            down_shape = down.get("weight_shape")
            gate_shape_t = gate_shape if isinstance(gate_shape, torch.Tensor) else None
            up_shape_t = up_shape if isinstance(up_shape, torch.Tensor) else None
            down_shape_t = down_shape if isinstance(down_shape, torch.Tensor) else None
            gate_dense_shape = _source_shape_tuple(gate_shape_t, gate_packed)
            up_dense_shape = _source_shape_tuple(up_shape_t, up_packed)
            down_dense_shape = _source_shape_tuple(down_shape_t, down_packed)

            w13_packed = params[prefix + ".w13_weight_packed"][local_expert]
            w13_scale = params[prefix + ".w13_weight_scale"][local_expert]
            w2_packed = params[prefix + ".w2_weight_packed"][local_expert]
            w2_scale = params[prefix + ".w2_weight_scale"][local_expert]
            source_device = w13_packed.device
            gate_packed = gate_packed.to(device=source_device, non_blocking=True)
            gate_scale = gate_scale.to(device=source_device, non_blocking=True)
            up_packed = up_packed.to(device=source_device, non_blocking=True)
            up_scale = up_scale.to(device=source_device, non_blocking=True)
            down_packed = down_packed.to(device=source_device, non_blocking=True)
            down_scale = down_scale.to(device=source_device, non_blocking=True)

            w13_preprocessed = (
                torch.cat((gate_packed, up_packed), dim=0).t().contiguous()
            )
            w13_refit_packed = _repack_kimi_expert_for_vllm_marlin(
                w13_preprocessed,
                w13_packed,
                prefix + f".w13_weight_packed[{local_expert}]",
            )

            w13_scale_interleaved = _interleave_tp_chunks(
                gate_scale.t().contiguous(),
                up_scale.t().contiguous(),
                tp_size=1,
            )
            w13_refit_scale = _permute_kimi_expert_scales_for_vllm_marlin(
                w13_scale_interleaved,
                w13_scale,
                "w13_fused_gate_up",
                prefix + f".w13_weight_scale[{local_expert}]",
            )

            w2_refit_packed = _repack_kimi_expert_for_vllm_marlin(
                down_packed.t().contiguous(),
                w2_packed,
                prefix + f".w2_weight_packed[{local_expert}]",
            )

            w2_refit_scale = _permute_kimi_expert_scales_for_vllm_marlin(
                down_scale.t().contiguous(),
                w2_scale,
                "down_proj",
                prefix + f".w2_weight_scale[{local_expert}]",
            )

            _copy_refit_tensor_(
                w13_packed,
                w13_refit_packed,
                prefix + f".w13_weight_packed[{local_expert}]",
            )
            _copy_refit_tensor_(
                w13_scale,
                w13_refit_scale,
                prefix + f".w13_weight_scale[{local_expert}]",
            )
            _copy_refit_tensor_(
                w2_packed,
                w2_refit_packed,
                prefix + f".w2_weight_packed[{local_expert}]",
            )
            _copy_refit_tensor_(
                w2_scale,
                w2_refit_scale,
                prefix + f".w2_weight_scale[{local_expert}]",
            )
            shape_updates += _copy_kimi_expert_weight_shape_if_present(
                target_tensors,
                prefix + ".w13_weight_shape",
                local_expert,
                (gate_dense_shape[0] + up_dense_shape[0], gate_dense_shape[1]),
            )
            shape_updates += _copy_kimi_expert_weight_shape_if_present(
                target_tensors,
                prefix + ".w2_weight_shape",
                local_expert,
                down_dense_shape,
            )
            updated += 2
            local_experts += 1
            processed_group_keys.add(group_key)
            cache.pop(group_key, None)

    for group_key, current_proj_map in grouped.items():
        if group_key in processed_group_keys:
            continue
        prefix, expert_id = group_key
        expert_map = buffers.get(prefix + "._expert_map")
        if expert_map is None or expert_id >= expert_map.numel():
            cache.pop(group_key, None)
            continue
        local_expert = int(expert_map[expert_id].item())
        if local_expert < 0:
            cache.pop(group_key, None)
            continue
        if not _kimi_proj_map_has_required_tensor(current_proj_map):
            continue

        cached_proj_map = cache.setdefault(group_key, {})
        for proj, kind_map in current_proj_map.items():
            cached_kind_map = cached_proj_map.setdefault(proj, {})
            for kind, tensor in kind_map.items():
                cached_kind_map[kind] = _cache_kimi_tensor(tensor)

    if updated:
        setattr(
            model,
            "_kimi_expert_refit_projection_updates",
            int(getattr(model, "_kimi_expert_refit_projection_updates", 0))
            + updated,
        )
        setattr(
            model,
            "_kimi_expert_refit_local_experts",
            int(getattr(model, "_kimi_expert_refit_local_experts", 0)) + local_experts,
        )
        setattr(
            model,
            "_kimi_expert_refit_shape_updates",
            int(getattr(model, "_kimi_expert_refit_shape_updates", 0)) + shape_updates,
        )

    return kept, updated


def _finalize_kimi_proven_tp8_expert_refit(model: torch.nn.Module) -> None:
    if not _is_truthy_env("KIMI_APPLY_PROVEN_TP8_EXPERT_REFIT"):
        return

    cache = getattr(model, "_kimi_proven_tp8_expert_refit_cache", None) or {}
    stale_shape_only = [
        group_key
        for group_key, proj_map in cache.items()
        if not _kimi_proj_map_has_required_tensor(proj_map)
    ]
    for group_key in stale_shape_only:
        cache.pop(group_key, None)

    if cache:
        samples = []
        for (prefix, expert_id), proj_map in list(cache.items())[:5]:
            missing = ",".join(_kimi_expert_refit_missing_required(proj_map))
            present = ",".join(
                f"{proj}:{'/'.join(sorted(kind_map))}"
                for proj, kind_map in sorted(proj_map.items())
            )
            samples.append(
                f"{prefix}.{expert_id} missing=[{missing}] present=[{present}]"
            )
        raise RuntimeError(
            "Kimi proven TP8 expert refit ended with incomplete local expert "
            f"groups: count={len(cache)} samples={samples}"
        )

    updated = int(getattr(model, "_kimi_proven_tp8_expert_refit_projection_updates", 0))
    if updated:
        local_experts = int(
            getattr(model, "_kimi_proven_tp8_expert_refit_local_experts", 0)
        )
        groups_seen = int(
            getattr(model, "_kimi_proven_tp8_expert_refit_groups_seen", 0)
        )
        shape_updates = int(
            getattr(model, "_kimi_proven_tp8_expert_refit_shape_updates", 0)
        )
        print(
            "KIMI proven TP8 expert refit complete: "
            f"groups_seen={groups_seen} local_experts={local_experts} "
            f"projection_updates={updated} shape_updates={shape_updates} "
            f"stale_shape_only={len(stale_shape_only)}",
            flush=True,
        )


def _apply_kimi_proven_tp8_expert_refit_in_place(
    weights: list[tuple[str, torch.Tensor]],
    model: torch.nn.Module,
    vllm_config: Any,
) -> tuple[list[tuple[str, torch.Tensor]], int]:
    """Apply the HF/Megatron -> vLLM Marlin TP8 mapping used by Kimi K2.6."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    target_tensors = dict(buffers)
    target_tensors.update(params)
    tp_rank, tp_size = _get_vllm_tp_rank_and_size(vllm_config)
    if tp_size != 8:
        raise RuntimeError(
            "KIMI_APPLY_PROVEN_TP8_EXPERT_REFIT=true requires vLLM TP=8, "
            f"but this worker reports TP rank/size {tp_rank}/{tp_size}."
        )
    full_expert_targets = _is_truthy_env("KIMI_PROVEN_TP8_EXPERT_REFIT_FULL_TARGETS")

    grouped: dict[tuple[str, int], dict[str, dict[str, torch.Tensor | None]]] = {}
    kept: list[tuple[str, torch.Tensor]] = []
    for key, tensor in weights:
        match = _KIMI_EXPERT_WEIGHT_RE.match(key)
        if match is None:
            kept.append((key, tensor))
            continue
        group_key = (match.group("prefix"), int(match.group("expert")))
        grouped.setdefault(group_key, {}).setdefault(match.group("proj"), {})[
            match.group("kind")
        ] = tensor

    if not grouped:
        return kept, 0

    setattr(
        model,
        "_kimi_proven_tp8_expert_refit_groups_seen",
        int(getattr(model, "_kimi_proven_tp8_expert_refit_groups_seen", 0))
        + len(grouped),
    )

    if not getattr(
        _apply_kimi_proven_tp8_expert_refit_in_place,
        "_kimi_seen_logged",
        False,
    ):
        print(
            "KIMI proven TP8 expert refit received routed expert tensors: "
            f"rank={tp_rank}/{tp_size} groups_in_chunk={len(grouped)}",
            flush=True,
        )
        setattr(
            _apply_kimi_proven_tp8_expert_refit_in_place,
            "_kimi_seen_logged",
            True,
        )

    cache = getattr(model, "_kimi_proven_tp8_expert_refit_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_kimi_proven_tp8_expert_refit_cache", cache)

    updated = 0
    shape_updates = 0
    local_experts = 0
    processed_group_keys: set[tuple[str, int]] = set()
    current_or_cached_group_keys = set(cache) | set(grouped)
    with torch.no_grad():
        for prefix, expert_id in list(current_or_cached_group_keys):
            group_key = (prefix, expert_id)
            proj_map = _merge_kimi_expert_proj_maps(
                cache.get(group_key),
                grouped.get(group_key),
            )
            expert_map = buffers.get(prefix + "._expert_map")
            if expert_map is None or expert_id >= expert_map.numel():
                cache.pop(group_key, None)
                continue
            local_expert = int(expert_map[expert_id].item())
            if local_expert < 0:
                cache.pop(group_key, None)
                continue

            missing = _kimi_expert_refit_missing_required(proj_map)
            if missing:
                continue

            gate = proj_map["gate_proj"]
            up = proj_map["up_proj"]
            down = proj_map["down_proj"]
            gate_packed = gate["weight_packed"]
            gate_scale = gate["weight_scale"]
            up_packed = up["weight_packed"]
            up_scale = up["weight_scale"]
            down_packed = down["weight_packed"]
            down_scale = down["weight_scale"]
            if not (
                isinstance(gate_packed, torch.Tensor)
                and isinstance(gate_scale, torch.Tensor)
                and isinstance(up_packed, torch.Tensor)
                and isinstance(up_scale, torch.Tensor)
                and isinstance(down_packed, torch.Tensor)
                and isinstance(down_scale, torch.Tensor)
            ):
                raise RuntimeError(f"Invalid Kimi expert tensor group for {prefix}.{expert_id}")

            gate_shape = gate.get("weight_shape")
            up_shape = up.get("weight_shape")
            down_shape = down.get("weight_shape")
            gate_shape_t = gate_shape if isinstance(gate_shape, torch.Tensor) else None
            up_shape_t = up_shape if isinstance(up_shape, torch.Tensor) else None
            down_shape_t = down_shape if isinstance(down_shape, torch.Tensor) else None
            gate_dense_shape = _source_shape_tuple(gate_shape_t, gate_packed)
            up_dense_shape = _source_shape_tuple(up_shape_t, up_packed)
            down_dense_shape = _source_shape_tuple(down_shape_t, down_packed)

            w13_packed_full = params[prefix + ".w13_weight_packed"][local_expert]
            w13_scale_full = params[prefix + ".w13_weight_scale"][local_expert]
            w2_packed_full = params[prefix + ".w2_weight_packed"][local_expert]
            w2_scale_full = params[prefix + ".w2_weight_scale"][local_expert]
            source_device = w13_packed_full.device
            gate_packed = gate_packed.to(device=source_device, non_blocking=True)
            gate_scale = gate_scale.to(device=source_device, non_blocking=True)
            up_packed = up_packed.to(device=source_device, non_blocking=True)
            up_scale = up_scale.to(device=source_device, non_blocking=True)
            down_packed = down_packed.to(device=source_device, non_blocking=True)
            down_scale = down_scale.to(device=source_device, non_blocking=True)

            if full_expert_targets:
                w13_packed_param = w13_packed_full
                w13_scale_param = w13_scale_full
                w2_packed_param = w2_packed_full
                w2_scale_param = w2_scale_full
                gate_packed_shard = gate_packed
                up_packed_shard = up_packed
            else:
                w13_packed_cols = w13_packed_full.shape[1] // tp_size
                w13_scale_cols = w13_scale_full.shape[1] // tp_size
                w2_packed_rows = w2_packed_full.shape[0] // tp_size
                w2_scale_rows_target = w2_scale_full.shape[0] // tp_size
                w13_packed_param = w13_packed_full[
                    :,
                    tp_rank * w13_packed_cols : (tp_rank + 1) * w13_packed_cols,
                ]
                w13_scale_param = w13_scale_full[
                    :,
                    tp_rank * w13_scale_cols : (tp_rank + 1) * w13_scale_cols,
                ]
                w2_packed_param = w2_packed_full[
                    tp_rank * w2_packed_rows : (tp_rank + 1) * w2_packed_rows,
                    :,
                ]
                w2_scale_param = w2_scale_full[
                    tp_rank * w2_scale_rows_target : (tp_rank + 1)
                    * w2_scale_rows_target,
                    :,
                ]

                gate_packed_shard, _, _ = _source_tp_shard_for_kimi_expert(
                    proj="gate_proj",
                    packed=gate_packed,
                    scale=gate_scale,
                    shape=gate_shape_t,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                )
                up_packed_shard, _, _ = _source_tp_shard_for_kimi_expert(
                    proj="up_proj",
                    packed=up_packed,
                    scale=up_scale,
                    shape=up_shape_t,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                )

            w13_preprocessed = (
                torch.cat((gate_packed_shard, up_packed_shard), dim=0)
                .t()
                .contiguous()
            )
            w13_packed = _repack_kimi_expert_for_vllm_marlin(
                w13_preprocessed,
                w13_packed_param,
                prefix + f".w13_weight_packed[{local_expert}]",
            )

            w13_scale_interleaved = _interleave_tp_chunks(
                gate_scale.t().contiguous(),
                up_scale.t().contiguous(),
                tp_size,
            )
            if full_expert_targets:
                w13_scale_shard = w13_scale_interleaved
            else:
                w13_scale_cols = w13_scale_interleaved.shape[1] // tp_size
                w13_scale_shard = w13_scale_interleaved[
                    :,
                    tp_rank * w13_scale_cols : (tp_rank + 1) * w13_scale_cols,
                ].contiguous()
            w13_scale = _permute_kimi_expert_scales_for_vllm_marlin(
                w13_scale_shard,
                w13_scale_param,
                "w13_fused_gate_up",
                prefix + f".w13_weight_scale[{local_expert}]",
            )

            if full_expert_targets:
                down_packed_shard = down_packed
            else:
                down_packed_shard, _, _ = _source_tp_shard_for_kimi_expert(
                    proj="down_proj",
                    packed=down_packed,
                    scale=down_scale,
                    shape=down_shape_t,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                )
            w2_packed = _repack_kimi_expert_for_vllm_marlin(
                down_packed_shard.t().contiguous(),
                w2_packed_param,
                prefix + f".w2_weight_packed[{local_expert}]",
            )

            down_scale_t = down_scale.t().contiguous()
            if full_expert_targets:
                w2_scale_shard = down_scale_t
            else:
                w2_scale_rows = down_scale_t.shape[0] // tp_size
                w2_scale_shard = down_scale_t[
                    tp_rank * w2_scale_rows : (tp_rank + 1) * w2_scale_rows,
                    :,
                ].contiguous()
            w2_scale = _permute_kimi_expert_scales_for_vllm_marlin(
                w2_scale_shard,
                w2_scale_param,
                "down_proj",
                prefix + f".w2_weight_scale[{local_expert}]",
            )

            _copy_refit_tensor_(
                w13_packed_param,
                w13_packed,
                prefix + f".w13_weight_packed[{local_expert}]",
            )
            _copy_refit_tensor_(
                w13_scale_param,
                w13_scale,
                prefix + f".w13_weight_scale[{local_expert}]",
            )
            _copy_refit_tensor_(
                w2_packed_param,
                w2_packed,
                prefix + f".w2_weight_packed[{local_expert}]",
            )
            _copy_refit_tensor_(
                w2_scale_param,
                w2_scale,
                prefix + f".w2_weight_scale[{local_expert}]",
            )
            shape_updates += _copy_kimi_expert_weight_shape_if_present(
                target_tensors,
                prefix + ".w13_weight_shape",
                local_expert,
                (gate_dense_shape[0] + up_dense_shape[0], gate_dense_shape[1]),
            )
            shape_updates += _copy_kimi_expert_weight_shape_if_present(
                target_tensors,
                prefix + ".w2_weight_shape",
                local_expert,
                down_dense_shape,
            )
            updated += 2
            local_experts += 1
            processed_group_keys.add(group_key)
            cache.pop(group_key, None)

    for group_key, current_proj_map in grouped.items():
        if group_key in processed_group_keys:
            continue
        prefix, expert_id = group_key
        expert_map = buffers.get(prefix + "._expert_map")
        if expert_map is None or expert_id >= expert_map.numel():
            cache.pop(group_key, None)
            continue
        local_expert = int(expert_map[expert_id].item())
        if local_expert < 0:
            cache.pop(group_key, None)
            continue
        if not _kimi_proj_map_has_required_tensor(current_proj_map):
            continue

        cached_proj_map = cache.setdefault(group_key, {})
        for proj, kind_map in current_proj_map.items():
            cached_kind_map = cached_proj_map.setdefault(proj, {})
            for kind, tensor in kind_map.items():
                cached_kind_map[kind] = _cache_kimi_tensor(tensor)

    if updated:
        setattr(
            model,
            "_kimi_proven_tp8_expert_refit_projection_updates",
            int(getattr(model, "_kimi_proven_tp8_expert_refit_projection_updates", 0))
            + updated,
        )
        setattr(
            model,
            "_kimi_proven_tp8_expert_refit_local_experts",
            int(getattr(model, "_kimi_proven_tp8_expert_refit_local_experts", 0))
            + local_experts,
        )
        setattr(
            model,
            "_kimi_proven_tp8_expert_refit_shape_updates",
            int(getattr(model, "_kimi_proven_tp8_expert_refit_shape_updates", 0))
            + shape_updates,
        )

    if local_experts and not getattr(
        _apply_kimi_proven_tp8_expert_refit_in_place,
        "_kimi_proven_tp8_logged",
        False,
    ):
        print(
            "KIMI proven TP8 expert refit mapping active: "
            f"rank={tp_rank}/{tp_size} local_experts={local_experts} "
            f"full_expert_targets={full_expert_targets} "
            "packed=transpose+gptq_marlin_moe_repack "
            "scale=transpose+marlin_moe_permute_scales",
            flush=True,
        )
        setattr(
            _apply_kimi_proven_tp8_expert_refit_in_place,
            "_kimi_proven_tp8_logged",
            True,
        )

    return kept, updated


def _apply_kimi_expert_refit(
    weights: list[tuple[str, torch.Tensor]],
    model: torch.nn.Module,
    vllm_config: Any,
) -> tuple[list[tuple[str, torch.Tensor]], int]:
    if not _is_truthy_env("KIMI_APPLY_PROVEN_TP8_EXPERT_REFIT"):
        return _apply_kimi_expert_refit_in_place(weights, model)
    if not any(".mlp.experts." in key for key, _ in weights):
        return weights, 0

    filtered, updated = _apply_kimi_proven_tp8_expert_refit_in_place(
        weights, model, vllm_config
    )
    if updated:
        print(
            "Applied proven Kimi TP8 Marlin expert refit tensors in-place: "
            f"{updated} projection tensors",
            flush=True,
        )
    leftovers = [(key, tensor) for key, tensor in filtered if ".mlp.experts." in key]
    if leftovers:
        samples = ",".join(key for key, _ in leftovers[:8])
        print(
            "KIMI proven TP8 expert refit consumed routed expert tensors before "
            f"generic vLLM load: dropped_unhandled={len(leftovers)} samples={samples}",
            flush=True,
        )
        filtered = [
            (key, tensor) for key, tensor in filtered if ".mlp.experts." not in key
        ]
    return filtered, updated


def _raise_if_kimi_expert_weights_reach_generic_loader(
    weights: list[tuple[str, torch.Tensor]],
) -> None:
    leaked = [key for key, _ in weights if ".mlp.experts." in key]
    if leaked:
        samples = ",".join(leaked[:8])
        raise RuntimeError(
            "Kimi expert refit guard: generic vLLM load received "
            f"{len(leaked)} routed expert tensors after expert refit filtering. "
            f"samples={samples}"
        )


def _kimi_nonexpert_name_variants(key: str) -> list[str]:
    variants = [key]
    if key.startswith("language_model."):
        variants.append(key[len("language_model.") :])
    else:
        variants.append("language_model." + key)
    if key.startswith("model."):
        variants.append("language_model." + key)
    if key.startswith("language_model.model."):
        variants.append(key[len("language_model.") :])
    if key.startswith("layers."):
        variants.append("model." + key)
        variants.append("language_model.model." + key)
    return list(dict.fromkeys(variants))


def _kimi_target_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    target_tensors = dict(model.named_parameters())
    target_tensors.update(dict(model.named_buffers()))
    return target_tensors


def _resolve_kimi_nonexpert_target(
    target_tensors: dict[str, torch.Tensor],
    key: str,
) -> tuple[str | None, torch.Tensor | None]:
    for target_name in _kimi_nonexpert_name_variants(key):
        target = target_tensors.get(target_name)
        if target is not None:
            return target_name, target
    return None, None


def _mark_kimi_nonexpert_refit_target_loaded(
    model: torch.nn.Module,
    target_name: str,
) -> None:
    loaded = getattr(model, "_kimi_nonexpert_refit_loaded_targets", None)
    if loaded is None:
        loaded = set()
        setattr(model, "_kimi_nonexpert_refit_loaded_targets", loaded)
    loaded.add(target_name)


def _is_kimi_vision_or_mm_key(key: str) -> bool:
    return key.startswith(("vision_tower.", "mm_projector.", "multi_modal_projector."))


def _get_kimi_nonexpert_fusion_cache(
    model: torch.nn.Module,
) -> dict[str, dict[str, torch.Tensor]]:
    cache = getattr(model, "_kimi_nonexpert_fusion_refit_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_kimi_nonexpert_fusion_refit_cache", cache)
    return cache


def _prepare_row_shard_for_kimi_nonexpert(
    tensor: torch.Tensor,
    target: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | None:
    if tuple(tensor.shape) == tuple(target.shape):
        return tensor.contiguous()
    if tensor.ndim != 2 or target.ndim != 2 or tp_size <= 1:
        return None
    if tensor.shape[0] != target.shape[0] * tp_size:
        return None
    if tensor.shape[1] != target.shape[1]:
        return None
    row_start = tp_rank * target.shape[0]
    return tensor[row_start : row_start + target.shape[0]].contiguous()


def _prepare_col_shard_for_kimi_nonexpert(
    tensor: torch.Tensor,
    target: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | None:
    if tuple(tensor.shape) == tuple(target.shape):
        return tensor.contiguous()
    if tensor.ndim != 2 or target.ndim != 2 or tp_size <= 1:
        return None
    if tensor.shape[0] != target.shape[0]:
        return None
    if tensor.shape[1] != target.shape[1] * tp_size:
        return None
    col_start = tp_rank * target.shape[1]
    return tensor[:, col_start : col_start + target.shape[1]].contiguous()


def _is_kimi_vocab_parallel_weight(key: str) -> bool:
    return key.endswith(("model.embed_tokens.weight", "lm_head.weight"))


def _is_kimi_mla_projection_weight(key: str) -> bool:
    return key.endswith(
        (
            ".self_attn.q_b_proj.weight",
            ".self_attn.kv_b_proj.weight",
            ".self_attn.o_proj.weight",
        )
    )


def _prepare_mla_projection_weight_for_kimi_nonexpert(
    key: str,
    tensor: torch.Tensor,
    target: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | None:
    if key.endswith((".self_attn.q_b_proj.weight", ".self_attn.kv_b_proj.weight")):
        return _prepare_row_shard_for_kimi_nonexpert(tensor, target, tp_rank, tp_size)
    if key.endswith(".self_attn.o_proj.weight"):
        return _prepare_col_shard_for_kimi_nonexpert(tensor, target, tp_rank, tp_size)
    return None


def _kimi_two_source_fusion_spec(
    key: str,
) -> tuple[str, str, tuple[str, str], str] | None:
    if key.endswith(".self_attn.q_a_proj.weight"):
        return (
            key[: -len(".q_a_proj.weight")] + ".fused_qkv_a_proj.weight",
            "q_a",
            ("q_a", "kv_a"),
            "row_full",
        )
    if key.endswith(".self_attn.kv_a_proj_with_mqa.weight"):
        return (
            key[: -len(".kv_a_proj_with_mqa.weight")] + ".fused_qkv_a_proj.weight",
            "kv_a",
            ("q_a", "kv_a"),
            "row_full",
        )
    if key.endswith(".mlp.shared_experts.gate_proj.weight"):
        return (
            key[: -len(".gate_proj.weight")] + ".gate_up_proj.weight",
            "gate",
            ("gate", "up"),
            "row_shard",
        )
    if key.endswith(".mlp.shared_experts.up_proj.weight"):
        return (
            key[: -len(".up_proj.weight")] + ".gate_up_proj.weight",
            "up",
            ("gate", "up"),
            "row_shard",
        )
    if (
        ".mlp.experts." not in key
        and ".mlp.shared_experts." not in key
        and key.endswith(".mlp.gate_proj.weight")
    ):
        return (
            key[: -len(".gate_proj.weight")] + ".gate_up_proj.weight",
            "gate",
            ("gate", "up"),
            "row_shard",
        )
    if (
        ".mlp.experts." not in key
        and ".mlp.shared_experts." not in key
        and key.endswith(".mlp.up_proj.weight")
    ):
        return (
            key[: -len(".up_proj.weight")] + ".gate_up_proj.weight",
            "up",
            ("gate", "up"),
            "row_shard",
        )
    return None


def _prepare_kimi_two_source_fusion(
    key: str,
    tensor: torch.Tensor,
    target_tensors: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tp_rank: int,
    tp_size: int,
) -> tuple[bool, bool]:
    spec = _kimi_two_source_fusion_spec(key)
    if spec is None:
        return False, False

    target_key, part_name, ordered_parts, shard_mode = spec
    target_name, target = _resolve_kimi_nonexpert_target(target_tensors, target_key)
    if target is None or target_name is None:
        return True, False

    cache = _get_kimi_nonexpert_fusion_cache(model)
    entry = cache.setdefault(target_name, {})
    entry[part_name] = tensor.detach().cpu().contiguous()
    if not all(part in entry for part in ordered_parts):
        return True, False

    fused = torch.cat([entry[part] for part in ordered_parts], dim=0)
    if shard_mode == "row_shard":
        prepared = _prepare_row_shard_for_kimi_nonexpert(
            fused, target, tp_rank, tp_size
        )
    else:
        prepared = fused.contiguous() if tuple(fused.shape) == tuple(target.shape) else None
    if prepared is None:
        shapes = {part: tuple(entry[part].shape) for part in ordered_parts}
        raise RuntimeError(
            "Kimi fused non-expert refit shape mismatch for "
            f"{target_name}: parts={shapes} fused={tuple(fused.shape)} "
            f"target={tuple(target.shape)} mode={shard_mode} "
            f"rank={tp_rank}/{tp_size}"
        )

    _copy_refit_tensor_(target, prepared, target_name)
    _mark_kimi_nonexpert_refit_target_loaded(model, target_name)
    cache.pop(target_name, None)
    return True, True


def _prepare_kimi_single_source_fused_nonexpert(
    key: str,
    tensor: torch.Tensor,
    target_tensors: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tp_rank: int,
    tp_size: int,
) -> tuple[bool, bool]:
    del model
    if (
        key.endswith(".mlp.shared_experts.down_proj.weight")
        or (
            ".mlp.experts." not in key
            and ".mlp.shared_experts." not in key
            and key.endswith(".mlp.down_proj.weight")
        )
    ):
        target_name, target = _resolve_kimi_nonexpert_target(target_tensors, key)
        if target is None or target_name is None:
            return True, False
        prepared = _prepare_col_shard_for_kimi_nonexpert(
            tensor, target, tp_rank, tp_size
        )
        if prepared is None:
            raise RuntimeError(
                "Kimi sharded non-expert refit shape mismatch for "
                f"{target_name}: source={tuple(tensor.shape)} "
                f"target={tuple(target.shape)} rank={tp_rank}/{tp_size}"
            )
        _copy_refit_tensor_(target, prepared, target_name)
        return True, True
    return False, False


def _finalize_kimi_nonexpert_fusion_refit(model: torch.nn.Module) -> None:
    cache = getattr(model, "_kimi_nonexpert_fusion_refit_cache", None) or {}
    if not cache:
        return
    samples = [
        f"{target_name}:parts={sorted(parts)}"
        for target_name, parts in list(cache.items())[:8]
    ]
    raise RuntimeError(
        "Kimi fused non-expert refit ended with incomplete source groups: "
        f"count={len(cache)} samples={samples}"
    )


def _apply_kimi_nonexpert_refit_in_place(
    weights: list[tuple[str, torch.Tensor]],
    model: torch.nn.Module,
    vllm_config: Any,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, int]]:
    target_tensors = _kimi_target_tensors(model)
    kept: list[tuple[str, torch.Tensor]] = []
    counts = {
        "exact": 0,
        "vocab": 0,
        "mla": 0,
        "fused": 0,
        "vision_or_mm": 0,
    }
    tp_rank, tp_size = _get_vllm_tp_rank_and_size(vllm_config)

    with torch.no_grad():
        for key, tensor in weights:
            if ".mlp.experts." in key or key.startswith("draft."):
                kept.append((key, tensor))
                continue
            if _is_kimi_vision_or_mm_key(key):
                counts["vision_or_mm"] += 1
                continue

            target_name, target = _resolve_kimi_nonexpert_target(target_tensors, key)
            if (
                target is not None
                and target_name is not None
                and tuple(target.shape) == tuple(tensor.shape)
            ):
                _copy_refit_tensor_(target, tensor, target_name)
                _mark_kimi_nonexpert_refit_target_loaded(model, target_name)
                counts["exact"] += 1
                continue

            if _is_kimi_vocab_parallel_weight(key) and target is not None:
                prepared = _prepare_row_shard_for_kimi_nonexpert(
                    tensor, target, tp_rank, tp_size
                )
                if prepared is not None and target_name is not None:
                    _copy_refit_tensor_(target, prepared, target_name)
                    _mark_kimi_nonexpert_refit_target_loaded(model, target_name)
                    counts["vocab"] += 1
                    continue

            if _is_kimi_mla_projection_weight(key) and target is not None:
                prepared = _prepare_mla_projection_weight_for_kimi_nonexpert(
                    key, tensor, target, tp_rank, tp_size
                )
                if prepared is not None and target_name is not None:
                    _copy_refit_tensor_(target, prepared, target_name)
                    _mark_kimi_nonexpert_refit_target_loaded(model, target_name)
                    counts["mla"] += 1
                    continue

            handled, copied = _prepare_kimi_two_source_fusion(
                key, tensor, target_tensors, model, tp_rank, tp_size
            )
            if handled:
                if copied:
                    counts["fused"] += 1
                continue

            handled, copied = _prepare_kimi_single_source_fused_nonexpert(
                key, tensor, target_tensors, model, tp_rank, tp_size
            )
            if handled:
                if copied:
                    counts["fused"] += 1
                continue

            kept.append((key, tensor))

    return kept, counts


def _is_kimi_moe_process_weights_target(obj: Any) -> bool:
    cls = type(obj)
    module_name = getattr(cls, "__module__", "")
    class_name = getattr(cls, "__name__", "")
    return (
        class_name == "CompressedTensorsWNA16MarlinMoEMethod"
        or (
            "compressed_tensors_moe" in module_name
            and "MoE" in class_name
            and hasattr(obj, "process_weights_after_loading")
        )
    )


def _kimi_noop_moe_process_weights_after_loading(self, layer):
    if not getattr(_kimi_noop_moe_process_weights_after_loading, "_kimi_logged", False):
        print(
            "Skipping Kimi MoE process_weights_after_loading after live refit.",
            flush=True,
        )
        setattr(_kimi_noop_moe_process_weights_after_loading, "_kimi_logged", True)
    return None


@contextlib.contextmanager
def _skip_kimi_moe_process_weights_after_refit(
    model: torch.nn.Module | None = None,
):
    """Bypass MoE post-load processing for Kimi tensors already written in Marlin layout."""
    class_patches: list[tuple[Any, Any]] = []
    instance_patches: list[tuple[Any, Any]] = []
    module_names = (
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe",
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_wna16_marlin",
    )
    class_name = "CompressedTensorsWNA16MarlinMoEMethod"
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            original = (
                getattr(cls, "process_weights_after_loading", None) if cls else None
            )
            if cls is not None and original is not None:
                setattr(
                    cls,
                    "process_weights_after_loading",
                    _kimi_noop_moe_process_weights_after_loading,
                )
                class_patches.append((cls, original))
        except Exception:
            continue

    seen_instances: set[int] = set()
    if model is not None:
        for module in model.modules():
            candidates = [
                module,
                getattr(module, "quant_method", None),
                getattr(getattr(module, "runner", None), "quant_method", None),
            ]
            for candidate in candidates:
                if candidate is None or id(candidate) in seen_instances:
                    continue
                seen_instances.add(id(candidate))
                original = getattr(candidate, "process_weights_after_loading", None)
                if original is None or not _is_kimi_moe_process_weights_target(
                    candidate
                ):
                    continue
                setattr(
                    candidate,
                    "process_weights_after_loading",
                    types.MethodType(
                        _kimi_noop_moe_process_weights_after_loading, candidate
                    ),
                )
                instance_patches.append((candidate, original))

    try:
        yield
    finally:
        for candidate, original in instance_patches:
            setattr(candidate, "process_weights_after_loading", original)
        for cls, original in class_patches:
            setattr(cls, "process_weights_after_loading", original)


def _read_mtp_layer_weights_from_checkpoint(
    model_path: str, mtp_layer_indices: set[int]
) -> list[tuple[str, torch.Tensor]]:
    """Read only the MTP draft layer weights from a sharded HF safetensors checkpoint.

    Uses the checkpoint's ``model.safetensors.index.json`` to open only the
    shards that contain the requested transformer layer indices, so the
    multi-terabyte base-model weights are never read from disk.

    Args:
        model_path: Path to the HF checkpoint directory.
        mtp_layer_indices: Transformer layer indices belonging to the MTP module(s).

    Returns:
        A list of ``(weight_name, tensor)`` pairs for the requested layers, with
        tensors on CPU.
    """
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        match = layer_re.search(name)
        if match is not None and int(match.group(1)) in mtp_layer_indices:
            shard_to_names.setdefault(shard, []).append(name)

    weights: list[tuple[str, torch.Tensor]] = []
    for shard, names in shard_to_names.items():
        with safe_open(
            os.path.join(model_path, shard), framework="pt", device="cpu"
        ) as reader:
            for name in names:
                weights.append((name, reader.get_tensor(name)))
    return weights


class VllmInternalWorkerExtension:
    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            timeout_ms = _zmq_timeout_ms()
            self.zmq_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
            return

        # FP8 KV cache: process KV scales after weight loading
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                self.model_runner.model_config,
                target_device,
            )

    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        This path is only used for the Eagle3 online-training flow, where the
        trainer exports draft parameters under a `draft.` prefix before sending
        them to vLLM.
        This implementation is specific to the eagle model. For MTP, we can add
        similar logic to this function to split weights and send it to the drafter.
        The "draft." prefix is added here https://github.com/isomap/RL/blob/d3a5e1396d00f82fb888d9ec6800687a23bb4017/nemo_rl/models/policy/workers/megatron_policy_worker.py#L967-L997
        """
        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    @staticmethod
    def _trim_vocab_padding(
        draft_model: torch.nn.Module,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Trim padded vocab dimensions from draft weights.

        Megatron pads vocab to a multiple, but vLLM 0.20's autoloader
        strictly asserts loaded_weight.shape[0] == org_vocab_size on
        VocabParallelEmbedding layers. Each such layer may have a
        different org_vocab_size (e.g. embed_tokens uses vocab_size
        while lm_head uses draft_vocab_size), so we match each weight
        to its target module by name.
        """
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        vocab_sizes: dict[str, int] = {}
        for name, module in draft_model.named_modules():
            if isinstance(module, VocabParallelEmbedding):
                vocab_sizes[name] = module.org_vocab_size

        if not vocab_sizes:
            return draft_weights

        trimmed = []
        for key, tensor in draft_weights:
            for mod_name, org_vocab_size in vocab_sizes.items():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in key and tensor.shape[0] > org_vocab_size:
                    tensor = tensor[:org_vocab_size]
                    break
            trimmed.append((key, tensor))
        return trimmed

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None

        if draft_model is None:
            print(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        draft_model.load_weights(weights=draft_weights)

    def load_mtp_weights_from_disk(self, model_path: str) -> bool:
        """Load only the MTP (multi-token-prediction) draft weights from disk.

        Used when an MTP speculative-decoding policy runs with
        ``load_format="dummy"``: the main model receives real weights via refit,
        but the MTP draft layer is not covered by refit (the trainer runs with
        ``mtp_num_layers=0``), so its weights must come from the checkpoint. Only
        the MTP layer(s) are read, avoiding a full base-model load (~1.3 TB for
        DeepSeek-V3) on every inference replica.

        Args:
            model_path: Path to the HF checkpoint directory.

        Returns:
            bool: True if MTP weights were loaded.
        """
        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None
        if draft_model is None:
            print("[mtp] Drafter unavailable; cannot load MTP weights from disk.")
            return False

        predictor = draft_model.model
        mtp_layer_indices = set(
            range(
                predictor.mtp_start_layer_idx,
                predictor.mtp_start_layer_idx + predictor.num_mtp_layers,
            )
        )
        weights = _read_mtp_layer_weights_from_checkpoint(model_path, mtp_layer_indices)
        if not weights:
            raise ValueError(
                f"No MTP layer weights for layers {sorted(mtp_layer_indices)} "
                f"found in checkpoint at {model_path}. The checkpoint must "
                f"include MTP layer weights to run deepseek_mtp speculative decoding."
            )

        self._load_draft_weights(weights)

        # The MTP block contains MoE experts whose weights need post-load
        # processing (e.g. grouped-GEMM layout), matching the main-model path.
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        draft_model_config = (
            self.model_runner.vllm_config.speculative_config.draft_model_config
        )
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(draft_model, draft_model_config, self.device)
        print(
            f"[mtp] Loaded MTP draft weights for layers "
            f"{sorted(mtp_layer_indices)} from {model_path}"
        )
        return True

    def _is_kimi_k25_model(self) -> bool:
        return _is_kimi_k25_architecture(
            self.model_runner.vllm_config.model_config.architectures
        )

    def _load_weights(self, weights):
        """Load weights with Gemma3 vision-tower weight name fix, FP8, and draft-weight support.

        Applies Gemma3 vision-tower weight name fix if needed, splits policy/draft
        weights, applies FP8 conversion if needed, and loads draft weights
        into the drafter model.
        """
        from nemo_rl.models.generation.vllm.quantization import fp8

        if (
            "Gemma3ForConditionalGeneration"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weights[idx] = (fix_gemma3_vision_weight_name(key), weight)

        if self._is_kimi_k25_model():
            weights, expert_updates = _apply_kimi_expert_refit(
                weights,
                self.model_runner.model,
                self.model_runner.vllm_config,
            )
            weights, nonexpert_counts = _apply_kimi_nonexpert_refit_in_place(
                weights,
                self.model_runner.model,
                self.model_runner.vllm_config,
            )
            if expert_updates or any(nonexpert_counts.values()):
                logged = getattr(self, "_kimi_refit_log_count", 0)
                if logged < 8:
                    print(
                        "Loaded Kimi K2.6 live-refit tensors in-place: "
                        f"expert_updates={expert_updates} "
                        f"nonexpert={nonexpert_counts}",
                        flush=True,
                    )
                setattr(self, "_kimi_refit_log_count", logged + 1)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if self._is_kimi_k25_model():
            _raise_if_kimi_expert_weights_reach_generic_loader(policy_weights)
            if not policy_weights:
                self._load_draft_weights(draft_weights)
                return

        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)

        self._load_draft_weights(draft_weights)

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weights = None

        try:
            self.maybe_init_zmq()
            while True:
                # Blocking receive with timeout (this is the main operation)
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    # means the update is done
                    from vllm.config import set_current_vllm_config
                    from vllm.model_executor.model_loader.utils import (
                        process_weights_after_loading,
                    )

                    with set_current_vllm_config(self.model_runner.vllm_config):
                        if self._is_kimi_k25_model():
                            if _is_truthy_env("KIMI_APPLY_PROVEN_TP8_EXPERT_REFIT"):
                                _finalize_kimi_proven_tp8_expert_refit(
                                    self.model_runner.model
                                )
                            else:
                                _finalize_kimi_expert_refit(self.model_runner.model)
                            _finalize_kimi_nonexpert_fusion_refit(
                                self.model_runner.model
                            )
                            if _is_truthy_env("KIMI_SKIP_PROCESS_WEIGHTS_AFTER_REFIT"):
                                print(
                                    "Skipping vLLM process_weights_after_loading after "
                                    "Kimi live refit because "
                                    "KIMI_SKIP_PROCESS_WEIGHTS_AFTER_REFIT=true.",
                                    flush=True,
                                )
                            else:
                                with _skip_kimi_moe_process_weights_after_refit(
                                    self.model_runner.model
                                ):
                                    process_weights_after_loading(
                                        self.model_runner.model,
                                        self.model_config,
                                        self.device,
                                    )
                        else:
                            process_weights_after_loading(
                                self.model_runner.model,
                                self.model_config,
                                self.device,
                            )
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, payload_entries, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weight = None
                weights = []
                offset = 0
                for entry in payload_entries:
                    key, shape, dtype = unpack_ipc_refit_payload_entry(
                        entry,
                        self.state_dict_info,
                    )

                    # Get the weight from the buffer
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weight = (
                        buffer[offset : offset + size_in_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    weights.append((key, weight))

                    # Move offset to the next weight
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size

                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )

                # Load weights into the model
                self._load_weights(weights)

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weight, weights, buffer
                weight = None
                weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        load_model_weight_func = self._load_weights

        try:
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_model_weight_func,
            )

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
