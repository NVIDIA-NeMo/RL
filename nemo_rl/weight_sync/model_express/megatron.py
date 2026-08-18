# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Megatron tensor selection for ModelExpress publication."""

from __future__ import annotations

from typing import Any


def _inherits(mapping: Any, class_name: str) -> bool:
    return any(base.__name__ == class_name for base in type(mapping).__mro__)


def model_express_role_for_mapping(
    *, mapping: Any, megatron_module: Any, tensor_ndim: int
) -> str:
    """Return the MX role for one supported Megatron-Bridge conversion mapping."""
    mapping_name = type(mapping).__name__
    if getattr(mapping, "is_expert", False):
        raise NotImplementedError(
            "ModelExpress NeMo-RL integration does not yet support "
            f"expert mapping {mapping_name}"
        )
    if _inherits(mapping, "QKVMapping"):
        if tensor_ndim != 2:
            raise NotImplementedError(
                "ModelExpress QKV publication currently supports weights only"
            )
        return "qkv_column"
    if _inherits(mapping, "GatedMLPMapping"):
        return "gated_mlp_column"
    if _inherits(mapping, "ColumnParallelMapping"):
        return "column"
    if _inherits(mapping, "RowParallelMapping"):
        return "row" if tensor_ndim > 1 else "replicated"
    if _inherits(mapping, "ReplicatedMapping") or _inherits(mapping, "DirectMapping"):
        return "replicated"
    if mapping_name == "AutoMapping":
        if getattr(mapping, "permute_dims", None) is not None:
            raise NotImplementedError(
                "ModelExpress publication does not yet support AutoMapping "
                "dimension permutation"
            )
        detected = mapping._detect_parallelism_type(megatron_module)
        if detected == "column":
            return "column"
        if detected == "row":
            return "row" if tensor_ndim > 1 else "replicated"
        if detected == "replicated":
            return "replicated"
        raise ValueError(f"AutoMapping returned unsupported parallelism {detected!r}")
    raise NotImplementedError(
        f"ModelExpress publication does not support Megatron mapping {mapping_name}"
    )


def _qkv_descriptor_extras(
    *, transformer_config: Any, tensor_parallel_size: int, global_rows: int
) -> dict[str, str]:
    """Build the global QKV geometry consumed by ModelExpress aliasing."""
    num_heads = int(transformer_config.num_attention_heads)
    num_kv_heads = int(transformer_config.num_query_groups)
    head_dim = int(transformer_config.kv_channels)
    if (
        tensor_parallel_size < 1
        or num_heads < 1
        or num_kv_heads < 1
        or head_dim < 1
        or num_heads % num_kv_heads
    ):
        raise ValueError("invalid global Q/KV geometry for ModelExpress")
    expected_rows = (num_heads + 2 * num_kv_heads) * head_dim
    if global_rows != expected_rows:
        raise ValueError(
            f"fused QKV rows {global_rows} disagree with global head geometry "
            f"{expected_rows}"
        )

    extras = {
        "qkv_interleave": "by_head",
        "head_dim": str(head_dim),
        "num_heads": str(num_heads),
        "num_kv_heads": str(num_kv_heads),
    }
    if (
        num_heads % tensor_parallel_size == 0
        and num_kv_heads % tensor_parallel_size == 0
    ):
        extras["num_heads_local"] = str(num_heads // tensor_parallel_size)
        extras["num_kv_heads_local"] = str(num_kv_heads // tensor_parallel_size)
    return extras


def build_megatron_tensor_specs(
    *,
    conversion_tasks: list[Any],
    transformer_config: Any,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> list[Any]:
    """Translate local Megatron-Bridge conversion tasks into MX tensor specs."""
    from modelexpress_rl.train.engines.megatron import MegatronTensorSpec

    specs = []
    for task in conversion_tasks:
        tensor = task.param_weight
        if tensor is None:
            continue
        role = model_express_role_for_mapping(
            mapping=task.mapping,
            megatron_module=task.megatron_module,
            tensor_ndim=tensor.ndim,
        )
        hf_param = task.mapping.hf_param
        extras: dict[str, str] = {}
        if role == "qkv_column":
            hf_names = tuple(hf_param[key] for key in ("q", "k", "v"))
            extras = _qkv_descriptor_extras(
                transformer_config=transformer_config,
                tensor_parallel_size=tensor_parallel_size,
                global_rows=int(tensor.shape[0]) * tensor_parallel_size,
            )
            shard_axis = 0
        elif role == "gated_mlp_column":
            hf_names = (hf_param["gate"], hf_param["up"])
            extras = {"gated_mlp_order": "gate_then_up"}
            shard_axis = 0
        elif role == "column":
            hf_names = (str(hf_param),)
            shard_axis = 0
        elif role == "row":
            hf_names = (str(hf_param),)
            shard_axis = 1
        else:
            hf_names = (str(hf_param),)
            shard_axis = None

        if role == "replicated":
            if tensor_parallel_rank != 0:
                continue
            global_shape = tuple(int(dim) for dim in tensor.shape)
            shard_range = None
            placement = "REPLICATE"
        else:
            assert shard_axis is not None
            local_extent = int(tensor.shape[shard_axis])
            global_shape_list = [int(dim) for dim in tensor.shape]
            global_shape_list[shard_axis] *= tensor_parallel_size
            global_shape = tuple(global_shape_list)
            shard_range = (
                tensor_parallel_rank * local_extent,
                (tensor_parallel_rank + 1) * local_extent,
            )
            placement = "SHARD"

        specs.append(
            MegatronTensorSpec(
                name=task.global_param_name,
                tensor=tensor.detach(),
                role=role,
                hf_names=hf_names,
                global_shape=global_shape,
                placement_kind=placement,
                shard_axis=shard_axis,
                local_shard_range=shard_range,
                extras=extras,
            )
        )
    if not specs:
        raise RuntimeError("this Megatron rank has no ModelExpress tensors")
    return specs


__all__ = ["build_megatron_tensor_specs", "model_express_role_for_mapping"]
