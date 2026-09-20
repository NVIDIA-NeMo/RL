# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply bounded Engram row updates to vLLM's existing local MXFP8 storage."""

import torch

from nemo_rl.models.engram_refit import parse_row_name


@torch.no_grad()
def load_engram_rows(model: torch.nn.Module, name: str, payload: torch.Tensor) -> bool:
    decoded = parse_row_name(name)
    if decoded is None:
        return False
    from nemo_rl.models.generation.vllm.quantization.deepseek_v41_fp8 import (
        map_checkpoint_name,
    )
    from nemo_rl.models.generation.vllm.quantization.fp8 import quantize_mxfp8_weight

    base, start, count = decoded
    mapped = map_checkpoint_name(model, base)
    module = model.get_submodule(mapped.removesuffix(".weight"))
    if payload.ndim != 2 or payload.shape != (count, module.dim):
        raise ValueError(f"Engram payload shape mismatch: {name} {payload.shape}")
    if start + count > module.num_embeddings:
        raise ValueError(f"Engram payload exceeds table: {name}")
    first = max(start, module.vocab_start_idx)
    last = min(start + count, module.vocab_end_idx)
    if first >= last:
        return True
    weight, scale = module.weight, module.weight_scale_inv
    if weight.is_meta or scale.is_meta:
        from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

        kernel_tensors = get_layerwise_info(module).kernel_tensors
        if kernel_tensors is None:
            raise RuntimeError("Engram row refit has no materialized kernel storage")
        params, _ = kernel_tensors
        weight, scale = params["weight"], params["weight_scale_inv"]
    if weight.is_meta or scale.is_meta:
        raise RuntimeError("Engram row refit destination is still on meta")
    rows = payload[first - start : last - start].to(torch.bfloat16).contiguous()
    # Packed NCCL payloads have dtype alignment only. A contiguous BF16 view
    # can still start at byte offset 2/4/6/...; MXFP8 requires 16-byte alignment.
    # Clone only the bounded local row chunk, not the complete Engram table.
    if rows.data_ptr() % 16:
        rows = rows.clone()
    values, scales = quantize_mxfp8_weight(rows)
    offset = first - module.vocab_start_idx
    # Blocking copies protect pinned CPU/UVA storage when the IPC buffer is reused.
    weight[offset : offset + last - first].copy_(values)
    scale[offset : offset + last - first].copy_(scales)
    return True


@torch.no_grad()
def load_frozen_engram_tables(model: torch.nn.Module, checkpoint: str) -> None:
    """Copy native FP8/scale bytes into local rollout shards, bounded to 32 MiB.

    The checkpoint must also back the trainer's frozen host table. This bypasses
    BF16 decode/requantization and is called once after engine construction.
    """
    import json
    from pathlib import Path

    from safetensors import safe_open

    root = Path(checkpoint)
    config = json.loads((root / "config.json").read_text())["text_config"]
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    layers = config["engram_layer_ids"]
    if not layers:
        raise ValueError("Frozen Engram checkpoint has no Engram layers")
    for layer, rows in zip(layers, config["engram_num_embeddings"], strict=True):
        prefix = f"layers.{layer}.engram.embed"
        mapped = model.hf_to_vllm_mapper.apply_list([prefix + ".weight"])[0]
        module = model.get_submodule(mapped.removesuffix(".weight"))
        first, last = module.vocab_start_idx, module.vocab_end_idx
        dim = config["engram_head_dim"]
        if module.num_embeddings != rows or module.dim != dim or not 0 <= first <= last <= rows:
            raise ValueError(f"Frozen Engram layout mismatch: {prefix}")
        for suffix, destination, width, dtype in (
            ("weight", module.weight, dim, "F8_E4M3"),
            ("scale", module.weight_scale_inv, dim // 32, "F8_E8M0"),
        ):
            name = prefix + "." + suffix
            if destination.is_meta or tuple(destination.shape) != (last - first, width):
                raise ValueError(f"Frozen Engram destination is not materialized: {name}")
            if destination.element_size() != 1:
                raise ValueError(f"Frozen Engram requires native byte storage: {name}")
            with safe_open(root / index[name], framework="pt", device="cpu") as reader:
                source = reader.get_slice(name)
                if source.get_shape() != [rows, width] or source.get_dtype() != dtype:
                    raise ValueError(f"Frozen Engram checkpoint shape/dtype mismatch: {name}")
                step = max(1, (32 * 1024 * 1024) // width)
                for start in range(first, last, step):
                    end = min(last, start + step)
                    destination[start - first:end - first].view(torch.uint8).copy_(
                        source[start:end].view(torch.uint8)
                    )
