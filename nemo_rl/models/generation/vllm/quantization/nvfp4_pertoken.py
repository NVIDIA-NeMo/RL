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
"""NVFP4 per-token W4A4 rollout support (no ModelOpt training dependency).

This module is importable from BOTH sides of the refit boundary:

- Megatron training workers (mcore venv, **no vLLM installed**) use the
  weight producer and the refit iterator filter. Everything at module scope
  is therefore vLLM-free; vLLM imports live inside functions.
- vLLM generation workers use the registered ``nvfp4_pertoken`` quantization
  config (weights pre-quantized by the producer, activation global scales
  derived per token inside the FlashInfer TRT-LLM fused-MoE kernel).

The producer matches vLLM's online-quant kernel
(``vllm._custom_ops.scaled_fp4_quant`` as used by
``_quantize_moe_weight_to_nvfp4``) bit for bit.
"""

import fnmatch
import re
from collections.abc import Iterator
from typing import Any, Optional

import torch

from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_config import (
    DEFAULT_NVFP4_IGNORE,
    NvFp4PerTokenRolloutConfig,
)

__all__ = ["DEFAULT_NVFP4_IGNORE", "NvFp4PerTokenRolloutConfig"]

_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

_FUSED_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<kind>w13|w2)_(?P<part>weight|weight_scale|weight_scale_2)$"
)

_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0
_AMAX_DENOMINATOR = _FP4_MAX * _FP8_E4M3_MAX

# E2M1 rounding boundaries. At a boundary, round-to-nearest-even selects the
# grid point with an even mantissa bit (0.25->0, 0.75->1.0, 1.25->1.0,
# 1.75->2.0, 2.5->2.0, 3.5->4.0, 5.0->4.0).
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Boundary indices whose tie resolves DOWN (toward the lower grid point).
_E2M1_TIE_DOWN = (0, 2, 4, 6)


def _round_e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    """Round |y| to E2M1 codes 0..7 with round-to-nearest-even semantics."""
    bounds = torch.tensor(_E2M1_BOUNDS, device=y.device, dtype=torch.float32)
    mag = y.abs()
    # searchsorted(right=True): ties land on the upper grid point...
    codes = torch.searchsorted(bounds, mag.reshape(-1).contiguous(), right=True)
    codes = codes.reshape(mag.shape).to(torch.uint8)
    # ...then push tie-down boundaries back to the lower point.
    for b_idx in _E2M1_TIE_DOWN:
        codes = torch.where(
            mag == _E2M1_BOUNDS[b_idx],
            torch.tensor(b_idx, device=y.device, dtype=torch.uint8),
            codes,
        )
    sign = (y < 0).to(torch.uint8) << 3
    # satfinite: values beyond the last boundary already clamp to code 7 (6.0)
    return sign | codes


def _quantize_blocks(scaled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-16 NVFP4 quantization of a pre-globally-scaled tensor.

    Mirrors ``scaled_fp4_quant(scaled, global_scale=1, non-swizzled)``:
    per-16-block e4m3 scale = RNE(block_amax / 6); elements are multiplied by
    the reciprocal of the decoded scale (multiply, not divide — matches the
    kernel) and rounded RNE onto the E2M1 grid, then nibble-packed with the
    even element in the low nibble.
    """
    *lead, k = scaled.shape
    if k % 16 != 0:
        raise ValueError(f"last dim must be a multiple of 16, got {k}")
    x = scaled.float().reshape(*lead, k // 16, 16)

    block_amax = x.abs().amax(dim=-1)
    block_scale = (block_amax / _FP4_MAX).to(torch.float8_e4m3fn)
    sf = block_scale.float()
    inv_sf = torch.where(sf > 0, sf.reciprocal(), torch.zeros_like(sf))
    y = x * inv_sf.unsqueeze(-1)

    codes = _round_e2m1_codes(y).reshape(*lead, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), block_scale.contiguous()


def quantize_nvfp4_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight to the NVFP4 (ModelOpt HF checkpoint) layout.

    Accepts ``(N, K)`` (one linear / one expert projection) or ``(E, N, K)``
    (stacked experts). Returns:

    - packed FP4 weight, uint8, ``(..., K // 2)``
    - block scales, float8_e4m3fn, ``(..., K // 16)``
    - global ``weight_scale_2``, float32, scalar for 2D / ``(E,)`` for 3D,
      stored as ``amax / (6 * 448)``

    Matches vLLM's ``_quantize_moe_weight_to_nvfp4`` numerics: per-tensor
    (per-expert) amax, global scale folded in with an intermediate cast back
    to the input dtype, then block-16 quantization under a unit global scale.
    """
    if weight.dim() == 2:
        amax = weight.abs().amax().float().clamp_min(1e-8)
        global_scale = _AMAX_DENOMINATOR / amax
        weight_scale_2 = (1.0 / global_scale).reshape(())
        scaled = (weight.float() * global_scale).to(weight.dtype)
    elif weight.dim() == 3:
        amax = weight.abs().amax(dim=(1, 2)).float().clamp_min(1e-8)
        global_scale = _AMAX_DENOMINATOR / amax
        weight_scale_2 = 1.0 / global_scale
        scaled = (weight.float() * global_scale[:, None, None]).to(weight.dtype)
    else:
        raise ValueError(f"expected 2D or 3D weight, got shape {tuple(weight.shape)}")

    packed, block_scale = _quantize_blocks(scaled)
    return packed, block_scale, weight_scale_2.to(torch.float32)


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def iter_nvfp4_pertoken_weights(
    base_iter: Iterator[tuple[str, torch.Tensor]],
    quant_patterns: list[str],
    ignore_patterns: Optional[list[str]] = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Refit filter: quantize matching ``*.weight`` tensors in an export stream.

    Wraps the ``(hf_name, tensor)`` iterator the Megatron policy worker already
    produces (TP-gathered, HF-named). Per-expert projections matching
    ``quant_patterns`` are collected per layer. Layers outside
    ``ignore_patterns`` are quantized per expert (gate+up share one global
    scale — vLLM's fused-MoE loader keeps only
    ``w13_weight_scale_2[:, 0]``) and emitted as fused stacked tensors in the
    ModelOpt fused-MoE convention::

        <...>.experts.w13_weight          uint8   (E, 2N, K/2)
        <...>.experts.w13_weight_scale    e4m3    (E, 2N, K/16)
        <...>.experts.w13_weight_scale_2  fp32    (E, 2)
        <...>.experts.w2_weight / _scale / _scale_2

    Ignored expert layers stay BF16 but are fused into
    ``experts.gate_up_proj`` and ``experts.down_proj`` tensors for transport.
    Everything else (non-matching weights, biases, kv scales, draft weights)
    passes through untouched. Assumes the export streams a layer's experts
    contiguously (HF checkpoint order), flushing on layer-prefix change.
    """
    ignore = ignore_patterns or []
    quantized = {"layers": 0, "experts": 0}
    bf16_fused_layers = 0
    passthrough = 0

    # Per-(layer-prefix) buffers of expert projections. Experts are stacked
    # and emitted as FUSED tensors (`<prefix>.w13_weight` etc.) purely for
    # transport: streaming per-expert names (~55k tensors on a 128-expert
    # 48-layer model) crawls through per-tensor IPC handshakes and reload
    # buffering and cannot finish a refit in tolerable time. vLLM-side, the
    # worker extension expands them back to per-expert checkpoint names via
    # expand_fused_expert_weights before model.load_weights (the fused names
    # match no entry in RoutedExperts' expert mapping and would be dropped).
    pending: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    pending_ignore: dict[str, bool] = {}

    def _flush(prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
        nonlocal bf16_fused_layers
        group = pending.pop(prefix)
        ignored = pending_ignore.pop(prefix)
        missing = {p for p in ("gate_proj", "up_proj", "down_proj") if p not in group}
        if missing:
            raise RuntimeError(
                f"[nvfp4_pertoken] incomplete expert group for {prefix}: "
                f"missing {sorted(missing)}"
            )
        counts = {p: sorted(group[p]) for p in group}
        num_experts = len(counts["gate_proj"])
        for p, eids in counts.items():
            if eids != list(range(num_experts)):
                raise RuntimeError(
                    f"[nvfp4_pertoken] non-contiguous expert ids for "
                    f"{prefix}.{p}: {eids[:5]}..."
                )

        def _stack(proj: str) -> torch.Tensor:
            return torch.stack([group[proj][e] for e in range(num_experts)], dim=0)

        if ignored:
            bf16_fused_layers += 1
            yield (
                f"{prefix}.gate_up_proj",
                torch.cat([_stack("gate_proj"), _stack("up_proj")], dim=1),
            )
            yield f"{prefix}.down_proj", _stack("down_proj")
            return

        # ONE global scale per expert across the fused gate+up tensor.
        # vLLM's ModelOptNvFp4FusedMoE.process_weights_after_loading collapses
        # w13_weight_scale_2 to column 0 (`[:, 0]`) — per-projection scales
        # would decode the up half with the gate scale, corrupting MoE outputs.
        # Quantizing the stacked (E, 2N, K) tensor matches upstream's
        # online-quant behavior;
        # the (E, 2) checkpoint-convention shape carries the shared scale in
        # both columns.
        w13 = torch.cat([_stack("gate_proj"), _stack("up_proj")], dim=1)
        w13_q, w13_bs, w13_s2 = quantize_nvfp4_weight(w13)
        d_q, d_bs, d_s2 = quantize_nvfp4_weight(_stack("down_proj"))

        quantized["layers"] += 1
        quantized["experts"] += num_experts
        yield f"{prefix}.w13_weight", w13_q
        yield f"{prefix}.w13_weight_scale", w13_bs
        yield (
            f"{prefix}.w13_weight_scale_2",
            w13_s2.unsqueeze(1).expand(-1, 2).contiguous(),
        )
        yield f"{prefix}.w2_weight", d_q
        yield f"{prefix}.w2_weight_scale", d_bs
        yield f"{prefix}.w2_weight_scale_2", d_s2

    current_prefix: Optional[str] = None
    for name, tensor in base_iter:
        m = _EXPERT_WEIGHT_RE.match(name)
        if m is None or not _matches_any(name, quant_patterns):
            passthrough += 1
            yield name, tensor
            continue
        prefix = m.group("prefix")
        if current_prefix is not None and prefix != current_prefix:
            yield from _flush(current_prefix)
        current_prefix = prefix
        ignored = _matches_any(name, ignore)
        previous_ignore = pending_ignore.setdefault(prefix, ignored)
        if previous_ignore != ignored:
            raise RuntimeError(
                f"[nvfp4_pertoken] partial expert-layer ignore for {prefix}; "
                "ignore patterns must cover the complete layer"
            )
        pending.setdefault(prefix, {}).setdefault(m.group("proj"), {})[
            int(m.group("eid"))
        ] = tensor

    for prefix in list(pending):
        yield from _flush(prefix)

    # Per-refit liveness proof: a config/name mismatch (e.g. quant_patterns
    # not matching the export's expert naming) would otherwise silently
    # degrade to an all-BF16 refit that vLLM then fails to load — or worse.
    # Use print because Ray workers default to WARNING-level logging.
    print(
        f"[nvfp4_pertoken] refit: quantized {quantized['layers']} expert layers "
        f"({quantized['experts']} experts) -> {6 * quantized['layers']} fused "
        f"tensors; fused {bf16_fused_layers} BF16 expert layers -> "
        f"{2 * bf16_fused_layers} tensors; passthrough {passthrough}",
        flush=True,
    )
    if quant_patterns and quantized["layers"] == 0:
        raise RuntimeError(
            "[nvfp4_pertoken] refit quantized 0 params although quant_patterns="
            f"{quant_patterns} is configured — export naming and patterns are "
            "out of sync."
        )


def expand_fused_expert_weights(
    weights: Iterator[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Expand fused expert tensors back to per-expert ModelOpt checkpoint names.

    Inverse of :func:`iter_nvfp4_pertoken_weights`'s fused emission, applied
    vLLM-side just before ``model.load_weights``. The fused tensors exist for
    TRANSPORT only (per-expert streaming crawls through per-tensor IPC): vLLM's
    ``RoutedExperts.load_weights`` expert mapping matches per-expert checkpoint
    names (``experts.{e}.gate_proj.weight`` ...) or BF16 HF fused names
    (``experts.gate_up_proj``), but NOT the ``w13_weight``/``w2_weight``
    parameter names — those pass through unmatched and the layerwise-reload
    finalize silently restores the previous kernel tensors
    ("RoutedExperts: Failed to load weights"). Expansion is local slicing
    (views, no copies), so it adds none of the per-tensor transport overhead
    the fusing removed.
    """
    for name, tensor in weights:
        m = _FUSED_EXPERT_RE.match(name)
        if m is None:
            yield name, tensor
            continue
        prefix, kind, part = m.group("prefix"), m.group("kind"), m.group("part")
        num_experts = tensor.shape[0]
        if kind == "w2" and part == "weight_scale_2":
            # Last tensor of a layer's fused group (filter emission order is
            # fixed). Also emit neutral input scales: the quant method
            # registers w13/w2_input_scale params, so the layerwise reloader
            # counts them in load_numel_total — without them every
            # RoutedExperts layer stays "incomplete", vLLM buffers the whole
            # model (~5.4GB/worker) and defers all processing to finalize.
            # The per-token method overwrites input scales with 1.0 in
            # process_weights_after_loading, so streamed 1.0s are consistent.
            one = torch.ones((), device=tensor.device, dtype=torch.float32)
            for e in range(num_experts):
                yield f"{prefix}.{e}.down_proj.weight_scale_2", tensor[e]
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield f"{prefix}.{e}.{proj}.input_scale", one
        elif kind == "w2":
            for e in range(num_experts):
                yield f"{prefix}.{e}.down_proj.{part}", tensor[e]
        elif part == "weight_scale_2":
            # (E, 2) with identical columns (one shared gate+up global scale).
            for e in range(num_experts):
                yield f"{prefix}.{e}.gate_proj.weight_scale_2", tensor[e, 0]
                yield f"{prefix}.{e}.up_proj.weight_scale_2", tensor[e, 1]
        else:
            n = tensor.shape[1] // 2
            for e in range(num_experts):
                yield f"{prefix}.{e}.gate_proj.{part}", tensor[e, :n]
                yield f"{prefix}.{e}.up_proj.{part}", tensor[e, n:]


def build_nvfp4_pertoken_hf_quant_config(ignore: list[str]) -> dict[str, Any]:
    """HF ``quantization_config`` override for the per-token W4A4 rollout.

    A literal dict (no ModelOpt conversion helper): NVFP4 weights with
    block-16 e4m3 scales; activations dynamic (per-token global scales are
    derived inside the kernel, no ``input_scale`` tensors exist).
    """
    # Mirrors the quantization_config of ModelOpt NVFP4 HF checkpoints
    # (e.g. nvidia/Qwen3-30B-A3B-NVFP4 config.json) key-for-key — vLLM's
    # ModelOpt config parser is shape-sensitive (`ignore`, not
    # `exclude_modules`; `targets` inside the group). Only delta:
    # input_activations.dynamic=True since no input_scale tensors exist.
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "producer": {"name": "modelopt"},
        "ignore": list(ignore),
        "config_groups": {
            "group_0": {
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["Linear"],
            }
        },
    }
