# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Runtime patch: inject DSV4-Flash-Base SwiGLU clamp into vLLM's DeepGEMM FP8 MoE.

vLLM's `DeepGemmExperts._act_mul_quant` does not propagate the model's
`swiglu_limit` (which DSV4-Flash-Base ships as 10.0) to the silu_mul kernel.
The sister class `DeepGemmFP4Experts` does pass `gemm1_clamp_limit` — but
the FP8-experts variant we use for Base was missed. Without the clamp,
routed-expert SwiGLU outputs are unbounded and rare-vocab tokens win argmax
at clause boundaries (`)Skip`, `<|begin_of_file|>`, etc.).

We pre-clamp the kernel's `input` (gate=first half, up=second half along dim 1)
out of place before delegating to the original `_act_mul_quant`. The math is
equivalent because `silu(gate)*up` reads gate and up from these halves; clamping
input upstream is the same as clamping inside the kernel.

Activation: set ``NRL_SWIGLU_LIMIT=10`` in the launcher. Unset → no-op.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _read_swiglu_limit() -> float | None:
    raw = os.environ.get("NRL_SWIGLU_LIMIT", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        logger.warning("[swiglu-clamp] NRL_SWIGLU_LIMIT=%r is not a float; patch disabled", raw)
        return None
    if v <= 0:
        return None
    return v


_PATCH_APPLIED = False


def apply_swiglu_clamp_patch() -> None:
    """Monkey-patch vLLM's DeepGemmExperts._act_mul_quant to apply SwiGLU clamp.

    Idempotent across calls in a single process (each Ray worker should call
    this once via apply_fp8_patches).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    limit = _read_swiglu_limit()
    if limit is None:
        logger.info("[swiglu-clamp] NRL_SWIGLU_LIMIT not set or 0; patch disabled (no-op)")
        return

    import torch

    # Import here so the module can be loaded even when vLLM isn't present
    # (e.g. unit tests) — this only fires inside the Ray worker venv where
    # vLLM is available.
    import vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe as dgm
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if not hasattr(dgm, "DeepGemmExperts"):
        logger.warning(
            "[swiglu-clamp] vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe "
            "lacks DeepGemmExperts; vLLM source has changed. Skipping patch."
        )
        return

    original_act_mul_quant = dgm.DeepGemmExperts._act_mul_quant

    def patched_act_mul_quant(self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation):
        # Only SiLU SwiGLU pairs use the gate/up split; other activations stay unchanged.
        if activation == MoEActivation.SILU:
            d = input.shape[1] // 2
            input = torch.cat(
                [
                    input[:, :d].clamp(max=limit),
                    input[:, d:].clamp(min=-limit, max=limit),
                ],
                dim=1,
            )
        return original_act_mul_quant(self, input, output, activation)

    dgm.DeepGemmExperts._act_mul_quant = patched_act_mul_quant
    _PATCH_APPLIED = True
    logger.info(
        "[swiglu-clamp] patched DeepGemmExperts._act_mul_quant with clamp limit=%.4f", limit
    )
    print(f"[swiglu-clamp] patched DeepGemmExperts._act_mul_quant with clamp limit={limit}", flush=True)
