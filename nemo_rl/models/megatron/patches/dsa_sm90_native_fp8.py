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

"""Native SM90 FP8 cuDNN scorer for compatible DSA Indexers.

Model adapters provide post-preprocessing Q, K, and raw head weights as FP32
carrier tensors. This module preserves those operands through MCore's normal
layout handling, quantizes the final layouts to E4M3 with UE8M0 descales, folds
the Q descale and score scales into FP32 weights, and calls cuDNN Frontend with
``precision="fp8"``.

The scorer is model-agnostic: it does not implement projections, normalization,
RoPE, or other model preprocessing. It is inference-only for Indexer selection;
the current integration does not implement an auxiliary-loss backward path.
"""

from __future__ import annotations

import functools
import inspect
from importlib import import_module
from types import ModuleType
from typing import Any, Optional

import torch

_CUDNN_KERNEL_MODULE = (
    "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels"
)
_DSA_MODULE = "megatron.core.transformer.experimental_attention_variant.dsa"
_REQUIRED_CUDNN_FP8_PARAMETERS = {
    "precision",
    "q_scale",
    "k_scale",
    "q_causal_offsets",
}
_SUPPORTED_INDEX_HEADS = frozenset({32, 64})
_INDEX_HEAD_DIM = 128
_FP8_E4M3_MAX = 448.0
_FP8_SCALE_AMAX_FLOOR = 1.0e-4


def _fp8_ue8m0_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to E4M3 with one UE8M0 scale per vector."""
    if not x.is_floating_point():
        raise TypeError(
            f"SM90 native FP8 DSA requires floating-point input, got {x.dtype}."
        )
    if x.shape[-1] != _INDEX_HEAD_DIM:
        raise ValueError(
            "SM90 native FP8 DSA requires "
            f"{_INDEX_HEAD_DIM}-wide vectors, got {x.shape[-1]}."
        )

    x_fp32 = x.float()
    amax = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(_FP8_SCALE_AMAX_FLOOR)
    scale = torch.exp2(torch.ceil(torch.log2(amax / _FP8_E4M3_MAX)))
    quantized = (
        (x_fp32 / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    )
    return quantized, scale


def _uses_native_fp8_operands(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
) -> bool:
    """Identify the opt-in FP32 carrier tensors and reject partial conversion."""
    if q.dtype == k.dtype == weights.dtype == torch.float32:
        return True
    # FP32 weights are also cuDNN's normal folded-weight representation for
    # already-quantized FP8 Q/K, so they must remain transparent to the proxy.
    # An FP32 Q or K, however, can only be one of this integration's carrier
    # tensors and therefore indicates an incomplete conversion.
    if q.dtype == torch.float32 or k.dtype == torch.float32:
        raise RuntimeError(
            "SM90 native FP8 DSA requires Q, K, and raw weights to all be "
            "FP32 before final quantization, got "
            f"{(q.dtype, k.dtype, weights.dtype)}."
        )
    return False


def _validate_native_operand_layout(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, ratio: int
) -> None:
    """Validate the SM90 cuDNN FP8 BSHD/THD operand contract."""
    if ratio != 1:
        raise RuntimeError(f"SM90 native FP8 DSA requires ratio=1, got {ratio}.")
    if q.ndim not in (3, 4) or k.ndim != q.ndim or weights.ndim != q.ndim - 1:
        raise RuntimeError(
            "SM90 native FP8 DSA expects THD or BSHD operands, got "
            f"Q/K/W ranks {q.ndim}/{k.ndim}/{weights.ndim}."
        )
    if q.shape[:-1] != weights.shape:
        raise RuntimeError(
            "SM90 native FP8 DSA Q/weight shape mismatch: "
            f"Q={tuple(q.shape)}, weights={tuple(weights.shape)}."
        )
    if k.size(-2) != 1:
        raise RuntimeError(
            f"SM90 native FP8 DSA requires one K head, got {k.size(-2)}."
        )
    if q.size(-2) not in _SUPPORTED_INDEX_HEADS or q.size(-1) != _INDEX_HEAD_DIM:
        raise RuntimeError(
            "SM90 native FP8 DSA requires H in {32, 64} and D=128, got "
            f"H={q.size(-2)}, D={q.size(-1)}."
        )
    if k.size(-1) != q.size(-1):
        raise RuntimeError(
            f"SM90 native FP8 DSA Q/K head dims differ: {q.size(-1)} vs {k.size(-1)}."
        )
    if q.device != k.device or q.device != weights.device:
        raise RuntimeError(
            "SM90 native FP8 DSA Q/K/weights must be on one device, got "
            f"{q.device}, {k.device}, and {weights.device}."
        )
    if not q.is_cuda:
        raise RuntimeError("SM90 native FP8 DSA requires CUDA tensors on SM90.")


def _derive_q_causal_offsets(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build bottom-right causal offsets required by cuDNN Frontend 1.28."""
    if q.ndim == 4:
        return torch.full(
            (q.size(0),),
            k.size(1) - q.size(1),
            dtype=torch.int32,
            device=q.device,
        )

    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise RuntimeError(
            "SM90 native FP8 THD DSA requires cu_seqlens_q and cu_seqlens_k."
        )
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise RuntimeError(
            "SM90 native FP8 THD q/k cu_seqlens shapes differ: "
            f"{tuple(cu_seqlens_q.shape)} vs {tuple(cu_seqlens_k.shape)}."
        )
    return (
        (
            torch.diff(cu_seqlens_k.to(device=q.device))
            - torch.diff(cu_seqlens_q.to(device=q.device))
        )
        .to(dtype=torch.int32)
        .contiguous()
    )


class _NativeFp8CudnnDsaProxy:
    """Delegate every cuDNN DSA API except native-FP8 Indexer scoring."""

    _dsa_sm90_native_fp8_proxy = True

    def __init__(self, delegate: Any):
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def indexer_forward_wrapper(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        ratio: int,
        sm_scale: float,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        q_causal_offsets: Optional[torch.Tensor] = None,
        precision: str = "bf16",
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if not _uses_native_fp8_operands(index_q, index_k, weights):
            return self._delegate.indexer_forward_wrapper(
                index_q,
                index_k,
                weights,
                ratio=ratio,
                sm_scale=sm_scale,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                q_causal_offsets=q_causal_offsets,
                precision=precision,
                q_scale=q_scale,
                k_scale=k_scale,
                **kwargs,
            )

        _validate_native_operand_layout(index_q, index_k, weights, ratio)
        q_fp8, q_descale = _fp8_ue8m0_quantize(index_q)
        k_fp8, k_descale = _fp8_ue8m0_quantize(index_k)
        q_descale = q_descale.squeeze(-1).contiguous()
        k_descale = k_descale.squeeze(-1).contiguous()

        # Fold the per-vector Q descale and score normalization into the FP32
        # head weights. Every multiplication stays in FP32, so the scorer does
        # not introduce another low-precision rounding point.
        folded_weights = weights * q_descale
        folded_weights = folded_weights * (index_q.size(-1) ** -0.5)
        folded_weights = folded_weights * (index_q.size(-2) ** -0.5)
        folded_weights = (folded_weights * float(sm_scale)).contiguous()

        if q_causal_offsets is None:
            q_causal_offsets = _derive_q_causal_offsets(
                index_q, index_k, cu_seqlens_q, cu_seqlens_k
            )
        else:
            q_causal_offsets = q_causal_offsets.to(
                device=index_q.device, dtype=torch.int32
            ).contiguous()

        return self._delegate.indexer_forward_wrapper(
            q_fp8.contiguous(),
            k_fp8.contiguous(),
            folded_weights,
            ratio=ratio,
            # FP32 W tells the SM90 kernel that q_descale*sm_scale is already
            # folded. Keep q_scale and sm_scale neutral; K still needs its
            # actual descale because it is not represented in the weights.
            sm_scale=1.0,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            precision="fp8",
            q_scale=torch.ones_like(q_descale),
            k_scale=k_descale,
            q_causal_offsets=q_causal_offsets,
            **kwargs,
        )


def _validate_sm90_runtime() -> None:
    """Fail at setup rather than after the first expensive training batch."""
    if not torch.cuda.is_available():
        raise RuntimeError("Native FP8 DSA requires a CUDA SM90 runtime.")
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability != (9, 0):
        raise RuntimeError(
            "Native FP8 DSA currently supports SM90 only, got "
            f"SM{capability[0]}{capability[1]}."
        )


def _validate_cudnn_fp8_api(namespace: Any) -> None:
    wrapper = getattr(namespace, "indexer_forward_wrapper", None)
    if not callable(wrapper):
        raise RuntimeError("cuDNN DSA does not expose indexer_forward_wrapper.")
    try:
        signature = inspect.signature(wrapper)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot verify the cuDNN DSA native-FP8 Indexer API at setup."
        ) from exc
    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    missing = _REQUIRED_CUDNN_FP8_PARAMETERS - set(signature.parameters)
    if missing and not has_var_kwargs:
        raise RuntimeError(
            "cuDNN DSA indexer_forward_wrapper lacks native-FP8 parameters "
            f"{sorted(missing)}; cudnn-frontend 1.28+ is required."
        )


def validate_dsa_sm90_fp8_indexer(indexer: Any) -> None:
    """Validate model-independent requirements of the SM90 FP8 scorer."""
    config = indexer.config
    requirements = {
        "experimental_attention_variant": (
            config.experimental_attention_variant,
            "dsa",
        ),
        "dsa_kernel_backend": (config.dsa_kernel_backend, "cudnn"),
        "dsa_indexer_head_dim": (
            config.dsa_indexer_head_dim,
            _INDEX_HEAD_DIM,
        ),
        "dsa_indexer_scoring_relu": (config.dsa_indexer_scoring_relu, True),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in requirements.items()
        if actual != expected
    ]
    attention_backend = getattr(
        config.attention_backend, "value", config.attention_backend
    )
    if attention_backend == "unfused":
        mismatches.append("attention_backend='unfused' (requires fused DSA dispatch)")
    if indexer.index_n_heads not in _SUPPORTED_INDEX_HEADS:
        mismatches.append(
            f"index_n_heads={indexer.index_n_heads!r} (required 32 or 64)"
        )
    if mismatches:
        raise ValueError(
            "SM90 native FP8 DSA received an unsupported scorer configuration: "
            + ", ".join(mismatches)
        )
    if (config.dsa_indexer_loss_coeff or 0.0) > 0.0:
        raise ValueError(
            "SM90 native FP8 DSA requires dsa_indexer_loss_coeff=0 because "
            "the quantized Indexer score path is non-differentiable."
        )


def _patch_global_row_score_helper(cudnn_kernels: ModuleType) -> None:
    """Route MCore's offset/chunk score helper through native cuDNN FP8."""
    original = cudnn_kernels._compute_indexer_scores_chunk_with_global_rows
    if getattr(original, "_dsa_sm90_native_fp8_wrapper", False):
        return

    @functools.wraps(original)
    def native_fp8_global_rows(
        q_chunk_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        w_chunk_bsh: torch.Tensor,
        *,
        row_start: int,
        indexer_ratio: int,
        sm_scale: float,
        seq_lens: Optional[torch.Tensor] = None,
        k_bdk: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not _uses_native_fp8_operands(q_chunk_bshd, k_bshd, w_chunk_bsh):
            return original(
                q_chunk_bshd,
                k_bshd,
                w_chunk_bsh,
                row_start=row_start,
                indexer_ratio=indexer_ratio,
                sm_scale=sm_scale,
                seq_lens=seq_lens,
                k_bdk=k_bdk,
                key_positions=key_positions,
            )

        del k_bdk
        batch, query_rows = q_chunk_bshd.shape[:2]
        key_rows = k_bshd.size(1)
        if indexer_ratio != 1:
            raise RuntimeError(
                "SM90 native FP8 DSA global-row scoring requires ratio=1, "
                f"got {indexer_ratio}."
            )
        if query_rows == 0:
            return torch.empty(
                (batch, 0, key_rows),
                dtype=torch.float32,
                device=q_chunk_bshd.device,
            )
        if seq_lens is None:
            query_positions = torch.arange(
                row_start,
                row_start + query_rows,
                device=q_chunk_bshd.device,
            )
            seq_lens = ((query_positions + 1) // indexer_ratio).clamp(max=key_rows)
        else:
            if seq_lens.ndim != 1 or seq_lens.numel() != query_rows:
                raise RuntimeError(
                    "cuDNN DSA native-FP8 score chunk seq_lens must have shape "
                    f"({query_rows},)."
                )
            seq_lens = seq_lens.to(device=q_chunk_bshd.device, dtype=torch.int64).clamp(
                max=key_rows
            )

        # cuDNN accepts one causal offset per BSHD batch, not per query row.
        # Split only where explicit packed/varlen bounds reset their causal
        # position; an ordinary causal chunk remains one kernel call.
        local_rows = torch.arange(query_rows, device=q_chunk_bshd.device)
        row_offsets = seq_lens - local_rows - 1
        valid_rows = seq_lens > 0
        boundaries = torch.nonzero(
            (row_offsets[1:] != row_offsets[:-1]) | (valid_rows[1:] != valid_rows[:-1]),
            as_tuple=False,
        ).flatten()
        boundary_rows = [0]
        boundary_rows.extend((boundaries + 1).cpu().tolist())
        boundary_rows.append(query_rows)

        score_parts = []
        for group_start, group_end in zip(boundary_rows[:-1], boundary_rows[1:]):
            group_offset = int(row_offsets[group_start].item()) + group_start
            if group_offset < 0:
                score_parts.append(
                    torch.full(
                        (batch, group_end - group_start, key_rows),
                        float("-inf"),
                        dtype=torch.float32,
                        device=q_chunk_bshd.device,
                    )
                )
                continue
            q_causal_offsets = torch.full(
                (batch,),
                group_offset,
                dtype=torch.int32,
                device=q_chunk_bshd.device,
            )
            score_parts.append(
                cudnn_kernels._cudnn_dsa.indexer_forward_wrapper(
                    q_chunk_bshd[:, group_start:group_end].contiguous(),
                    k_bshd,
                    w_chunk_bsh[:, group_start:group_end].contiguous(),
                    ratio=indexer_ratio,
                    sm_scale=sm_scale,
                    q_causal_offsets=q_causal_offsets,
                )["scores"]
            )
        scores = torch.cat(score_parts, dim=1)

        if key_positions is None:
            key_positions = torch.arange(key_rows, device=q_chunk_bshd.device)
        scores.masked_fill_(
            key_positions.view(1, 1, key_rows) >= seq_lens.view(1, query_rows, 1),
            float("-inf"),
        )
        return scores

    native_fp8_global_rows._dsa_sm90_native_fp8_wrapper = True
    cudnn_kernels._compute_indexer_scores_chunk_with_global_rows = (
        native_fp8_global_rows
    )


def _call_operands(
    args: tuple[Any, ...], kwargs: dict[str, Any], names: tuple[str, str, str]
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    values: list[Any] = []
    for position, name in enumerate(names):
        values.append(
            kwargs.get(name, args[position] if position < len(args) else None)
        )
    if not all(isinstance(value, torch.Tensor) for value in values):
        return None
    return values[0], values[1], values[2]


def _patch_backend_entrypoints(cudnn_kernels: ModuleType) -> None:
    """Prevent native operands from ever falling back to a non-FP8 scorer."""
    original_topk = cudnn_kernels.run_fused_qk_topk
    if not getattr(original_topk, "_dsa_sm90_native_fp8_wrapper", False):

        @functools.wraps(original_topk)
        def checked_topk(*args: Any, **kwargs: Any) -> Any:
            operands = _call_operands(args, kwargs, ("q", "k", "weights"))
            native = operands is not None and _uses_native_fp8_operands(*operands)
            result = original_topk(*args, **kwargs)
            if native and result is None:
                raise RuntimeError(
                    "cuDNN declined SM90 native FP8 DSA top-k; refusing to "
                    "fall back to the non-FP8 scorer."
                )
            return result

        checked_topk._dsa_sm90_native_fp8_wrapper = True
        cudnn_kernels.run_fused_qk_topk = checked_topk

    original_full = cudnn_kernels.run_fused_dsa_attention
    if not getattr(original_full, "_dsa_sm90_native_fp8_wrapper", False):

        @functools.wraps(original_full)
        def split_native_fp8_attention(*args: Any, **kwargs: Any) -> Any:
            operands = _call_operands(
                args, kwargs, ("q_indexer", "k_indexer", "indexer_weights")
            )
            if operands is not None and _uses_native_fp8_operands(*operands):
                # Keep Indexer scoring isolated in checked_topk. MCore will still
                # use its fused sparse-attention kernel after top-k selection.
                return None
            return original_full(*args, **kwargs)

        split_native_fp8_attention._dsa_sm90_native_fp8_wrapper = True
        cudnn_kernels.run_fused_dsa_attention = split_native_fp8_attention

    original_topk_with_loss = cudnn_kernels.run_fused_qk_topk_with_loss
    if not getattr(original_topk_with_loss, "_dsa_sm90_native_fp8_wrapper", False):

        @functools.wraps(original_topk_with_loss)
        def reject_native_fp8_loss(*args: Any, **kwargs: Any) -> Any:
            operands = _call_operands(args, kwargs, ("q", "k", "weights"))
            if operands is not None and _uses_native_fp8_operands(*operands):
                raise RuntimeError(
                    "SM90 native FP8 DSA does not support Indexer auxiliary loss."
                )
            return original_topk_with_loss(*args, **kwargs)

        reject_native_fp8_loss._dsa_sm90_native_fp8_wrapper = True
        cudnn_kernels.run_fused_qk_topk_with_loss = reject_native_fp8_loss


def _patch_naive_score_fallback(dsa_module: ModuleType) -> None:
    """Reject MCore paths that bypass the native cuDNN scorer entirely."""
    original = dsa_module.fused_qk_topk_naive
    if getattr(original, "_dsa_sm90_native_fp8_wrapper", False):
        return

    @functools.wraps(original)
    def reject_native_fp8_naive(*args: Any, **kwargs: Any) -> Any:
        operands = _call_operands(args, kwargs, ("q", "k", "weights"))
        if operands is not None and _uses_native_fp8_operands(*operands):
            raise RuntimeError(
                "MCore bypassed the cuDNN scorer for SM90 native FP8 DSA; "
                "this mask/layout is unsupported instead of silently falling back "
                "to naive FP32 top-k."
            )
        return original(*args, **kwargs)

    reject_native_fp8_naive._dsa_sm90_native_fp8_wrapper = True
    dsa_module.fused_qk_topk_naive = reject_native_fp8_naive


def install_dsa_sm90_fp8_scorer() -> None:
    """Install the process-wide cuDNN scorer after validating its runtime API."""
    _validate_sm90_runtime()
    cudnn_kernels = import_module(_CUDNN_KERNEL_MODULE)
    required_helpers = (
        "_ensure_dsa_namespace",
        "_compute_indexer_scores_chunk_with_global_rows",
        "run_fused_qk_topk",
        "run_fused_qk_topk_with_loss",
        "run_fused_dsa_attention",
    )
    missing_helpers = [
        name for name in required_helpers if not hasattr(cudnn_kernels, name)
    ]
    if missing_helpers:
        raise RuntimeError(
            "Unsupported Megatron cuDNN DSA integration; missing helpers "
            f"{missing_helpers}."
        )

    cudnn_kernels._ensure_dsa_namespace()
    namespace = cudnn_kernels._cudnn_dsa
    if namespace is None:
        raise RuntimeError("Megatron failed to initialize the cuDNN DSA namespace.")
    if not getattr(namespace, "_dsa_sm90_native_fp8_proxy", False):
        _validate_cudnn_fp8_api(namespace)
        cudnn_kernels._cudnn_dsa = _NativeFp8CudnnDsaProxy(namespace)

    _patch_global_row_score_helper(cudnn_kernels)
    _patch_backend_entrypoints(cudnn_kernels)
    dsa_module = import_module(_DSA_MODULE)
    if not hasattr(dsa_module, "fused_qk_topk_naive"):
        raise RuntimeError(
            "Unsupported Megatron DSA integration; missing fused_qk_topk_naive."
        )
    _patch_naive_score_fallback(dsa_module)
