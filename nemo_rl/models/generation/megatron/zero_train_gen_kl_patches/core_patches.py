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

"""TE cuBLAS and train/logprob log-softmax patches for zero train/gen KL."""

from __future__ import annotations

import importlib
from typing import Callable, Optional

import torch

_TE_CUBLAS_WS_SIZE_FN_ORIG: Optional[Callable[[], int]] = None
_DISTRIBUTED_LOG_SOFTMAX_ORIG: Optional[Callable[..., torch.Tensor]] = None
_TE_BIK_ATTENTION_ASSERT_ORIG: Optional[Callable[[], None]] = None
_LOG_SOFTMAX_PATCHED = False
_TE_BIK_ATTENTION_ASSERT_PATCHED = False

# Minimum workspace that satisfies TE's NVFP4 alpha-scratch guard in cublaslt_gemm.cu.
_TE_CUBLAS_WS_PINNED_BYTES: int = 4


def apply_te_gemm_cublas_pinned_patch(
    target_bytes: int = _TE_CUBLAS_WS_PINNED_BYTES,
) -> None:
    """Shrink TE's cuBLAS workspace so cuBLASLt picks workspace-free algorithms.

    Mirrors megatron.core.transformer.custom_layers.batch_invariant_kernels.
    ``_shrink_te_cublas_workspace_for_invariance``. Intended for zero-KL /
    ``zero_train_gen_mismatch`` only — call from ``_apply_zero_train_gen_mismatch``
    in setup.py, not from generic batch-invariant mode.
    """
    global _TE_CUBLAS_WS_SIZE_FN_ORIG
    if _TE_CUBLAS_WS_SIZE_FN_ORIG is not None:
        return
    try:
        te_gemm_mod = importlib.import_module(
            "transformer_engine.pytorch.cpp_extensions.gemm"
        )
    except ImportError:
        print(
            "te_gemm_cublas_pinned: transformer_engine.pytorch.cpp_extensions.gemm "
            "is not importable; skipping workspace shrink."
        )
        return
    if not hasattr(te_gemm_mod, "get_cublas_workspace_size_bytes"):
        print(
            "te_gemm_cublas_pinned: TE gemm module has no get_cublas_workspace_size_bytes "
            "(TE version mismatch?); skipping workspace shrink."
        )
        return

    _TE_CUBLAS_WS_SIZE_FN_ORIG = te_gemm_mod.get_cublas_workspace_size_bytes
    te_gemm_mod.get_cublas_workspace_size_bytes = lambda: int(target_bytes)
    ws_fn = getattr(te_gemm_mod, "get_cublas_workspace", None)
    if ws_fn is not None and hasattr(ws_fn, "cache_clear"):
        try:
            ws_fn.cache_clear()
        except Exception:  # pylint: disable=broad-except
            pass
    print(
        f"[zero_train_gen_mismatch] shrunk TE cuBLAS workspace to {target_bytes} bytes "
        "(te_gemm_cublas_pinned via core_patches.py)."
    )


def restore_te_gemm_cublas_pinned_patch() -> None:
    """Restore TE's original cuBLAS workspace sizer (for tests)."""
    global _TE_CUBLAS_WS_SIZE_FN_ORIG
    if _TE_CUBLAS_WS_SIZE_FN_ORIG is None:
        return
    try:
        te_gemm_mod = importlib.import_module(
            "transformer_engine.pytorch.cpp_extensions.gemm"
        )
    except ImportError:
        _TE_CUBLAS_WS_SIZE_FN_ORIG = None
        return
    if hasattr(te_gemm_mod, "get_cublas_workspace_size_bytes"):
        te_gemm_mod.get_cublas_workspace_size_bytes = _TE_CUBLAS_WS_SIZE_FN_ORIG
    ws_fn = getattr(te_gemm_mod, "get_cublas_workspace", None)
    if ws_fn is not None and hasattr(ws_fn, "cache_clear"):
        try:
            ws_fn.cache_clear()
        except Exception:  # pylint: disable=broad-except
            pass
    _TE_CUBLAS_WS_SIZE_FN_ORIG = None


def _nrl_inference_compatible_log_softmax(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    if torch.distributed.get_world_size(group) == 1:
        return torch.nn.functional.log_softmax(vocab_parallel_logits, dim=-1)

    assert _DISTRIBUTED_LOG_SOFTMAX_ORIG is not None
    return _DISTRIBUTED_LOG_SOFTMAX_ORIG(vocab_parallel_logits, group)


def apply_log_softmax_determinism_patch() -> None:
    """Match TP=1 train/logprob normalization to Megatron ``raw_logprobs`` inference.

    Intended for ``zero_train_gen_mismatch`` only — call from
    ``_apply_zero_train_gen_mismatch`` in setup.py alongside TE cuBLAS pinning.
    """
    global _DISTRIBUTED_LOG_SOFTMAX_ORIG, _LOG_SOFTMAX_PATCHED
    if _LOG_SOFTMAX_PATCHED:
        return

    from nemo_rl.distributed import model_utils

    _DISTRIBUTED_LOG_SOFTMAX_ORIG = (
        model_utils._compute_distributed_log_softmax_with_grad
    )
    model_utils._compute_distributed_log_softmax_with_grad = (
        _nrl_inference_compatible_log_softmax
    )
    _LOG_SOFTMAX_PATCHED = True
    print(
        "[zero_train_gen_mismatch] patched TP=1 train/logprob to use "
        "inference-compatible fp32 F.log_softmax."
    )


def restore_log_softmax_determinism_patch() -> None:
    """Restore the original distributed log-softmax helper (for tests)."""
    global _DISTRIBUTED_LOG_SOFTMAX_ORIG, _LOG_SOFTMAX_PATCHED
    if not _LOG_SOFTMAX_PATCHED or _DISTRIBUTED_LOG_SOFTMAX_ORIG is None:
        return

    from nemo_rl.distributed import model_utils

    model_utils._compute_distributed_log_softmax_with_grad = (
        _DISTRIBUTED_LOG_SOFTMAX_ORIG
    )
    _DISTRIBUTED_LOG_SOFTMAX_ORIG = None
    _LOG_SOFTMAX_PATCHED = False


def apply_te_bik_attention_assert_skip_patch() -> None:
    """Skip Megatron's TE>=2.18 batch-invariant attention gate for zero-KL on TE 2.15.

    ``zero_train_gen_mismatch`` sets ``config.batch_invariant_mode=True`` (MoE
    DeepGEMM validation) but does not call global ``enable_batch_invariant_mode()``.
    Determinism comes from ``apply_te_gemm_cublas_pinned_patch`` and MoE fixed-order
    combine instead of Megatron's TE FA version pinning.
    """
    global _TE_BIK_ATTENTION_ASSERT_ORIG, _TE_BIK_ATTENTION_ASSERT_PATCHED
    if _TE_BIK_ATTENTION_ASSERT_PATCHED:
        return
    try:
        bik_mod = importlib.import_module(
            "megatron.core.transformer.custom_layers.batch_invariant_kernels"
        )
    except ImportError:
        print(
            "te_bik_attention_assert_skip: Megatron batch_invariant_kernels is not "
            "importable; skipping TE attention assert bypass."
        )
        return

    _TE_BIK_ATTENTION_ASSERT_ORIG = bik_mod.assert_te_supports_batch_invariant_attention
    bik_mod.assert_te_supports_batch_invariant_attention = lambda: None
    _TE_BIK_ATTENTION_ASSERT_PATCHED = True
    print(
        "[zero_train_gen_mismatch] skipped Megatron TE batch-invariant attention "
        "assert (using core_patches on TE 2.15; not enable_batch_invariant_mode)",
        flush=True,
    )


def restore_te_bik_attention_assert_skip_patch() -> None:
    """Restore Megatron's TE batch-invariant attention assert (for tests)."""
    global _TE_BIK_ATTENTION_ASSERT_ORIG, _TE_BIK_ATTENTION_ASSERT_PATCHED
    if not _TE_BIK_ATTENTION_ASSERT_PATCHED or _TE_BIK_ATTENTION_ASSERT_ORIG is None:
        return
    try:
        bik_mod = importlib.import_module(
            "megatron.core.transformer.custom_layers.batch_invariant_kernels"
        )
    except ImportError:
        _TE_BIK_ATTENTION_ASSERT_ORIG = None
        _TE_BIK_ATTENTION_ASSERT_PATCHED = False
        return
    bik_mod.assert_te_supports_batch_invariant_attention = _TE_BIK_ATTENTION_ASSERT_ORIG
    _TE_BIK_ATTENTION_ASSERT_ORIG = None
    _TE_BIK_ATTENTION_ASSERT_PATCHED = False
