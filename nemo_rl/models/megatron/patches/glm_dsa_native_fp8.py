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

"""GLM-5.2 preprocessing adapter for native SM90 FP8 DSA scoring.

The vLLM GLM-5.2 Indexer keeps LayerNorm and RoPE in FP32 registers before
quantizing Q and K. Megatron's regular path materializes BF16 operands between
these operations, which changes scores near the top-k boundary.

This adapter reproduces the GLM-5.2 preprocessing order and returns FP32 Q, K,
and raw head-weight carrier tensors. The model-independent scorer in
``nemo_rl.models.megatron.patches.dsa_sm90_native_fp8`` owns final E4M3/UE8M0
quantization, scale folding, cuDNN dispatch, and MCore compatibility routing.
"""

from __future__ import annotations

import inspect
import logging
import math
from types import MethodType
from typing import Any, Iterator, Optional

import torch
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer

from nemo_rl.models.megatron.patches.dsa_sm90_native_fp8 import (
    install_dsa_sm90_fp8_scorer,
    validate_dsa_sm90_fp8_indexer,
)

logger = logging.getLogger(__name__)

_GLM52_MODEL_BASENAME = "glm-5.2"
_GLM52_HF_CACHE_COMPONENT = "models--zai-org--glm-5.2"
_OTHER_GLM5_MODEL_BASENAMES = frozenset(
    {"glm-5", "glm-5.1", "glm-5.3", "glm-5.3-flash"}
)
_OTHER_GLM5_HF_CACHE_COMPONENTS = frozenset(
    f"models--zai-org--{name}" for name in _OTHER_GLM5_MODEL_BASENAMES
)
# GLM-5.3 currently shares these fields. Explicit 5.3 identities are excluded
# below; an entirely renamed model with this contract is treated as compatible.
_GLM52_COMPATIBILITY_FINGERPRINT = (
    ("experimental_attention_variant", "dsa"),
    ("dsa_indexer_head_dim", 128),
    ("dsa_indexer_n_heads", 32),
    ("dsa_indexer_topk_freq", 4),
    ("dsa_indexer_skip_topk_offset", 3),
    ("dsa_indexer_rope_interleaved", True),
    ("qk_pos_emb_head_dim", 64),
    ("rotary_base", 8_000_000),
)
_SM90_COMPUTE_CAPABILITY = (9, 0)


def _hf_model_id_parts(hf_model_id: str) -> list[str]:
    """Normalize an HF model ID or local cache path into lowercase components."""
    return [
        part
        for part in hf_model_id.rstrip("/").replace("\\", "/").casefold().split("/")
        if part
    ]


def _is_glm52_model_id(hf_model_id: str) -> bool:
    """Return whether an HF model ID or local cache path identifies GLM-5.2."""
    parts = _hf_model_id_parts(hf_model_id)
    return bool(parts) and (
        _GLM52_MODEL_BASENAME in parts or _GLM52_HF_CACHE_COMPONENT in parts
    )


def _has_glm52_preprocessing_contract(model_config: Any) -> bool:
    """Recognize local or renamed GLM-5.2-compatible model configurations."""
    return all(
        getattr(model_config, name, None) == expected
        for name, expected in _GLM52_COMPATIBILITY_FINGERPRINT
    )


def _selects_glm52_adapter(hf_model_id: str, model_config: Any) -> bool:
    """Select GLM-5.2 while excluding known sibling model identities."""
    if _is_glm52_model_id(hf_model_id):
        return True

    parts = _hf_model_id_parts(hf_model_id)
    if parts and (
        any(part in _OTHER_GLM5_MODEL_BASENAMES for part in parts)
        or any(part in _OTHER_GLM5_HF_CACHE_COMPONENTS for part in parts)
    ):
        return False
    return _has_glm52_preprocessing_contract(model_config)


def _is_sm90_runtime() -> bool:
    """Return whether the current CUDA device supports the scorer kernel."""
    if not torch.cuda.is_available():
        return False
    return (
        torch.cuda.get_device_capability(torch.cuda.current_device())
        == _SM90_COMPUTE_CAPABILITY
    )


def _supports_glm52_native_fp8_config(model_config: Any) -> bool:
    """Return whether the configured DSA path can safely use this patch."""
    requirements = {
        "experimental_attention_variant": "dsa",
        "dsa_kernel_backend": "cudnn",
        "dsa_indexer_head_dim": 128,
        "dsa_indexer_scoring_relu": True,
        "dsa_indexer_rotate_activation": False,
        "rotary_interleaved": False,
        "apply_rope_fusion": False,
        "bf16": True,
        "fp16": False,
    }
    if any(
        getattr(model_config, name, None) != expected
        for name, expected in requirements.items()
    ):
        return False

    attention_backend = getattr(model_config, "attention_backend", None)
    if getattr(attention_backend, "value", attention_backend) == "unfused":
        return False
    if (getattr(model_config, "dsa_indexer_loss_coeff", None) or 0.0) > 0.0:
        return False
    if getattr(model_config, "dsa_indexer_n_heads", None) not in (32, 64):
        return False

    rope_dim = getattr(model_config, "qk_pos_emb_head_dim", None)
    if (
        not isinstance(rope_dim, int)
        or rope_dim <= 0
        or rope_dim > 128
        or rope_dim % 2 != 0
    ):
        return False
    rope_type = getattr(model_config, "rope_type", None)
    if rope_type not in ("rope", "yarn"):
        return False
    if rope_type == "yarn" and any(
        getattr(model_config, name, None) != 1.0
        for name in ("rotary_scaling_factor", "mscale", "mscale_all_dim")
    ):
        return False

    eps = getattr(model_config, "dsa_indexer_k_norm_epsilon", None)
    if eps is None:
        eps = getattr(model_config, "layernorm_epsilon", None)
    return isinstance(eps, (float, int)) and math.isfinite(eps) and eps > 0.0


def _layer_norm_fp32(indexer: DSAIndexer, x: torch.Tensor) -> torch.Tensor:
    """Run the GLM Indexer K LayerNorm entirely in FP32."""
    weight = getattr(indexer.k_norm, "weight", None)
    bias = getattr(indexer.k_norm, "bias", None)
    if weight is None or bias is None:
        raise RuntimeError(
            "GLM-5.2 native FP8 DSA requires affine Indexer K LayerNorm."
        )

    eps = indexer.config.dsa_indexer_k_norm_epsilon
    if eps is None:
        eps = indexer.config.layernorm_epsilon

    x_fp32 = x.float()
    mean = x_fp32.mean(dim=-1, keepdim=True)
    centered = x_fp32 - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + eps)

    norm_weight = weight.float()
    if indexer.config.layernorm_zero_centered_gamma:
        norm_weight = norm_weight + 1.0
    return normalized * norm_weight + bias.float()


def _apply_glm52_rope_fp32(
    indexer: DSAIndexer,
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    mscale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the configured GLM Indexer RoPE without a BF16 materialization."""
    rope_dim = indexer.config.qk_pos_emb_head_dim
    x_pe, x_nope = torch.split(
        x,
        [
            rope_dim,
            indexer.index_head_dim - rope_dim,
        ],
        dim=-1,
    )
    squeezed_batch_dim = False
    if cu_seqlens is not None and cu_seqlens.device != x_pe.device:
        cu_seqlens = cu_seqlens.to(device=x_pe.device)
    if cu_seqlens is not None and x_pe.ndim == 4 and x_pe.size(1) == 1:
        x_pe = x_pe.squeeze(1)
        squeezed_batch_dim = True

    x_pe = apply_rotary_pos_emb(
        x_pe,
        rotary_pos_emb,
        config=indexer.config,
        cu_seqlens=cu_seqlens,
        mscale=mscale,
        cp_group=indexer.pg_collection.cp,
        mla_rotary_interleaved=indexer.config.dsa_indexer_rope_interleaved,
        # MLA-style RoPE groups adjacent inputs as [all-even, all-odd]. Restore
        # the model-configured adjacent layout instead of exposing that internal
        # permutation to the FP8 quantizer.
        mla_output_remove_interleaving=indexer.config.dsa_indexer_rope_interleaved,
    )
    if squeezed_batch_dim:
        x_pe = x_pe.unsqueeze(1)
    return torch.cat([x_pe, x_nope], dim=-1)


def _iter_modules(model: Any) -> Iterator[torch.nn.Module]:
    if isinstance(model, (list, tuple)):
        for chunk in model:
            yield from _iter_modules(chunk)
        return
    if isinstance(model, torch.nn.Module):
        yield from model.modules()


def _glm52_native_fp8_forward_before_topk(
    self: DSAIndexer,
    x: torch.Tensor,
    qr: torch.Tensor,
    packed_seq_params: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 carrier operands after GLM-5.2 LayerNorm and RoPE."""
    packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
        None, None, x, self.config, packed_seq_params
    )
    if self.config.rope_type == "rope":
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        mscale = 1.0
    elif self.config.rope_type == "yarn":
        rotary_pos_emb, mscale = self.rotary_pos_emb(
            rotary_seq_len, packed_seq=packed_seq
        )
    else:  # Guarded by _validate_glm52_native_fp8_indexer.
        raise RuntimeError(f"Unsupported GLM-5.2 RoPE type: {self.config.rope_type!r}.")

    if packed_seq:
        cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(
            packed_seq_params
        )
    else:
        cu_seqlens_q = cu_seqlens_kv = None

    if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
        x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
        qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

    seqlen, batch, _ = x.size()
    q, _ = self.linear_wq_b(qr)
    if q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"GLM-5.2 native FP8 DSA requires BF16 Q projection output, got {q.dtype}."
        )
    q = q.reshape(seqlen, batch, self.index_n_heads, self.index_head_dim).float()
    q = _apply_glm52_rope_fp32(self, q, rotary_pos_emb, mscale, cu_seqlens_q)

    k, _ = self.linear_wk(x)
    if k.dtype != torch.bfloat16:
        raise RuntimeError(
            f"GLM-5.2 native FP8 DSA requires BF16 K projection output, got {k.dtype}."
        )
    k = _layer_norm_fp32(self, k)
    k = k.reshape(seqlen, batch, 1, self.index_head_dim)
    k = _apply_glm52_rope_fp32(self, k, rotary_pos_emb, mscale, cu_seqlens_kv)
    k = k.reshape(seqlen, batch, self.index_head_dim)

    weights, _ = self.linear_weights_proj(x)
    if weights.dtype != torch.bfloat16:
        raise RuntimeError(
            "GLM-5.2 native FP8 DSA requires BF16 weight projection output, got "
            f"{weights.dtype}."
        )
    return q, k, weights.float()


def _native_fp8_forward_with_scores_unsupported(
    self: DSAIndexer, *args: Any, **kwargs: Any
) -> Any:
    del self, args, kwargs
    raise RuntimeError(
        "DSAIndexer.forward_with_scores bypasses the configured native cuDNN FP8 "
        "scorer and is unsupported while the GLM-5.2 SM90 patch is active."
    )


def _validate_glm52_native_fp8_indexer(indexer: DSAIndexer) -> None:
    validate_dsa_sm90_fp8_indexer(indexer)
    config = indexer.config
    requirements = {
        "dsa_indexer_rotate_activation": (config.dsa_indexer_rotate_activation, False),
        "rotary_interleaved": (config.rotary_interleaved, False),
        "apply_rope_fusion": (config.apply_rope_fusion, False),
        "bf16": (config.bf16, True),
        "fp16": (config.fp16, False),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in requirements.items()
        if actual != expected
    ]
    rope_dim = config.qk_pos_emb_head_dim
    if rope_dim <= 0 or rope_dim > indexer.index_head_dim or rope_dim % 2 != 0:
        mismatches.append(
            f"qk_pos_emb_head_dim={rope_dim!r} (required a positive even value "
            f"no larger than dsa_indexer_head_dim={indexer.index_head_dim})"
        )
    if config.rope_type not in ("rope", "yarn"):
        mismatches.append(
            f"rope_type={config.rope_type!r} (required 'rope' or unscaled 'yarn')"
        )
    elif config.rope_type == "yarn":
        neutral_yarn = {
            "rotary_scaling_factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
        mismatches.extend(
            f"{name}={getattr(config, name, None)!r} (required {expected!r} for unscaled yarn)"
            for name, expected in neutral_yarn.items()
            if getattr(config, name, None) != expected
        )
    if mismatches:
        raise ValueError(
            "GLM-5.2 native FP8 DSA received an unsupported preprocessing "
            "configuration: " + ", ".join(mismatches)
        )

    eps = config.dsa_indexer_k_norm_epsilon
    if eps is None:
        eps = config.layernorm_epsilon
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(
            f"DSA Indexer K LayerNorm epsilon must be finite and positive, got {eps}."
        )


def enable_glm52_dsa_native_fp8(model: Any) -> int:
    """Attach the GLM-5.2 adapter and install the generic SM90 FP8 scorer."""
    expected_params = ["self", "x", "qr", "packed_seq_params"]
    actual_params = list(inspect.signature(DSAIndexer.forward_before_topk).parameters)
    if actual_params != expected_params:
        raise RuntimeError(
            "Unsupported Megatron DSAIndexer.forward_before_topk signature for "
            f"GLM-5.2 native FP8: expected {expected_params}, got {actual_params}."
        )

    indexers: list[DSAIndexer] = []
    seen: set[int] = set()
    for module in _iter_modules(model):
        if not isinstance(module, DSAIndexer) or id(module) in seen:
            continue
        seen.add(id(module))
        _validate_glm52_native_fp8_indexer(module)
        indexers.append(module)
    # Pipeline stages that contain only embeddings or the output layer have no
    # local Indexer to patch. Automatic activation must leave those stages alone.
    if not indexers:
        return 0

    pending = [
        indexer
        for indexer in indexers
        if getattr(indexer.forward_before_topk, "__func__", None)
        is not _glm52_native_fp8_forward_before_topk
    ]
    if not pending:
        return 0

    install_dsa_sm90_fp8_scorer()
    for indexer in pending:
        indexer.forward_before_topk = MethodType(
            _glm52_native_fp8_forward_before_topk, indexer
        )
        indexer.forward_with_scores = MethodType(
            _native_fp8_forward_with_scores_unsupported, indexer
        )
    return len(pending)


def maybe_enable_glm52_dsa_native_fp8(
    model: Any, *, hf_model_id: str, model_config: Any
) -> Optional[int]:
    """Enable the GLM-5.2 scorer automatically on a supported SM90 device.

    Returns:
        The number of patched local Indexers when this is a GLM-5.2 SM90
        runtime, including zero for a pipeline stage without an Indexer.
        Returns ``None`` when the model, DSA config, or device does not select
        this patch.
    """
    if not _selects_glm52_adapter(hf_model_id, model_config) or not _is_sm90_runtime():
        return None
    if not _supports_glm52_native_fp8_config(model_config):
        logger.info(
            "Skipping the automatic GLM-5.2 native SM90 FP8 DSA patch because "
            "the configured Indexer path is not compatible."
        )
        return None
    return enable_glm52_dsa_native_fp8(model)
