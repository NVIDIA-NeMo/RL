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

r"""Run Megatron forward-only and save last-token logits.

Run a batch of real prompts through a Megatron model and save the last-token
logits for each prompt. Defaults to Llama-3.1-8B; pass ``--model
nemotron3-nano`` or the full
``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`` id for Nemotron 3 Nano.

BF16 (default):
    uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py

MXFP8 (Megatron consumes the BF16 HF ckpt unchanged and quantizes activations
and GEMM weights on the fly via TE's MXFP8BlockScaling recipe):
    uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py \\
        --mxfp8 [--fp8-format e4m3|hybrid]

Setting ``--mxfp8`` sets ``provider.fp8`` to the chosen FP8 element format and
``provider.fp8_recipe = "mxfp8"``. ``TransformerBlock.forward`` then triggers
``fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling(...))`` for every
decoder layer, so all linears run as MXFP8 GEMMs. ``fp8_param`` is left at
default (False) — it's only required for the ``inference_optimized`` impl,
which we are not using.
"""

import argparse
import os

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, print_rank_0
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.module import Float16Module
from tensor_capture import get_debug_tensor_capture, install_debug_tensor_hooks
from transformers import AutoTokenizer

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NEMOTRON3_NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_ALIASES = {
    "llama": DEFAULT_MODEL,
    "llama3.1-8b": DEFAULT_MODEL,
    "llama-3.1-8b": DEFAULT_MODEL,
    "nemotron3-nano": NEMOTRON3_NANO_MODEL,
    "nemotron-3-nano": NEMOTRON3_NANO_MODEL,
    "nemotron3-nano-30b-a3b": NEMOTRON3_NANO_MODEL,
}
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_SUBSET = "main"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_FIELD = "question"
DEFAULT_NUM_PROMPTS = 32

G_VLLM_STYLE_SDPA_SEQ_LENS: list[int] | None = None


def resolve_model_ref(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def is_nemotron3_nano_ref(value: str | None) -> bool:
    if value is None:
        return False
    value = resolve_model_ref(value)
    normalised = value.lower().replace("_", "-")
    return (
        value == NEMOTRON3_NANO_MODEL or "nvidia-nemotron-3-nano-30b-a3b" in normalised
    )


def model_output_tag(model: str, tokenizer: str | None) -> str:
    if is_nemotron3_nano_ref(model) or is_nemotron3_nano_ref(tokenizer):
        return "_nemotron3_nano"
    return ""


def parse_layer_indices(value: str) -> list[int]:
    layers: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid layer range {item!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    if not layers:
        raise argparse.ArgumentTypeError("--capture-layers parsed to an empty set")
    return sorted(layers)


def load_prompts(dataset, subset, split, field, n, seed):
    """Load `n` non-empty prompts from a HuggingFace dataset, deterministically."""
    from datasets import load_dataset

    kwargs = {"split": split}
    if subset:
        ds = load_dataset(dataset, subset, **kwargs)
    else:
        ds = load_dataset(dataset, **kwargs)
    ds = ds.shuffle(seed=seed)
    prompts = []
    for row in ds:
        text = row.get(field)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        prompts.append(text)
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        raise RuntimeError(
            f"only found {len(prompts)} non-empty '{field}' rows in "
            f"{dataset}:{subset}:{split}, needed {n}"
        )
    return prompts


def _normalise_token_ids_list(value, lengths=None):
    if isinstance(value, torch.Tensor):
        token_ids_rows = value.detach().cpu().to(torch.long)
        if token_ids_rows.dim() == 1:
            token_ids_rows = token_ids_rows.unsqueeze(0)
        if token_ids_rows.dim() != 2:
            raise ValueError(
                f"expected 1D or 2D token id tensor, got {tuple(token_ids_rows.shape)}"
            )
        if lengths is None:
            return [
                [int(token_id) for token_id in row.tolist()] for row in token_ids_rows
            ]
        length_values = [int(length) for length in lengths]
        return [
            [int(token_id) for token_id in token_ids_rows[i, :length].tolist()]
            for i, length in enumerate(length_values)
        ]

    token_ids_list = []
    for row in value:
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().to(torch.long).tolist()
        token_ids_list.append([int(token_id) for token_id in row])
    return token_ids_list


def load_token_ids_from_file(path: str, token_ids_key: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if token_ids_key not in payload:
            raise KeyError(f"{path} does not contain token id key {token_ids_key!r}")
        lengths = payload.get("seq_lens")
        if lengths is None:
            lengths = payload.get("input_lengths")
        token_ids_list = _normalise_token_ids_list(payload[token_ids_key], lengths)
        prompts = payload.get("prompts")
        if not isinstance(prompts, list) or len(prompts) != len(token_ids_list):
            prompts = [f"{token_ids_key}[{i}]" for i in range(len(token_ids_list))]
        else:
            prompts = [str(prompt) for prompt in prompts]
        return prompts, token_ids_list, payload

    token_ids_list = _normalise_token_ids_list(payload)
    prompts = [f"{token_ids_key}[{i}]" for i in range(len(token_ids_list))]
    return prompts, token_ids_list, {"token_ids_list": token_ids_list}


def default_output(
    batch_invariant: bool,
    split_fused: bool,
    no_rope_fusion: bool = False,
    vllm_rope: bool = False,
    vllm_swiglu: bool = False,
    vllm_sdpa: bool = False,
    vllm_rmsnorm: bool = False,
    split_all_fused: bool = False,
    mxfp8: bool = False,
    vllm_paged_sdpa: bool = False,
    model_tag: str = "",
) -> str:
    suffix = model_tag
    if mxfp8:
        suffix += "_mxfp8"
    if split_all_fused:
        suffix += "_splitall"
    elif split_fused:
        suffix += "_split"
    if no_rope_fusion:
        suffix += "_norope"
    if vllm_rope:
        suffix += "_vllmrope"
    if vllm_swiglu:
        suffix += "_vllmswiglu"
    if vllm_paged_sdpa:
        suffix += "_vllmpagedsdpa"
    if vllm_sdpa:
        suffix += "_vllmsdpa"
    if vllm_rmsnorm:
        suffix += "_vllmrmsnorm"
    if batch_invariant:
        suffix += "_bi"
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"megatron_capture{suffix}.pt",
    )


def install_mxfp8_compact_scales():
    """Force all MXFP8 tensors to use compact (non-swizzled) scales.

    TE's MXFP8 quantizer normally stores per-block scales in a swizzled
    cuBLASLt-friendly layout when ``optimize_for_gemm=True``. cuBLASLt's
    block-scaled fp8 GEMM consumes that layout directly. But the C++ kernel
    behind ``tex.dequantize`` asserts on swizzled tensors:

        Assertion failed: !input.with_gemm_swizzled_scales.
        Input must have scales in compact format.

    Since the BI MXFP8 path dequants both inputs back to bf16 before calling
    the BF16 BI matmul, we don't need the swizzled layout at all.

    TE's basic_linear / forward_grouped_mlp / backward_grouped_mlp paths set
    ``input_quantizer.optimize_for_gemm = True`` right before quantization,
    so patching ``make_empty`` (which doesn't fire on the tex.quantize path)
    isn't enough. We replace ``optimize_for_gemm`` on the ``MXFP8Quantizer``
    class with a property that always reads ``False`` and silently swallows
    ``True`` writes. The C++ side reads this attribute via
    ``quantizer.attr("optimize_for_gemm").cast<bool>()`` per call
    (``quantizer.cpp:116``), so every newly-allocated MXFP8 tensor
    inherits ``with_gemm_swizzled_scales=False``.

    Must be called BEFORE ``model_provider.finalize()`` /
    ``provide_distributed_model`` so the first MXFP8 quantizers picked up
    inside ``fp8_autocast`` see the patched property.
    """
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    def _get(self):
        return False

    def _set(self, value):
        # Silently swallow `True` writes; we always want compact scales.
        pass

    MXFP8Quantizer.optimize_for_gemm = property(_get, _set)


def install_mxfp8_passthrough_for_bi_gemm():
    """Route MXFP8 GEMMs around the BI matmul; BF16 GEMMs still hit BI Triton.

    Interim fix that lets ``--batch-invariant`` coexist with ``--mxfp8``
    without attempting any vLLM bit-identity. Megatron's BI ``general_gemm``
    patch (``_te_general_gemm_patched``) immediately reads ``A.is_cuda`` /
    ``B.is_cuda`` on every call, but TE's quantised storage types
    (``MXFP8TensorStorage`` / ``Float8TensorStorage``) do not expose
    ``is_cuda``, so MXFP8 linears under BI mode raise
    ``AttributeError: 'MXFP8TensorStorage' object has no attribute 'is_cuda'``.
    This wrapper detects non-``torch.Tensor`` GEMM operands (TE quantised
    storages) and forwards the call to TE's original ``general_gemm``;
    regular CUDA tensors still go through the BI Triton matmul.
    """
    import megatron.core.extensions.transformer_engine as meg_te
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod
    import transformer_engine.pytorch.module.linear as te_linear_mod
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )

    if bik_mod._TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError(
            "enable_batch_invariant_mode() must run before "
            "install_mxfp8_passthrough_for_bi_gemm()"
        )

    orig_gemm = bik_mod._TE_GENERAL_GEMM_ORIG
    bi_gemm = bik_mod._te_general_gemm_patched
    extract = bik_mod._extract_te_gemm_args

    def _wrapper(*args, **kwargs):
        a, b, _, _, _, _, _ = extract(args, kwargs)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return orig_gemm(*args, **kwargs)
        return bi_gemm(*args, **kwargs)

    for mod, attr in (
        (te_cpp, "general_gemm"),
        (te_linear_mod, "general_gemm"),
        (te_layernorm_linear_mod, "general_gemm"),
        (meg_te, "general_gemm"),
    ):
        if hasattr(mod, attr):
            setattr(mod, attr, _wrapper)


def install_mxfp8_dequant_for_bi_gemm():
    """Route MXFP8 GEMMs through BF16 batch-invariant matmul.

    When ``--batch-invariant`` and ``--mxfp8`` are both on, dequantise both
    inputs to bf16 first.

    Mirrors vLLM's ``BatchInvariantMxfp8LinearKernel`` (which dequants weight
    via ``dequant_mxfp8_to_bf16`` and calls ``matmul_persistent``):
    elementwise dequant is naturally batch-invariant, and the BF16 BI matmul
    has a fixed K-reduction order, so the result is deterministic and
    matches across engines if both sides dequant identically.

    On TE side, the FP8 input is an ``MXFP8TensorStorage`` (or similar TE
    quantised-tensor type); ``.dequantize(dtype=torch.bfloat16)`` returns a
    regular bf16 tensor via TE's C++ dequant kernel. The result is then fed
    into ``BatchInvariantTEGemmFn.apply`` — the same path BF16 layers take.

    Must be called AFTER ``enable_batch_invariant_mode()``.

    Replaces the earlier ``install_mxfp8_passthrough_for_bi_gemm`` which
    simply fell through to TE's non-BI ``general_gemm`` for MXFP8 calls.
    """
    import megatron.core.extensions.transformer_engine as meg_te
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod
    import transformer_engine.pytorch.module.linear as te_linear_mod
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )

    if bik_mod._TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError(
            "enable_batch_invariant_mode() must run before "
            "install_mxfp8_dequant_for_bi_gemm()"
        )

    extract = bik_mod._extract_te_gemm_args
    bi_gemm_fn = bik_mod.BatchInvariantTEGemmFn

    def _maybe_dequant_to_bf16(t):
        # Regular tensor (or None): leave alone. MXFP8/Float8 TE tensor: call
        # .dequantize(dtype=bf16). Detection uses the .dequantize attribute,
        # which TE's quantised-tensor storages expose but regular tensors do
        # not.
        if t is None or isinstance(t, torch.Tensor):
            return t
        if hasattr(t, "dequantize"):
            return t.dequantize(dtype=torch.bfloat16)
        return t

    def _wrapper(*args, **kwargs):
        a, b, out_dtype, layout, out_tensor, bias, grad = extract(args, kwargs)
        extra_output = kwargs.get("extra_output", None)
        ub = kwargs.get("ub", None)
        ub_type = kwargs.get("ub_type", None)
        bulk_overlap = kwargs.get("bulk_overlap", False)
        if (
            extra_output is not None
            or ub is not None
            or ub_type is not None
            or bulk_overlap
        ):
            raise RuntimeError(
                "Batch-invariant GEMM does not support Userbuffers/overlap "
                "(extra_output/ub/ub_type/bulk_overlap)."
            )

        a_bf16 = _maybe_dequant_to_bf16(a)
        b_bf16 = _maybe_dequant_to_bf16(b)

        result = bi_gemm_fn.apply(
            a_bf16, b_bf16, bias if not grad else None, out_dtype, layout
        )

        bias_grad = None
        if grad and bias is not None:
            b_flat = (
                b_bf16.reshape(-1, b_bf16.shape[-1]) if b_bf16.dim() > 2 else b_bf16
            )
            bias_grad = b_flat.sum(dim=0)

        if out_tensor is not None:
            out_tensor.copy_(result)
            return (out_tensor, bias_grad, None, extra_output)
        return (result, bias_grad, None, extra_output)

    for mod, attr in (
        (te_cpp, "general_gemm"),
        (te_linear_mod, "general_gemm"),
        (te_layernorm_linear_mod, "general_gemm"),
        (meg_te, "general_gemm"),
    ):
        if hasattr(mod, attr):
            setattr(mod, attr, _wrapper)


def install_vllm_style_rmsnorm():
    """Route Megatron's BI RMSNorm through vLLM's BI Triton kernel.

    Monkey-patch Megatron's BI RMSNorm to call vLLM's
    `rms_norm_batch_invariant` directly, so both engines run the *exact same*
    CUDA kernel.

    Megatron's default `BatchInvariantRMSNormFn` is a PyTorch implementation
    (`mean_dim(x*x)` -> `torch.sqrt` -> multiply). vLLM's `rms_norm_batch_invariant`
    is a single fused Triton kernel (`_rms_norm_kernel`: load-cast-square-reduce-
    rsqrt-multiply-cast). Even though both nominally use `BLOCK_SIZE=1024`
    sequential outer reductions and `1.0/sqrt(...)`, they differ in:
      - `tl.sqrt` (Triton, may compile to `sqrt.approx.f32`) vs `torch.sqrt`
        (PyTorch, IEEE-compliant `__fsqrt_rn`) — diverges by 1 ULP on some inputs.
      - Single fused kernel vs separate `x*x` materialisation + `mean_dim` call —
        same fp32 values in principle, but the Triton compiler may pick a
        different reduction tree than `mean_kernel`.

    By dispatching Megatron's BI RMSNorm through vLLM's exact Triton kernel,
    every RMSNorm call on either engine resolves to byte-identical kernel
    invocations on byte-identical inputs, guaranteeing bit-identical outputs.

    Must be called AFTER `enable_batch_invariant_mode()` — that's the call
    that installs `_te_rmsnorm_forward_patched` onto TE's RMSNorm; this patch
    swaps the autograd function that `_te_rmsnorm_forward_patched` invokes by
    global-name lookup.
    """
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )
    from vllm.model_executor.layers.batch_invariant import (
        rms_norm as vllm_rms_norm_triton,
    )

    class _VllmStyleBatchInvariantRMSNormFn(torch.autograd.Function):
        """RMSNorm autograd Fn that dispatches to vLLM's exact BI Triton kernel."""

        @staticmethod
        def forward(ctx, x, weight, eps, zero_centered_gamma):
            if not x.is_cuda:
                raise RuntimeError("Batch-invariant RMSNorm requires CUDA tensors.")
            w_eff = (weight + 1.0) if zero_centered_gamma else weight
            # vLLM's rms_norm wrapper reshapes (..., H) to (-1, H), runs the
            # Triton kernel, and reshapes back. We just need to pass the input
            # and weight in the original dtype.
            out = vllm_rms_norm_triton(x, w_eff, eps)

            ctx.eps = eps
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.save_for_backward(x, weight)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            raise NotImplementedError(
                "Backward not implemented for the vllm-style RMSNorm patch. "
                "This script is forward-only."
            )

    bik_mod.BatchInvariantRMSNormFn = _VllmStyleBatchInvariantRMSNormFn


def install_vllm_style_sdpa(*, paged_kv=False, paged_block_size=16):
    """Route TE `DotProductAttention.forward` through vLLM FA2.

    Monkey-patch TE's `DotProductAttention.forward` to dispatch to vLLM's FA2
    (`vllm.vllm_flash_attn.flash_attn_varlen_func`) with `num_splits=1` and
    `fa_version=2` — the exact kernel vLLM uses under
    `VLLM_BATCH_INVARIANT=1` on Blackwell.

    Why FA2 and not FA4: vLLM rejects FA4 in BI mode because FA4 uses
    batch-shape-dependent scheduling heuristics on SM100+ (see
    `vllm/v1/attention/backends/fa_utils.py:137-142`). Under BI, vLLM falls
    back to FA2. So matching vLLM's BI output bit-for-bit requires Megatron
    to also call FA2 — not FA4. (An earlier version of this patch called the
    FA4 cute kernel; that no longer matches what vLLM actually runs in BI.)

    This patch:
      1. Unpads Megatron/TE's (s, b, n, d) tensors into FA2 varlen's packed
         (total_tokens, n, d) layout with the actual prompt-length
         `cu_seqlens` that vLLM uses.
      2. Calls `flash_attn_varlen_func(..., fa_version=2, num_splits=1)`.
      3. Scatters the (total, n, d) output back to TE's expected padded
         (s, b, n*d) (or (b, s, n*d) for bshd) layout.

    If ``paged_kv`` is enabled, step 2 mirrors vLLM decoder attention more
    closely: K/V are written through ``reshape_and_cache_flash`` into a
    block-table KV cache, then FA2 reads ``key_cache`` / ``value_cache`` with
    ``block_table`` and ``seqused_k``. This is a diagnostic path for matching
    vLLM's normal decoder prefill path.

    Inference-only; no bias / sliding window / fp8 / sinks.
    """
    import math

    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as dpa_mod,
    )
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )

    def _seq_lens_for_batch(batch_size, seqlen):
        if G_VLLM_STYLE_SDPA_SEQ_LENS is None:
            return None
        if len(G_VLLM_STYLE_SDPA_SEQ_LENS) != batch_size:
            raise RuntimeError(
                "vllm-style SDPA seq_lens batch mismatch: "
                f"{len(G_VLLM_STYLE_SDPA_SEQ_LENS)} vs {batch_size}"
            )
        if max(G_VLLM_STYLE_SDPA_SEQ_LENS) > seqlen:
            raise RuntimeError(
                "vllm-style SDPA seq_lens exceed padded sequence length: "
                f"{max(G_VLLM_STYLE_SDPA_SEQ_LENS)} > {seqlen}"
            )
        return G_VLLM_STYLE_SDPA_SEQ_LENS

    def _pack_bshd(tensor, seq_lens):
        return torch.cat(
            [tensor[i, :seq_len] for i, seq_len in enumerate(seq_lens)],
            dim=0,
        ).contiguous()

    def _cu_seqlens(seq_lens, device):
        lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        zeros = torch.zeros(1, dtype=torch.int32, device=device)
        return torch.cat([zeros, torch.cumsum(lens, dim=0, dtype=torch.int32)])

    def _scatter_bshd(packed, batch_size, seqlen, seq_lens):
        out = packed.new_zeros((batch_size, seqlen, packed.size(1), packed.size(2)))
        offset = 0
        for batch_idx, seq_len in enumerate(seq_lens):
            next_offset = offset + seq_len
            out[batch_idx, :seq_len] = packed[offset:next_offset]
            offset = next_offset
        return out

    def _block_metadata(seq_lens, block_size, device):
        max_blocks = math.ceil(max(seq_lens) / block_size)
        block_table = torch.zeros(
            (len(seq_lens), max_blocks), dtype=torch.int32, device=device
        )
        slot_mapping = []
        next_block = 0
        for batch_idx, seq_len in enumerate(seq_lens):
            num_blocks = math.ceil(seq_len / block_size)
            blocks = torch.arange(
                next_block,
                next_block + num_blocks,
                dtype=torch.int32,
                device=device,
            )
            block_table[batch_idx, :num_blocks] = blocks
            for pos in range(seq_len):
                slot_mapping.append(
                    (next_block + pos // block_size) * block_size + pos % block_size
                )
            next_block += num_blocks
        return (
            block_table,
            torch.tensor(slot_mapping, dtype=torch.int64, device=device),
            next_block,
        )

    def _paged_fa2(q, k, v, seq_lens, max_q, max_k, softmax_scale, causal):
        block_table, slot_mapping, num_blocks = _block_metadata(
            seq_lens,
            paged_block_size,
            q.device,
        )
        key_cache = torch.empty(
            (num_blocks, paged_block_size, k.size(1), k.size(2)),
            dtype=k.dtype,
            device=k.device,
        )
        value_cache = torch.empty_like(key_cache)
        scale = torch.ones((), dtype=torch.float32, device=q.device)
        reshape_and_cache_flash(
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            scale,
            scale,
        )
        out = torch.empty_like(q)
        result = flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=out,
            max_seqlen_q=max_q,
            cu_seqlens_q=_cu_seqlens(seq_lens, q.device),
            max_seqlen_k=max_k,
            seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device=q.device),
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            num_splits=1,  # batch-invariant K reduction
            fa_version=2,  # match vLLM BI path
        )
        if isinstance(result, tuple):
            return result[0]
        return out

    def _vllm_fa2_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask=None,
        qkv_format=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        attn_mask_type=None,
        window_size=None,
        bottom_right_diagonal=None,
        checkpoint_core_attention=False,
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
        alibi_slopes=None,
        fast_zero_fill=True,
        inference_params=None,
        pad_between_seqs=None,
        fp8_output=False,
        num_splits=1,
    ):
        assert core_attention_bias is None, "bias not supported in vllm-fa2 patch"
        assert alibi_slopes is None, "alibi not supported in vllm-fa2 patch"
        assert inference_params is None, (
            "kv-cache inference not supported in vllm-fa2 patch"
        )
        assert fp8_output is False, "fp8 output not supported in vllm-fa2 patch"

        fmt = qkv_format or getattr(self, "qkv_format", "sbhd")
        if fmt == "sbhd":
            s_q, b_q, n_q, d = query_layer.shape
            s_kv, b_kv, n_kv, _ = key_layer.shape
            q_bshd = query_layer.transpose(0, 1).contiguous()
            k_bshd = key_layer.transpose(0, 1).contiguous()
            v_bshd = value_layer.transpose(0, 1).contiguous()
        elif fmt == "bshd":
            b_q, s_q, n_q, d = query_layer.shape
            b_kv, s_kv, n_kv, _ = key_layer.shape
            q_bshd = query_layer.contiguous()
            k_bshd = key_layer.contiguous()
            v_bshd = value_layer.contiguous()
        else:
            raise NotImplementedError(f"qkv_format={fmt!r} not supported")

        if b_q != b_kv:
            raise NotImplementedError("cross-attention is not supported")

        seq_lens = _seq_lens_for_batch(b_q, s_q)
        if seq_lens is None:
            q = q_bshd.reshape(b_q * s_q, n_q, d).contiguous()
            k = k_bshd.reshape(b_kv * s_kv, n_kv, d).contiguous()
            v = v_bshd.reshape(b_kv * s_kv, n_kv, d).contiguous()
            cu_q = torch.arange(
                0, (b_q + 1) * s_q, s_q, dtype=torch.int32, device=q.device
            )
            cu_k = torch.arange(
                0, (b_kv + 1) * s_kv, s_kv, dtype=torch.int32, device=q.device
            )
            max_q = s_q
            max_k = s_kv
        else:
            q = _pack_bshd(q_bshd, seq_lens)
            k = _pack_bshd(k_bshd, seq_lens)
            v = _pack_bshd(v_bshd, seq_lens)
            cu_q = _cu_seqlens(seq_lens, q.device)
            cu_k = cu_q
            max_q = max(seq_lens)
            max_k = max_q

        softmax_scale = 1.0 / math.sqrt(d)
        mt = attn_mask_type or "causal"
        causal = mt.startswith("causal")

        if paged_kv:
            seq_lens_for_paged = (
                seq_lens if seq_lens is not None else [s_q for _ in range(b_q)]
            )
            out = _paged_fa2(
                q,
                k,
                v,
                seq_lens_for_paged,
                max_q,
                max_k,
                softmax_scale,
                causal,
            )
        else:
            out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                max_seqlen_q=max_q,
                cu_seqlens_q=cu_q,
                max_seqlen_k=max_k,
                cu_seqlens_k=cu_k,
                softmax_scale=softmax_scale,
                causal=causal,
                num_splits=1,  # batch-invariant K reduction
                fa_version=2,  # match vLLM BI path
                deterministic=False,
            )
            if isinstance(out, tuple):
                out = out[0]
        if seq_lens is None:
            out = out.reshape(b_q, s_q, n_q, d)
        else:
            out = _scatter_bshd(out, b_q, s_q, seq_lens)
        if fmt == "sbhd":
            out = out.transpose(0, 1)  # -> (s, b, n_q, d)
            return out.reshape(s_q, b_q, n_q * d).contiguous()
        return out.reshape(b_q, s_q, n_q * d).contiguous()

    dpa_mod.DotProductAttention.forward = _vllm_fa2_forward


def set_vllm_style_sdpa_seq_lens(seq_lens):
    global G_VLLM_STYLE_SDPA_SEQ_LENS
    G_VLLM_STYLE_SDPA_SEQ_LENS = list(seq_lens)


def install_vllm_style_swiglu():
    """Monkey-patch Megatron's SwiGLU to match vLLM's `silu_and_mul` CUDA kernel.

    vLLM's kernel (`csrc/activation_kernels.cu`) does:
      silu_bf16 = (bf16)((float)gate / (1 + exp(-(float)gate)))   # round #1
      out_bf16  = silu_bf16 * up_bf16                              # round #2

    Megatron's default uses `@jit_fuser` (= `torch.compile`) on:
      def swiglu(y):
          y_1, y_2 = torch.chunk(y, 2, -1)
          return F.silu(y_1) * y_2
    which TorchInductor fuses into one Triton kernel keeping the silu result
    in fp32 registers — only ONE bf16 round at the final store. That extra
    precision is what creates the ~1-bf16-ULP drift at `linear_fc2` input
    vs vLLM's `down_proj` input.

    This patch replaces `swiglu` with an eager-mode version that runs as two
    separate kernels (silu materialises bf16, then bf16*bf16 multiply), so
    Megatron matches vLLM's two-rounding-event behaviour bit-for-bit.

    NOTE: This is a *downgrade* on Megatron's side for the sake of cross-
    engine bit equality. The "correct" long-term fix is to make vLLM's CUDA
    kernel keep silu in fp32 until the multiply, matching Megatron's compiled
    path. See `skills/debug-generation-training-mismatch/SKILL.md`.
    """
    import megatron.core.fusions.fused_bias_swiglu as swg_mod
    import torch.nn.functional as F

    def _vllm_style_swiglu(y):
        y_1, y_2 = torch.chunk(y, 2, -1)
        silu_out = F.silu(y_1)  # eager: fp32 compute, materialise bf16
        return silu_out * y_2  # bf16 * bf16

    def _vllm_style_bias_swiglu(y, bias):
        return _vllm_style_swiglu(y + bias)

    swg_mod.swiglu = _vllm_style_swiglu
    swg_mod.bias_swiglu = _vllm_style_bias_swiglu


def install_vllm_style_rope():
    """Match vLLM RoPE precision and multiply-add order.

    Monkey-patch Megatron's `apply_rotary_pos_emb`.

    vLLM:
      1. Precompute cos/sin in fp32 from fp32 freqs.
      2. Cast cos/sin to bf16 at module init (lossy).
      3. Inside the C++ wrapper, upcast cos_sin_cache back to fp32
         (`pos_encoding_kernels.cu:171`).
      4. CUDA kernel: read bf16 q/k -> fp32 -> apply
         `out_first  = x1*cos - x2*sin` (fp32)
         `out_second = x2*cos + x1*sin` (fp32)
         single bf16 cast per output element.

    This patch replicates all four steps in PyTorch so Megatron's intermediate
    bf16 rounding events disappear and cos/sin precision matches vLLM exactly.
    """
    from megatron.core.models.common.embeddings import rope_utils

    def _vllm_style_apply_rope(
        t: torch.Tensor,
        freqs: torch.Tensor,
        config,
        cu_seqlens=None,
        mscale: float = 1.0,
        cp_group=None,
    ):
        assert cu_seqlens is None, "vllm-style RoPE patch only supports non-thd"
        rot_dim = freqs.shape[-1]
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

        # Step 1+2+3: compute cos/sin, cast through bf16 to bf16 (precision loss),
        # then back to fp32 (matches vLLM's `cache.to(kFloat32)` in the C++ wrapper).
        target_dtype = t.dtype
        cos = (torch.cos(freqs) * mscale).to(target_dtype).to(torch.float32)
        sin = (torch.sin(freqs) * mscale).to(target_dtype).to(torch.float32)

        # Megatron passes freqs already duplicated via cat(f, f). The first half
        # carries the real frequencies; the second half is just a copy. vLLM's
        # cache stores only the half-dim cos/sin. Take the first half so we apply
        # the same per-element cos to x1 and x2.
        cos_h = cos[..., : rot_dim // 2]
        sin_h = sin[..., : rot_dim // 2]

        # Step 4: rotation in fp32, single bf16 cast per output element.
        t_rot_fp32 = t_rot.to(torch.float32)
        if not getattr(config, "rotary_interleaved", False):
            # NeoX layout: t_rot = cat(x1, x2)
            x1, x2 = torch.chunk(t_rot_fp32, 2, dim=-1)
            o1 = x1 * cos_h - x2 * sin_h
            o2 = x2 * cos_h + x1 * sin_h
            out_fp32 = torch.cat((o1, o2), dim=-1)
        else:
            # GPT-J / interleaved layout
            x1 = t_rot_fp32[..., 0::2]
            x2 = t_rot_fp32[..., 1::2]
            o1 = x1 * cos_h - x2 * sin_h
            o2 = x2 * cos_h + x1 * sin_h
            out_fp32 = torch.stack((o1, o2), dim=-1).flatten(-2)

        out = out_fp32.to(target_dtype)
        return torch.cat((out, t_pass), dim=-1) if t_pass.numel() > 0 else out

    rope_utils.apply_rotary_pos_emb = _vllm_style_apply_rope
    # Also patch the re-export in transformer.attention if present so the
    # `attention.py` import binding is updated, not just the rope_utils module.
    import importlib

    for mod_name in ("megatron.core.transformer.attention",):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "apply_rotary_pos_emb"):
                mod.apply_rotary_pos_emb = _vllm_style_apply_rope
        except ImportError:
            pass


class SplitNormLinear(torch.nn.Module):
    """Materialise the post-norm tensor before the wrapped linear.

    Drop-in replacement for `TELayerNormColumnParallelLinear` that lets a
    forward hook on `linear` capture vLLM-comparable `qkv_proj` /
    `gate_up_proj` input.
    """

    def __init__(self, norm, linear):
        super().__init__()
        self.norm = norm
        self.linear = linear

    def forward(self, x):
        return self.linear(self.norm(x))


def split_layer_fused(layer, config):
    """Unfuse the first QKV and MLP projection fused modules.

    Unfuse `self_attention.linear_qkv` and `mlp.linear_fc1` on one decoder layer
    using the upstream `split_te_layernorm_column_parallel_linear` primitive,
    replacing each with a tiny `SplitNormLinear(norm, linear)` wrapper. This
    forces the post-norm activation to be materialised in bf16 (an extra
    round-trip) so the layer's numerics match vLLM's standalone `RMSNorm` +
    `qkv_proj` / `gate_up_proj` path exactly.
    """
    from megatron.core.extensions.transformer_engine import (
        split_te_layernorm_column_parallel_linear,
    )

    split_count = 0
    for owner, attr in (
        (getattr(layer, "self_attention", None), "linear_qkv"),
        (getattr(layer, "mlp", None), "linear_fc1"),
    ):
        if owner is None or not hasattr(owner, attr):
            continue
        fused = getattr(owner, attr)
        norm, linear = split_te_layernorm_column_parallel_linear(fused, config)
        norm = norm.to(device=fused.weight.device, dtype=fused.weight.dtype)
        linear = linear.to(device=fused.weight.device, dtype=fused.weight.dtype)
        setattr(owner, attr, SplitNormLinear(norm, linear))
        split_count += 1
    return split_count


def split_first_layer_fused(first_layer, config):
    """Back-compat wrapper around `split_layer_fused` for layer 0 only."""
    return split_layer_fused(first_layer, config)


def split_all_layers_fused(decoder, config):
    """Apply `split_layer_fused` to every decoder layer (not just layer 0)."""
    split_layer_count = 0
    split_module_count = 0
    for layer in decoder.layers:
        layer_count = split_layer_fused(layer, config)
        if layer_count:
            split_layer_count += 1
            split_module_count += layer_count
    return split_layer_count, split_module_count


class SingleBatchIterator:
    def __init__(self, input_ids, position_ids):
        self.batch = dict(tokens=input_ids, position_ids=position_ids)
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def forward_step(data_iterator, model, **kwargs):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "runtime_gather_output": True,
    }

    def loss_func(x, **_):
        return x

    return model(**forward_args), loss_func


def unwrap(m):
    if isinstance(m, Float16Module):
        m = m.module
    if hasattr(m, "language_model"):
        m = m.language_model
    return m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer source (HF id or path). Defaults to --model.",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help="Number of prompts to draw from --dataset (default: 32).",
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dataset-subset", default=DEFAULT_DATASET_SUBSET)
    p.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    p.add_argument("--dataset-field", default=DEFAULT_DATASET_FIELD)
    p.add_argument("--dataset-seed", type=int, default=0)
    p.add_argument(
        "--token-ids-file",
        default=None,
        help=(
            "Torch .pt file containing token id rows to replay instead of "
            "loading prompts from --dataset."
        ),
    )
    p.add_argument(
        "--token-ids-key",
        default="token_ids_list",
        help="Payload key to read from --token-ids-file (default: token_ids_list).",
    )
    p.add_argument("--output", default=None)
    p.add_argument("--batch-invariant", action="store_true")
    p.add_argument(
        "--split-fused",
        action="store_true",
        help="Unfuse LN+Linear on first layer (linear_qkv, linear_fc1)",
    )
    p.add_argument(
        "--split-all-fused",
        action="store_true",
        help="Unfuse LN+Linear on ALL decoder layers (implies "
        "--split-fused for layer 0). Required for full-depth "
        "bit-equality with vLLM's standalone-norm + linear path.",
    )
    p.add_argument(
        "--no-rope-fusion",
        action="store_true",
        help="Disable TE fused RoPE (apply_rope_fusion=False) to "
        "match vLLM's bf16-cast-of-cos/sin precision behavior.",
    )
    p.add_argument(
        "--vllm-rope",
        action="store_true",
        help="Monkey-patch Megatron's apply_rotary_pos_emb with a "
        "PyTorch RoPE that reproduces vLLM's exact precision "
        "behaviour (bf16 cos/sin cache, fp32 rotation, single "
        "bf16 cast at end).",
    )
    p.add_argument(
        "--vllm-swiglu",
        action="store_true",
        help="Monkey-patch Megatron's swiglu to bypass torch.compile "
        "fusion and match vLLM's two-rounding-event SwiGLU "
        "(eager F.silu then bf16 multiply).",
    )
    p.add_argument(
        "--vllm-sdpa",
        action="store_true",
        help="Monkey-patch TE DotProductAttention to call vLLM's "
        "FA2 (flash_attn_varlen_func, fa_version=2, "
        "num_splits=1) — the kernel vLLM actually uses under "
        "VLLM_BATCH_INVARIANT=1 on Blackwell.",
    )
    p.add_argument(
        "--vllm-paged-sdpa",
        action="store_true",
        help="Monkey-patch TE DotProductAttention to mirror vLLM "
        "decoder prefill attention: write K/V with "
        "reshape_and_cache_flash, then call FA2 with "
        "key_cache/value_cache, block_table, and seqused_k.",
    )
    p.add_argument(
        "--vllm-paged-block-size",
        type=int,
        default=16,
        help="KV-cache block size for --vllm-paged-sdpa. vLLM's "
        "default FlashAttention block size is 16.",
    )
    p.add_argument(
        "--vllm-rmsnorm",
        action="store_true",
        help="Monkey-patch Megatron's BI RMSNorm to use 1/sqrt(...) "
        "instead of torch.rsqrt(...), matching vLLM's original "
        "Triton BI RMSNorm kernel bit-for-bit. Only meaningful "
        "in combination with --batch-invariant.",
    )
    p.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Megatron tensor model parallel size (default: 1).",
    )
    p.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=1,
        help="Megatron expert model parallel size (default: 1).",
    )
    p.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        help="Megatron expert tensor parallel size (default: 1).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Force trust_remote_code=True for HF model/tokenizer loading.",
    )
    p.add_argument(
        "--mxfp8",
        action="store_true",
        help="Enable MXFP8 (Blackwell-only). Configures the model "
        "provider with fp8=<format> and fp8_recipe='mxfp8' so "
        "TE's fp8_autocast wraps every decoder layer.",
    )
    p.add_argument(
        "--fp8-format",
        default="e4m3",
        choices=["e4m3", "hybrid"],
        help="FP8 element format used by the MXFP8 recipe (default: "
        "e4m3). 'hybrid' uses e4m3 for fwd and e5m2 for bwd. "
        "Only meaningful with --mxfp8.",
    )
    p.add_argument(
        "--mxfp8-bi-dequant",
        action="store_true",
        help="Under --mxfp8 --batch-invariant, dequant MXFP8 operands "
        "to bf16 and route through the BF16 BI matmul "
        "(install_mxfp8_dequant_for_bi_gemm). Default is the "
        "passthrough patch which just lets MXFP8 GEMMs use TE's "
        "original cuBLASLt kernel.",
    )
    p.add_argument(
        "--capture-debug-tensors",
        action="store_true",
        help="Save layer-entry tensors for every decoder layer and "
        "input/output tensors for every module in --capture-layers.",
    )
    p.add_argument(
        "--capture-layers",
        type=parse_layer_indices,
        default=[0],
        help="Comma-separated layer indices/ranges to fully hook when "
        "--capture-debug-tensors is set, e.g. 0,1,5 or 0-2. Default: 0.",
    )
    args = p.parse_args()
    args.model = resolve_model_ref(args.model)
    if args.tokenizer is None:
        args.tokenizer = args.model
    else:
        args.tokenizer = resolve_model_ref(args.tokenizer)
    args.trust_remote_code = (
        args.trust_remote_code
        or is_nemotron3_nano_ref(args.model)
        or is_nemotron3_nano_ref(args.tokenizer)
    )
    if args.vllm_sdpa and args.vllm_paged_sdpa:
        raise SystemExit("--vllm-sdpa and --vllm-paged-sdpa are mutually exclusive")
    if args.split_all_fused:
        args.split_fused = True  # imply layer-0 split when all-layers split
    if args.output is None:
        args.output = default_output(
            args.batch_invariant,
            args.split_fused,
            args.no_rope_fusion,
            args.vllm_rope,
            args.vllm_swiglu,
            args.vllm_sdpa,
            args.vllm_rmsnorm,
            args.split_all_fused,
            args.mxfp8,
            args.vllm_paged_sdpa,
            model_output_tag(args.model, args.tokenizer),
        )
    return args


def main():
    args = parse_args()

    if args.batch_invariant:
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            enable_batch_invariant_mode,
        )

        enable_batch_invariant_mode()
        print_rank_0("[megatron] batch_invariant_mode ENABLED")
        if args.mxfp8:
            if args.mxfp8_bi_dequant:
                install_mxfp8_compact_scales()
                install_mxfp8_dequant_for_bi_gemm()
                print_rank_0("[megatron] MXFP8 quantizers forced to compact scales")
                print_rank_0(
                    "[megatron] BI GEMM patch wrapped: MXFP8 tensors "
                    "dequant -> bf16 -> BF16 BI matmul (matmul_persistent)"
                )
            else:
                install_mxfp8_passthrough_for_bi_gemm()
                print_rank_0(
                    "[megatron] BI GEMM patch wrapped: MXFP8 tensors "
                    "passthrough to TE's original general_gemm; "
                    "BF16 ops still hit BI Triton"
                )

    if args.vllm_rmsnorm:
        if not args.batch_invariant:
            raise SystemExit(
                "--vllm-rmsnorm requires --batch-invariant "
                "(patches the BI RMSNorm autograd function)"
            )
        install_vllm_style_rmsnorm()
        print_rank_0("[megatron] vllm-style RMSNorm patched (1/sqrt instead of rsqrt)")

    if args.vllm_rope:
        # Force apply_rope_fusion off so the python apply_rotary_pos_emb path
        # is taken (which is what we monkey-patch).
        install_vllm_style_rope()
        print_rank_0("[megatron] vllm-style RoPE patched into apply_rotary_pos_emb")

    if args.vllm_swiglu:
        install_vllm_style_swiglu()
        print_rank_0("[megatron] vllm-style SwiGLU patched (eager F.silu + bf16 mul)")

    if args.vllm_sdpa:
        install_vllm_style_sdpa(paged_kv=False)
        print_rank_0(
            "[megatron] TE DotProductAttention.forward patched -> vLLM FA2 (num_splits=1)"
        )
    if args.vllm_paged_sdpa:
        install_vllm_style_sdpa(
            paged_kv=True,
            paged_block_size=args.vllm_paged_block_size,
        )
        print_rank_0(
            "[megatron] TE DotProductAttention.forward patched -> "
            "vLLM paged-KV FA2 "
            f"(block_size={args.vllm_paged_block_size}, num_splits=1)"
        )

    print_rank_0(f"[megatron] loading bridge for {args.model}")
    bridge = AutoBridge.from_hf_pretrained(
        args.model,
        trust_remote_code=is_safe_repo(
            trust_remote_code=True if args.trust_remote_code else None,
            hf_path=args.model,
        ),
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    model_provider.pipeline_model_parallel_size = 1
    model_provider.expert_model_parallel_size = args.expert_model_parallel_size
    model_provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    model_provider.pipeline_dtype = torch.bfloat16
    # Forward-only inference — disable grad-fusion knobs that need APEX.
    model_provider.gradient_accumulation_fusion = False
    if args.no_rope_fusion or args.vllm_rope:
        # Use the unfused PyTorch RoPE path so our monkey-patch is actually
        # called (TE fused path is a CUDA kernel, not patchable from Python).
        model_provider.apply_rope_fusion = False

    if args.mxfp8:
        # Setting `fp8` to a non-None format flips on TE's `fp8_autocast` inside
        # `TransformerBlock.forward`; `fp8_recipe="mxfp8"` selects
        # `MXFP8BlockScaling` (Blackwell-only). Weights stay BF16 on disk/host —
        # TE quantizes per-block (1x32, E8M0 scales) at the GEMM boundary.
        model_provider.fp8 = args.fp8_format
        model_provider.fp8_recipe = "mxfp8"
        print_rank_0(
            f"[megatron] MXFP8 enabled: fp8={args.fp8_format}, fp8_recipe=mxfp8"
        )

    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    model_list = model_provider.provide_distributed_model(wrap_with_ddp=False)
    model_list = [m.cuda() for m in model_list]
    for m in model_list:
        m.eval()
        disable_mtp_for_inference(m)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids_payload = None
    if args.token_ids_file:
        prompts, token_ids_list, token_ids_payload = load_token_ids_from_file(
            args.token_ids_file,
            args.token_ids_key,
        )
    else:
        prompts = load_prompts(
            args.dataset,
            args.dataset_subset,
            args.dataset_split,
            args.dataset_field,
            args.num_prompts,
            args.dataset_seed,
        )
        token_ids_list = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    seq_lens = [len(ids) for ids in token_ids_list]
    if args.token_ids_file:
        print_rank_0(
            f"[megatron] token ids: {args.token_ids_file} "
            f"key={args.token_ids_key!r} n={len(prompts)}"
        )
    else:
        print_rank_0(
            f"[megatron] dataset: {args.dataset}:{args.dataset_subset}:"
            f"{args.dataset_split} field={args.dataset_field!r} "
            f"n={len(prompts)}"
        )
    print_rank_0(
        f"[megatron] seq_len min/mean/max = {min(seq_lens)}/"
        f"{sum(seq_lens) / len(seq_lens):.1f}/{max(seq_lens)}"
    )
    if args.vllm_sdpa or args.vllm_paged_sdpa:
        set_vllm_style_sdpa_seq_lens(seq_lens)
        if args.vllm_paged_sdpa:
            print_rank_0("[megatron] vllm-paged SDPA using actual seq_lens")
        else:
            print_rank_0("[megatron] vllm-style SDPA using packed actual seq_lens")

    # Pad all prompts to a common length so they fit in one (B, S) tensor. Under
    # MXFP8 the per-block scaling kernel requires every dim to be a multiple of
    # 32 — round seq_len up accordingly. Causal attention makes the pad
    # positions invisible to the real token positions, so per-prompt last-token
    # logits are unaffected.
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    padded_seq_len = max(seq_lens)
    if args.mxfp8 and padded_seq_len % 32 != 0:
        padded_seq_len = ((padded_seq_len + 31) // 32) * 32
        print_rank_0(
            f"[megatron] padded seq_len {max(seq_lens)} -> {padded_seq_len} "
            "for MXFP8 (TE MXFP8 block size requires divisibility by 32)"
        )

    padded_ids = [
        ids + [pad_id] * (padded_seq_len - len(ids)) for ids in token_ids_list
    ]
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device="cuda")
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )

    assert len(model_list) == 1, "expected one model chunk with pp=1"
    inner = unwrap(model_list[0])

    if args.split_all_fused:
        n_layers, n_modules = split_all_layers_fused(inner.decoder, inner.config)
        print_rank_0(
            f"[megatron] split fused LN+Linear on {n_layers} decoder layers "
            f"({n_modules} modules) "
            "(linear_qkv -> SplitNormLinear, linear_fc1 -> SplitNormLinear)"
        )
    elif args.split_fused:
        n_modules = split_first_layer_fused(inner.decoder.layers[0], inner.config)
        print_rank_0(
            f"[megatron] split fused LN+Linear on first layer ({n_modules} modules) "
            "(linear_qkv -> SplitNormLinear(norm, linear), "
            "linear_fc1 -> SplitNormLinear(norm, linear))"
        )

    debug_capture = {}
    if args.capture_debug_tensors:
        hook_info = install_debug_tensor_hooks(
            inner,
            capture_layers=args.capture_layers,
        )
        print_rank_0(f"[megatron] debug tensor hooks installed: {hook_info}")

    with torch.no_grad():
        fwd_bwd = get_forward_backward_func()
        iterator = SingleBatchIterator(input_ids, position_ids)
        output = fwd_bwd(
            forward_step_func=forward_step,
            data_iterator=iterator,
            model=model_list,
            num_microbatches=1,
            forward_only=True,
            seq_length=input_ids.size(1),
            micro_batch_size=input_ids.size(0),
            collect_non_loss_data=True,
        )

    if isinstance(output, list) and len(output) > 0:
        output = output[0]

    last_token_logits_cpu = None
    if parallel_state.is_pipeline_last_stage() and isinstance(output, torch.Tensor):
        logits_cpu = output.detach().to(torch.float32).cpu()
        print_rank_0(f"[megatron] logits shape: {tuple(logits_cpu.shape)}")
        # Megatron returns logits as (B, S, V). Gather row[i] at position
        # seq_lens[i] - 1 (the last real token of prompt i).
        assert logits_cpu.dim() == 3 and logits_cpu.size(0) == len(prompts), (
            f"unexpected logits shape {tuple(logits_cpu.shape)}; "
            f"expected (B={len(prompts)}, S, V)"
        )
        idx = torch.tensor([sl - 1 for sl in seq_lens], dtype=torch.long)
        last_token_logits_cpu = logits_cpu[torch.arange(len(prompts)), idx]

    if dist.is_initialized() and dist.get_rank() == 0:
        if args.capture_debug_tensors:
            debug_capture = get_debug_tensor_capture(inner)
            num_first_layer = len(debug_capture.get("first_layer_inputs", {}))
            num_layers = debug_capture.get("num_layers", "?")
            print(
                f"[megatron] captured {num_first_layer} layer-0 modules; "
                f"module_layers={debug_capture.get('captured_module_layers')} "
                f"num_layers={num_layers}"
            )
        payload = {
            "model": args.model,
            "tokenizer": args.tokenizer,
            "model_family": "nemotron3_nano"
            if is_nemotron3_nano_ref(args.model)
            or is_nemotron3_nano_ref(args.tokenizer)
            else "llama",
            "prompts": prompts,
            "token_ids_list": token_ids_list,
            "seq_lens": seq_lens,
            "last_token_logits": last_token_logits_cpu,
        }
        if token_ids_payload is not None:
            payload["token_ids_source_file"] = args.token_ids_file
            payload["token_ids_source_key"] = args.token_ids_key
            for key in (
                "offline_target_token_ids",
                "offline_generation_logprobs",
                "offline_metadata",
                "sample_rows",
                "sample_prompt_token_ids_list",
                "sample_response_token_ids_list",
                "sample_full_token_ids_list",
            ):
                if key in token_ids_payload:
                    payload[key] = token_ids_payload[key]
        payload.update(debug_capture)
        torch.save(payload, args.output)
        print(f"[megatron] saved capture to {args.output}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
