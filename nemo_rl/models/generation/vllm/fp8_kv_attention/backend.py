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
"""vLLM TRITON_ATTN override for ``kv_cache_dtype=fp8_per_token_head``.

Quantization is unchanged from vLLM (one FP32 scale ``max|x| / 448`` per token
and KV head, written by vLLM's own cache writer); only the bytes inside each
64-token page are rearranged and the attention kernels are swapped:

* Page-local layout. vLLM packs ``[K(128) | K scale | V(128) | V scale]`` per
  (head, token). This backend reinterprets every page of ``P = 64 * H * 132``
  bytes per K/V half as ``[K data (64, H, 128)][K scales (64, H)]`` followed by
  the same for V, so head rows are 128-byte aligned. The page size, and with it
  vLLM's allocation, block copies and prefix caching, is unchanged; caches
  written by the stock backend are not readable by this one (and vice versa).
* Prefill / mixed batches: vLLM's ``unified_attention`` launched with
  ``num_stages=1`` (otherwise identical math).
* Pure decode (``max_query_len == 1``): the SM100 Gluon kernel in ``decode.py``.
"""

import types

import torch
from vllm.v1.attention.backends import triton_attn
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.ops import triton_unified_attention as _unified
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.v1.kv_cache_layout import KVCacheLayout

from nemo_rl.models.generation.vllm.fp8_kv_attention.decode import (
    HEAD_SIZE,
    PAGE_SIZE,
    SUPPORTED_HEADS,
    decode,
)

SCALE_BYTES = 4


def page_views(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(k, v, k_scale, v_scale)`` page-local views of one layer's cache.

    ``kv_cache`` is vLLM 0.29's logical ``[num_blocks, H, 64, 2 * 132]`` byte
    tensor. Any physical layout is accepted as long as the ``H x 64 x 264`` bytes
    of a block form one dense run (block stride may be larger, e.g. padded).
    Data views are ``[num_blocks, 64, H, 128]`` FP8 and scale views
    ``[num_blocks, 64, H]`` FP32, matching vLLM's per-token-head kernel ABI.
    """
    blocks, heads, page, content = kv_cache.shape
    padded = HEAD_SIZE + SCALE_BYTES
    if kv_cache.element_size() != 1 or page != PAGE_SIZE or content != 2 * padded:
        raise ValueError(
            f"Unsupported FP8 per-token-head cache {tuple(kv_cache.shape)} {kv_cache.dtype}"
        )
    stride = kv_cache.stride()
    dense = 1
    for step, size in sorted(zip(stride[1:], (heads, page, content))):
        if size == 1:  # e.g. one KV head per rank: its stride is irrelevant
            continue
        if step != dense:
            raise ValueError(f"KV cache block is not one dense page (strides {stride})")
        dense *= size
    base = kv_cache.storage_offset()
    if stride[0] % SCALE_BYTES or base % SCALE_BYTES:
        raise ValueError("KV cache pages are not 4-byte aligned")

    storage = kv_cache.untyped_storage()
    as_bytes = torch.empty(0, dtype=torch.uint8, device=kv_cache.device).set_(storage)
    as_floats = torch.empty(0, dtype=torch.float32, device=kv_cache.device).set_(
        storage
    )
    half = page * heads * padded  # bytes of the K (or V) half of a page
    data_bytes = page * heads * HEAD_SIZE

    def data(offset):
        return torch.as_strided(
            as_bytes,
            (blocks, page, heads, HEAD_SIZE),
            (stride[0], heads * HEAD_SIZE, HEAD_SIZE, 1),
            base + offset,
        ).view(torch.float8_e4m3fn)

    def scales(offset):
        return torch.as_strided(
            as_floats,
            (blocks, page, heads),
            (stride[0] // SCALE_BYTES, heads, 1),
            (base + offset) // SCALE_BYTES,
        )

    return data(0), data(half), scales(data_bytes), scales(half + data_bytes)


class _StagedLaunch:
    """Stand-in for ``kernel_unified_attention`` that launches with fixed ``num_stages``."""

    def __init__(self, kernel, num_stages: int):
        self.kernel, self.num_stages = kernel, num_stages

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        return lambda *args, **kwargs: launch(
            *args, **{"num_stages": self.num_stages, **kwargs}
        )


def _unified_attention_with_stages(num_stages: int) -> types.FunctionType:
    """Copy of vLLM's ``unified_attention`` whose kernel launch uses ``num_stages``.

    ``unified_attention`` has no launch-option argument, so the function is
    rebuilt over a copy of its module globals in which only the kernel name is
    replaced; tile sizes, split thresholds and reduction are vLLM's own.
    """
    fn = _unified.unified_attention
    if "kernel_unified_attention" not in fn.__code__.co_names or fn.__closure__:
        raise RuntimeError(
            "vLLM unified_attention no longer launches kernel_unified_attention directly"
        )
    namespace = dict(vars(_unified))
    namespace["kernel_unified_attention"] = _StagedLaunch(
        _unified.kernel_unified_attention, num_stages
    )
    staged = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__)
    staged.__kwdefaults__ = fn.__kwdefaults__
    return staged


prefill_attention = _unified_attention_with_stages(num_stages=1)


class Fp8KvAttentionImpl(triton_attn.TritonAttentionImpl):
    """TritonAttentionImpl over page-local FP8 caches with SM100 decode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.num_heads, self.num_kv_heads)
        if (
            self._kv_quant_mode != KVQuantMode.FP8_PER_TOKEN_HEAD
            or shape not in SUPPORTED_HEADS
            or self.head_size != HEAD_SIZE
            or self.attn_type != triton_attn.AttentionType.DECODER
        ):
            raise ValueError(
                f"Unsupported FP8 KV attention: heads {shape}, head {self.head_size}, {self.attn_type}"
            )
        assert (
            self.alibi_slopes is None
            and self.sinks is None
            and self.logits_soft_cap == 0
        )
        assert (
            self.sliding_window == (-1, -1)
            and self.chunk_lookback == -1
            and not self.use_td
        )
        self._page_key_cache: torch.Tensor | None = None
        self._page_value_cache: torch.Tensor | None = None

    def _ensure_scale_caches(self, kv_cache):
        # vLLM resets _k_scale_cache to None whenever it reallocates KV caches.
        if self._k_scale_cache is not None:
            return
        k, v, k_scale, v_scale = page_views(kv_cache)
        k_scale.fill_(1.0)
        v_scale.fill_(1.0)
        self._page_key_cache, self._page_value_cache = k, v
        self._v_scale_cache = v_scale
        self._k_scale_cache = k_scale

    def _pth_key_value_caches(self, kv_cache):
        self._ensure_scale_caches(kv_cache)
        return self._page_key_cache, self._page_value_cache

    def fused_output_quant_supported(self, quant_key):
        return False

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        k, v = self._pth_key_value_caches(kv_cache)
        triton_reshape_and_cache_flash_per_token_head_quant(
            key,
            value,
            k,
            v,
            self._k_scale_cache,
            self._v_scale_cache,
            slot_mapping,
            kv_quant_mode=self._kv_quant_mode,
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None:  # profiling run
            return output.fill_(0)
        assert (
            output_scale is None
            and output_block_scale is None
            and not attn_metadata.use_cascade
        )
        assert (
            attn_metadata.mm_prefix_range_tensor is None
            and attn_metadata.rswa_window is None
        )
        k, v = self._pth_key_value_caches(kv_cache)
        n = attn_metadata.num_actual_tokens
        if attn_metadata.max_query_len == 1:
            decode(
                query[:n],
                k=k,
                v=v,
                k_scale=self._k_scale_cache,
                v_scale=self._v_scale_cache,
                out=output[:n],
                cu_seqlens_q=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                softmax_scale=self.scale,
            )
            return output
        prefill_attention(
            q=query[:n],
            k=k,
            v=v,
            out=output[:n],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=(-1, -1),
            block_table=attn_metadata.block_table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=self._k_scale_cache,
            v_scale_cache=self._v_scale_cache,
        )
        return output


class Fp8KvAttentionBackend(triton_attn.TritonAttentionBackend):
    """TRITON_ATTN backend restricted to fp8_per_token_head on 64-token pages."""

    supported_kv_cache_dtypes = ["fp8_per_token_head"]

    @staticmethod
    def get_impl_cls():
        return Fp8KvAttentionImpl

    @staticmethod
    def get_supported_kernel_block_sizes():
        # Kernels address 64-token pages; larger (e.g. hybrid-model) manager
        # blocks are split into these by vLLM.
        return [PAGE_SIZE]

    @classmethod
    def supports_block_size(cls, block_size):
        return block_size is None or block_size % PAGE_SIZE == 0

    @classmethod
    def supported_kv_cache_layouts(cls):
        # Layouts whose per-layer [H, N, C] block is one dense page.
        return (
            KVCacheLayout.LBNHC,
            KVCacheLayout.LBHNC,
            KVCacheLayout.BLNHC,
            KVCacheLayout.BLHNC,
        )


def install() -> None:
    """Route AttentionBackendEnum.TRITON_ATTN to this backend in the current process."""
    register_backend(
        AttentionBackendEnum.TRITON_ATTN,
        f"{__name__}.{Fp8KvAttentionBackend.__qualname__}",
    )
