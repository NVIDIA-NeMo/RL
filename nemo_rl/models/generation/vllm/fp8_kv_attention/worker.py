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
"""vLLM ``worker_cls`` that installs the FP8 KV attention backend in each engine worker.

Enable with ``policy.generation.vllm_kwargs``::

    worker_cls: nemo_rl.models.generation.vllm.fp8_kv_attention.worker.Worker
    attention_config: {backend: TRITON_ATTN}
    block_size: 64
    kv_cache_dtype: fp8_per_token_head   # policy.generation.vllm_cfg.kv_cache_dtype

Supported: Qwen3-30B-A3B at TP1 (48 layers, 32/4 heads) and Nemotron 3 Super at
TP2 (8 attention layers, 16/1 heads per rank) on SM100 (GB200).
"""

import logging

import torch
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from nemo_rl.models.generation.vllm.fp8_kv_attention import backend
from nemo_rl.models.generation.vllm.fp8_kv_attention.decode import (
    MAX_ROWS,
    reserve_scratch,
)

logger = logging.getLogger(__name__)

# Attention module suffix -> (layers, Q heads, KV heads, head size) per rank.
SHAPES = {".self_attn.attn": (48, 32, 4, 128), ".mixer.attn": (8, 16, 1, 128)}


def _attention_layers(model):
    for suffix, (count, *shape) in SHAPES.items():
        layers = [
            module for name, module in model.named_modules() if name.endswith(suffix)
        ]
        if layers:
            assert len(layers) == count, (suffix, len(layers))
            return layers, tuple(shape)
    raise AssertionError(
        "No supported attention layers for the FP8 KV attention backend"
    )


class Worker(GPUWorker):
    """GPU worker with the page-local FP8 KV backend registered for TRITON_ATTN."""

    def __init__(self, *args, **kwargs):
        backend.install()  # before the model (and its attention backends) is built
        super().__init__(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        assert torch.cuda.get_device_capability() == (10, 0), (
            "SM100 decode kernel requires GB200/B200"
        )
        assert self.vllm_config.scheduler_config.max_num_seqs <= MAX_ROWS
        capture = getattr(
            self.vllm_config.compilation_config, "max_cudagraph_capture_size", None
        )
        assert capture is None or capture <= MAX_ROWS, capture
        layers, (heads, kv_heads, head_size) = _attention_layers(
            self.model_runner.get_model()
        )
        for layer in layers:
            impl = layer.impl
            assert type(impl) is backend.Fp8KvAttentionImpl, type(impl)
            assert (
                layer.kv_cache_dtype == "fp8_per_token_head"
                and layer.query_quant is None
            )
            assert (impl.num_heads, impl.num_kv_heads, impl.head_size) == (
                heads,
                kv_heads,
                head_size,
            )
        # Allocate the shared decode scratch before vLLM sizes the KV cache.
        reserve_scratch(self.device, heads, kv_heads)
        logger.info(
            "FP8 KV attention backend active on %d layers (%d/%d heads)",
            len(layers),
            heads,
            kv_heads,
        )
