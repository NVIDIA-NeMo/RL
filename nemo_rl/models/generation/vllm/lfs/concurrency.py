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

"""Concurrency limits that guarantee the configured KV cache can fit."""

from typing import Any


def get_engine_kv_cache_shape(engine: Any) -> tuple[int, int]:
    """Read initialized vLLM KV token capacity and block size.

    vLLM's sync and async front ends expose the same cache config at
    different object paths. Only an initialized config has a positive
    ``num_gpu_blocks``; returning the actual value lets benchmarks validate
    TP-dependent concurrency without relying on a TP=1 estimate.
    """
    candidate_paths = (
        ("vllm_config", "cache_config"),
        ("llm_engine", "vllm_config", "cache_config"),
        ("llm_engine", "cache_config"),
        ("engine_core", "vllm_config", "cache_config"),
    )
    for path in candidate_paths:
        candidate = engine
        try:
            for attribute in path:
                candidate = getattr(candidate, attribute)
            num_gpu_blocks = int(candidate.num_gpu_blocks)
            block_size = int(candidate.block_size)
        except (AttributeError, TypeError, ValueError):
            continue
        if num_gpu_blocks > 0 and block_size > 0:
            return num_gpu_blocks * block_size, block_size
    raise RuntimeError("could not read initialized vLLM KV cache capacity")


def max_concurrency_without_preemption(
    *,
    kv_cache_tokens: int,
    max_sequence_length: int,
    block_size: int,
    reserve_blocks_per_sequence: int = 0,
) -> int:
    """Return a block-aligned concurrency cap that fits the KV cache.

    The calculation assumes every live request reaches ``max_sequence_length``.
    This is intentionally more conservative than using an observed mean length:
    the result is suitable for experiments that must exclude KV-capacity-driven
    preemption.

    Args:
        kv_cache_tokens: Total token slots in the GPU KV cache.
        max_sequence_length: Maximum prompt-plus-output length per request.
        block_size: Number of token slots in one KV cache block.
        reserve_blocks_per_sequence: Extra blocks reserved per live request for
            scheduler lookahead or an explicit safety margin.

    Returns:
        The largest positive number of concurrent requests that fits.

    Raises:
        ValueError: If an input is invalid or one maximum-length request cannot
            fit in the cache.
    """
    if kv_cache_tokens <= 0:
        raise ValueError("kv_cache_tokens must be positive")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if reserve_blocks_per_sequence < 0:
        raise ValueError("reserve_blocks_per_sequence must be non-negative")

    available_blocks = kv_cache_tokens // block_size
    sequence_blocks = (
        (max_sequence_length + block_size - 1) // block_size
        + reserve_blocks_per_sequence
    )
    concurrency = available_blocks // sequence_blocks
    if concurrency < 1:
        raise ValueError(
            "KV cache cannot fit one maximum-length request: "
            f"available_blocks={available_blocks}, sequence_blocks={sequence_blocks}"
        )
    return concurrency
