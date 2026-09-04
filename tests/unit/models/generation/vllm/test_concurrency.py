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

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.lfs.concurrency import (
    get_engine_kv_cache_shape,
    max_concurrency_without_preemption,
)


@pytest.mark.parametrize(
    "engine",
    [
        SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(num_gpu_blocks=100, block_size=16)
            )
        ),
        SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    cache_config=SimpleNamespace(
                        num_gpu_blocks=100, block_size=16
                    )
                )
            )
        ),
    ],
)
def test_get_engine_kv_cache_shape_supports_vllm_frontends(engine) -> None:
    assert get_engine_kv_cache_shape(engine) == (1_600, 16)


def test_get_engine_kv_cache_shape_requires_initialized_blocks() -> None:
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks=0, block_size=16)
        )
    )
    with pytest.raises(RuntimeError, match="initialized vLLM KV cache"):
        get_engine_kv_cache_shape(engine)


def test_max_concurrency_uses_block_aligned_max_sequence_length() -> None:
    concurrency = max_concurrency_without_preemption(
        kv_cache_tokens=1_069_984,
        max_sequence_length=16_532,
        block_size=16,
    )

    assert concurrency == 64


def test_max_concurrency_can_reserve_lookahead_blocks() -> None:
    concurrency = max_concurrency_without_preemption(
        kv_cache_tokens=1_069_984,
        max_sequence_length=16_532,
        block_size=16,
        reserve_blocks_per_sequence=8,
    )

    assert concurrency == 64


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kv_cache_tokens": 0}, "kv_cache_tokens"),
        ({"max_sequence_length": 0}, "max_sequence_length"),
        ({"block_size": 0}, "block_size"),
        ({"reserve_blocks_per_sequence": -1}, "reserve_blocks_per_sequence"),
    ],
)
def test_max_concurrency_rejects_invalid_inputs(
    kwargs: dict[str, int], match: str
) -> None:
    valid = {
        "kv_cache_tokens": 1_069_984,
        "max_sequence_length": 16_532,
        "block_size": 16,
        "reserve_blocks_per_sequence": 0,
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=match):
        max_concurrency_without_preemption(**valid)


def test_max_concurrency_rejects_cache_smaller_than_one_sequence() -> None:
    with pytest.raises(ValueError, match="cannot fit one"):
        max_concurrency_without_preemption(
            kv_cache_tokens=1_024,
            max_sequence_length=2_048,
            block_size=16,
        )
