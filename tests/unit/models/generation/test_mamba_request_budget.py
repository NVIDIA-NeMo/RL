# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Allocate native KV and recurrent state tensors within the configured pool budget."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from nemo_rl.models.generation.megatron.config import resolve_mamba_memory_ratio

if TYPE_CHECKING:
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

pytestmark = [
    pytest.mark.mcore,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs native Megatron CUDA context"
    ),
]


@pytest.fixture
def native_groups(tmp_path: Path) -> Iterator[None]:
    # Native context allocation belongs to the optional Megatron GPU test lane.
    from megatron.core import parallel_state

    assert not torch.distributed.is_initialized(), "use an isolated native test process"
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=120),
    )
    try:
        parallel_state.initialize_model_parallel()
        yield
        torch.cuda.synchronize()
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _context(
    *, max_requests: int | None, ratio: float | None, ssm_dtype: torch.dtype
) -> DynamicInferenceContext:
    # Exercise real inference allocation, including its request-state sizing branch.
    from megatron.core.inference.config import (
        InferenceConfig,
        MambaInferenceStateConfig,
    )
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
    from megatron.core.transformer.transformer_config import TransformerConfig

    model_config = TransformerConfig(
        num_layers=2,
        hidden_size=1024,
        num_attention_heads=8,
        num_query_groups=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        mamba_num_heads=8,
        mamba_head_dim=128,
        mamba_state_dim=1024,
    )
    inference_config = InferenceConfig(
        buffer_size_gb=1,
        max_requests=max_requests,
        max_tokens=256,
        max_sequence_length=512,
        num_cuda_graphs=None,
        enable_prefix_caching=False,
        mamba_memory_ratio=ratio,
        mamba_inference_state_config=MambaInferenceStateConfig(
            layer_type_list=[Symbols.MAMBA, Symbols.ATTENTION],
            conv_states_shape=(17408, 4),
            ssm_states_shape=(8, 128, 1024),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=ssm_dtype,
        ),
    )
    context = DynamicInferenceContext(model_config, inference_config)
    context.initialize_all_tensors()
    return context


def _resident_pool_bytes(context: DynamicInferenceContext) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            context.memory_buffer,
            context.mamba_conv_states,
            context.mamba_ssm_states,
        )
    )


@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32])
def test_explicit_request_states_are_deducted_from_native_kv_pool(
    native_groups: None,
    ssm_dtype: torch.dtype,
) -> None:
    legacy = _context(max_requests=64, ratio=0.1, ssm_dtype=ssm_dtype)
    assert _resident_pool_bytes(legacy) > 2**30
    actual = _context(
        max_requests=64,
        ssm_dtype=ssm_dtype,
        ratio=resolve_mamba_memory_ratio(
            is_hybrid_model=True,
            max_requests=64,
            num_speculative_tokens=0,
        ),
    )
    assert actual.max_requests == legacy.max_requests == 64
    assert _resident_pool_bytes(actual) <= 2**30
    assert actual.kv_block_allocator.pool_size < legacy.kv_block_allocator.pool_size


@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32])
def test_automatically_derived_native_capacity_is_preserved(
    native_groups: None,
    ssm_dtype: torch.dtype,
) -> None:
    legacy = _context(max_requests=None, ratio=0.1, ssm_dtype=ssm_dtype)
    actual = _context(
        max_requests=None,
        ssm_dtype=ssm_dtype,
        ratio=resolve_mamba_memory_ratio(
            is_hybrid_model=True,
            max_requests=None,
            num_speculative_tokens=0,
        ),
    )
    assert actual.max_requests == legacy.max_requests > 0
    assert actual.kv_block_allocator.pool_size == legacy.kv_block_allocator.pool_size
    assert _resident_pool_bytes(actual) == _resident_pool_bytes(legacy) <= 2**30


@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32])
def test_impossible_request_state_budget_is_rejected_before_allocation(
    native_groups: None,
    ssm_dtype: torch.dtype,
) -> None:
    with pytest.raises(
        AssertionError, match="Not enough memory for 1024 mamba requests"
    ):
        _context(
            max_requests=1024,
            ssm_dtype=ssm_dtype,
            ratio=resolve_mamba_memory_ratio(
                is_hybrid_model=True,
                max_requests=1024,
                num_speculative_tokens=0,
            ),
        )
