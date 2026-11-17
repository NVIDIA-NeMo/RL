# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# [CUSTOM] Tests for vllm_worker._patch_vllm_swa_admission_cap (vLLM 0.20.x
# sliding-window block accounting, "Cannot get N free blocks from the pool").
# vllm is only imported inside the test bodies (marked @pytest.mark.vllm), so the
# module stays collectable in the non-vllm unit lane. No GPU needed.

import pytest

BLOCK_SIZE = 16
SLIDING_WINDOW = 1024
MAX_NUM_BATCHED_TOKENS = 2048
MAX_MODEL_LEN = 131072


@pytest.fixture
def swa_cap_patch(monkeypatch):
    """Apply the patch for one test and restore vLLM's classes afterwards."""
    import inspect

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    if (
        "apply_admission_cap"
        in inspect.signature(
            SingleTypeKVCacheManager.get_num_blocks_to_allocate
        ).parameters
    ):
        pytest.skip("vLLM already has the upstream fix; the patch is a no-op")

    from nemo_rl.models.generation.vllm.vllm_worker import (
        _patch_vllm_swa_admission_cap,
    )

    # Record the current attributes so that monkeypatch restores them on teardown.
    monkeypatch.setattr(
        SingleTypeKVCacheManager,
        "get_num_blocks_to_allocate",
        SingleTypeKVCacheManager.get_num_blocks_to_allocate,
    )
    monkeypatch.setattr(
        KVCacheManager, "can_fit_full_sequence", KVCacheManager.can_fit_full_sequence
    )
    monkeypatch.setattr(
        SingleTypeKVCacheManager, "_nemo_rl_swa_cap_patched", False, raising=False
    )
    _patch_vllm_swa_admission_cap()


def _make_swa_kv_cache_manager(num_blocks: int):
    """A real KVCacheManager with one DomynEdge-like sliding-window group."""
    import torch
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    spec = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=SLIDING_WINDOW,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["swa"], spec)],
    )
    return KVCacheManager(
        config,
        max_model_len=MAX_MODEL_LEN,
        hash_block_size=BLOCK_SIZE,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_caching=True,
    )


@pytest.mark.vllm
def test_running_long_swa_request_needs_a_block_at_full_pool(swa_cap_patch):
    """The per-step check must report the block a long decoding request needs.

    Then allocate_slots returns None and the scheduler preempts, instead of
    allocate_new_blocks raising and killing the EngineCore.
    """
    from vllm.utils.math_utils import cdiv

    kv_manager = _make_swa_kv_cache_manager(num_blocks=4096)
    manager = kv_manager.coordinator.single_type_managers[0]
    pool = kv_manager.block_pool
    assert manager._max_admission_blocks_per_request is not None

    # Far beyond window + max_num_batched_tokens, so the cap would bite.
    # 20000 tokens fill exactly 1250 blocks: token 20001 opens a new block.
    num_tokens = 20_000
    num_blocks = cdiv(num_tokens, BLOCK_SIZE)
    num_slid_out = (num_tokens - (SLIDING_WINDOW - 1)) // BLOCK_SIZE
    # Running request in decode: blocks that slid out of the window are null
    # placeholders, only the in-window ones are real.
    manager.req_to_blocks["req"] = [pool.null_block] * num_slid_out + (
        pool.get_new_blocks(num_blocks - num_slid_out)
    )
    manager.num_cached_block["req"] = 0
    # Other requests hold the rest of the pool.
    pool.get_new_blocks(pool.get_num_free_blocks())

    # One decode step, as allocate_slots does it.
    manager.remove_skipped_blocks("req", num_tokens)
    needed = manager.get_num_blocks_to_allocate(
        "req",
        num_tokens=num_tokens + 1,
        new_computed_blocks=[],
        total_computed_tokens=num_tokens,
        num_tokens_main_model=num_tokens + 1,
    )

    assert pool.get_num_free_blocks() == 0
    # Unpatched vLLM 0.20.1 returns 0 here and then crashes in the allocation.
    assert needed == 1
    with pytest.raises(ValueError, match="free blocks"):
        manager.allocate_new_blocks("req", num_tokens + 1, num_tokens + 1)


@pytest.mark.vllm
def test_admission_gate_keeps_swa_cap(swa_cap_patch):
    """The full-sequence admission gate still counts SWA groups at window size.

    A long new request must be admitted even if the pool could not hold its
    whole length: the patch must not remove the cap from the gate.
    """
    from types import SimpleNamespace

    from vllm.utils.math_utils import cdiv

    prompt_len = 100_000
    num_blocks = 1000
    assert cdiv(prompt_len, BLOCK_SIZE) > num_blocks  # would not fit uncapped
    kv_manager = _make_swa_kv_cache_manager(num_blocks=num_blocks)
    request = SimpleNamespace(
        request_id="new", num_computed_tokens=0, num_tokens=prompt_len
    )

    assert kv_manager.can_fit_full_sequence(request)
