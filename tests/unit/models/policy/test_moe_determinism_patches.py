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

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.models.policy.workers.moe_zero_kl_patches import (
    _nrl_dynamic_step_context_bookkeeping,
    _patched_unpermute,
    _unpermute_fixed_order_combine,
    _unpermute_gather_combine,
    _unpermute_gather_combine_droppad,
    apply_moe_unpermute_determinism_patch,
    apply_router_replay_inference_patches,
    restore_moe_determinism_patches,
)


class TestUnpermuteFixedOrderCombine:
    def test_sums_per_token(self):
        permuted = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        sorted_indices = torch.tensor([0, 0, 1])
        out = _unpermute_fixed_order_combine(permuted, sorted_indices, torch.Size([2, 2]))
        assert torch.allclose(out, torch.tensor([[3.0, 0.0], [3.0, 0.0]]))


class TestGatherCombineDroppad:
    def setup_method(self):
        restore_moe_determinism_patches()

    def teardown_method(self):
        restore_moe_determinism_patches()

    def test_routes_one_token_per_expert(self):
        routing_map = torch.tensor([[True, False], [False, True]])
        permuted = torch.tensor([[10.0], [20.0]])
        sorted_indices = torch.tensor([0, 1])
        out = _unpermute_gather_combine_droppad(
            permuted, sorted_indices, torch.Size([2, 1]), routing_map
        )
        assert torch.allclose(out, torch.tensor([[10.0], [20.0]]))

    def test_patched_unpermute_uses_droppad_when_flag_set(self):
        routing_map = torch.tensor([[True, False], [False, True]])
        probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        permuted = torch.tensor([[3.0], [4.0]])
        sorted_indices = torch.tensor([0, 1])
        out = _patched_unpermute(
            permuted,
            sorted_indices,
            torch.Size([2, 1]),
            probs=probs,
            routing_map=routing_map,
            drop_and_pad=True,
        )
        assert torch.allclose(out, torch.tensor([[3.0], [4.0]]))


class TestGatherCombineUnified:
    def setup_method(self):
        restore_moe_determinism_patches()

    def teardown_method(self):
        restore_moe_determinism_patches()

    def test_packed_gather_matches_droppad_gather(self):
        routing_map = torch.tensor([[True, False], [False, True]])
        packed = torch.tensor([[1.5], [2.5]])
        packed_idx = torch.tensor([0, 1])
        droppad = torch.tensor([[1.5], [0.0], [2.5], [0.0]])
        droppad_idx = torch.tensor([0, 0, 1, 1])
        packed_out = _unpermute_gather_combine(
            packed, packed_idx, torch.Size([2, 1]), routing_map
        )
        droppad_out = _unpermute_gather_combine_droppad(
            droppad, droppad_idx, torch.Size([2, 1]), routing_map
        )
        expected = torch.tensor([[1.5], [2.5]])
        assert torch.allclose(packed_out, expected)
        assert torch.allclose(droppad_out, expected)

    def test_patched_unpermute_routes_gather_for_train_and_droppad_for_decode(
        self, monkeypatch
    ):
        monkeypatch.setenv("NRL_COMBINE_IMPL", "gather")
        monkeypatch.setenv("NRL_COMBINE_GATHER_DROPPAD", "1")
        routing_map = torch.tensor([[True, False], [False, True]])
        probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        train_out = _patched_unpermute(
            torch.tensor([[3.0], [4.0]]),
            torch.tensor([0, 1]),
            torch.Size([2, 1]),
            probs=probs,
            routing_map=routing_map,
            drop_and_pad=False,
        )
        decode_out = _patched_unpermute(
            torch.tensor([[3.0], [0.0], [4.0], [0.0]]),
            torch.tensor([0, 0, 1, 1]),
            torch.Size([2, 1]),
            probs=probs,
            routing_map=routing_map,
            drop_and_pad=True,
        )
        assert torch.allclose(train_out, torch.tensor([[3.0], [4.0]]))
        assert torch.allclose(decode_out, torch.tensor([[3.0], [4.0]]))


class TestApplyMoeDeterminismPatches:
    def setup_method(self):
        restore_moe_determinism_patches()

    def teardown_method(self):
        restore_moe_determinism_patches()

    def test_unpermute_patch_is_idempotent(self):
        fake_mod = MagicMock()
        fake_mod.HAVE_TE = False
        fake_mod.fused_unpermute = None
        fake_mod.is_te_min_version = lambda _v: False
        fake_mod.unpermute = MagicMock(return_value="orig")
        fake_dispatcher = MagicMock()
        fake_dispatcher.unpermute = MagicMock(return_value="dispatcher_orig")

        with patch.dict(
            sys.modules,
            {
                "megatron.core.transformer.moe.moe_utils": fake_mod,
                "megatron.core.transformer.moe.token_dispatcher": fake_dispatcher,
            },
        ):
            apply_moe_unpermute_determinism_patch()
            apply_moe_unpermute_determinism_patch()
            assert fake_mod.unpermute is _patched_unpermute
            assert fake_dispatcher.unpermute is _patched_unpermute
            restore_moe_determinism_patches()
            assert fake_mod.unpermute() == "orig"
            assert fake_dispatcher.unpermute() == "dispatcher_orig"

    def test_patched_unpermute_uses_fixed_order(self):
        permuted = torch.tensor([[1.0], [2.0]])
        sorted_indices = torch.tensor([0, 0])
        out = _patched_unpermute(permuted, sorted_indices, torch.Size([1, 1]))
        assert torch.allclose(out, torch.tensor([[3.0]]))

    def test_router_replay_inference_patch_replaces_methods(self):
        pytest.importorskip("megatron")
        from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
        from megatron.core.inference.text_generation_controllers.text_generation_controller import (
            TextGenerationController,
        )

        orig_bookkeeping = TextGenerationController._dynamic_step_context_bookkeeping
        orig_async_bookkeep = DynamicInferenceEngine.async_bookkeep
        try:
            apply_router_replay_inference_patches()
            assert TextGenerationController._dynamic_step_context_bookkeeping is (
                _nrl_dynamic_step_context_bookkeeping
            )
            assert DynamicInferenceEngine.async_bookkeep is not orig_async_bookkeep
        finally:
            restore_moe_determinism_patches()
            assert (
                TextGenerationController._dynamic_step_context_bookkeeping is orig_bookkeeping
            )
            assert DynamicInferenceEngine.async_bookkeep is orig_async_bookkeep
