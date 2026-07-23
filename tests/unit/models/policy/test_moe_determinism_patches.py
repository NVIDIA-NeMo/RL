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

from nemo_rl.models.policy.workers.moe_determinism_patches import (
    _nrl_dynamic_step_context_bookkeeping,
    _patched_unpermute,
    _unpermute_fixed_order_combine,
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
