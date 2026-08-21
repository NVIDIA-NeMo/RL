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

from nemo_rl.models.generation.megatron.zero_train_gen_kl_patches.moe_zero_kl_patches import (
    _patched_unpermute,
    _unpermute_fixed_order_combine,
    apply_moe_determinism_patches,
    restore_moe_determinism_patches,
)


class TestFixedOrderCombine:
    def test_routes_one_token_per_expert(self):
        routing_map = torch.tensor([[True, False], [False, True]])
        permuted = torch.tensor([[10.0], [20.0]])
        sorted_indices = torch.tensor([0, 1])
        out = _unpermute_fixed_order_combine(
            permuted, sorted_indices, torch.Size([2, 1])
        )
        assert torch.allclose(out, torch.tensor([[10.0], [20.0]]))

    def test_patched_unpermute_uses_fixed_order_for_train_and_decode(self):
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
            routing_map=routing_map,
            drop_and_pad=True,
        )
        assert torch.allclose(train_out, torch.tensor([[3.0], [4.0]]))
        assert torch.allclose(decode_out, torch.tensor([[3.0], [4.0]]))

    def test_patched_unpermute_accepts_megatron_only_kwargs(self):
        out = _patched_unpermute(
            torch.tensor([[3.0], [4.0]]),
            torch.tensor([0, 1]),
            torch.Size([2, 1]),
            batch_invariant_inverse_map=torch.zeros(2, 2, 2, dtype=torch.long),
            future_megatron_kwarg=None,
        )
        assert torch.allclose(out, torch.tensor([[3.0], [4.0]]))


class TestApplyMoeDeterminismPatches:
    def setup_method(self):
        restore_moe_determinism_patches()

    def teardown_method(self):
        restore_moe_determinism_patches()

    def test_unpermute_patch_is_idempotent(self):
        fake_mod = MagicMock()
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
            apply_moe_determinism_patches()
            apply_moe_determinism_patches()
            assert fake_mod.unpermute is _patched_unpermute
            assert fake_dispatcher.unpermute is _patched_unpermute
            restore_moe_determinism_patches()
            assert fake_mod.unpermute() == "orig"
            assert fake_dispatcher.unpermute() == "dispatcher_orig"

    def test_patched_unpermute_rejects_fused(self):
        with pytest.raises(ValueError, match="moe_permute_fusion=false"):
            _patched_unpermute(
                torch.tensor([[1.0]]),
                torch.tensor([0]),
                torch.Size([1, 1]),
                fused=True,
            )
