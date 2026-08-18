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

from nemo_rl.models.policy.workers import moe_zero_kl_patches as moe_patches
from nemo_rl.models.policy.workers.moe_zero_kl_patches import (
    _build_sparse_routing_with_index_put,
    _nrl_dynamic_step_context_bookkeeping,
    _patched_topk_routing_with_score_function,
    _patched_unpermute,
    _unpermute_gather_combine,
    _unpermute_gather_combine_droppad,
    _unpermute_segmented_combine,
    apply_moe_unpermute_determinism_patch,
    apply_router_replay_inference_patches,
    configure_moe_combine_for_cuda_graph_inference,
    configure_moe_combine_for_zero_kl,
    restore_moe_determinism_patches,
)


class TestSparseRoutingIndexPut:
    def test_builds_expected_routing_map(self):
        logits = torch.zeros(2, 4)
        probs = torch.tensor([[0.7, 0.3], [1.0, 0.0]])
        top_indices = torch.tensor([[0, 2], [1, 3]])
        routing_probs, routing_map = _build_sparse_routing_with_index_put(
            logits, probs, top_indices
        )
        assert routing_map[0, 0] and routing_map[0, 2]
        assert routing_map[1, 1] and routing_map[1, 3]
        assert torch.allclose(routing_probs[0, 0], torch.tensor(0.7))
        assert torch.allclose(routing_probs[1, 1], torch.tensor(1.0))


class TestTopkRoutingPatch:
    def setup_method(self):
        restore_moe_determinism_patches()

    def teardown_method(self):
        restore_moe_determinism_patches()

    def test_uses_index_put_for_sparse_output(self):
        logits = torch.randn(3, 5)
        probs = torch.tensor([[0.6, 0.4], [1.0, 0.0], [0.2, 0.8]])
        top_indices = torch.tensor([[1, 3], [0, 2], [4, 1]])

        def fake_orig(*args, **kwargs):
            if kwargs.get("dense_output"):
                return probs, top_indices
            raise AssertionError("expected dense_output=True call")

        moe_patches._TOPK_ROUTING_ORIG = fake_orig
        routing_probs, routing_map = _patched_topk_routing_with_score_function(
            logits, topk=2
        )
        assert routing_map.shape == logits.shape
        assert routing_map.dtype == torch.bool
        assert routing_probs[0, 1] == probs[0, 0]

    def test_rejects_fused_topk(self):
        with pytest.raises(ValueError, match="moe_permute_fusion=false"):
            _patched_topk_routing_with_score_function(
                torch.randn(2, 4), topk=2, fused=True
            )


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
        configure_moe_combine_for_cuda_graph_inference()
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
        configure_moe_combine_for_cuda_graph_inference()
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

    def test_zero_kl_config_uses_gather_train_and_droppad_decode(self):
        configure_moe_combine_for_zero_kl(gather_droppad=False)
        routing_map = torch.tensor([[True, False], [False, True]])
        probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        with (
            patch.object(
                moe_patches,
                "_unpermute_gather_combine",
                wraps=_unpermute_gather_combine,
            ) as gather_combine,
            patch.object(
                moe_patches,
                "_unpermute_gather_combine_droppad",
                wraps=_unpermute_gather_combine_droppad,
            ) as droppad_combine,
        ):
            _patched_unpermute(
                torch.tensor([[3.0], [4.0]]),
                torch.tensor([0, 1]),
                torch.Size([2, 1]),
                probs=probs,
                routing_map=routing_map,
                drop_and_pad=False,
            )
            configure_moe_combine_for_cuda_graph_inference()
            _patched_unpermute(
                torch.tensor([[3.0], [0.0], [4.0], [0.0]]),
                torch.tensor([0, 0, 1, 1]),
                torch.Size([2, 1]),
                probs=probs,
                routing_map=routing_map,
                drop_and_pad=True,
            )

        gather_combine.assert_called_once()
        droppad_combine.assert_called_once()


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
        fake_mod.topk_routing_with_score_function = MagicMock(return_value=("topk", "map"))
        fake_dispatcher = MagicMock()
        fake_dispatcher.unpermute = MagicMock(return_value="dispatcher_orig")

        with patch.dict(
            sys.modules,
            {
                "megatron.core.transformer.moe.moe_utils": fake_mod,
                "megatron.core.transformer.moe.token_dispatcher": fake_dispatcher,
                "megatron.core.transformer.moe.router": MagicMock(
                    topk_routing_with_score_function=MagicMock(return_value=("r", "m"))
                ),
            },
        ):
            apply_moe_unpermute_determinism_patch()
            apply_moe_unpermute_determinism_patch()
            assert fake_mod.unpermute is _patched_unpermute
            assert fake_dispatcher.unpermute is _patched_unpermute
            assert fake_mod.topk_routing_with_score_function is (
                _patched_topk_routing_with_score_function
            )
            restore_moe_determinism_patches()
            assert fake_mod.unpermute() == "orig"
            assert fake_dispatcher.unpermute() == "dispatcher_orig"

    def test_patched_unpermute_falls_back_to_segmented_without_routing_map(self):
        permuted = torch.tensor([[1.0], [2.0]])
        sorted_indices = torch.tensor([0, 0])
        expected = _unpermute_segmented_combine(
            permuted, sorted_indices, torch.Size([1, 1])
        )
        out = _patched_unpermute(permuted, sorted_indices, torch.Size([1, 1]))
        assert torch.allclose(out, expected)

    def test_patched_unpermute_rejects_fused(self):
        with pytest.raises(ValueError, match="moe_permute_fusion=false"):
            _patched_unpermute(
                torch.tensor([[1.0]]),
                torch.tensor([0]),
                torch.Size([1, 1]),
                fused=True,
            )

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


class TestPadResetBeforeScoring:
    def setup_method(self):
        restore_moe_determinism_patches()

    def teardown_method(self):
        restore_moe_determinism_patches()

    def test_should_reset_when_zero_kl_colocated_graphs_and_moe_pad(self):
        cfg = {
            "megatron_cfg": {
                "zero_train_gen_mismatch": True,
                "moe_pad_experts_for_cuda_graph_inference": True,
            },
            "generation": {
                "colocated": {"enabled": True},
                "mcore_generation_config": {"cuda_graph_impl": "local"},
            },
        }
        assert moe_patches._should_reset_pad_before_scoring(cfg)

    def test_sleep_wrapper_resets_padding(self, monkeypatch):
        from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

        calls: list[bool] = []

        def fake_sleep(self):
            calls.append(True)

        monkeypatch.setattr(MegatronGenerationMixin, "_sleep", fake_sleep)
        moe_patches.apply_colocated_pad_reset_patch()

        worker = MagicMock()
        worker.cfg = {
            "megatron_cfg": {
                "zero_train_gen_mismatch": True,
                "moe_pad_experts_for_cuda_graph_inference": True,
            },
            "generation": {
                "colocated": {"enabled": True},
                "mcore_generation_config": {"cuda_graph_impl": "local"},
            },
        }
        worker.model = MagicMock()

        with patch.object(moe_patches, "_reset_decode_expert_padding") as reset_pad:
            MegatronGenerationMixin._sleep(worker)
            assert calls == [True]
            reset_pad.assert_called_once_with(worker.model)


class TestForceEagerDims:
    def setup_method(self):
        restore_moe_determinism_patches()
        moe_patches.set_inference_cuda_graphs_enabled(False)

    def teardown_method(self):
        restore_moe_determinism_patches()

    def test_should_suppress_when_inference_graphs_off(self):
        moe_patches.set_inference_cuda_graphs_enabled(False)
        assert moe_patches._should_suppress_spurious_cuda_graph_match()

    def test_should_allow_match_when_inference_graphs_on(self):
        moe_patches.set_inference_cuda_graphs_enabled(True)
        assert not moe_patches._should_suppress_spurious_cuda_graph_match()

    def test_env_override_forces_suppression(self, monkeypatch):
        monkeypatch.setenv("NRL_FORCE_EAGER_DIMS", "1")
        moe_patches.set_inference_cuda_graphs_enabled(True)
        assert moe_patches._should_suppress_spurious_cuda_graph_match()

    def test_env_override_can_disable_suppression(self, monkeypatch):
        monkeypatch.setenv("NRL_FORCE_EAGER_DIMS", "0")
        moe_patches.set_inference_cuda_graphs_enabled(False)
        assert not moe_patches._should_suppress_spurious_cuda_graph_match()

    def test_match_graph_config_returns_none_when_graphs_off(self):
        pytest.importorskip("megatron")
        from megatron.core.inference.batch_dimensions_utils import (
            CUDAGraphBatchDimensionBuilder,
        )

        sentinel = object()
        orig_match = CUDAGraphBatchDimensionBuilder.match_graph_config

        def fake_match(cls, *args, **kwargs):
            return sentinel

        CUDAGraphBatchDimensionBuilder.match_graph_config = classmethod(fake_match)  # type: ignore[method-assign]
        try:
            moe_patches.apply_force_eager_dims_patch()
            result = CUDAGraphBatchDimensionBuilder.match_graph_config(None, [])
            assert result is None
        finally:
            restore_moe_determinism_patches()
            CUDAGraphBatchDimensionBuilder.match_graph_config = orig_match


class TestForceEagerLifecycle:
    _COLocated_CFG = {
        "megatron_cfg": {"zero_train_gen_mismatch": True},
        "generation": {
            "colocated": {"enabled": True},
            "mcore_generation_config": {"cuda_graph_impl": "local"},
        },
    }

    def setup_method(self):
        restore_moe_determinism_patches()

    def teardown_method(self):
        restore_moe_determinism_patches()

    def test_should_track_colocated_zero_kl_with_graphs(self):
        assert moe_patches._should_track_inference_cuda_graph_lifecycle(
            self._COLocated_CFG
        )

    def test_sync_prepare_enables_inference_graph_flag(self):
        moe_patches.set_inference_cuda_graphs_enabled(False)
        moe_patches._sync_inference_cuda_graphs_enabled_for_prepare(self._COLocated_CFG)
        assert moe_patches._INFERENCE_CUDA_GRAPHS_ENABLED is True

    def test_lifecycle_patch_wraps_generation_entrypoints(self):
        from nemo_rl.models.generation.megatron.megatron_worker import (
            MegatronGenerationMixin,
        )

        orig_prepare = MegatronGenerationMixin.prepare_for_generation
        orig_finish = MegatronGenerationMixin.finish_generation
        moe_patches.apply_colocated_force_eager_lifecycle_patch()
        try:
            assert MegatronGenerationMixin.prepare_for_generation is not orig_prepare
            assert MegatronGenerationMixin.finish_generation is not orig_finish
        finally:
            moe_patches.restore_colocated_force_eager_lifecycle_patch()
            assert MegatronGenerationMixin.prepare_for_generation is orig_prepare
            assert MegatronGenerationMixin.finish_generation is orig_finish
