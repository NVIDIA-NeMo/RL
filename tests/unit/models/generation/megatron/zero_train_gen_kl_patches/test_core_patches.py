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

from unittest.mock import MagicMock, patch

import torch

from nemo_rl.models.generation.megatron.zero_train_gen_kl_patches.core_patches import (
    apply_log_softmax_determinism_patch,
    apply_te_bik_attention_assert_skip_patch,
    apply_te_gemm_cublas_pinned_patch,
    restore_log_softmax_determinism_patch,
    restore_te_bik_attention_assert_skip_patch,
    restore_te_gemm_cublas_pinned_patch,
)


class TestApplyTeGemmCublasPinnedPatch:
    """Tests for apply_te_gemm_cublas_pinned_patch."""

    def setup_method(self):
        restore_te_gemm_cublas_pinned_patch()

    def teardown_method(self):
        restore_te_gemm_cublas_pinned_patch()

    def test_shrinks_workspace_and_is_idempotent(self, capsys):
        orig_fn = MagicMock(return_value=4096)
        mock_ws_fn = MagicMock()
        mock_ws_fn.cache_clear = MagicMock()
        mock_gemm_mod = MagicMock()
        mock_gemm_mod.get_cublas_workspace_size_bytes = orig_fn
        mock_gemm_mod.get_cublas_workspace = mock_ws_fn

        with patch(
            "nemo_rl.models.generation.megatron.zero_train_gen_kl_patches."
            "core_patches.importlib.import_module",
            return_value=mock_gemm_mod,
        ):
            apply_te_gemm_cublas_pinned_patch()
            apply_te_gemm_cublas_pinned_patch()

        assert mock_gemm_mod.get_cublas_workspace_size_bytes() == 4
        mock_ws_fn.cache_clear.assert_called_once()
        captured = capsys.readouterr()
        assert captured.out.count("[zero_train_gen_mismatch] shrunk TE cuBLAS workspace") == 1

    def test_skips_when_te_gemm_module_missing(self, capsys):
        with patch(
            "nemo_rl.models.generation.megatron.zero_train_gen_kl_patches."
            "core_patches.importlib.import_module",
            side_effect=ImportError("no te"),
        ):
            apply_te_gemm_cublas_pinned_patch()

        captured = capsys.readouterr()
        assert "is not importable" in captured.out

    def test_restore_puts_back_original(self):
        orig_fn = MagicMock(return_value=4096)
        mock_gemm_mod = MagicMock()
        mock_gemm_mod.get_cublas_workspace_size_bytes = orig_fn
        mock_gemm_mod.get_cublas_workspace = MagicMock()

        with patch(
            "nemo_rl.models.generation.megatron.zero_train_gen_kl_patches."
            "core_patches.importlib.import_module",
            return_value=mock_gemm_mod,
        ):
            apply_te_gemm_cublas_pinned_patch()
            restore_te_gemm_cublas_pinned_patch()

        assert mock_gemm_mod.get_cublas_workspace_size_bytes is orig_fn


class TestApplyLogSoftmaxDeterminismPatch:
    def setup_method(self):
        restore_log_softmax_determinism_patch()

    def teardown_method(self):
        restore_log_softmax_determinism_patch()

    def test_matches_inference_for_tp1(self):
        from nemo_rl.distributed import model_utils

        original = model_utils._compute_distributed_log_softmax_with_grad
        logits = torch.randn(2, 5, dtype=torch.float32)

        with patch("torch.distributed.get_world_size", return_value=1):
            apply_log_softmax_determinism_patch()
            apply_log_softmax_determinism_patch()
            actual = model_utils._compute_distributed_log_softmax_with_grad(
                logits, MagicMock()
            )

        torch.testing.assert_close(
            actual,
            torch.nn.functional.log_softmax(logits, dim=-1),
            rtol=0,
            atol=0,
        )
        restore_log_softmax_determinism_patch()
        assert model_utils._compute_distributed_log_softmax_with_grad is original


class TestApplyTeBikAttentionAssertSkipPatch:
    def setup_method(self):
        restore_te_bik_attention_assert_skip_patch()

    def teardown_method(self):
        restore_te_bik_attention_assert_skip_patch()

    def test_noops_assert_and_is_idempotent(self, capsys):
        orig_assert = MagicMock(side_effect=AssertionError("TE too old"))
        mock_bik_mod = MagicMock()
        mock_bik_mod.assert_te_supports_batch_invariant_attention = orig_assert

        with patch(
            "nemo_rl.models.generation.megatron.zero_train_gen_kl_patches."
            "core_patches.importlib.import_module",
            return_value=mock_bik_mod,
        ):
            apply_te_bik_attention_assert_skip_patch()
            apply_te_bik_attention_assert_skip_patch()
            mock_bik_mod.assert_te_supports_batch_invariant_attention()

        orig_assert.assert_not_called()
        captured = capsys.readouterr()
        assert captured.out.count("skipped Megatron TE batch-invariant attention") == 1

    def test_restore_puts_back_original(self):
        orig_assert = MagicMock()
        mock_bik_mod = MagicMock()
        mock_bik_mod.assert_te_supports_batch_invariant_attention = orig_assert

        with patch(
            "nemo_rl.models.generation.megatron.zero_train_gen_kl_patches."
            "core_patches.importlib.import_module",
            return_value=mock_bik_mod,
        ):
            apply_te_bik_attention_assert_skip_patch()
            restore_te_bik_attention_assert_skip_patch()

        assert mock_bik_mod.assert_te_supports_batch_invariant_attention is orig_assert
