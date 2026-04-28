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

"""Unit tests for pipeline_parallel.py utilities."""

from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.models.automodel.pipeline_parallel import (
    PPLogitsCapturer,
    PPLossAdapter,
    _reset_pp_schedule_state,
    pad_batch_for_pp,
)

# =====================
# Fixtures
# =====================


@pytest.fixture
def mock_loss_fn():
    loss_fn = MagicMock(spec=LossFunction)
    loss_fn.return_value = (torch.tensor(0.5), {"loss": 0.5, "num_valid_samples": 4})
    loss_fn.input_type = LossInputType.LOGIT
    return loss_fn


@pytest.fixture
def mock_device_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    mesh.__getitem__ = MagicMock(return_value=mesh)
    return mesh


@pytest.fixture
def mock_cp_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    return mesh


@pytest.fixture
def mock_tp_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = MagicMock()
    return mesh


@pytest.fixture
def base_cfg():
    return {
        "dtensor_cfg": {"sequence_parallel": False},
        "sequence_packing": {"train_mb_tokens": 256},
        "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": None},
    }


# =====================
# Test PPLogitsCapturer
# =====================
@pytest.mark.automodel
class TestPPLogitsCapturer:
    def test_captures_logits(self):
        capturer = PPLogitsCapturer()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(4, 64, 32000)
        target = torch.randint(0, 32000, (4, 64))

        result = capturer(mock_output, target)

        assert len(capturer.captured_logits) == 1
        assert torch.equal(capturer.captured_logits[0], mock_output.logits)
        assert result.item() == 0.0

    def test_captures_multiple_microbatches(self):
        capturer = PPLogitsCapturer()
        for i in range(3):
            mock_output = MagicMock()
            mock_output.logits = torch.randn(2, 32, 1000)
            capturer(mock_output, torch.zeros(2, 32, dtype=torch.long))

        assert len(capturer.captured_logits) == 3

    def test_reset_clears_state(self):
        capturer = PPLogitsCapturer()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(2, 32, 1000)
        capturer(mock_output, torch.zeros(2, 32, dtype=torch.long))

        capturer.reset()
        assert len(capturer.captured_logits) == 0

    def test_captures_raw_tensor(self):
        """When output is a tensor (not an object with .logits), capture directly."""
        capturer = PPLogitsCapturer()
        raw_logits = torch.randn(4, 64, 32000)
        target = torch.randint(0, 32000, (4, 64))

        capturer(raw_logits, target)

        assert len(capturer.captured_logits) == 1
        assert torch.equal(capturer.captured_logits[0], raw_logits)

    def test_captured_logits_are_detached(self):
        capturer = PPLogitsCapturer()
        logits = torch.randn(2, 4, 100, requires_grad=True)
        capturer(logits, torch.zeros(2, 4, dtype=torch.long))

        assert not capturer.captured_logits[0].requires_grad


# =====================
# Test PPLossAdapter
# =====================
@pytest.mark.automodel
class TestPPLossAdapter:
    def test_set_microbatches_splits_data(
        self, mock_loss_fn, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        adapter = PPLossAdapter(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        data = {
            "input_ids": torch.randint(0, 1000, (8, 64)),
            "input_lengths": torch.full((8,), 64),
            "sample_mask": torch.ones(8),
        }
        n_microbatches = 2

        adapter.set_microbatches(
            data,
            n_microbatches,
            global_valid_seqs=torch.tensor(8),
            global_valid_toks=torch.tensor(512),
        )

        assert len(adapter._microbatches) == 2
        assert adapter._microbatches[0]["input_ids"].shape[0] == 4
        assert adapter._microbatches[1]["input_ids"].shape[0] == 4

    def test_reset_clears_state(
        self, mock_loss_fn, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        adapter = PPLossAdapter(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        data = {"input_ids": torch.randint(0, 1000, (4, 32))}
        adapter.set_microbatches(data, 1, torch.tensor(4), torch.tensor(128))

        adapter.reset()
        assert adapter._call_idx == 0
        assert adapter._all_metrics == []

    def test_call_scales_loss_by_dp_cp(
        self, mock_loss_fn, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        """Loss should be scaled by dp_size * cp_size to cancel FSDP averaging."""
        dp_size = 4
        cp_size = 2
        adapter = PPLossAdapter(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=cp_size,
            dp_size=dp_size,
        )

        data = {
            "input_ids": torch.randint(0, 1000, (4, 32)),
            "input_lengths": torch.full((4,), 32),
            "sample_mask": torch.ones(4),
        }
        adapter.set_microbatches(data, 1, torch.tensor(4), torch.tensor(128))

        logits = torch.randn(4, 32, 32000)
        target = torch.randint(0, 32000, (4, 32))

        mock_output = MagicMock()
        mock_output.logits = logits

        result = adapter(mock_output, target)

        base_loss = 0.5  # from mock_loss_fn
        expected_scale = dp_size * cp_size
        assert abs(result.item() - base_loss * expected_scale) < 1e-5

    def test_call_increments_index(
        self, mock_loss_fn, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        adapter = PPLossAdapter(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        data = {
            "input_ids": torch.randint(0, 1000, (4, 32)),
            "input_lengths": torch.full((4,), 32),
            "sample_mask": torch.ones(4),
        }
        adapter.set_microbatches(data, 2, torch.tensor(4), torch.tensor(128))

        logits = torch.randn(2, 32, 32000)
        target = torch.randint(0, 32000, (2, 32))
        mock_output = MagicMock()
        mock_output.logits = logits

        adapter(mock_output, target)
        assert adapter._call_idx == 1

        adapter(mock_output, target)
        assert adapter._call_idx == 2

    def test_unsqueezes_2d_logits(
        self, mock_loss_fn, base_cfg, mock_device_mesh, mock_cp_mesh, mock_tp_mesh
    ):
        """THD format produces 2D logits [total_tokens, vocab] — adapter should unsqueeze to 3D."""
        adapter = PPLossAdapter(
            loss_fn=mock_loss_fn,
            cfg=base_cfg,
            device_mesh=mock_device_mesh,
            cp_mesh=mock_cp_mesh,
            tp_mesh=mock_tp_mesh,
            cp_size=1,
            dp_size=1,
        )

        # THD: 1 packed row of 128 tokens
        data = {
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "input_lengths": torch.tensor([128]),
            "sample_mask": torch.ones(1),
        }
        adapter.set_microbatches(data, 1, torch.tensor(1), torch.tensor(128))

        # 2D logits (THD format: [total_tokens, vocab])
        logits_2d = torch.randn(128, 32000)
        # Target must match the unsqueezed batch dim
        target = torch.randint(0, 32000, (1, 128))
        mock_output = MagicMock()
        mock_output.logits = logits_2d

        # Should not crash — adapter handles 2D by unsqueezing to [1, 128, vocab]
        result = adapter(mock_output, target)
        assert isinstance(result, torch.Tensor)


# =====================
# Test pad_batch_for_pp
# =====================
@pytest.mark.automodel
class TestPadBatchForPP:
    def test_no_padding_needed(self):
        input_ids = torch.randint(0, 1000, (8, 64))
        padded, actual = pad_batch_for_pp(input_ids, pp_batch_size=8)

        assert actual == 8
        assert padded.shape == (8, 64)
        assert torch.equal(padded, input_ids)

    def test_padding_added(self):
        input_ids = torch.randint(0, 1000, (3, 64))
        padded, actual = pad_batch_for_pp(input_ids, pp_batch_size=8)

        assert actual == 3
        assert padded.shape == (8, 64)
        # Original rows preserved
        assert torch.equal(padded[:3], input_ids)
        # Padding rows are zeros
        assert (padded[3:] == 0).all()

    def test_single_sample(self):
        input_ids = torch.randint(0, 1000, (1, 32))
        padded, actual = pad_batch_for_pp(input_ids, pp_batch_size=4)

        assert actual == 1
        assert padded.shape == (4, 32)

    def test_preserves_dtype(self):
        input_ids = torch.randint(0, 1000, (2, 16), dtype=torch.int32)
        padded, _ = pad_batch_for_pp(input_ids, pp_batch_size=4)

        assert padded.dtype == torch.int32


# =====================
# Test _reset_pp_schedule_state
# =====================
@pytest.mark.automodel
class TestResetPPScheduleState:
    def test_raises_without_update_seq_len(self):
        model = MagicMock(spec=[])  # No update_seq_len attribute

        with pytest.raises(RuntimeError, match="update_seq_len.*not found"):
            _reset_pp_schedule_state(model, seq_len=64)

    def test_calls_update_seq_len_for_bshd(self):
        model = MagicMock()
        model._pp_current_seq_len = None

        _reset_pp_schedule_state(model, seq_len=64)

        model.update_seq_len.assert_called_once_with(64)

    def test_calls_reset_thd_for_seqpack(self):
        model = MagicMock()
        model._pp_current_seq_len = None

        with patch(
            "nemo_rl.models.automodel.pipeline_parallel.reset_pp_stage_shapes_for_thd"
        ) as mock_reset_thd:
            _reset_pp_schedule_state(
                model, seq_len=128, seqpack=True, is_hf_model=False
            )

            mock_reset_thd.assert_called_once_with(model, 128)
            # _pp_current_seq_len should be cleared for THD
            # (THD always needs fresh shapes)

    def test_hf_model_uses_update_seq_len_even_with_seqpack(self):
        """HF models don't use THD format, so update_seq_len is used."""
        model = MagicMock()
        model._pp_current_seq_len = None

        _reset_pp_schedule_state(model, seq_len=64, seqpack=True, is_hf_model=True)

        model.update_seq_len.assert_called_once_with(64)

    def test_force_clears_cached_seq_len(self):
        """force=True clears _pp_current_seq_len so update_seq_len re-initializes."""
        model = MagicMock()
        model._pp_current_seq_len = 64

        _reset_pp_schedule_state(model, seq_len=64, force=True)

        # force=True should clear the cache, causing update_seq_len to be called
        # even though seq_len matches the cached value
        model.update_seq_len.assert_called_once_with(64)

    def test_update_seq_len_skips_when_unchanged(self):
        """Without force, update_seq_len handles its own skip logic."""
        model = MagicMock()
        model._pp_current_seq_len = 64

        _reset_pp_schedule_state(model, seq_len=64)

        # update_seq_len is still called — it has its own internal skip
        model.update_seq_len.assert_called_once_with(64)
