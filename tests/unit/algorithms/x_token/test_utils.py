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
"""Unit tests for ``nemo_rl/algorithms/x_token/utils.py`` (CPU-only)."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.x_token.utils import (
    assert_teacher_student_batch_grid,
    assert_xtoken_ipc_node_local,
    pad_distillation_val_batch,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class TestAssertTeacherStudentBatchGrid:
    def _check(self, **kw):
        args = dict(
            global_batch_size=8,
            student_gbs=8,
            teacher_gbs=8,
            student_dp=4,
            teacher_dp=2,
            student_mbs=1,
            teacher_mbs=1,
        )
        args.update(kw)
        assert_teacher_student_batch_grid(**args)

    def test_valid_passes(self):
        self._check()

    def test_gbs_disagreement_raises(self):
        with pytest.raises(AssertionError):
            self._check(student_gbs=4)

    def test_not_divisible_by_dp_raises(self):
        with pytest.raises(AssertionError):
            self._check(global_batch_size=6, student_gbs=6, teacher_gbs=6)

    def test_not_divisible_by_mbs_raises(self):
        with pytest.raises(AssertionError):
            self._check(student_mbs=3)


class TestAssertXtokenIpcNodeLocal:
    def _check(self, **kw):
        args = dict(
            num_nodes=2,
            gpus_per_node=4,
            student_tp=2,
            student_cp=2,
            teacher_tp=2,
            teacher_cp=2,
            student_dp=1,
            teacher_dp=1,
        )
        args.update(kw)
        assert_xtoken_ipc_node_local(**args)

    def test_single_node_always_ok(self):
        # Even an otherwise-disallowed multi-node grid passes on one node.
        self._check(num_nodes=1, student_dp=1, teacher_dp=2, teacher_tp=4, teacher_cp=2)

    def test_multinode_matched_grid_ok(self):
        self._check()

    def test_multinode_mismatched_dp_raises(self):
        with pytest.raises(AssertionError):
            self._check(student_dp=1, teacher_dp=2)

    def test_multinode_mismatched_group_raises(self):
        with pytest.raises(AssertionError):
            self._check(student_tp=1, teacher_tp=2)  # groups 2 vs 4

    def test_multinode_group_exceeds_node_raises(self):
        with pytest.raises(AssertionError):
            self._check(
                gpus_per_node=2, student_tp=2, student_cp=2, teacher_tp=2, teacher_cp=2
            )  # group 4 > 2

    def test_multinode_group_not_node_aligned_raises(self):
        with pytest.raises(AssertionError):
            self._check(
                gpus_per_node=6, student_tp=2, student_cp=2, teacher_tp=2, teacher_cp=2
            )  # 6 % 4 != 0


class TestPadDistillationValBatch:
    def test_pads_tensors_lists_and_zeros_mask(self):
        batch = BatchedDataDict(
            {
                "x": torch.arange(2 * 3).reshape(2, 3),
                "sample_mask": torch.ones(2),
                "meta": ["a", "b"],
            }
        )
        out = pad_distillation_val_batch(batch, 4)
        assert out["x"].shape == (4, 3)
        assert torch.equal(out["x"][:2], batch["x"])
        assert out["meta"] == ["a", "b", "b", "b"]
        assert out["sample_mask"].tolist() == [1, 1, 0, 0]

    def test_noop_when_target_equals_size(self):
        batch = BatchedDataDict({"sample_mask": torch.ones(2)})
        assert pad_distillation_val_batch(batch, 2) is batch
