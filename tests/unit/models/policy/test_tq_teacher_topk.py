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
"""CPU tests for the TQ-mediated distillation teacher forward.

Same shape as tests/unit/models/value/test_tq_value.py. The teacher differs
from every other ``*_presharded`` entrypoint in one way worth covering: it
writes back **two** tensors rather than one, and both carry a third axis
(``[B, S, k]``) that the single-tensor entrypoints never produce.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin


class _TeacherStubWorker(TQWorkerMixin):
    """Mixin host recording backend calls; fetch/attach are stubbed."""

    def __init__(
        self,
        is_leader: bool = True,
        logits: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ):
        self.calls: list[tuple] = []
        self._leader = is_leader
        self._dp_client = MagicMock()
        self._logits = logits if logits is not None else torch.ones(2, 3, 4)
        self._indices = (
            indices if indices is not None else torch.zeros(2, 3, 4, dtype=torch.long)
        )

    def _fetch(self, meta):
        self.calls.append(("fetch", meta))
        return {"data_from": meta}

    def _attach_or_repack_pack_metadata(self, data, meta):
        self.calls.append(("attach", meta))
        return data

    def _is_replica_leader(self) -> bool:
        return self._leader

    def get_topk_logits(self, data, k, micro_batch_size=None):
        self.calls.append(("get_topk_logits", data, k, micro_batch_size))
        return {"topk_logits": self._logits, "topk_indices": self._indices}


def _meta(sample_ids: list[str] | None = None) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=sample_ids if sample_ids is not None else ["s0", "s1"],
    )


class TestGetTopkLogitsPresharded:
    def test_writes_both_tensors_back_and_returns_nothing(self):
        w = _TeacherStubWorker()
        meta = _meta()

        with patch("nemo_rl.data_plane.column_io.write_columns") as write_columns:
            out = w.get_topk_logits_presharded(meta=meta, k=4, micro_batch_size=8)

        assert out is None
        assert [c[0] for c in w.calls] == ["fetch", "attach", "get_topk_logits"]
        assert w.calls[2][2] == 4
        assert w.calls[2][3] == 8

        # Two write-backs, not one: logits and indices are separate columns.
        assert write_columns.call_count == 2
        written = {}
        for call in write_columns.call_args_list:
            written.update(call.args[2])
        assert set(written) == {"teacher_topk_logits", "teacher_topk_indices"}
        assert torch.equal(written["teacher_topk_logits"], torch.ones(2, 3, 4))
        assert written["teacher_topk_indices"].dtype == torch.long

    def test_the_k_axis_survives_the_write_back(self):
        """The single-tensor entrypoints all write [B, S]; this one must not
        flatten or truncate the extra axis on the way through."""
        w = _TeacherStubWorker(logits=torch.randn(2, 5, 7))

        with patch("nemo_rl.data_plane.column_io.write_columns") as write_columns:
            w.get_topk_logits_presharded(meta=_meta(), k=7)

        written = {}
        for call in write_columns.call_args_list:
            written.update(call.args[2])
        assert written["teacher_topk_logits"].shape == (2, 5, 7)

    def test_non_leader_twin_does_not_write(self):
        """TP/CP/PP twins hold identical copies; a second writer is the
        duplicate-write bug the leader gate exists to prevent. Doubly relevant
        here, where one call writes two columns."""
        w = _TeacherStubWorker(is_leader=False)

        with patch("nemo_rl.data_plane.column_io.write_columns") as write_columns:
            w.get_topk_logits_presharded(meta=_meta(), k=4)

        write_columns.assert_not_called()

    def test_rejects_batch_dim_mismatch(self):
        w = _TeacherStubWorker(logits=torch.ones(3, 3, 4))

        with pytest.raises(ValueError, match="shape mismatch"):
            w.get_topk_logits_presharded(meta=_meta(["s0", "s1"]), k=4)
