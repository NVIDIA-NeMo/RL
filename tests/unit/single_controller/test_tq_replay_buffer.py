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

"""Unit tests for TQReplayBuffer (plain SC-process buffer + TQ proxy)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

import nemo_rl.algorithms.async_utils.replay_buffer as _replay_buffer_module
from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord

# Each record yields _N_GENS training rows.
_N_GENS = 2


def _stub_record_to_train_batch(
    record: PromptGroupRecord, *, pad_value_dict: Any
) -> BatchedDataDict[Any]:
    del record, pad_value_dict
    return BatchedDataDict[Any](
        {
            "input_ids": torch.ones((_N_GENS, 3), dtype=torch.long),
            "input_lengths": torch.full((_N_GENS,), 3, dtype=torch.long),
            "total_reward": torch.zeros(_N_GENS, dtype=torch.float32),
        }
    )


@pytest.fixture(autouse=True)
def _patch_converter(monkeypatch):
    """Bypass the real ``record_to_train_batch`` so tests can use empty records."""
    monkeypatch.setattr(
        _replay_buffer_module,
        "record_to_train_batch",
        _stub_record_to_train_batch,
    )


class FakeDataPlaneClient:
    """Sync in-memory DataPlaneClient stub used by TQReplayBuffer tests."""

    def __init__(self, partition_id: str = "rollout_data") -> None:
        self._partition_id = partition_id
        self._rows: dict[str, dict[str, Any]] = {}
        self.put_calls: list[dict[str, Any]] = []
        self.clear_calls: list[list[str]] = []

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: Any = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        assert partition_id == self._partition_id
        self.put_calls.append(
            {
                "sample_ids": list(sample_ids),
                "fields": fields,
                "tags": [dict(t) for t in tags] if tags is not None else None,
            }
        )
        for i, sid in enumerate(sample_ids):
            self._rows[sid] = {
                "tag": dict(tags[i]) if tags is not None else {},
            }
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            sample_ids=list(sample_ids),
            fields=None,
            tags=[dict(t) for t in tags] if tags is not None else None,
        )

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        assert partition_id == self._partition_id
        ids = list(sample_ids) if sample_ids is not None else list(self._rows)
        self.clear_calls.append(list(ids))
        for sid in ids:
            self._rows.pop(sid, None)

    def depth(self) -> int:
        return len(self._rows)


def _run(coro):
    return asyncio.run(coro)


def _make_record() -> PromptGroupRecord:
    """Opaque PromptGroupRecord — converter is stubbed, so contents are unused."""
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[],
        extra_env_info=None,
        metadata={},
        completions=[],
        rollout_metrics={},
    )


def _make_buffer(dp: FakeDataPlaneClient) -> TQReplayBuffer:
    return TQReplayBuffer(
        dp, partition_id="rollout_data", pad_value_dict={"token_ids": 0}
    )


def _add_group(
    buf: TQReplayBuffer, weight: int, end_weight: int | None = None
) -> KVBatchMeta:
    if end_weight is None:
        end_weight = weight
    group_id = buf.reserve(weight_version=weight)
    return _run(
        buf.commit(
            group_id,
            _make_record(),
            start_weight_version=weight,
            end_weight_version=end_weight,
        )
    )


class TestTQReplayBufferReserveCommit:
    def test_reserve_appends_placeholder_unready(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        group_id = buf.reserve(weight_version=3)

        assert isinstance(group_id, str) and group_id
        assert buf.size() == 1
        assert buf.start_weight_list == [3]
        assert buf.end_weight_list == [-1]
        assert buf.ready_list == [False]
        assert buf.meta_list == [None]
        assert dp.depth() == 0
        assert dp.put_calls == []

    def test_commit_writes_tq_then_fills_meta(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        group_id = buf.reserve(weight_version=3)
        meta = _run(
            buf.commit(
                group_id,
                _make_record(),
                start_weight_version=3,
                end_weight_version=4,
            )
        )

        # pack_payload stamps sample_ids as ``{group_uuid}_g{i}``.
        assert len(meta.sample_ids) == _N_GENS
        head, _, idx = meta.sample_ids[0].rpartition("_g")
        assert head == group_id and idx == "0"
        assert all(sid.startswith(group_id + "_g") for sid in meta.sample_ids)
        assert dp.depth() == _N_GENS
        assert buf.size() == 1
        assert buf.start_weight_list == [3]
        assert buf.end_weight_list == [4]
        assert buf.ready_list == [True]
        assert buf.meta_list[0].sample_ids == meta.sample_ids
        # TQ tag uses start_weight_version (dispatch time).
        assert meta.tags == [{"weight_version": 3}] * _N_GENS
        assert len(dp.put_calls) == 1

    def test_commit_raises_for_unknown_group_id(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        buf.reserve(weight_version=3)

        with pytest.raises(ValueError):
            _run(
                buf.commit(
                    "not-a-real-id",
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

    def test_reserve_then_commit_preserves_dispatch_order(self):
        """Reserve in dispatch order, commit out of order; insertion order holds."""
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        weights = (1, 2, 3)
        gids = [buf.reserve(weight_version=w) for w in weights]
        # Commit out of order: 2, 0, 1 — buffer order must still match reserve order.
        for i in (2, 0, 1):
            _run(
                buf.commit(
                    gids[i],
                    _make_record(),
                    start_weight_version=weights[i],
                    end_weight_version=weights[i],
                )
            )

        assert buf.size() == 3
        assert buf.start_weight_list == [1, 2, 3]
        assert buf.end_weight_list == [1, 2, 3]
        assert buf.ready_list == [True, True, True]
        # sample_id head equals reserved group_id at each slot.
        for i, gid in enumerate(gids):
            assert buf.meta_list[i] is not None
            assert buf.meta_list[i].sample_ids[0].startswith(gid + "_g")

    def test_commit_appends_multiple_records_in_order(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        metas = [_add_group(buf, weight=w) for w in (1, 2, 3)]

        assert buf.size() == 3
        assert buf.start_weight_list == [1, 2, 3]
        assert buf.end_weight_list == [1, 2, 3]
        assert [m.sample_ids for m in buf.meta_list] == [
            list(metas[0].sample_ids),
            list(metas[1].sample_ids),
            list(metas[2].sample_ids),
        ]


class TestTQReplayBufferRemove:
    def test_remove_drops_indices_and_clears_dp_when_requested(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(3)]

        n = _run(buf.remove([0, 2], remove_in_dp=True))

        assert n == 2
        assert buf.size() == 1
        assert buf.start_weight_list == [1]
        assert buf.end_weight_list == [1]
        assert buf.meta_list[0].sample_ids == list(metas[1].sample_ids)
        assert dp.depth() == _N_GENS
        assert set(dp._rows) == set(metas[1].sample_ids)

    def test_remove_without_dp_keeps_rows(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(2)]

        n = _run(buf.remove([0], remove_in_dp=False))

        assert n == 1
        assert buf.size() == 1
        assert buf.start_weight_list == [1]
        assert buf.end_weight_list == [1]
        assert buf.meta_list[0].sample_ids == list(metas[1].sample_ids)
        assert dp.clear_calls == []
        assert dp.depth() == 2 * _N_GENS

    def test_remove_rejects_out_of_range_before_mutating(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(2)]

        with pytest.raises(IndexError, match=r"out of range: 5; size=2"):
            _run(buf.remove([0, 5], remove_in_dp=True))

        assert buf.size() == 2
        assert [m.sample_ids for m in buf.meta_list] == [
            list(metas[0].sample_ids),
            list(metas[1].sample_ids),
        ]
        assert dp.depth() == 2 * _N_GENS
        assert dp.clear_calls == []

    def test_remove_empty_is_noop(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=0)
        _add_group(buf, weight=0)

        n = _run(buf.remove([], remove_in_dp=True))

        assert n == 0
        assert buf.size() == 2
        assert dp.depth() == 2 * _N_GENS
        assert dp.clear_calls == []


class TestTQReplayBufferSize:
    def test_size_and_len(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        assert buf.size() == 0
        assert len(buf) == 0

        _add_group(buf, weight=0)
        assert buf.size() == 1
        assert len(buf) == 1

        _add_group(buf, weight=0)
        assert buf.size() == 2
        assert len(buf) == 2

        _run(buf.remove([0], remove_in_dp=True))
        assert buf.size() == 1
        assert len(buf) == 1
