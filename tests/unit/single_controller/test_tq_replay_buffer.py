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
        self.get_calls: list[dict[str, Any]] = []

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

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        assert partition_id == self._partition_id
        self.get_calls.append(
            {
                "sample_ids": list(sample_ids),
                "select_fields": (
                    list(select_fields) if select_fields is not None else None
                ),
            }
        )
        # Opaque per-group payload; load_state_dict must re-put it verbatim.
        return {"payload_for": list(sample_ids)}

    def depth(self) -> int:
        return len(self._rows)


class _PutThenFailDataPlaneClient(FakeDataPlaneClient):
    def put_samples(self, *args: Any, **kwargs: Any) -> KVBatchMeta:
        super().put_samples(*args, **kwargs)
        raise RuntimeError("injected DataPlane put failure")


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

    def test_cancel_reservation_drops_only_unready_slot(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        committed = _add_group(buf, weight=2)
        pending_id = buf.reserve(weight_version=3)

        assert buf.cancel_reservation(pending_id)
        assert not buf.cancel_reservation(pending_id)
        assert not buf.cancel_reservation("missing")
        assert not buf.cancel_reservation(buf._group_ids[0])

        assert buf.size() == 1
        assert buf.ready_list == [True]
        assert buf.meta_list[0].sample_ids == committed.sample_ids

    def test_reserve_rejects_duplicate_group_id(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        buf.reserve(weight_version=3, group_id="stable-group")

        with pytest.raises(ValueError, match="already reserved"):
            buf.reserve(weight_version=3, group_id="stable-group")

        assert buf.size() == 1

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

        assert dp.put_calls == []
        assert dp.depth() == 0

    def test_failed_commit_clears_rows_and_reservation(self):
        dp = _PutThenFailDataPlaneClient()
        buf = _make_buffer(dp)
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(RuntimeError, match="injected DataPlane put failure"):
            _run(
                buf.commit(
                    group_id,
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        assert buf.size() == 0
        assert dp.depth() == 0
        assert len(dp.clear_calls) == 1

    def test_commit_rejects_reservation_weight_mismatch_before_writing(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(ValueError, match="reserved at weight version 3, not 4"):
            _run(
                buf.commit(
                    group_id,
                    _make_record(),
                    start_weight_version=4,
                    end_weight_version=4,
                )
            )

        assert buf.size() == 1
        assert dp.put_calls == []

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


# ── state_dict / load_state_dict (Phase 3 checkpointing) ─────────────────────


def _group_id_of(meta: KVBatchMeta) -> str:
    head, _, _ = meta.sample_ids[0].rpartition("_g")
    return head


def _make_group_entry(
    group_id: str,
    weight: int,
    *,
    n: int = _N_GENS,
    target_step: int | None = None,
    sample_ids: list[str] | None = None,
    sequence_lengths: list[int] | None = None,
    partition_id: str = "rollout_data",
) -> dict[str, Any]:
    """Hand-built envelope group (bypasses commit) for preflight tests."""
    sids = (
        list(sample_ids)
        if sample_ids is not None
        else [f"{group_id}_g{i}" for i in range(n)]
    )
    meta = KVBatchMeta(
        partition_id=partition_id,
        task_name="train",
        sample_ids=sids,
        fields=["input_ids", "input_lengths", "total_reward"],
        sequence_lengths=(
            sequence_lengths if sequence_lengths is not None else [3] * len(sids)
        ),
        tags=[{"weight_version": weight}] * len(sids),
    )
    return {
        "meta": meta,
        "start_weight": weight,
        "end_weight": weight,
        "target_step": target_step,
        "group_id": group_id,
        "fields_data": {"payload_for": sids},
    }


def _make_envelope(
    groups: list[dict[str, Any]],
    *,
    partition_id: str = "rollout_data",
    saved_capacity: int = 8,
) -> dict[str, Any]:
    return {
        "partition_id": partition_id,
        "saved_capacity": saved_capacity,
        "groups": list(groups),
    }


def _load(
    buf: TQReplayBuffer,
    state: dict[str, Any],
    *,
    max_groups: int = 8,
    expected_partition_id: str = "rollout_data",
    expected_group_size: int = _N_GENS,
) -> int:
    return _run(
        buf.load_state_dict(
            state,
            max_groups=max_groups,
            expected_partition_id=expected_partition_id,
            expected_group_size=expected_group_size,
        )
    )


class TestTQReplayBufferStateDict:
    def test_state_dict_serializes_ready_and_skips_unready(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=w) for w in (1, 2)]
        buf.reserve(weight_version=3)  # in-flight: must be excluded

        state = _run(buf.state_dict(saved_capacity=8))

        assert state["partition_id"] == "rollout_data"
        assert state["saved_capacity"] == 8
        assert len(state["groups"]) == 2
        assert [g["start_weight"] for g in state["groups"]] == [1, 2]
        assert [g["end_weight"] for g in state["groups"]] == [1, 2]
        assert [g["target_step"] for g in state["groups"]] == [None, None]
        assert [g["group_id"] for g in state["groups"]] == [
            _group_id_of(metas[0]),
            _group_id_of(metas[1]),
        ]
        # Payloads are fetched from the DataPlane rows of each group.
        assert [c["sample_ids"] for c in dp.get_calls] == [
            list(metas[0].sample_ids),
            list(metas[1].sample_ids),
        ]
        assert dp.get_calls[0]["select_fields"] == list(metas[0].fields)
        assert state["groups"][0]["fields_data"] == {
            "payload_for": list(metas[0].sample_ids)
        }

    def test_round_trip_restores_lists_and_rows(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=w) for w in (1, 2)]
        state = _run(buf.state_dict(saved_capacity=8))

        dp2 = FakeDataPlaneClient()
        buf2 = _make_buffer(dp2)
        restored = _load(buf2, state)

        assert restored == 2
        assert buf2.size() == 2
        # Parallel lists rebuilt in order, all ready.
        assert buf2.start_weight_list == [1, 2]
        assert buf2.end_weight_list == [1, 2]
        assert buf2.target_step_list == [None, None]
        assert buf2.ready_list == [True, True]
        assert buf2._group_ids == [_group_id_of(m) for m in metas]
        assert [m.sample_ids for m in buf2.meta_list] == [
            list(metas[0].sample_ids),
            list(metas[1].sample_ids),
        ]
        # Rows re-put with identical sample_ids / fields payload / tags.
        assert len(dp2.put_calls) == 2
        for put, meta in zip(dp2.put_calls, metas):
            assert put["sample_ids"] == list(meta.sample_ids)
            assert put["fields"] == {"payload_for": list(meta.sample_ids)}
            assert put["tags"] == [dict(t) for t in meta.tags]


class TestTQReplayBufferLoadPreflight:
    """Malformed envelopes raise ValueError before any DataPlane write."""

    def _assert_rejected(self, state: dict[str, Any], match: str, **load_kwargs):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        with pytest.raises(ValueError, match=match):
            _load(buf, state, **load_kwargs)
        assert dp.put_calls == []
        assert buf.size() == 0

    def test_missing_envelope_keys(self):
        self._assert_rejected({"groups": []}, match="missing required keys")

    def test_partition_id_mismatch(self):
        state = _make_envelope([], partition_id="other_partition")
        self._assert_rejected(state, match="partition_id mismatch")

    def test_group_missing_keys(self):
        group = _make_group_entry("g0", weight=1)
        del group["fields_data"]
        self._assert_rejected(_make_envelope([group]), match="group missing keys")

    def test_group_misaligned_sequence_lengths(self):
        group = _make_group_entry("g0", weight=1, sequence_lengths=[3])
        self._assert_rejected(_make_envelope([group]), match="misaligned")

    def test_group_size_mismatch(self):
        state = _make_envelope([_make_group_entry("g0", weight=1, n=2)])
        self._assert_rejected(state, match="misaligned", expected_group_size=3)

    def test_duplicate_sample_ids_across_groups(self):
        g0 = _make_group_entry("g0", weight=1)
        g1 = _make_group_entry(
            "g1", weight=2, sample_ids=["g0_g0", "g1_g1"]
        )  # g0_g0 collides
        self._assert_rejected(_make_envelope([g0, g1]), match="duplicate sample_id")


class TestTQReplayBufferLoadTruncation:
    def test_capacity_change_truncates_to_freshest(self, monkeypatch):
        state = _make_envelope(
            [_make_group_entry(f"g{w}", weight=w) for w in (1, 2, 3)],
            saved_capacity=8,
        )
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        printed: list[str] = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)),
        )

        restored = _load(buf, state, max_groups=2)

        assert restored == 2
        # The freshest max_groups groups survive, original order preserved.
        assert buf.start_weight_list == [2, 3]
        put_sample_ids = [sid for c in dp.put_calls for sid in c["sample_ids"]]
        assert "g1_g0" not in put_sample_ids and "g1_g1" not in put_sample_ids
        assert any("capacity changed" in line for line in printed)
