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

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.data_plane import KVBatchMeta


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

    def clear_samples(
        self, sample_ids: list[str] | None, partition_id: str
    ) -> None:
        assert partition_id == self._partition_id
        ids = list(sample_ids) if sample_ids is not None else list(self._rows)
        self.clear_calls.append(list(ids))
        for sid in ids:
            self._rows.pop(sid, None)

    def depth(self) -> int:
        return len(self._rows)


def _run(coro):
    return asyncio.run(coro)


def _add_group(buf: TQReplayBuffer, group_id: str, group_size: int, weight: int):
    sample_ids = [f"{group_id}_s{j}" for j in range(group_size)]
    tags = [{"weight_version": weight, "group_id": group_id}] * group_size
    return _run(
        buf.add(
            sample_ids=sample_ids,
            fields=None,
            tags=tags,
            weight_version=weight,
        )
    )


class TestTQReplayBufferAdd:
    def test_add_writes_tq_then_appends_meta(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")

        meta = _add_group(buf, "g0", group_size=2, weight=3)

        assert meta.sample_ids == ["g0_s0", "g0_s1"]
        assert dp.depth() == 2
        assert buf.size() == 1
        assert buf.weight_list == [3]
        assert buf.meta_list[0].sample_ids == ["g0_s0", "g0_s1"]
        assert len(dp.put_calls) == 1

    def test_add_rejects_non_int_weight_version(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")
        with pytest.raises(TypeError):
            _run(
                buf.add(
                    sample_ids=["g0_s0"],
                    fields=None,
                    tags=None,
                    weight_version=None,  # type: ignore[arg-type]
                )
            )
        assert dp.depth() == 0
        assert buf.size() == 0

    def test_add_rejects_bool_weight_version(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")
        with pytest.raises(TypeError):
            _run(
                buf.add(
                    sample_ids=["g0_s0"],
                    fields=None,
                    tags=None,
                    weight_version=True,  # type: ignore[arg-type]
                )
            )

    def test_add_appends_multiple_groups_in_order(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")

        _add_group(buf, "g0", group_size=2, weight=1)
        _add_group(buf, "g1", group_size=2, weight=2)
        _add_group(buf, "g2", group_size=2, weight=3)

        assert buf.size() == 3
        assert buf.weight_list == [1, 2, 3]
        assert [m.sample_ids[0] for m in buf.meta_list] == [
            "g0_s0",
            "g1_s0",
            "g2_s0",
        ]


class TestTQReplayBufferRemove:
    def test_remove_drops_fully_covered_entries(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")
        for g in range(3):
            _add_group(buf, f"g{g}", group_size=2, weight=g)

        n = _run(
            buf.remove(["g0_s0", "g0_s1", "g2_s0", "g2_s1"])
        )

        assert n == 2
        assert buf.size() == 1
        assert buf.weight_list == [1]
        assert buf.meta_list[0].sample_ids == ["g1_s0", "g1_s1"]
        # TQ cleared too
        assert dp.depth() == 2
        assert set(dp._rows) == {"g1_s0", "g1_s1"}

    def test_remove_rejects_partial_group(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")
        _add_group(buf, "g0", group_size=2, weight=0)
        _add_group(buf, "g1", group_size=2, weight=0)

        with pytest.raises(ValueError, match="whole-entry boundaries"):
            _run(buf.remove(["g0_s0"]))

        # Local state and TQ unchanged
        assert buf.size() == 2
        assert dp.depth() == 4
        assert dp.clear_calls == []

    def test_remove_rejects_unknown_sample_id(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")
        _add_group(buf, "g0", group_size=2, weight=0)

        with pytest.raises(ValueError, match="whole-entry boundaries"):
            _run(buf.remove(["g0_s0", "g0_s1", "ghost_s0"]))

        assert buf.size() == 1
        assert dp.depth() == 2

    def test_remove_empty_is_noop(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")
        _add_group(buf, "g0", group_size=2, weight=0)
        _add_group(buf, "g1", group_size=2, weight=0)

        n = _run(buf.remove([]))

        assert n == 0
        assert buf.size() == 2
        assert dp.depth() == 4


class TestTQReplayBufferSize:
    def test_size_and_len(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(dp, partition_id="rollout_data")
        assert buf.size() == 0
        assert len(buf) == 0

        _add_group(buf, "g0", group_size=2, weight=0)
        assert buf.size() == 1
        assert len(buf) == 1

        _add_group(buf, "g1", group_size=2, weight=0)
        assert buf.size() == 2
        assert len(buf) == 2

        _run(buf.remove(["g0_s0", "g0_s1"]))
        assert buf.size() == 1
        assert len(buf) == 1
