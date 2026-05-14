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
"""Recovery API tests for the TransferQueue adapter without booting Ray/TQ."""

from __future__ import annotations

import pytest

from nemo_rl.data_plane.adapters.transfer_queue import TQDataPlaneClient
from nemo_rl.data_plane.interfaces import (
    DataPlaneClearError,
    DataPlaneReadError,
    DataPlaneTimeout,
)


class _FakeTQ:
    def __init__(self, listing=None, *, clear_error=None, list_error=None):
        self.listing = listing or {}
        self.clear_error = clear_error
        self.list_error = list_error
        self.cleared: list[tuple[list[str], str]] = []

    def kv_list(self, partition_id):
        if self.list_error is not None:
            raise self.list_error
        return {partition_id: self.listing.get(partition_id, {})}

    def kv_clear(self, keys, partition_id):
        if self.clear_error is not None:
            raise self.clear_error
        self.cleared.append((list(keys), partition_id))


def _client(tq) -> TQDataPlaneClient:
    client = TQDataPlaneClient.__new__(TQDataPlaneClient)
    client._tq = tq
    client._partitions = {}
    client._closed = False
    return client


def test_tq_list_metadata_groups_key_tags():
    client = _client(
        _FakeTQ(
            {
                "p": {
                    "a": {
                        "group_id": "g",
                        "weight_version": "4",
                        "created_at": "12.5",
                        "committed": "true",
                        "expected_num_keys": "2",
                    },
                    "b": {
                        "group_id": "g",
                        "weight_version": 4,
                        "committed": True,
                        "expected_num_keys": 2,
                    },
                    "c": {"group_id": "partial", "committed": False},
                }
            }
        )
    )

    groups = {group.group_id: group for group in client.list_metadata("p")}

    assert groups["g"].keys == ["a", "b"]
    assert groups["g"].weight_version == 4
    assert groups["g"].created_at == 12.5
    assert groups["g"].committed
    assert groups["g"].is_complete
    assert not groups["partial"].committed
    assert client.depth("p") == 1


def test_tq_clear_errors_are_typed():
    client = _client(_FakeTQ({"p": {"a": {}}}, clear_error=RuntimeError("clear failed")))

    with pytest.raises(DataPlaneClearError):
        client.pop(keys=["a"], partition_id="p")


def test_tq_list_errors_are_typed():
    client = _client(_FakeTQ(list_error=RuntimeError("bad metadata")))

    with pytest.raises(DataPlaneReadError):
        client.list_metadata("p")


def test_tq_call_timeout_is_typed():
    client = _client(_FakeTQ())

    with pytest.raises(DataPlaneTimeout):
        client._call_tq(
            "hang",
            lambda: __import__("time").sleep(1.0),
            DataPlaneReadError,
            timeout_s=0.001,
        )
