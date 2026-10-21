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
import asyncio
from types import SimpleNamespace

from transfer_queue.storage.managers.base import KVStorageManager

from nemo_rl.data_plane.adapters.tq_nixl import NixlStorageManager


class FakeClient:
    def __init__(self):
        self.cleared = []

    def clear(self, keys, custom_backend_meta=None):
        self.cleared.append(custom_backend_meta)


def put(monkeypatch, old_meta, fields):
    """Run put_data on a manager whose TQ put overwrites the meta, as TQ does."""
    mgr = object.__new__(NixlStorageManager)
    mgr.storage_client = FakeClient()
    md = SimpleNamespace(_custom_backend_meta=old_meta)

    async def tq_put(self, data, metadata, data_parser=None):
        metadata._custom_backend_meta = [{f: {"b": "new"} for f in data} for _ in old_meta]

    monkeypatch.setattr(KVStorageManager, "put_data", tq_put)
    monkeypatch.setattr(NixlStorageManager, "__del__", lambda self: None)  # __init__ skipped
    asyncio.run(mgr.put_data(dict.fromkeys(fields), md))
    return mgr.storage_client.cleared


def test_overwrite_releases_old_blob(monkeypatch):
    assert put(monkeypatch, [{"x": {"b": "old"}}], ["x"]) == [[{"b": "old"}]]


def test_first_put_releases_nothing(monkeypatch):
    assert put(monkeypatch, [{}], ["x"]) == []


def test_only_written_fields_are_released(monkeypatch):
    old = [{"x": {"b": "ox"}, "y": {"b": "oy"}}]
    assert put(monkeypatch, old, ["x"]) == [[{"b": "ox"}]]
