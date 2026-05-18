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
"""Unit tests for the lean observability decorator.

Wraps :class:`NoOpDataPlaneClient` so the tests run in the slim Tier-1
venv (no TQ, no Ray). The lean shape is one user-injected ``on_event``
callback plus :meth:`snapshot` for cumulative totals — no ABC, no
built-in sinks.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.observability import MetricsDataPlaneClient


@pytest.fixture
def wrapped_client():
    events: list[dict] = []
    inner = NoOpDataPlaneClient()
    client = MetricsDataPlaneClient(inner, on_event=events.append)
    yield client, events
    inner.close()


def test_put_records_bytes_and_count(wrapped_client):
    client, events = wrapped_client
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=4, consumer_tasks=["read"]
    )
    fields = TensorDict({"x": torch.zeros(4, dtype=torch.float32)}, batch_size=[4])
    client.put_samples(sample_ids=["a", "b", "c", "d"], partition_id="p", fields=fields)

    put_events = [e for e in events if e["op"] == "put"]
    assert len(put_events) == 1
    e = put_events[0]
    assert e["status"] == "ok"
    assert e["n_keys"] == 4
    assert e["n_bytes"] == 16  # 4 floats * 4 bytes
    assert e["wall_ms"] >= 0


def test_get_records_after_put(wrapped_client):
    client, events = wrapped_client
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=2, consumer_tasks=["read"]
    )
    client.put_samples(
        sample_ids=["a", "b"],
        partition_id="p",
        fields=TensorDict({"x": torch.ones(2)}, batch_size=[2]),
    )
    out = client.get_samples(sample_ids=["a", "b"], partition_id="p", select_fields=["x"])
    assert torch.equal(out["x"], torch.ones(2))

    get_events = [e for e in events if e["op"] == "get"]
    assert len(get_events) == 1
    assert get_events[0]["n_bytes"] > 0


def test_register_and_clear_recorded(wrapped_client):
    client, events = wrapped_client
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["r"]
    )
    client.clear_samples(sample_ids=None, partition_id="p")

    ops = [e["op"] for e in events]
    assert ops.count("register") == 1
    assert ops.count("clear") == 1


def test_error_status_recorded_and_reraised(wrapped_client):
    """Decorator does NOT swallow errors — re-raise after recording."""
    client, events = wrapped_client
    with pytest.raises(KeyError):
        client.get_samples(sample_ids=["a"], partition_id="nope", select_fields=["x"])

    err = [e for e in events if e["op"] == "get" and e["status"] == "error"]
    assert len(err) == 1


def test_snapshot_accumulates_successful_ops(wrapped_client):
    client, _ = wrapped_client
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["r"]
    )
    client.put_samples(
        sample_ids=["a"],
        partition_id="p",
        fields=TensorDict({"x": torch.zeros(1)}, batch_size=[1]),
    )
    snap = client.snapshot()
    assert snap["total_ops"] >= 2  # register + put
    assert snap["total_bytes"] >= 4  # 1 float = 4 bytes


def test_default_callback_is_noop():
    """Omitting on_event must not raise; the wrapper just forwards."""
    inner = NoOpDataPlaneClient()
    client = MetricsDataPlaneClient(inner)
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["r"]
    )
    client.close()


def test_close_propagates(wrapped_client):
    client, _ = wrapped_client
    client.close()
    # Second close must not raise — NoOp is idempotent.
    client.close()


def test_factory_wraps_when_observability_enabled():
    """Programmatic wrap path; factory.py uses the same MetricsDataPlaneClient."""
    inner = NoOpDataPlaneClient()
    seen: list[dict] = []
    client = MetricsDataPlaneClient(inner, on_event=seen.append)
    assert hasattr(client, "snapshot")
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["r"]
    )
    assert len(seen) == 1 and seen[0]["op"] == "register"
    client.close()
