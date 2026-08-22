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
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.observability import (
    MetricsDataPlaneClient,
    _estimate_encoded_bytes,
    _td_bytes,
)

from ._rollout_shapes import make_rollout_batch


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
    out = client.get_samples(
        sample_ids=["a", "b"], partition_id="p", select_fields=["x"]
    )
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


def test_observability_records_realistic_rollout_put() -> None:
    """Metrics middleware records put-bytes correctly when the put carries a
    realistic rollout-shaped batch (bf16 logprobs, int32 masks, int64 ids)."""

    inner = NoOpDataPlaneClient()
    seen: list[dict] = []
    client = MetricsDataPlaneClient(inner, on_event=seen.append)

    n = 4
    batch = make_rollout_batch(n=n, max_seqlen=64, seed=71)
    client.register_partition(
        partition_id="train",
        fields=["input_ids", "input_lengths", "generation_logprobs"],
        num_samples=n,
        consumer_tasks=["train"],
    )
    fields = TensorDict(
        {
            "input_ids": batch["input_ids"],
            "input_lengths": batch["input_lengths"],
            "generation_logprobs": batch["generation_logprobs"],
        },
        batch_size=[n],
    )
    client.put_samples(
        sample_ids=[f"u{i}" for i in range(n)],
        partition_id="train",
        fields=fields,
    )

    put_events = [e for e in seen if e["op"] == "put"]
    assert len(put_events) == 1
    # Bytes should reflect bf16 logprobs (2 bytes/elem) + int64 ids (8 bytes/elem),
    # not a fixed-dtype assumption. Lower bound: at least one full int64 batch.
    min_expected = n * 64 * 8  # input_ids alone
    assert put_events[0]["n_bytes"] >= min_expected
    client.close()


# ── byte accounting ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,td,expected",
    [
        ("flat", TensorDict({"x": torch.zeros(8, 16)}, batch_size=[8]), 8 * 16 * 4),
        (
            "sliced-view",
            TensorDict({"x": torch.zeros(8, 32)[:, :8]}, batch_size=[8]),
            8 * 8 * 4,
        ),
        (
            "transposed",
            TensorDict({"x": torch.zeros(4, 8).t()}, batch_size=[8]),
            8 * 4 * 4,
        ),
        (
            "stride-0-expand",
            TensorDict({"x": torch.zeros(8, 1).expand(8, 9)}, batch_size=[8]),
            8 * 9 * 4,
        ),
        (
            "mixed-dtype",
            TensorDict(
                {
                    "i": torch.zeros(8, 4, dtype=torch.int64),
                    "b": torch.zeros(8, 4, dtype=torch.bool),
                    "f": torch.zeros(8, 4, dtype=torch.bfloat16),
                },
                batch_size=[8],
            ),
            8 * 4 * 8 + 8 * 4 * 1 + 8 * 4 * 2,
        ),
        (
            "nested-container",
            TensorDict(
                {
                    "a": torch.zeros(8, 2),
                    "sub": TensorDict({"b": torch.zeros(8, 3)}, batch_size=[8]),
                },
                batch_size=[8],
            ),
            8 * 2 * 4 + 8 * 3 * 4,
        ),
        ("empty", TensorDict({}, batch_size=[8]), 0),
        ("none", None, 0),
    ],
)
def test_td_bytes_counts_wire_payload(name, td, expected):
    """Tensor leaves count as ``contiguous().nbytes``, containers count once."""
    assert _td_bytes(td) == expected


def test_td_bytes_nontensordata_is_not_broadcast():
    """``NonTensorData`` holds ONE object; counting it per batch row would
    inflate a 64-row put by 64x. Its bytes must not scale with batch size."""
    payload = {"tool": "bash", "text": "x" * 100}
    small = TensorDict({"m": NonTensorData(payload, batch_size=[2])}, batch_size=[2])
    large = TensorDict({"m": NonTensorData(payload, batch_size=[64])}, batch_size=[64])
    assert _td_bytes(small) == _td_bytes(large)
    assert _td_bytes(small) >= 100  # the string itself is still counted


def test_td_bytes_nontensorstack_scales_with_rows():
    """``NonTensorStack`` genuinely holds one object per row, so its estimate
    must scale — and stay close to the exact walk it extrapolates from."""
    row = {"turns": ["hello"] * 4, "n": 3}
    stack_8 = NonTensorStack(*[NonTensorData(dict(row)) for _ in range(8)])
    stack_64 = NonTensorStack(*[NonTensorData(dict(row)) for _ in range(64)])
    bytes_8 = _td_bytes(TensorDict({"s": stack_8}, batch_size=[8]))
    bytes_64 = _td_bytes(TensorDict({"s": stack_64}, batch_size=[64]))
    assert bytes_8 > 0
    assert bytes_64 == pytest.approx(8 * bytes_8, rel=0.05)
    # And it agrees with summing every row explicitly.
    exact = sum(_estimate_encoded_bytes(dict(row), [10_000]) for _ in range(64))
    assert bytes_64 == pytest.approx(exact, rel=0.05)


def test_estimate_encoded_bytes_walk_is_bounded():
    """The node budget caps the walk so one pathological payload cannot make
    a put O(payload size)."""
    huge = {"k": list(range(100_000))}
    bounded = _estimate_encoded_bytes(huge, [64])
    unbounded = _estimate_encoded_bytes(huge, [10_000_000])
    assert bounded < unbounded
    assert bounded <= 4 * 64  # ≤2 leaves per budget unit, ≤2 bytes each here


def test_outstanding_bytes_reconcile_exactly():
    """Put then clear must return ``bytes_outstanding`` to zero: the per-key
    split drops its division remainder on one key rather than spreading it,
    so the total has to be preserved for the accounting to close."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    ids = [f"u{i}" for i in range(7)]  # 7 keys => non-zero remainder
    fields = TensorDict({"x": torch.zeros(7, 5)}, batch_size=[7])
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=7, consumer_tasks=["t"]
    )
    client.put_samples(sample_ids=ids, partition_id="p", fields=fields)
    assert client.snapshot()["bytes_outstanding"] == 7 * 5 * 4
    client.clear_samples(sample_ids=ids, partition_id="p")
    assert client.snapshot()["bytes_outstanding"] == 0
    client.close()


def test_no_callback_still_accumulates_stats():
    """``on_event=None`` skips building the event dict; the counters that
    ``snapshot()`` reports must not depend on a sink being registered."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=2, consumer_tasks=["t"]
    )
    client.put_samples(
        sample_ids=["a", "b"],
        partition_id="p",
        fields=TensorDict({"x": torch.zeros(2, 3)}, batch_size=[2]),
    )
    snap = client.snapshot()
    assert snap["total_bytes"] == 2 * 3 * 4
    assert snap["by_op"]["put"]["calls"] == 1
    assert snap["total_wall_ms"] > 0
    client.close()


# ── wire-in / wire-out hash verification ───────────────────────────────


class _CorruptingClient(NoOpDataPlaneClient):
    """Flips one element of one field on read — a stand-in for a wire bug."""

    def __init__(self, field: str, row: int) -> None:
        super().__init__()
        self._corrupt_field = field
        self._corrupt_row = row

    def get_samples(self, sample_ids, partition_id, select_fields):
        out = super().get_samples(sample_ids, partition_id, select_fields)
        if self._corrupt_field in out.keys():
            out[self._corrupt_field][self._corrupt_row] += 1
        return out


def _hash_client(inner=None):
    client = MetricsDataPlaneClient(
        inner or NoOpDataPlaneClient(), verify_tensor_hash=True
    )
    client.register_partition(
        partition_id="p", fields=["ids", "lp"], num_samples=4, consumer_tasks=["t"]
    )
    return client


def _hash_fields(n=4):
    return TensorDict(
        {
            "ids": torch.arange(n * 6, dtype=torch.int64).reshape(n, 6),
            "lp": torch.linspace(0, 1, n * 6, dtype=torch.bfloat16).reshape(n, 6),
        },
        batch_size=[n],
    )


def test_hash_verification_clean_roundtrip():
    client = _hash_client()
    ids = [f"u{i}" for i in range(4)]
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids", "lp"])

    hv = client.snapshot()["hash_verify"]
    assert hv["rows_recorded"] == 4
    assert hv["rows_checked"] == 4
    assert hv["rows_unverified"] == 0
    assert hv["mismatches"] == 0
    client.close()


def test_hash_verification_detects_corruption():
    client = _hash_client(_CorruptingClient(field="ids", row=2))
    ids = [f"u{i}" for i in range(4)]
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids", "lp"])

    hv = client.snapshot()["hash_verify"]
    assert hv["mismatches"] == 1
    assert client.get_step_metrics(1.0)["hash/mismatches"] == 1
    client.close()


def test_hash_verification_survives_shard_readback():
    """A 4-row put read back two rows at a time must still line up: the
    fingerprint is per row, not per batch."""
    client = _hash_client()
    ids = [f"u{i}" for i in range(4)]
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
    for shard in (ids[:2], ids[2:]):
        client.get_samples(sample_ids=shard, partition_id="p", select_fields=["ids"])

    hv = client.snapshot()["hash_verify"]
    assert hv["rows_checked"] == 4
    assert hv["mismatches"] == 0
    client.close()


def test_hash_verification_reports_rows_it_never_wrote():
    """A consumer-side client sees only wire-out. Those rows must land in
    ``rows_unverified`` — reporting 0 mismatches would read as 'clean'."""
    inner = NoOpDataPlaneClient()
    writer = _hash_client(inner)
    ids = [f"u{i}" for i in range(4)]
    writer.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())

    reader = MetricsDataPlaneClient(inner, verify_tensor_hash=True)
    reader.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids"])

    hv = reader.snapshot()["hash_verify"]
    assert hv["rows_unverified"] == 4
    assert hv["rows_checked"] == 0
    assert hv["mismatches"] == 0
    writer.close()


def test_hash_fingerprints_released_on_clear():
    """Fingerprints must be bounded by the live key population, not by
    cumulative traffic."""
    client = _hash_client()
    ids = [f"u{i}" for i in range(4)]
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
    assert client._hash_by_partition["p"]
    client.clear_samples(sample_ids=ids, partition_id="p")
    assert client._hash_by_partition == {}
    client.close()


def test_hash_fingerprint_covers_jagged_fields():
    """The per-token fields on this wire are jagged by the time they reach
    ``put_samples`` (``codec.pack_jagged_fields``). Skipping nested leaves
    would leave the entire bulk payload unguarded while still reporting zero
    mismatches — a guard that reads as clean because it checked nothing."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient(), verify_tensor_hash=True)
    ids = ["a", "b", "c"]
    rows = [torch.arange(n, dtype=torch.int64) + n for n in (3, 5, 4)]
    jagged = TensorDict(
        {"x": torch.nested.nested_tensor(rows, layout=torch.jagged)}, batch_size=[3]
    )
    digests = client._row_fingerprints(jagged, ids)
    assert "x" in digests, "jagged leaf was skipped"
    assert client.snapshot()["hash_verify"]["fields_skipped"] == 0
    assert len(set(digests["x"])) == 3


def test_hash_fingerprint_matches_across_jagged_and_dense():
    """``_from_wire`` densifies a jagged field whose rows are uniform, so a
    jagged put has to reconcile against a dense get. Zero padding is XOR-
    neutral, which is what makes the two layouts agree."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient(), verify_tensor_hash=True)
    ids = ["a", "b"]
    dense = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    jagged = torch.nested.nested_tensor(list(dense.unbind()), layout=torch.jagged)
    assert (
        client._row_fingerprints(TensorDict({"x": jagged}, batch_size=[2]), ids)["x"]
        == client._row_fingerprints(TensorDict({"x": dense}, batch_size=[2]), ids)["x"]
    )


def test_hash_fingerprint_separates_dtype():
    """``hash_tensor`` upcasts to 64 bits before reducing, so bf16 and fp32
    holding the same values reduce identically. The dtype salt is the only
    thing that makes a precision change visible."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient(), verify_tensor_hash=True)
    ids = ["a", "b"]
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    as_fp32 = client._row_fingerprints(TensorDict({"x": values}, batch_size=[2]), ids)[
        "x"
    ]
    as_bf16 = client._row_fingerprints(
        TensorDict({"x": values.to(torch.bfloat16)}, batch_size=[2]), ids
    )["x"]
    assert as_fp32 != as_bf16


def test_hash_verification_off_by_default():
    """Default construction must do no hashing work at all."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id="p", fields=["ids", "lp"], num_samples=4, consumer_tasks=["t"]
    )
    ids = [f"u{i}" for i in range(4)]
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids"])

    assert client.snapshot()["hash_verify"]["rows_recorded"] == 0
    assert client._hash_by_partition == {}
    assert "hash/mismatches" not in client.get_step_metrics(1.0)
    client.close()
