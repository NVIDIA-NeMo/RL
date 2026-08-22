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

from time import monotonic

import pytest
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.observability import (
    MetricsDataPlaneClient,
    cluster_step_metrics,
    merge_snapshots,
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


def _jagged(rows):
    return TensorDict(
        {"x": torch.nested.nested_tensor(rows, layout=torch.jagged)},
        batch_size=[len(rows)],
    )


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


def test_td_bytes_jagged_matches_the_public_count():
    """The nested fast path reads the packed values buffer instead of the
    tensor's own (dispatched, ~16us) ``nbytes``. It must agree exactly."""
    rows = [torch.arange(n, dtype=torch.int64) for n in (3, 7, 2, 5)]
    td = _jagged(rows)
    assert _td_bytes(td) == td["x"].nbytes == sum(r.nbytes for r in rows)


def test_td_bytes_does_not_overcount_a_narrow_view():
    """``torch.nested.narrow`` yields a tensor whose values buffer views a
    larger allocation; trusting it would silently inflate ``n_bytes``."""
    lengths = torch.tensor([3, 4, 5, 6])
    narrowed = torch.nested.narrow(
        torch.zeros(4, 10),
        1,
        torch.zeros(4, dtype=torch.int64),
        lengths,
        layout=torch.jagged,
    )
    assert narrowed._values.nbytes > narrowed.nbytes, "fixture must be a view"
    td = TensorDict({"x": narrowed}, batch_size=[4])
    assert _td_bytes(td) == narrowed.nbytes == int(lengths.sum()) * 4


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


def test_step_metrics_use_one_unit_per_dimension():
    """Every duration is ms, every volume MB. A chart mixing `wall_s` with
    `p99_ms` shows 0.008 beside 24.85 and reads as a data-plane bug rather
    than an axis one."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=2, consumer_tasks=["t"]
    )
    client.put_samples(
        sample_ids=["a", "b"],
        partition_id="p",
        fields=TensorDict({"x": torch.zeros(2, 3)}, batch_size=[2]),
    )
    keys = set(client.get_step_metrics(1.0))
    assert not [k for k in keys if k.endswith(("_s", "_gb", "_kb", "_us"))], keys
    assert "step/wall_ms" in keys and "step/comm_volume_mb" in keys
    client.close()


def test_step_metrics_tail_is_exact_not_bucketed():
    """Per-step percentiles came off a histogram that is never reset, so
    they went flat and quantised to bucket edges (p99 of a single sample in
    (10, 25] is always 10 + 15*0.99 = 24.85). ``max_ms`` is exact and
    tracks the slowest call actually seen."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["t"]
    )
    client._emit("put", "p", 1, 8, monotonic() - 0.030, "ok")  # a 30 ms call
    metrics = client.get_step_metrics(1.0)

    assert "put/p99_ms" not in metrics and "step/put/p50_ms" not in metrics
    assert metrics["step/put/max_ms"] >= 30.0
    assert metrics["step/put/max_ms"] != pytest.approx(24.85, abs=0.5), "bucket edge"
    # the cumulative view still carries percentiles for one-off inspection
    assert "p99_ms" in client.snapshot()["by_op"]["put"]
    client.close()


def test_latency_breakdown_stacks_to_wall_ms():
    """The fit is reported as two ms components rather than a ratio, so a
    chart can stack them against the measured ``wall_ms``.

    A ratio would have been both flat (the fit is cumulative) and unitless
    on an axis of milliseconds. These carry the coefficients from the
    cumulative fit but attribute them to *this* step's calls and bytes.
    """
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    fixed_ms, mb_per_s = 8.0, 500.0
    now = monotonic()
    for i in range(12):  # varied sizes, or the fit is unidentifiable
        n_bytes = 50_000 * (i + 1)
        wall_ms = fixed_ms + n_bytes / (mb_per_s * 1e3)
        client._emit("put", "p", 1, n_bytes, now - wall_ms / 1e3, "ok")

    metrics = client.get_step_metrics(1.0)
    fit = client.snapshot()["by_op"]["put"]["fit"]
    assert fit["model_trustworthy"], fit
    assert fit["fixed_ms"] == pytest.approx(fixed_ms, rel=0.05)
    assert fit["bandwidth_mb_s"] == pytest.approx(mb_per_s, rel=0.05)

    # the two components are the split, in ms, and they add up
    total = metrics["step/put/overhead_ms"] + metrics["step/put/transfer_ms"]
    assert total == pytest.approx(metrics["step/put/wall_ms"], rel=0.05)
    assert "step/put/overhead_frac" not in metrics, "a ratio is derivable from these"
    client.close()


def test_step_max_is_scoped_to_the_step():
    """A lifetime max is monotonic and goes flat the moment the worst call
    has been seen — the same defect as logging a cumulative percentile. The
    reported max must fall again when a step is quicker."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client._emit("put", "p", 1, 8, monotonic() - 0.050, "ok")  # slow step
    slow = client.get_step_metrics(1.0)["step/put/max_ms"]
    client._emit("put", "p", 1, 8, monotonic() - 0.001, "ok")  # quick step
    quick = client.get_step_metrics(1.0)["step/put/max_ms"]

    assert slow >= 50.0
    assert quick < slow, "step max must reset, not carry the lifetime worst"
    # the lifetime worst is still available for a one-off look
    assert client.snapshot()["by_op"]["put"]["max_ms"] >= 50.0
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
    assert client.get_step_metrics(1.0)["step/hash/mismatches"] == 1
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
    rows = [torch.arange(n, dtype=torch.int64) + n for n in (3, 5, 4)]
    digest = client._row_fingerprints(_jagged(rows), ["a", "b", "c"])["x"]

    assert client.snapshot()["hash_verify"]["fields_skipped"] == 0
    assert digest.batch_scoped, "jagged digests only reconcile per batch"
    assert len(digest.per_row) == 3
    # A jagged digest carries the row length, so a change in any row's
    # payload moves every row's value and a length change moves one.
    changed = list(rows)
    changed[1] = changed[1] + 1
    assert client._row_fingerprints(_jagged(changed), ["a", "b", "c"])["x"] != digest


def test_hash_fingerprint_matches_across_jagged_and_dense():
    """``_from_wire`` densifies a jagged field whose rows are uniform, so a
    jagged put has to reconcile against a dense get.

    Regression guard: picking the scheme from the layout in hand rather than
    from what was recorded made every row of a uniform batch report a
    mismatch — 940 false alarms over the verification soak.
    """
    client = MetricsDataPlaneClient(NoOpDataPlaneClient(), verify_tensor_hash=True)
    ids = ["a", "b"]
    dense = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    jagged = _jagged(list(dense.unbind()))

    put_side = client._row_fingerprints(jagged, ids)["x"]
    get_side = client._row_fingerprints(
        TensorDict({"x": dense}, batch_size=[2]), ids, batch_scoped_fields={"x"}
    )["x"]
    assert put_side == get_side


def test_hash_shard_read_of_jagged_field_is_unverified_not_a_mismatch():
    """A batch-scoped digest covers the whole buffer, so a shard read cannot
    reproduce it. That has to report as unverified — reporting it as a
    mismatch would make the guard cry wolf on every sharded fetch."""
    client = _hash_client()
    ids = [f"u{i}" for i in range(4)]
    rows = [torch.arange(3, dtype=torch.int64) + i for i in range(4)]
    client.put_samples(sample_ids=ids, partition_id="p", fields=_jagged(rows))
    client.get_samples(sample_ids=ids[:2], partition_id="p", select_fields=["x"])

    assert client.snapshot()["hash_verify"]["mismatches"] == 0
    client.close()


def test_hash_incomparable_field_is_counted_not_dropped():
    """A rectangular put can come back jagged — truncating one row makes the
    batch ragged. Its per-row digests are not comparable against a
    batch-scoped read, so the field is dropped; dropping it *silently* is
    the exact shape of the bug that let this check pass while covering
    nothing, so the drop has to land in ``fields_skipped``."""
    client = _hash_client(_RaggedOnReadClient())
    ids = [f"u{i}" for i in range(4)]
    dense = TensorDict(
        {"x": torch.arange(16, dtype=torch.int64).reshape(4, 4)}, batch_size=[4]
    )
    client.put_samples(sample_ids=ids, partition_id="p", fields=dense)
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["x"])

    hv = client.snapshot()["hash_verify"]
    assert hv["mismatches"] == 0, "must not cry wolf on an incomparable field"
    assert hv["fields_skipped"] == 1, "the drop must be visible"
    assert client.get_step_metrics(1.0)["step/hash/fields_skipped"] == 1


class _RaggedOnReadClient(NoOpDataPlaneClient):
    """Returns one row shorter than it was written — a truncation on the wire."""

    def get_samples(self, sample_ids, partition_id, select_fields):
        out = super().get_samples(sample_ids, partition_id, select_fields)
        rows = list(out["x"].unbind())
        rows[1] = rows[1][:-1]
        return TensorDict(
            {"x": torch.nested.nested_tensor(rows, layout=torch.jagged)},
            batch_size=[len(sample_ids)],
        )


def test_hash_fingerprint_separates_dtype():
    """The values reduce identically once bitcast, so only the dtype salt
    makes a precision change visible."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient(), verify_tensor_hash=True)
    ids = ["a", "b"]
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    as_fp32 = client._row_fingerprints(TensorDict({"x": values}, batch_size=[2]), ids)[
        "x"
    ]
    as_bf16 = client._row_fingerprints(
        TensorDict({"x": values.to(torch.bfloat16)}, batch_size=[2]), ids
    )["x"]
    assert as_fp32.per_row != as_bf16.per_row


def test_hash_fingerprint_handles_float8():
    """``hash_tensor`` has no float8 kernel. Without the integer bitcast the
    ``NotImplementedError`` propagates out of ``put_samples`` and takes the
    transfer down with it."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient(), verify_tensor_hash=True)
    fp8 = TensorDict(
        {"x": torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)},
        batch_size=[2],
    )
    digest = client._row_fingerprints(fp8, ["a", "b"])["x"]
    assert len(digest.per_row) == 2


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
    assert "step/hash/mismatches" not in client.get_step_metrics(1.0)
    client.close()


# ── cross-process aggregation ──────────────────────────────────────────


def _rank_client(latencies_ms):
    """A client that has seen one put per entry, at that latency."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    now = monotonic()
    for ms in latencies_ms:
        client._emit("put", "p", 1, 1_000, now - ms / 1e3, "ok")
    return client


def _rank_with(latencies_ms):
    """Its snapshot, for the merge tests."""
    return _rank_client(latencies_ms).snapshot()


def test_merge_sums_counters_and_rederives_percentiles():
    """The accumulators are shaped to add: histograms and regression sums
    from every rank combine into the true cluster distribution. Averaging
    per-rank percentiles could not do this, which is the whole reason the
    latency lives in fixed buckets rather than retained samples."""
    ranks = [_rank_client([5.0] * 4) for _ in range(3)]
    merged = merge_snapshots([c.snapshot() for c in ranks])

    assert merged["n_processes"] == 3
    assert merged["by_op"]["put"]["calls"] == 12  # 3 ranks x 4 puts
    assert merged["by_op"]["put"]["n_bytes"] == 12_000
    assert sum(merged["by_op"]["put"]["latency_hist"]) == 12
    # derived from the summed histogram, not averaged from the ranks
    single = ranks[0].snapshot()["by_op"]["put"]
    assert merged["by_op"]["put"]["p50_ms"] == pytest.approx(single["p50_ms"], rel=0.3)
    for c in ranks:
        c.close()


def test_merge_takes_max_for_max_fields():
    """A cluster's worst call is the worst any rank saw, not their sum."""
    slow = _rank_client([40.0])
    fast = _rank_client([1.0])
    merged = merge_snapshots([slow.snapshot(), fast.snapshot()])
    assert merged["by_op"]["put"]["max_ms"] >= 40.0
    assert merged["by_op"]["put"]["max_ms"] < 41.0, "max, not sum"
    slow.close()
    fast.close()


def test_merge_of_nothing_is_empty():
    assert merge_snapshots([]) == {}


def test_cluster_step_metrics_report_their_own_cost():
    """``observability_overhead_ms`` is the wrapper's own wall time minus
    the inner client's, summed over processes — the bill for measuring,
    sitting beside what it bought."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=4, consumer_tasks=["t"]
    )
    for i in range(4):
        client.put_samples(
            sample_ids=[f"u{i}"],
            partition_id="p",
            fields=TensorDict({"x": torch.zeros(1, 512)}, batch_size=[1]),
        )
    merged = merge_snapshots([client.snapshot()])
    metrics = cluster_step_metrics(merged, {}, 1.0)

    assert metrics["now/n_processes"] == 1
    assert metrics["step/observability_overhead_ms"] > 0, "measuring is never free"
    # Deliberately not clamped to 1. Against this no-op inner client the RPC
    # is instant, so measuring costs more than the thing measured and the
    # ratio exceeds 100% -- which is exactly the signal worth surfacing.
    # Against a real backend it lands near 0.01.
    assert metrics["step/observability_overhead_frac"] > 0
    assert "step/wall_ms" in metrics and "step/comm_volume_mb" in metrics
    client.close()


def test_cluster_time_is_per_process_not_a_bare_fraction():
    """``wall_ms`` sums processes that ran concurrently, so dividing it by
    one step's wall clock exceeded 1 whenever they overlapped and read as
    "105% of the step". Reported per process it is a duration in ms, like
    everything else, and needs no gloss."""
    ranks = [_rank_with([100.0] * 5) for _ in range(10)]
    metrics = cluster_step_metrics(merge_snapshots(ranks), {}, 1.0)

    assert "busy_frac_mean" not in metrics and "frac_of_step" not in metrics
    assert metrics["step/wall_ms_per_process"] == pytest.approx(500.0, rel=0.1)
    assert metrics["now/n_processes"] == 10


def test_cluster_overhead_includes_the_collection_fan_out():
    """The fan-out is the larger half of the bill. Reporting only the per-op
    wrapper understated the real cost by ~19x in the cross-process e2e
    (0.13 ms reported against 2.44 ms actually spent)."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["t"]
    )
    client.put_samples(
        sample_ids=["a"],
        partition_id="p",
        fields=TensorDict({"x": torch.zeros(1, 8)}, batch_size=[1]),
    )
    merged = merge_snapshots([client.snapshot()])
    without = cluster_step_metrics(merged, {}, 1.0)
    with_gather = cluster_step_metrics(merged, {}, 1.0, collect_ms=2.31)

    delta = (
        with_gather["step/observability_overhead_ms"]
        - without["step/observability_overhead_ms"]
    )
    assert delta == pytest.approx(2.31, rel=1e-6)
    client.close()


def test_cluster_percentiles_never_exceed_the_measured_max():
    """Bucket interpolation spreads a bucket's samples uniformly across it,
    so calls clustered low in a wide bucket read high: 160 calls of 120 ms
    all land in (100, 250] and interpolate to a p50 of 175 — above every
    call observed, and above the exact max reported beside it. The max is
    the tighter bound, so the percentiles are clamped to it."""
    merged = merge_snapshots([_rank_with([120.0] * 20) for _ in range(8)])
    metrics = cluster_step_metrics(merged, {}, 1.0)

    assert metrics["step/put/max_ms"] == pytest.approx(120.0, abs=2.0)
    assert metrics["step/put/p50_ms"] <= metrics["step/put/max_ms"]
    assert metrics["step/put/p99_ms"] <= metrics["step/put/max_ms"]
    assert metrics["step/put/p50_ms"] <= metrics["step/put/p99_ms"]


def test_cluster_percentiles_withheld_below_a_useful_sample_count():
    """A percentile off a handful of calls is bucket geometry, not data.
    Silence beats a number that looks like an answer."""
    few = merge_snapshots([_rank_with([120.0] * 3) for _ in range(2)])  # 6 calls
    many = merge_snapshots([_rank_with([120.0] * 20) for _ in range(8)])  # 160

    assert "step/put/p50_ms" not in cluster_step_metrics(few, {}, 1.0)
    assert "step/put/max_ms" in cluster_step_metrics(few, {}, 1.0), "max always works"
    assert "step/put/p50_ms" in cluster_step_metrics(many, {}, 1.0)


def test_cluster_series_declare_delta_or_level():
    """A per-step delta and an instantaneous level shared the ``_mb`` suffix
    and a chart, with nothing to tell them apart. Every series now sits
    under ``step/`` or ``now/`` so the kind is on the axis label."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["t"]
    )
    client.put_samples(
        sample_ids=["a"],
        partition_id="p",
        fields=TensorDict({"x": torch.zeros(1, 8)}, batch_size=[1]),
    )
    metrics = cluster_step_metrics(merge_snapshots([client.snapshot()]), {}, 1.0)

    unlabelled = [k for k in metrics if not k.startswith(("step/", "now/"))]
    assert not unlabelled, unlabelled
    assert "now/bytes_outstanding_mb" in metrics, "a level"
    assert "step/comm_volume_mb" in metrics, "a delta"
    client.close()


def test_clear_frees_only_what_was_actually_live():
    """A clear may name uids already dropped, or belonging to another
    partition. Billing those releases bytes this partition never held:
    clearing 50 live keys alongside 50 unknown ones freed two thirds of a
    partition that had lost half its keys."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    ids = [f"u{i}" for i in range(100)]
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=100, consumer_tasks=["t"]
    )
    client.put_samples(
        sample_ids=ids,
        partition_id="p",
        fields=TensorDict({"x": torch.zeros(100, 250)}, batch_size=[100]),
    )
    total = client.snapshot()["bytes_outstanding"]

    client.clear_samples(
        sample_ids=ids[:50] + [f"unknown{i}" for i in range(50)], partition_id="p"
    )
    assert client.snapshot()["bytes_outstanding"] == total // 2

    client.clear_samples(sample_ids=ids[50:], partition_id="p")
    assert client.snapshot()["bytes_outstanding"] == 0
    client.close()


def test_outstanding_reconciles_over_random_put_clear_sequences():
    """Interleaved puts and partial clears must always land back at zero;
    the pro-rata release is only sound if it does."""
    import random

    rng = random.Random(0)
    for _ in range(50):
        client = MetricsDataPlaneClient(NoOpDataPlaneClient())
        client.register_partition(
            partition_id="p", fields=["x"], num_samples=5000, consumer_tasks=["t"]
        )
        live: set[str] = set()
        for _ in range(rng.randint(1, 6)):
            batch = list(
                dict.fromkeys(
                    f"k{rng.randint(0, 60)}" for _ in range(rng.randint(1, 20))
                )
            )
            client.put_samples(
                sample_ids=batch,
                partition_id="p",
                fields=TensorDict(
                    {"x": torch.zeros(len(batch), 250)}, batch_size=[len(batch)]
                ),
            )
            live |= set(batch)
            if live and rng.random() < 0.5:
                drop = rng.sample(sorted(live), k=rng.randint(1, len(live)))
                client.clear_samples(sample_ids=drop, partition_id="p")
                live -= set(drop)
        if live:
            client.clear_samples(sample_ids=sorted(live), partition_id="p")
        assert client.snapshot()["bytes_outstanding"] == 0
        client.close()


def test_snapshot_percentiles_never_exceed_the_measured_max():
    """The clamp lives in ``_derive_op_metrics``, not at one call site, so
    ``snapshot()`` and ``merge_snapshots()`` inherit it. Without it five
    register calls of 0.011 ms all land in (0, 0.1] and interpolate to a
    p50 of 0.05 — four times the slowest call that happened — on a public
    surface documented as the cumulative view."""
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    now = monotonic()
    for _ in range(5):
        client._emit("register", "p", 1, 0, now - 0.011 / 1e3, "ok")

    for view in (client.snapshot(), merge_snapshots([client.snapshot()])):
        stats = view["by_op"]["register"]
        assert stats["p50_ms"] <= stats["max_ms"], stats
        assert stats["p99_ms"] <= stats["max_ms"], stats
        assert stats["p50_ms"] <= stats["p99_ms"], stats
    client.close()
