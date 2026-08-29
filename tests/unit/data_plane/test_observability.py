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

import logging
import random
from time import monotonic

import pytest
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.observability import (
    _QUANTILES,
    MetricsDataPlaneClient,
    _estimate_encoded_bytes,
    _hash_deltas,
    _td_bytes,
    breakdown_table,
    cluster_step_metrics,
    headline_series,
    merge_snapshots,
)

# ── helpers ────────────────────────────────────────────────────────────


def _ids(n, prefix="u"):
    return [f"{prefix}{i}" for i in range(n)]


def _client(inner=None, *, register=True, **kwargs):
    """A wrapped no-op client, with partition ``p`` registered by default.

    ``NoOpDataPlaneClient`` validates neither ``num_samples`` nor ``fields``,
    so one registration serves every caller. Pass ``register=False`` in tests
    that synthesise calls with :func:`_emit`: a real register call would add
    an op to ``by_op`` and skew the per-op shares under test.
    """
    client = MetricsDataPlaneClient(inner or NoOpDataPlaneClient(), **kwargs)
    if register:
        client.register_partition(
            partition_id="p",
            fields=["x", "ids", "lp"],
            num_samples=5_000,
            consumer_tasks=["t"],
        )
    return client


def _put(client, ids, width=4):
    """Put ``len(ids)`` rows of ``width`` float32 columns; return the bytes billed."""
    client.put_samples(
        sample_ids=list(ids),
        partition_id="p",
        fields=TensorDict({"x": torch.zeros(len(ids), width)}, batch_size=[len(ids)]),
    )
    return len(ids) * width * 4


def _emit(client, latencies_ms, op="put", n_bytes=1_000):
    """Record one synthetic ``op`` call per entry in ``latencies_ms``.

    ``n_bytes`` takes an int, or a callable of the call index.
    """
    now = monotonic()
    for i, ms in enumerate(latencies_ms):
        size = n_bytes(i) if callable(n_bytes) else n_bytes
        client._emit(op, "p", 1, size, now - ms / 1e3, "ok")
    return client


def _rank(latencies_ms, op="put", n_bytes=1_000):
    """One finished rank's snapshot: a call per entry, at that latency."""
    client = _client(register=False)
    try:
        return _emit(client, latencies_ms, op=op, n_bytes=n_bytes).snapshot()
    finally:
        client.close()


def _busy(op_calls):
    """A client that ran ``n`` calls of each named op, at ``ms`` each."""
    client = _client(register=False)
    for op, (n, ms) in op_calls.items():
        _emit(client, [ms] * n, op=op, n_bytes=lambda i: 1_000_000 * (1 + i % 5))
    return client


def _jagged(rows, field="x"):
    return TensorDict(
        {field: torch.nested.nested_tensor(rows, layout=torch.jagged)},
        batch_size=[len(rows)],
    )


def _hash_fields(n=4):
    return TensorDict(
        {
            "ids": torch.arange(n * 6, dtype=torch.int64).reshape(n, 6),
            "lp": torch.linspace(0, 1, n * 6, dtype=torch.bfloat16).reshape(n, 6),
        },
        batch_size=[n],
    )


def _jagged_ids(lengths, seed=0, with_dense=False):
    """Rows of pseudorandom token ids, optionally beside a uniform ``lp`` field.

    Deliberately not ``arange``: ``hash_tensor`` is an XOR reduction, and
    aligned runs of consecutive integers collide under it — ``XOR(6..11)``
    and ``XOR(12..17)`` are both 1 — which would make two visibly different
    rows fingerprint the same.
    """
    g = torch.Generator().manual_seed(seed)
    fields = {
        "ids": torch.nested.nested_tensor(
            [torch.randint(0, 32000, (n,), generator=g) for n in lengths],
            layout=torch.jagged,
        )
    }
    if with_dense:
        fields["lp"] = torch.zeros(len(lengths), 6)
    return TensorDict(fields, batch_size=[len(lengths)])


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


class _JaggedEcho(NoOpDataPlaneClient):
    """Returns whatever was put, jagged, so row lengths survive the trip."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def put_samples(self, sample_ids, partition_id, fields=None, tags=None):
        for key in fields.keys():
            v = fields.get(key)
            rows = v.unbind() if v.is_nested else list(v)
            for sid, row in zip(sample_ids, rows):
                self.rows.setdefault((partition_id, sid), {})[str(key)] = row.clone()
        return super().put_samples(
            sample_ids=sample_ids, partition_id=partition_id, fields=fields, tags=tags
        )

    def get_samples(self, sample_ids, partition_id, select_fields):
        out = {}
        for f in select_fields:
            rows = [self.rows[(partition_id, sid)][f] for sid in sample_ids]
            out[f] = (
                torch.stack(rows)
                if all(r.shape == rows[0].shape for r in rows[1:])
                else torch.nested.nested_tensor(rows, layout=torch.jagged)
            )
        return TensorDict(out, batch_size=[len(sample_ids)])


@pytest.fixture
def wrapped_client():
    """A registered client plus the list of events it emitted."""
    events: list[dict] = []
    client = _client(on_event=events.append)
    yield client, events
    client.close()


# ── the wrapper's event stream ─────────────────────────────────────────


def test_put_records_bytes_and_count(wrapped_client):
    client, events = wrapped_client
    client.put_samples(
        sample_ids=_ids(4, "a"),
        partition_id="p",
        fields=TensorDict({"x": torch.zeros(4, dtype=torch.float32)}, batch_size=[4]),
    )

    (e,) = [e for e in events if e["op"] == "put"]
    assert e["status"] == "ok"
    assert e["n_keys"] == 4
    assert e["n_bytes"] == 16  # 4 floats * 4 bytes
    assert e["wall_ms"] >= 0


def test_get_records_after_put(wrapped_client):
    client, events = wrapped_client
    client.put_samples(
        sample_ids=["a", "b"],
        partition_id="p",
        fields=TensorDict({"x": torch.ones(2)}, batch_size=[2]),
    )
    out = client.get_samples(
        sample_ids=["a", "b"], partition_id="p", select_fields=["x"]
    )

    assert torch.equal(out["x"], torch.ones(2))
    (e,) = [e for e in events if e["op"] == "get"]
    assert e["n_bytes"] > 0


def test_register_and_clear_recorded(wrapped_client):
    client, events = wrapped_client
    client.clear_samples(sample_ids=None, partition_id="p")

    assert [e["op"] for e in events] == ["register", "clear"]


def test_list_sample_ids_is_forwarded_and_recorded(wrapped_client):
    client, events = wrapped_client
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=2, consumer_tasks=["r"]
    )
    client.put_samples(
        sample_ids=["b", "a"],
        partition_id="p",
        fields=TensorDict({"x": torch.ones(2)}, batch_size=[2]),
    )

    assert client.list_sample_ids("p") == ["a", "b"]
    assert events[-1]["op"] == "list_sample_ids"
    assert events[-1]["status"] == "ok"


def test_error_status_recorded_and_reraised(wrapped_client):
    """The wrapper records the failure and re-raises rather than swallowing it."""
    client, events = wrapped_client
    with pytest.raises(KeyError):
        client.get_samples(sample_ids=["a"], partition_id="nope", select_fields=["x"])

    assert [(e["op"], e["status"]) for e in events if e["op"] == "get"] == [
        ("get", "error")
    ]


def test_no_callback_still_accumulates_stats():
    """``on_event=None`` skips building the event dict; the counters that
    ``snapshot()`` reports must not depend on a sink being registered."""
    client = _client()
    expected = _put(client, ["a", "b"], width=3)
    snap = client.snapshot()

    assert snap["total_bytes"] == expected
    assert snap["by_op"]["put"]["calls"] == 1
    assert snap["total_wall_ms"] > 0
    client.close()


def test_checkpoint_lifecycle_is_forwarded_and_recorded(tmp_path) -> None:
    checkpoint_dir = tmp_path / "data-plane"
    source_events: list[dict] = []
    source = MetricsDataPlaneClient(
        NoOpDataPlaneClient(),
        on_event=source_events.append,
    )
    source.register_partition(
        partition_id="p",
        fields=["x"],
        num_samples=1,
        consumer_tasks=["train"],
    )
    source.put_samples(
        sample_ids=["a"],
        partition_id="p",
        fields=TensorDict({"x": torch.tensor([1])}, batch_size=[1]),
    )
    source.save_checkpoint(checkpoint_dir, metadata={"step": 3})

    restore_events: list[dict] = []
    restored = MetricsDataPlaneClient(
        NoOpDataPlaneClient(),
        on_event=restore_events.append,
    )
    metadata = restored.load_checkpoint(checkpoint_dir)

    assert metadata == {"step": 3}
    assert [event["op"] for event in source_events][-1] == "save_checkpoint"
    assert source_events[-1]["status"] == "ok"
    assert [event["op"] for event in restore_events] == ["load_checkpoint"]
    assert restore_events[-1]["status"] == "ok"
    source.close()
    restored.close()


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
    inflate a 64-row put by 64x."""
    payload = {"tool": "bash", "text": "x" * 100}
    small = TensorDict({"m": NonTensorData(payload, batch_size=[2])}, batch_size=[2])
    large = TensorDict({"m": NonTensorData(payload, batch_size=[64])}, batch_size=[64])

    assert _td_bytes(small) == _td_bytes(large)
    assert _td_bytes(small) >= 100  # the string itself is still counted


def test_td_bytes_nontensorstack_scales_with_rows():
    """``NonTensorStack`` genuinely holds one object per row, so its estimate
    must scale — and stay close to the exact walk it extrapolates from."""
    row = {"turns": ["hello"] * 4, "n": 3}

    def stack_bytes(n):
        stack = NonTensorStack(*[NonTensorData(dict(row)) for _ in range(n)])
        return _td_bytes(TensorDict({"s": stack}, batch_size=[n]))

    bytes_8, bytes_64 = stack_bytes(8), stack_bytes(64)
    exact = sum(_estimate_encoded_bytes(dict(row), [10_000]) for _ in range(64))

    assert bytes_8 > 0
    assert bytes_64 == pytest.approx(8 * bytes_8, rel=0.05)
    assert bytes_64 == pytest.approx(exact, rel=0.05)


def test_estimate_encoded_bytes_walk_is_bounded():
    """The node budget caps the walk so one pathological payload cannot make
    a put O(payload size)."""
    huge = {"k": list(range(100_000))}

    assert _estimate_encoded_bytes(huge, [64]) < _estimate_encoded_bytes(
        huge, [10_000_000]
    )
    assert _estimate_encoded_bytes(huge, [64]) <= 4 * 64  # <=2 leaves/unit, <=2B each


def test_clear_frees_only_what_was_actually_live():
    """A clear may name uids already dropped, or belonging to another
    partition. Billing those releases bytes this partition never held:
    clearing 50 live keys alongside 50 unknown ones freed two thirds of a
    partition that had lost half its keys."""
    client = _client()
    ids = _ids(100)
    total = _put(client, ids, width=250)
    assert client.snapshot()["bytes_outstanding"] == total, "the put is billed in full"

    client.clear_samples(sample_ids=ids[:50] + _ids(50, "unknown"), partition_id="p")
    assert client.snapshot()["bytes_outstanding"] == total // 2

    client.clear_samples(sample_ids=ids[50:], partition_id="p")
    assert client.snapshot()["bytes_outstanding"] == 0
    client.close()


@pytest.mark.parametrize("seed", range(8))
def test_outstanding_reconciles_over_random_put_clear_sequences(seed):
    """Interleaved puts and partial clears must always land back at zero; the
    pro-rata release drops its division remainder, so only clearing the last
    live key can settle the account."""
    rng = random.Random(seed)
    client = _client()
    live: set[str] = set()
    for _ in range(rng.randint(1, 6)):
        batch = list(
            dict.fromkeys(f"k{rng.randint(0, 60)}" for _ in range(rng.randint(1, 20)))
        )
        _put(client, batch, width=250)
        live |= set(batch)
        if live and rng.random() < 0.5:
            drop = rng.sample(sorted(live), k=rng.randint(1, len(live)))
            client.clear_samples(sample_ids=drop, partition_id="p")
            live -= set(drop)
    if live:
        client.clear_samples(sample_ids=sorted(live), partition_id="p")

    assert client.snapshot()["bytes_outstanding"] == 0
    client.close()


# ── per-step series: units, windows, and the latency split ─────────────


def test_step_metrics_tail_is_exact_not_bucketed():
    """Per-step percentiles came off a histogram that is never reset, so they
    went flat and quantised to bucket edges. ``max_ms`` is exact, and one call
    supports no percentile at all — in either view."""
    client = _client(register=False)
    _emit(client, [30.0], n_bytes=8)
    metrics = client.get_step_metrics(1.0)

    assert "put/p90_ms" not in metrics and "step/by_op/put/p50_ms" not in metrics
    assert metrics["step/by_op/put/max_ms"] >= 30.0
    assert metrics["step/by_op/put/max_ms"] != pytest.approx(24.85, abs=0.5), (
        "bucket edge"
    )
    assert "p90_ms" not in client.snapshot()["by_op"]["put"]
    client.close()


def test_snapshot_leaves_the_step_window_alone_unless_asked():
    """``snapshot()`` is also how a human inspects a live client. Resetting the
    step window on every call would let an inspection blank the next step."""
    client = _client(register=False)
    _emit(client, [30.0])

    assert client.snapshot()["by_op"]["put"]["step_max_ms"] >= 30.0
    assert client.snapshot()["by_op"]["put"]["step_max_ms"] >= 30.0, "still there"
    assert client.snapshot(reset_step_window=True)["by_op"]["put"]["step_max_ms"] >= 30
    assert client.snapshot()["by_op"]["put"]["step_max_ms"] == 0.0, "window reopened"
    client.close()


def test_cluster_step_max_reopens_each_step():
    """A maximum cannot be differenced out of a cumulative counter, so the
    cluster path reported the lifetime max: after one 50 ms call every later
    step still read 50 ms. The reader resets the window as it reads."""
    client = _client(register=False)
    prev, seen = {}, []
    for slowest_ms in (5.0, 50.0, 5.0, 5.0):
        _emit(client, [slowest_ms, 5.0, 5.0, 5.0])
        merged = merge_snapshots([client.snapshot(reset_step_window=True)])
        seen.append(cluster_step_metrics(merged, prev, 1.0)["step/by_op/put/max_ms"])
        prev = merged

    assert seen[1] == pytest.approx(50.0, abs=1.0), "the spike shows"
    assert seen[2] == pytest.approx(5.0, abs=1.0), "and does not latch"
    client.close()


# ── percentiles: clamping and sample gates ─────────────────────────────


def test_cluster_percentiles_never_exceed_the_measured_max():
    """The same clamp reached through ``cluster_step_metrics``: 160 calls of
    120 ms all land in (100, 250] and interpolate to a p50 of 175 — above every
    call observed, and above the exact max reported beside it."""
    metrics = cluster_step_metrics(
        merge_snapshots([_rank([120.0] * 20) for _ in range(8)]), {}, 1.0
    )
    max_ms = metrics["step/by_op/put/max_ms"]

    assert max_ms == pytest.approx(120.0, abs=2.0)
    p50, p90 = metrics["step/by_op/put/p50_ms"], metrics["step/by_op/put/p90_ms"]
    assert p50 <= p90 <= max_ms


def test_each_quantile_waits_for_the_samples_it_needs():
    """Each quantile needs about four observations above its rank to mean
    anything, so they cannot share one gate: 48 calls carry a real median and
    no usable tail, and a single threshold for both reported neither. A
    percentile off a handful of calls is bucket geometry, not data."""

    def metrics_for(n_calls):
        client = _client(register=False)
        _emit(client, [5.0 + i % 7 for i in range(n_calls)])
        try:
            return cluster_step_metrics(merge_snapshots([client.snapshot()]), {}, 1.0)
        finally:
            client.close()

    thin = metrics_for(10)
    assert "step/by_op/put/p50_ms" not in thin, "too thin for either"
    assert "step/by_op/put/max_ms" in thin, "max always works"

    mid = metrics_for(30)
    assert "step/by_op/put/p50_ms" in mid, "a median off 30 calls is real"
    assert "step/by_op/put/p90_ms" not in mid, "a p90 off 30 calls is not"

    both = metrics_for(58)
    assert "step/by_op/put/p50_ms" in both and "step/by_op/put/p90_ms" in both


def test_no_quantile_finer_than_the_sample_size_can_resolve():
    """Guard the choice itself, not just the gate. The tail is p90 rather than
    p99 because a p99 off the ~58 calls a step holds equalled ``max_ms`` 80% of
    the time — the maximum twice, under a more precise-sounding name."""
    assert 0.99 not in {q for q, _, _ in _QUANTILES}
    for q, name, min_samples in _QUANTILES:
        above_the_rank = min_samples * (1 - q)
        assert above_the_rank >= 4 - 1e-9, (
            f"{name} is gated at {min_samples}, leaving only "
            f"{above_the_rank:.1f} observations above its rank"
        )


# ── cross-process aggregation ──────────────────────────────────────────


def test_merge_of_nothing_is_empty():
    assert merge_snapshots([]) == {}


def test_merge_sums_counters_and_rederives_percentiles():
    """The accumulators are shaped to add: histograms and regression sums from
    every rank combine into the true cluster distribution. Averaging per-rank
    percentiles could not, which is why latency lives in fixed buckets rather
    than retained samples."""
    snaps = [_rank([5.0] * 4) for _ in range(3)]
    merged = merge_snapshots(snaps)

    assert merged["n_processes"] == 3
    assert merged["by_op"]["put"]["calls"] == 12  # 3 ranks x 4 puts
    assert merged["by_op"]["put"]["n_bytes"] == 12_000
    assert merged["by_op"]["put"]["latency_hist"] == [
        sum(counts)
        for counts in zip(*(s["by_op"]["put"]["latency_hist"] for s in snaps))
    ]
    assert "p50_ms" not in merged["by_op"]["put"], "12 calls supports no percentile"


def test_merge_takes_max_for_max_fields():
    """A cluster's worst call is the worst any rank saw, not their sum."""
    merged = merge_snapshots([_rank([40.0]), _rank([1.0])])
    assert 40.0 <= merged["by_op"]["put"]["max_ms"] < 41.0, "max, not sum"


def test_cluster_frac_of_step_is_per_process_and_bounded():
    """``wall_ms`` sums processes that ran concurrently, so dividing it by one
    step's wall clock exceeded 1 whenever they overlapped and read as "105% of
    the step". Divided per process it is the mean share of the step a process
    spent in the data plane: 10 ranks x 5 calls x 100 ms over a 5 s step is
    500 ms each, or 10%."""
    metrics = cluster_step_metrics(
        merge_snapshots([_rank([100.0] * 5) for _ in range(10)]), {}, 5.0
    )

    assert "busy_frac_mean" not in metrics
    assert metrics["step/frac_of_step"] == pytest.approx(0.10, rel=0.1)
    assert metrics["now/n_processes"] == 10


def test_cluster_per_op_time_is_reported_per_call():
    """``wall_ms`` sums concurrent processes, so it scales with DP degree;
    dividing by the process count trades one arbitrary denominator for another.
    Per call is invariant to both DP degree and batch size, so it describes the
    wire rather than the shape of the run."""
    small = cluster_step_metrics(
        merge_snapshots([_rank([10.0] * 5) for _ in range(8)]), {}, 1.0
    )
    large = cluster_step_metrics(
        merge_snapshots([_rank([10.0] * 5) for _ in range(32)]), {}, 1.0
    )

    assert small["step/by_op/put/mean_ms"] == pytest.approx(10.0, rel=0.15)
    assert large["step/by_op/put/mean_ms"] == pytest.approx(
        small["step/by_op/put/mean_ms"], rel=0.15
    ), "mean must not move with cluster size"
    assert large["step/by_op/put/wall_ms"] == pytest.approx(
        4 * small["step/by_op/put/wall_ms"], rel=0.15
    ), "the sum does move with cluster size"

    columns, _ = breakdown_table(small)
    assert "mean_ms" in columns and "percent_of_dataplane" in columns


# ── what gets charted: shares, volume, and the breakdown table ─────────


def test_percent_of_dataplane_names_the_bottleneck_and_says_of_what():
    """The denominator is data-plane time, not the step: ``by_op`` sums to 100
    by construction, so the largest is the bottleneck *within the data plane*.
    Whether the data plane mattered at all is ``frac_of_step`` — here a tenth of
    a second of data-plane work inside a 10 s step is 9% of one, 100% of the
    other. 32 per-op line charts answer neither question."""
    client = _busy({"get": (100, 9.0), "put": (10, 1.0), "clear": (10, 0.1)})
    metrics = cluster_step_metrics(
        merge_snapshots([client.snapshot(reset_step_window=True)]), {}, 10.0
    )
    by_op = {
        k: v
        for k, v in metrics.items()
        if k.startswith("step/percent_of_dataplane/by_op/")
    }

    assert sum(by_op.values()) == pytest.approx(100.0), "percent of one total"
    assert max(by_op, key=by_op.__getitem__) == "step/percent_of_dataplane/by_op/get"
    assert by_op["step/percent_of_dataplane/by_op/get"] == pytest.approx(
        100 * 900 / 911, rel=0.05
    )
    # the two denominators are different questions and must not agree
    assert metrics["step/frac_of_step"] == pytest.approx(0.0911, rel=0.1)
    client.close()


def test_headline_drops_per_op_detail_but_keeps_the_percentages():
    """Four ops times eight fields is 32 series saying one thing. The detail is
    still computed — the breakdown table is built from the same dict, so the two
    cannot disagree — but only the totals and percentages are charted."""
    client = _busy({"get": (100, 9.0), "put": (10, 1.0), "clear": (10, 0.1)})
    metrics = cluster_step_metrics(
        merge_snapshots([client.snapshot(reset_step_window=True)]), {}, 1.0
    )
    head = headline_series(metrics)

    # the property, not a ratio: a ratio drifts as series are added on either
    # side, while "no per-op series is charted" is the thing being claimed
    assert len(head) < len(metrics), f"{len(head)} of {len(metrics)}"
    assert not [k for k in head if k.split("/")[1:2] in (["get"], ["put"], ["clear"])]
    assert "step/percent_of_dataplane/by_op/get" in head
    assert "step/wall_ms" in head and "step/frac_of_step" in head
    assert breakdown_table(metrics)[1], "the table still has rows"
    client.close()


def test_per_op_volume_replaces_the_written_read_split():
    """``comm_volume_mb`` alone hides which direction the traffic went — on a
    real step get moved 20.8 MB against put's 2.7 MB — while the old
    ``bytes_written``/``bytes_read`` pair was charted by nobody and absent from
    the table. One key per op says it finer, in both scopes."""
    client = _client(register=False)
    _emit(client, [5.0] * 6, op="get", n_bytes=3_000_000)
    _emit(client, [5.0] * 2, op="put", n_bytes=1_000_000)
    _emit(client, [1.0], op="clear", n_bytes=0)

    driver = client.get_step_metrics(1.0)
    assert "step/bytes_written_mb" not in driver
    assert "step/bytes_read_mb" not in driver
    assert driver["step/volume_mb/by_op/put"] == pytest.approx(2.0)

    head = headline_series(
        cluster_step_metrics(
            merge_snapshots([client.snapshot(reset_step_window=True)]), {}, 1.0
        )
    )
    assert head["step/volume_mb/by_op/get"] == pytest.approx(18.0)
    assert head["step/volume_mb/by_op/put"] == pytest.approx(2.0)
    assert "step/volume_mb/by_op/clear" not in head, "no payload, not a zero"
    per_op = sum(v for k, v in head.items() if k.startswith("step/volume_mb/"))
    assert per_op == pytest.approx(head["step/comm_volume_mb"]), "parts make the whole"
    client.close()


def test_breakdown_table_rows_by_op_worst_first():
    """One row per op, ordered by wall time, so the expensive op is the first
    line read rather than the alphabetically luckiest. Share of data-plane time
    is the second column for the same reason."""
    columns, rows = breakdown_table(
        {
            "step/wall_ms": 100.0,
            "step/percent_of_dataplane/by_op/get": 10.0,
            "step/percent_of_dataplane/by_op/put": 90.0,
            "step/by_op/get/calls": 8,
            "step/by_op/get/wall_ms": 10.0,
            "step/by_op/get/max_ms": 2.0,
            "step/by_op/put/calls": 2,
            "step/by_op/put/wall_ms": 90.0,
            "step/by_op/put/max_ms": 50.0,
            "step/comm_volume_mb": 1.0,  # not per-op, must not become a row
            "now/bytes_outstanding_mb": 0.0,  # a level, likewise
        }
    )

    assert columns[0] == "op"
    assert columns[1] == "percent_of_dataplane", "the bottleneck reads first"
    assert [r[0] for r in rows] == ["put", "get"], "worst first"
    assert rows[0][columns.index("wall_ms")] == 90.0
    assert rows[0][columns.index("percent_of_dataplane")] == pytest.approx(90.0)


def test_breakdown_table_leaves_withheld_series_empty():
    """A percentile below the sample gate is absent from the series — the table
    must carry None there rather than a zero that would read as a measurement."""
    columns, rows = breakdown_table(
        {
            "step/by_op/put/calls": 3,
            "step/by_op/put/wall_ms": 5.0,
            "step/by_op/put/max_ms": 2.0,
        }
    )

    assert rows[0][columns.index("p90_ms")] is None
    assert rows[0][columns.index("calls")] == 3


def test_breakdown_table_is_empty_when_nothing_ran():
    assert breakdown_table({"step/wall_ms": 0.0})[1] == []


def test_breakdown_table_ignores_reserved_namespaces():
    """``step/self/overhead_ms`` and ``step/volume_mb/by_op/get`` share the
    three-part shape of a per-op series; they must feed the right row (or none)
    rather than invent a "self" or "volume_mb" op beside put and get."""
    columns, rows = breakdown_table(
        {
            "step/by_op/get/calls": 3,
            "step/by_op/get/wall_ms": 9.0,
            "step/by_op/get/mb": 18.0,
            "step/volume_mb/by_op/get": 18.0,
            "step/self/overhead_ms": 6.2,
            "step/hash/mismatches": 0,
        }
    )

    assert columns[0] == "op"
    assert [r[0] for r in rows] == ["get"], rows
    assert rows[0][columns.index("mb")] == 18.0


# ── wire-in / wire-out hash verification ───────────────────────────────


def test_hash_state_and_counters_absent_when_the_guard_is_off():
    """Default construction does no hashing work and emits no hash series.

    Always-zero counters on every run that never asked for the guard would read
    as "checked, nothing wrong" rather than "not checked"."""
    client = _client()
    ids = _ids(4)
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids"])
    merged = merge_snapshots([client.snapshot(reset_step_window=True)])

    assert client.snapshot()["hash_verify"]["rows_recorded"] == 0
    assert client._hash_by_partition == {}
    assert not [k for k in client.get_step_metrics(1.0) if "hash" in k]
    assert not [k for k in cluster_step_metrics(merged, {}, 1.0) if "hash" in k]
    client.close()


def test_hash_verification_clean_roundtrip():
    client = _client(verify_tensor_hash=True)
    ids = _ids(4)
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids", "lp"])

    assert client.snapshot()["hash_verify"] == {
        "rows_recorded": 4,
        "rows_checked": 4,
        "rows_unverified": 0,
        "mismatches": 0,
        "fields_skipped": 0,
        "guard_failures": 0,  # the guard itself never raised
    }
    client.close()


def test_guard_failure_is_absorbed_counted_and_charted(caplog, monkeypatch):
    """A bug in the guard must not take the transfer down — and must not read
    as clean either.

    Both failures this check has produced were exactly this: an unhandled dtype
    inside ``_row_fingerprints`` propagating out of ``put_samples`` and killing
    the run. Absorbing them is only safe if the absorption is visible, so the
    count has to reach the series even though no rows were ever recorded.
    """
    client = _client(verify_tensor_hash=True)
    ids = _ids(4)

    def boom(*_args, **_kwargs):
        raise NotImplementedError("no hash_tensor kernel for this dtype")

    monkeypatch.setattr(client, "_row_fingerprints", boom)
    with caplog.at_level(logging.WARNING):
        # neither call may raise, on either side of the wire
        client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
        client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids"])

    hv = client.snapshot()["hash_verify"]
    assert hv["guard_failures"] == 2, "put and get each failed once"
    assert hv["rows_recorded"] == 0 and hv["rows_checked"] == 0, "nothing checked"
    assert caplog.text.count("hash guard failed") == 1, "logged once, not per call"
    # and it is a series, not just a counter -- the gate cannot key on rows
    assert client.get_step_metrics(1.0)["step/hash/guard_failures"] == 2
    client.close()


def test_hash_mismatch_reaches_every_scope():
    """A guard whose findings are not reported is not a guard.
    ``_log_data_plane_metrics`` prefers the cluster path whenever the fan-out
    reaches more than one process — every real run — and that path once emitted
    no hash counters at all."""
    client = _client(_CorruptingClient(field="ids", row=2), verify_tensor_hash=True)
    ids = _ids(4)
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids", "lp"])

    assert client.snapshot()["hash_verify"]["mismatches"] == 1
    assert client.get_step_metrics(1.0)["step/hash/mismatches"] == 1

    cluster = cluster_step_metrics(
        merge_snapshots([client.snapshot(reset_step_window=True)]), {}, 1.0
    )
    assert cluster["step/hash/mismatches"] == 1
    assert "step/hash/fields_skipped" in cluster, "abstentions visible too"
    assert headline_series(cluster)["step/hash/mismatches"] == 1, "and charted"
    client.close()


def test_hash_verification_survives_shard_readback():
    """A 4-row put read back two rows at a time must still line up: the
    fingerprint is per row, not per batch."""
    client = _client(verify_tensor_hash=True)
    ids = _ids(4)
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
    writer = _client(inner, verify_tensor_hash=True)
    ids = _ids(4)
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
    client = _client(verify_tensor_hash=True)
    ids = _ids(4)
    client.put_samples(sample_ids=ids, partition_id="p", fields=_hash_fields())

    assert client._hash_by_partition["p"]
    client.clear_samples(sample_ids=ids, partition_id="p")
    assert client._hash_by_partition == {}
    client.close()


def test_hash_fingerprint_covers_jagged_fields():
    """The per-token fields on this wire are jagged by the time they reach
    ``put_samples``. Skipping nested leaves would leave the entire bulk payload
    unguarded while still reporting zero mismatches — a guard that reads as
    clean because it checked nothing."""
    client = _client(verify_tensor_hash=True)
    rows = [torch.arange(n, dtype=torch.int64) + n for n in (3, 5, 4)]
    digest = client._row_fingerprints(_jagged(rows), ["a", "b", "c"])["x"]

    assert client.snapshot()["hash_verify"]["fields_skipped"] == 0
    assert digest.batch_scoped, "jagged digests only reconcile per batch"
    assert len(digest.per_row) == 3
    changed = list(rows)
    changed[1] = changed[1] + 1
    assert client._row_fingerprints(_jagged(changed), ["a", "b", "c"])["x"] != digest
    client.close()


def test_hash_fingerprint_matches_across_jagged_and_dense():
    """``_from_wire`` densifies a jagged field whose rows are uniform, so a
    jagged put has to reconcile against a dense get. Picking the scheme from the
    layout in hand rather than from what was recorded made every row of a
    uniform batch report a mismatch — 940 false alarms over the soak."""
    client = _client(verify_tensor_hash=True)
    ids = ["a", "b"]
    dense = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

    # uniform rows: both sides reduce per row, and a densified read agrees
    put_side = client._row_fingerprints(_jagged(list(dense.unbind())), ids)["x"]
    assert not put_side.batch_scoped
    assert put_side == client._row_fingerprints(
        TensorDict({"x": dense}, batch_size=[2]), ids
    )["x"]

    # ragged rows: batch-scoped, and the read replays that scheme rather than
    # choosing one from the dense tensor in hand
    ragged = _jagged([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    assert client._row_fingerprints(ragged, ids)["x"].batch_scoped
    assert client._row_fingerprints(
        TensorDict({"x": dense}, batch_size=[2]), ids, batch_scoped_fields={"x"}
    )["x"].batch_scoped
    client.close()


def test_hash_fingerprint_separates_dtype():
    """The values reduce identically once bitcast, so only the dtype salt makes
    a precision change visible."""
    client = _client(verify_tensor_hash=True)
    ids = ["a", "b"]
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    as_fp32 = client._row_fingerprints(TensorDict({"x": values}, batch_size=[2]), ids)
    as_bf16 = client._row_fingerprints(
        TensorDict({"x": values.to(torch.bfloat16)}, batch_size=[2]), ids
    )
    assert as_fp32["x"].per_row != as_bf16["x"].per_row
    client.close()


def test_hash_fingerprint_handles_float8():
    """``hash_tensor`` has no float8 kernel. Without the integer bitcast the
    ``NotImplementedError`` propagates out of ``put_samples`` and takes the
    transfer down with it."""
    client = _client(verify_tensor_hash=True)
    fp8 = TensorDict(
        {"x": torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)},
        batch_size=[2],
    )
    # through the public path, because that is where the exception surfaced
    client.put_samples(sample_ids=["a", "b"], partition_id="p", fields=fp8)

    assert client.snapshot()["hash_verify"]["rows_recorded"] == 2
    client.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hash_fingerprint_handles_device_tensors():
    """Torch has no ``bitwise_xor`` CUDA kernel for UInt64, so salting the digest
    tensor before it leaves the device raises ``NotImplementedError`` for any
    backend whose get returns device tensors — register mode under GDR does. The
    digests must also match the host's, or a device-resident get would verify
    against a host put as a mismatch on every row."""
    client = _client(verify_tensor_hash=True)
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ids = ["a", "b"]

    on_host = client._row_fingerprints(TensorDict({"x": values}, batch_size=[2]), ids)
    on_device = client._row_fingerprints(
        TensorDict({"x": values.cuda()}, batch_size=[2]), ids
    )
    assert on_device["x"].per_row == on_host["x"].per_row
    client.close()


# ── hash verification: what counts as a mismatch vs an abstention ──────


def test_hash_shard_of_a_ragged_field_is_skipped_not_a_mismatch():
    """The abstention that *is* legitimate: a batch-scoped digest covers the
    whole buffer it was reduced over, so a shard of it is genuinely
    incomparable. That must land in ``fields_skipped`` — visible, but not
    crying wolf on every sharded fetch."""
    client = _client(verify_tensor_hash=True)
    ids = _ids(4)
    rows = [torch.arange(3 + i, dtype=torch.int64) for i in range(4)]
    client.put_samples(sample_ids=ids, partition_id="p", fields=_jagged(rows))
    client.get_samples(sample_ids=ids[:2], partition_id="p", select_fields=["x"])

    hv = client.snapshot()["hash_verify"]
    assert hv["mismatches"] == 0
    assert hv["fields_skipped"] == 1
    assert client.get_step_metrics(1.0)["step/hash/fields_skipped"] == 1
    client.close()


def test_uniform_jagged_rows_are_fingerprinted_per_row():
    """A jagged field whose rows happen to be uniform is a rectangle already —
    its values buffer reshapes to one as a view — so it earns per-row
    attribution for free. The batch-scoped fallback is an XOR over one shared
    buffer, which cannot see a permutation: two equal-length rows swapped by a
    mis-shard would round-trip clean."""
    inner = _JaggedEcho()
    client = _client(inner, verify_tensor_hash=True)
    ids = _ids(4)
    client.put_samples(
        sample_ids=ids, partition_id="p", fields=_jagged_ids([6, 6, 6, 6])
    )
    a, b = inner.rows[("p", "u1")]["ids"], inner.rows[("p", "u2")]["ids"]
    inner.rows[("p", "u1")]["ids"], inner.rows[("p", "u2")]["ids"] = b, a
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids"])

    hv = client.snapshot()["hash_verify"]
    assert hv["mismatches"] == 2, "the two swapped rows, named individually"
    assert hv["fields_skipped"] == 0
    client.close()


def test_uniform_put_read_back_ragged_checks_row_lengths():
    """A row that changed length between wire-in and wire-out is a divergence.
    The content is no longer comparable, so the field counts as an abstention,
    but the lengths still are — and reporting the whole field as a skip would
    leave mismatches reading zero, exactly the shape of a guard that covers
    nothing. Failing the whole field instead reported 3584 mismatches per step
    on a healthy 5-process run."""
    inner = _JaggedEcho()
    client = _client(inner, verify_tensor_hash=True)
    ids = _ids(4)
    client.put_samples(
        sample_ids=ids, partition_id="p", fields=_jagged_ids([6, 6, 6, 6])
    )
    inner.rows[("p", "u2")]["ids"] = inner.rows[("p", "u2")]["ids"][:-2]
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids"])

    hv = client.snapshot()["hash_verify"]
    assert hv["mismatches"] > 0, "a truncated row must not read as clean"
    assert hv["fields_skipped"] == 1, "and the content it could not compare"
    client.close()


def test_uniform_write_read_back_inside_a_ragged_batch_is_clean():
    """The false positive this cost: a shard written with uniform rows, read
    back inside a batch whose *other* rows are ragged. Nothing diverged — every
    recorded row still has the length it was written with — and the guard
    reported 3584 mismatches per step on a healthy run until it compared lengths
    instead of failing the field outright."""
    inner = _JaggedEcho()
    client = _client(inner, verify_tensor_hash=True)
    ids = _ids(4)
    client.put_samples(
        sample_ids=ids, partition_id="p", fields=_jagged_ids([6, 6, 6, 6])
    )
    # a later writer adds rows of a different length to the same partition
    other = _ids(2, "v")
    inner.rows[("p", other[0])] = {"ids": torch.randint(0, 32000, (3,))}
    inner.rows[("p", other[1])] = {"ids": torch.randint(0, 32000, (9,))}
    client.get_samples(sample_ids=ids + other, partition_id="p", select_fields=["ids"])

    hv = client.snapshot()["hash_verify"]
    assert hv["mismatches"] == 0, "no row changed length; nothing diverged"
    assert hv["fields_skipped"] == 1, "content uncomparable, and counted"
    client.close()


def test_delta_put_does_not_restate_the_scheme_of_untouched_fields():
    """``write_columns`` puts one field into a partition written ragged earlier.
    Holding the jagged/per-row choice per *partition* let that delta hand the
    read side the wrong scheme for a field it never touched, and every row of
    that field came back a false alarm.

    The second field is the whole point: with only one, a per-partition and a
    per-field scheme are indistinguishable, because the only put there is
    restates its own field either way.
    """
    inner = _JaggedEcho()
    client = _client(inner, verify_tensor_hash=True)
    ids = _ids(4)
    client.put_samples(
        sample_ids=ids,
        partition_id="p",
        fields=_jagged_ids([2, 4, 6, 3], with_dense=True),
    )
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids"])
    assert client.snapshot()["hash_verify"]["mismatches"] == 0, "baseline"

    # the delta names only ``lp``; ``ids`` must keep the ragged scheme it was
    # written with, or its next read replays the uniform one and cries wolf
    client.put_samples(
        sample_ids=ids,
        partition_id="p",
        fields=TensorDict({"lp": torch.ones(4, 6)}, batch_size=[4]),
    )
    client.get_samples(sample_ids=ids, partition_id="p", select_fields=["ids"])

    hv = client.snapshot()["hash_verify"]
    assert hv["mismatches"] == 0, "a delta put must not restate ids's scheme"
    assert hv["fields_skipped"] == 0, "and the read stays comparable"
    client.close()


@pytest.mark.parametrize(
    "mismatches,warns",
    [
        # Every row of every field wrong, identically, every step is not what a
        # broken wire looks like -- it is what a broken guard looks like. Both
        # false alarms this check has produced had that shape.
        (300, True),
        # A handful of bad rows is exactly what the guard exists to report.
        (3, False),
    ],
)
def test_implausible_mismatch_rates_are_called_out(caplog, mismatches, warns):
    hv = {
        "rows_recorded": 100,
        "rows_checked": 100,
        "mismatches": mismatches,
        "rows_unverified": 0,
        "fields_skipped": 0,
        "guard_failures": 0,
    }
    with caplog.at_level(logging.WARNING):
        deltas = _hash_deltas(hv, {})

    assert deltas["step/hash/mismatches"] == mismatches
    assert ("more likely a bug in the check" in caplog.text) is warns


# ── call-site ordering ─────────────────────────────────────────────────


def test_codec_pack_unpack_time_is_reported_separately():
    """Jagged pad/unpad is real CPU cost the per-op metrics cannot see.

    ``pack_jagged_fields`` runs in the caller before ``put_samples`` is
    entered, so it never reaches ``by_op``; ``_from_wire`` runs inside the
    adapter's ``get_samples``, where it would otherwise be billed as
    transport. Both are drained from the codec timer into their own series,
    deliberately outside ``total_wall_ms`` so ``frac_of_step`` keeps meaning
    time spent in the data plane rather than time spent on CPU around it.
    """
    from nemo_rl.data_plane import codec

    client = _client(register=False)
    codec.record_codec_s("pack", 0.010)  # 10 ms
    codec.record_codec_s("unpack", 0.004)  # 4 ms

    metrics = client.get_step_metrics(1.0)
    assert metrics["step/codec/pack_ms"] == pytest.approx(10.0, rel=1e-3)
    assert metrics["step/codec/unpack_ms"] == pytest.approx(4.0, rel=1e-3)
    # not folded into the transport totals
    assert metrics["step/wall_ms"] == 0.0
    # and charted, not just tabulated
    assert "step/codec/pack_ms" in headline_series(metrics)

    # drained exactly once: a second step reports zero, not the same 10 ms
    assert client.get_step_metrics(1.0)["step/codec/pack_ms"] == 0.0
    client.close()


def test_inspection_snapshot_does_not_steal_codec_time():
    """The drain is destructive, so only the reader that closes the step window
    may take it. ``snapshot()`` is also how a human inspects a live client, and
    an unguarded drain there would delete the time the step reader is about to
    report -- the same hazard ``step_max_ms`` is gated for."""
    from nemo_rl.data_plane import codec

    client = _client(register=False)
    codec.record_codec_s("pack", 0.010)

    client.snapshot()  # inspection: must not consume it
    assert client.get_step_metrics(1.0)["step/codec/pack_ms"] == pytest.approx(
        10.0, rel=1e-3
    )
    client.close()
