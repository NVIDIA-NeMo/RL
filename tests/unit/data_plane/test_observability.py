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

from contextlib import contextmanager

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.observability import MetricsDataPlaneClient

from ._rollout_shapes import make_rollout_batch

try:
    from nemo.lens import NemoLensConfig, setup_telemetry
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _HAS_LENS = True
except ImportError:
    _HAS_LENS = False

requires_lens = pytest.mark.skipif(
    not _HAS_LENS, reason="nemo-lens (+ opentelemetry sdk) not installed"
)


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


# --------------------------------------------------------------------------- #
# OTel spans for data-plane operations                                        #
# --------------------------------------------------------------------------- #
@contextmanager
def _exporting(span_groups: str):
    """Exporting telemetry with *span_groups* enabled.

    Yields a callable rather than the exporter because the span processor
    batches: without a force-flush the exporter is still empty at the point a
    test looks at it.
    """
    import nemo_rl.telemetry.setup as setup_mod

    # Importing span_groups is what registers the names ``span_groups`` resolves
    # against; lens ships none of its own.
    import nemo_rl.telemetry.span_groups  # noqa: F401

    exporter = InMemorySpanExporter()
    cfg = NemoLensConfig(enabled=True, span_groups=span_groups)
    handle = setup_telemetry(cfg, span_exporter=exporter)
    setup_mod._TELEMETRY_HANDLE = handle

    def _spans():
        from opentelemetry import trace

        trace.get_tracer_provider().force_flush()
        return exporter.get_finished_spans()

    try:
        yield _spans
    finally:
        handle.shutdown()
        setup_mod._TELEMETRY_HANDLE = None
        _reset_otel_globals()


@pytest.fixture
def finished_spans():
    """Only ``data_plane`` on: the batch-stage case."""
    with _exporting("data_plane") as spans:
        yield spans


@pytest.fixture
def finished_spans_with_per_prompt():
    """Both groups on, so a test can see which one a rollout op picks."""
    with _exporting("data_plane,per_prompt") as spans:
        yield spans


def _reset_otel_globals() -> None:
    """Drop the process-global providers this fixture installed.

    The OTel API only lets a provider be set once per process, so leaving ours
    in place would silently disable telemetry for every later test in the run.
    """
    import nemo.lens.handle as handle_mod
    import opentelemetry.metrics._internal as metrics_mod
    import opentelemetry.trace as trace_mod
    from nemo.lens.state import set_enabled_span_groups
    from opentelemetry.util._once import Once

    trace_mod._TRACER_PROVIDER = None
    trace_mod._TRACER_PROVIDER_SET_ONCE = Once()
    metrics_mod._METER_PROVIDER = None
    metrics_mod._METER_PROVIDER_SET_ONCE = Once()
    handle_mod._INITIALIZED = False
    set_enabled_span_groups(frozenset())


def _put_one(client) -> None:
    client.register_partition(
        partition_id="train", fields=["x"], num_samples=2, consumer_tasks=["read"]
    )
    client.put_samples(
        sample_ids=["a", "b"],
        partition_id="train",
        fields=TensorDict({"x": torch.zeros(2, dtype=torch.float32)}, batch_size=[2]),
    )


@requires_lens
def test_each_data_plane_op_emits_a_span(finished_spans):
    """Transfer-queue time was invisible in the waterfall before this.

    A step that stalls on a ``get`` looked like a gap between phases, with
    nothing to say whether the queue or the producer was responsible.
    """
    from nemo_rl.telemetry.instrumentation import RL_BUCKET_ATTR

    inner = NoOpDataPlaneClient()
    client = MetricsDataPlaneClient(inner, on_event=None)
    _put_one(client)
    client.close()
    inner.close()

    spans = finished_spans()
    names = [s.name for s in spans]
    assert "rl.data_plane.put" in names

    put = next(s for s in spans if s.name == "rl.data_plane.put")
    assert put.attributes["rl.data_plane.op"] == "put"
    assert put.attributes["rl.data_plane.partition"] == "train"
    assert put.attributes["rl.data_plane.keys"] == 2
    assert put.attributes["rl.data_plane.bytes"] > 0
    assert put.attributes["rl.data_plane.status"] == "ok"
    # Moving bytes between fleets is plumbing, not training progress, so it
    # must not be credited to goodput.
    assert put.attributes[RL_BUCKET_ATTR] == "overhead"


@requires_lens
def test_a_rollout_put_is_gated_apart_from_a_batch_put(finished_spans):
    """The same op, from the per-prompt path, is not a ``data_plane`` span.

    One client serves both paths, so with only ``data_plane`` on, the rollout's
    put has to disappear while the batch stages' puts keep coming -- that is
    the whole point of the split, since it is the per-prompt one whose count
    scales with the dataset.
    """
    from nemo_rl.telemetry.instrumentation import per_prompt_scope

    inner = NoOpDataPlaneClient()
    client = MetricsDataPlaneClient(inner, on_event=None)
    client.register_partition(
        partition_id="train", fields=["x"], num_samples=2, consumer_tasks=["read"]
    )
    with per_prompt_scope():
        client.put_samples(
            sample_ids=["a"],
            partition_id="train",
            fields=TensorDict(
                {"x": torch.zeros(1, dtype=torch.float32)}, batch_size=[1]
            ),
        )
    client.close()
    inner.close()

    names = [s.name for s in finished_spans()]
    assert "rl.data_plane.put" not in names
    # The batch-shaped ops around it are unaffected.
    assert "rl.data_plane.register" in names


@requires_lens
def test_a_rollout_put_carries_no_bucket(finished_spans_with_per_prompt):
    """A rollout's put overlaps training and every other in-flight rollout.

    Bucketing it ``overhead`` the way a batch put is would sum past the wall
    clock it happened in, by up to ``max_inflight_prompts``.
    """
    from nemo_rl.telemetry.instrumentation import RL_BUCKET_ATTR, per_prompt_scope

    inner = NoOpDataPlaneClient()
    client = MetricsDataPlaneClient(inner, on_event=None)
    client.register_partition(
        partition_id="train", fields=["x"], num_samples=1, consumer_tasks=["read"]
    )
    with per_prompt_scope():
        client.put_samples(
            sample_ids=["a"],
            partition_id="train",
            fields=TensorDict(
                {"x": torch.zeros(1, dtype=torch.float32)}, batch_size=[1]
            ),
        )
    client.close()
    inner.close()

    spans = finished_spans_with_per_prompt()
    put = next(s for s in spans if s.name == "rl.data_plane.put")
    assert RL_BUCKET_ATTR not in put.attributes
    # The op attributes are still recorded; only the bucket differs.
    assert put.attributes["rl.data_plane.op"] == "put"
    # A batch-stage op in the same run keeps its bucket.
    register = next(s for s in spans if s.name == "rl.data_plane.register")
    assert register.attributes[RL_BUCKET_ATTR] == "overhead"


@requires_lens
def test_leaving_the_scope_restores_the_batch_grouping(finished_spans):
    """The scope is a region, not a mode: a later batch put must not inherit it."""
    from nemo_rl.telemetry.instrumentation import per_prompt_scope

    inner = NoOpDataPlaneClient()
    client = MetricsDataPlaneClient(inner, on_event=None)
    with per_prompt_scope():
        client.register_partition(
            partition_id="rollout", fields=["x"], num_samples=1, consumer_tasks=["r"]
        )
    client.register_partition(
        partition_id="batch", fields=["x"], num_samples=1, consumer_tasks=["r"]
    )
    client.close()
    inner.close()

    partitions = [
        s.attributes["rl.data_plane.partition"]
        for s in finished_spans()
        if s.name == "rl.data_plane.register"
    ]
    assert partitions == ["batch"]


@requires_lens
def test_a_failed_op_is_recorded_rather_than_left_open(finished_spans):
    """An op that raises is exactly the one worth finding in a trace."""

    class _Boom(NoOpDataPlaneClient):
        def put_samples(self, *args, **kwargs):
            raise RuntimeError("rdma write failed")

    inner = _Boom()
    client = MetricsDataPlaneClient(inner, on_event=None)
    client.register_partition(
        partition_id="train", fields=["x"], num_samples=2, consumer_tasks=["read"]
    )
    with pytest.raises(RuntimeError):
        client.put_samples(
            sample_ids=["a"],
            partition_id="train",
            fields=TensorDict(
                {"x": torch.zeros(1, dtype=torch.float32)}, batch_size=[1]
            ),
        )
    inner.close()

    put = next(s for s in finished_spans() if s.name == "rl.data_plane.put")
    assert put.attributes["rl.data_plane.status"] == "error"


@requires_lens
def test_spans_are_emitted_without_the_event_callback(finished_spans):
    """Spans and the event log are independently switchable.

    The factory installs this wrapper for telemetry alone, leaving ``on_event``
    unset, so span emission must not be routed through the event path.
    """
    inner = NoOpDataPlaneClient()
    client = MetricsDataPlaneClient(inner, on_event=None)
    _put_one(client)
    client.close()
    inner.close()

    assert finished_spans()


def test_spans_cost_nothing_when_telemetry_is_off():
    """The wrapper is installed on every data-plane client, so the disabled
    path is the one that actually runs in production today."""
    inner = NoOpDataPlaneClient()
    client = MetricsDataPlaneClient(inner, on_event=None)
    _put_one(client)
    client.close()
    inner.close()
