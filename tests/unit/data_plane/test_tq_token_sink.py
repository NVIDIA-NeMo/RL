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

"""TQTokenSink / TQTokenSource against a live TQ backend.

Runs NeMo-Gym's installable conformance kit (golden call sequences →
byte-exact digests, manifests, and linearized rows) over the TransferQueue
implementations — the framework-CI half of Gym's published conformance
contract (the other half runs inside Gym itself, verifying its digest
implementation against the same golden_vectors.json) — plus the
protocol edges the kit does not cover (missing keys, stage failure shape).
"""

from __future__ import annotations

import pytest

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.digest import (  # noqa: E402
    compute_extras_digest,
    compute_staging_digest,
)
from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSink as TokenSinkProtocol,
)
from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSource as TokenSourceProtocol,
)

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    STAGING_FIELDS,
    TQTokenSink,
    TQTokenSource,
    generation_cut_staging_key,
)
from tests.unit.data_plane.token_capture_test_fixtures import (  # noqa: E402
    build_fixture_artifacts,
    fixture_names,
)
from tools.benchmark_tq_prefix_writes import (  # noqa: E402
    BenchmarkConfig,
    _data_plane_config,
    _write_ranges,
    run_prefix_write_benchmark,
)

STAGING_PARTITION = "rollout_staging_test"

pytestmark = pytest.mark.nemo_gym


@pytest.mark.nemo_gym
def test_gym_staging_package_is_importable_in_the_nemo_gym_lane():
    """The --nemo-gym-only lane installs the extra; a missing staging package
    means the Gym pin moved off the capture branch and every capture test
    below has silently degraded to a skip."""
    import nemo_gym.token_id_capture.staging  # noqa: F401


def test_tq_sink_source_passes_gym_golden_vectors():
    """Gym publishes a fixed wire contract independent of this repo's own
    fixtures; this catches drift in Gym's digest scheme that
    token_capture_test_fixtures.py cannot, since it computes its expected
    digests by calling Gym's own digest functions."""
    from nemo_gym.token_id_capture.staging.conformance import assert_golden_vectors

    assert_golden_vectors()


@pytest.fixture()
def staging_partition(tq_client):
    tq_client.register_partition(
        partition_id=STAGING_PARTITION,
        fields=list(STAGING_FIELDS),
        num_samples=64,
        consumer_tasks=["finalize"],
    )
    yield STAGING_PARTITION
    tq_client.clear_samples(sample_ids=None, partition_id=STAGING_PARTITION)


def test_implementations_satisfy_protocols(tq_client, staging_partition):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    assert isinstance(sink, TokenSinkProtocol)
    assert isinstance(source, TokenSourceProtocol)


@pytest.mark.parametrize(
    "fixture_name", ["worked_example", "single_call", "mixed_weight_versions"]
)
def test_tq_sink_source_passes_conformance(tq_client, staging_partition, fixture_name):
    assert fixture_name in fixture_names()
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts(fixture_name)
    for record in records:
        assert sink.stage(record).ok
    snapshots = source.fetch([record.staging_key for record in records])
    # The source returns extras-free base snapshots; every base field (all
    # digest inputs) must round-trip byte-exactly.
    assert [snapshot.model_dump() for snapshot in snapshots] == [
        record.model_dump(exclude={"extras"}) for record in records
    ]


def test_fetch_missing_key_raises_keyerror(tq_client, staging_partition):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError):
        source.fetch(["ghost_rollout/ghost_call"])


def test_fetch_for_finalization_is_small_typed_and_identity_preserving(
    tq_client, staging_partition
):
    class RecordingClient:
        def __init__(self, client):
            self.client = client
            self.select_fields = None

        def get_samples(self, **kwargs):
            self.select_fields = list(kwargs["select_fields"])
            return self.client.get_samples(**kwargs)

    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    assert sink.stage(records[0]).ok
    recording_client = RecordingClient(tq_client)
    source = TQTokenSource(recording_client, staging_partition=staging_partition)

    fetched = source.fetch_for_finalization([records[0].staging_key])

    assert recording_client.select_fields == STAGING_FIELDS
    assert "routed_experts" not in recording_client.select_fields
    assert len(fetched) == 1
    assert fetched[0].staging_key == records[0].staging_key
    assert fetched[0].snapshot.model_call_id == records[0].model_call_id
    assert fetched[0].routed_len == 0
    assert fetched[0].fragment is None
    # The snapshot is a normally validated base model, never model_construct'd.
    from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot

    assert type(fetched[0].snapshot) is StagedCallBaseSnapshot
    assert not hasattr(fetched[0].snapshot, "extras")


def test_fetch_for_finalization_rejects_duplicate_request_keys(
    tq_client, staging_partition
):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError, match="duplicate keys"):
        source.fetch_for_finalization(["r/c", "r/c"])


def test_stage_failure_reports_not_raises(staging_partition):
    class ExplodingClient:
        def put_samples(self, **kwargs):
            raise RuntimeError("controller down")

    sink = TQTokenSink(ExplodingClient(), staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    result = sink.stage(records[0])
    assert not result.ok
    assert result.staging_key == records[0].staging_key
    assert "controller down" in (result.error or "")


def test_sink_clear_drops_rows(tq_client, staging_partition):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    for record in records:
        assert sink.stage(record).ok
    keys = [record.staging_key for record in records]
    assert len(source.fetch(keys)) == len(keys)
    sink.clear(keys)
    with pytest.raises(KeyError):
        source.fetch(keys)


def test_generation_prefix_uses_checkpoint_and_chunk_scoped_key(
    tq_client, staging_partition
):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    record = records[0]

    first = sink.stage_generation_prefix(
        record, checkpoint_id="checkpoint-1", chunk_sequence=0
    )
    second = sink.stage_generation_prefix(
        record, checkpoint_id="checkpoint-1", chunk_sequence=1
    )

    first_key = generation_cut_staging_key(
        "checkpoint-1",
        record.rollout_id,
        record.model_call_id,
        chunk_sequence=0,
    )
    second_key = generation_cut_staging_key(
        "checkpoint-1",
        record.rollout_id,
        record.model_call_id,
        chunk_sequence=1,
    )
    assert first.ok and second.ok
    assert first.staging_key == first_key
    assert second.staging_key == second_key
    assert first_key != second_key
    assert [
        snapshot.model_dump() for snapshot in source.fetch([first_key, second_key])
    ] == [
        record.model_dump(exclude={"extras"}),
        record.model_dump(exclude={"extras"}),
    ]


def test_fetch_prefix_token_ids_empty(tq_client, staging_partition):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    assert source.fetch_prefix_token_ids([]) == []


def test_fetch_prefix_token_ids_single_key(tq_client, staging_partition):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    record = records[0]
    assert sink.stage(record).ok
    result = source.fetch_prefix_token_ids([record.staging_key])
    assert result == record.token_ids_delta


def test_fetch_prefix_token_ids_three_keys_concatenates(tq_client, staging_partition):
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("worked_example")
    for record in records:
        assert sink.stage(record).ok
    keys = [record.staging_key for record in records]
    result = source.fetch_prefix_token_ids(keys)
    expected = [t for record in records for t in record.token_ids_delta]
    assert result == expected


def test_fetch_prefix_token_ids_missing_key_raises_keyerror(
    tq_client, staging_partition
):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError):
        source.fetch_prefix_token_ids(["ghost_rollout/ghost_call"])


def test_fetch_prefix_token_ids_rejects_duplicates(tq_client, staging_partition):
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError, match="duplicates"):
        source.fetch_prefix_token_ids(["r/c", "r/c"])


def test_tq_prefix_write_benchmark_exercises_live_unbatched_path(tq_client):
    partition_id = "rollout_staging_prefix_write_benchmark_test"
    result = run_prefix_write_benchmark(
        tq_client,
        BenchmarkConfig(
            rows=8,
            prefix_tokens=4,
            writers_per_client=2,
            num_storage_units=1,
            verify_rows=3,
            verify_batch_size=2,
            partition_id=partition_id,
            checkpoint_id="benchmark-test",
        ),
    )

    assert result["put_calls"] == 8
    assert result["stored_keys"] == 8
    assert result["verified_rows"] == 3
    assert result["writers_used"] == 2
    assert result["rows_per_second"] > 0
    assert result["cleanup_seconds"] is not None
    assert tq_client.list_sample_ids(partition_id) == []


@pytest.mark.parametrize("batch_size,expected_calls", [(1, 8), (3, 4)])
def test_tq_prefix_write_benchmark_uses_process_isolated_clients(
    tq_client, batch_size, expected_calls
):
    partition_id = "rollout_staging_prefix_write_multi_client_test"
    config = BenchmarkConfig(
        rows=8,
        prefix_tokens=4,
        clients=2,
        writers_per_client=1,
        batch_size=batch_size,
        num_storage_units=1,
        verify_rows=3,
        verify_batch_size=2,
        partition_id=partition_id,
        checkpoint_id="multi-client-benchmark-test",
    )
    result = run_prefix_write_benchmark(
        tq_client,
        config,
        dp_config=_data_plane_config(config),
    )

    assert result["put_calls"] == expected_calls
    assert result["stored_keys"] == 8
    assert result["verified_rows"] == 3
    assert result["clients_used"] == 2
    assert result["writers_used"] == 2
    assert result["client_setup_seconds"] > 0
    assert result["rows_per_second"] > 0
    assert result["cleanup_seconds"] is not None
    assert tq_client.list_sample_ids(partition_id) == []


def test_generation_prefix_batch_round_trips_ragged_rows(tq_client, staging_partition):
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def put_samples(self, **kwargs):
            self.calls.append(kwargs)
            return tq_client.put_samples(**kwargs)

    client = RecordingClient()
    sink = TQTokenSink(client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("worked_example")
    more_records, _, _ = build_fixture_artifacts(
        "single_call", rollout_id="a-different-length-rollout-id"
    )
    records.extend(more_records)
    results = sink.stage_generation_prefix_batch(
        records, checkpoint_id="batch-checkpoint", chunk_sequences=[0, 2, 3]
    )
    assert all(result.ok for result in results)
    assert len(client.calls) == 1
    assert client.calls[0]["sample_ids"] == [result.staging_key for result in results]
    assert [tag["digest"] for tag in client.calls[0]["tags"]] == [
        record.digest for record in records
    ]
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    restored = source.fetch([result.staging_key for result in results])
    assert [row.model_dump() for row in restored] == [
        record.model_dump(exclude={"extras"}) for record in records
    ]


def test_generation_prefix_batch_rejects_bad_inventory_before_writing():
    class UnexpectedClient:
        def put_samples(self, **kwargs):
            pytest.fail("invalid inventory must not reach TQ")

    sink = TQTokenSink(UnexpectedClient(), staging_partition="test")
    records, _, _ = build_fixture_artifacts("single_call")
    assert (
        sink.stage_generation_prefix_batch(
            [], checkpoint_id="checkpoint", chunk_sequences=[]
        )
        == []
    )
    with pytest.raises(ValueError, match="equal lengths"):
        sink.stage_generation_prefix_batch(
            records, checkpoint_id="checkpoint", chunk_sequences=[]
        )
    with pytest.raises(ValueError, match="duplicate"):
        sink.stage_generation_prefix_batch(
            records * 2, checkpoint_id="checkpoint", chunk_sequences=[0, 0]
        )


def test_generation_prefix_batch_preserves_optional_routes(
    tq_client, staging_partition
):
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def put_samples(self, **kwargs):
            self.calls.append(kwargs)
            return tq_client.put_samples(**kwargs)

    records, _, _ = build_fixture_artifacts("worked_example")
    values = records[0].model_dump(exclude={"digest"})
    values["extras"] = {"routed_experts": [[[1, 2]]] * records[0].delta_len}
    values["extras_digest"] = compute_extras_digest(values["extras"])
    values["digest"] = compute_staging_digest(
        **{key: value for key, value in values.items() if key != "extras"}
    )
    records[0] = type(records[0]).model_validate(values)
    client = RecordingClient()
    sink = TQTokenSink(client, staging_partition=staging_partition)
    results = sink.stage_generation_prefix_batch(
        records, checkpoint_id="routes", chunk_sequences=[0, 0]
    )
    assert all(result.ok for result in results)
    assert len(client.calls) == 2
    assert "routed_experts" in client.calls[0]["fields"]
    assert "routed_experts" not in client.calls[1]["fields"]
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    routed = source.fetch_for_finalization(
        [results[0].staging_key], include_route_fragments=True
    )[0]
    assert routed.fragment is not None
    assert routed.routed_len == records[0].delta_len
    assert routed.snapshot.model_dump() == records[0].model_dump(exclude={"extras"})
    plain = source.fetch([results[1].staging_key])[0]
    assert plain.model_dump() == records[1].model_dump(exclude={"extras"})


def test_generation_prefix_batch_failure_does_not_advertise_success():
    class FailingClient:
        def __init__(self):
            self.calls = 0

        def put_samples(self, **kwargs):
            self.calls += 1
            raise RuntimeError("injected partial batch write")

    client = FailingClient()
    sink = TQTokenSink(client, staging_partition="test")
    records, _, _ = build_fixture_artifacts("worked_example")
    results = sink.stage_generation_prefix_batch(
        records, checkpoint_id="checkpoint", chunk_sequences=[0, 1]
    )
    assert client.calls == 1
    assert len(results) == len(records)
    assert all(not result.ok for result in results)
    assert all(result.staging_key for result in results)
    assert all("injected partial batch write" in result.error for result in results)


def test_prefix_batch_benchmark_1000_rows_issue_four_puts():
    class RecordingClient:
        def __init__(self):
            self.batches = []

        def put_samples(self, **kwargs):
            self.batches.append(list(kwargs["sample_ids"]))

    client = RecordingClient()
    sink = TQTokenSink(client, staging_partition="test")
    timings = _write_ranges(
        sink,
        range(1000),
        checkpoint_id="checkpoint",
        prefix_tokens=4,
        writers=1,
        batch_size=256,
    )
    assert [len(batch) for batch in client.batches] == [256, 256, 256, 232]
    assert len({key for batch in client.batches for key in batch}) == 1000
    assert sum(timing.put_calls for timing in timings) == 4


@pytest.mark.parametrize("batch_size,expected_calls", [(1, 11), (4, 3), (256, 1)])
def test_tq_prefix_write_benchmark_batches_and_flushes_tail(
    tq_client, batch_size, expected_calls
):
    partition_id = f"prefix_batch_benchmark_{batch_size}"
    result = run_prefix_write_benchmark(
        tq_client,
        BenchmarkConfig(
            rows=11,
            prefix_tokens=4,
            batch_size=batch_size,
            partition_id=partition_id,
            verify_rows=11,
        ),
    )
    assert result["put_calls"] == expected_calls
    assert result["stored_keys"] == 11
    assert result["verified_rows"] == 11
    assert tq_client.list_sample_ids(partition_id) == []
