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
implementations — the framework-CI half of the § 3.0 contract — plus the
protocol edges the kit does not cover (missing keys, stage failure shape).
"""

from __future__ import annotations

import pytest

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

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
)
from tests.unit.data_plane.token_capture_test_fixtures import (  # noqa: E402
    build_fixture_artifacts,
    fixture_names,
)

STAGING_PARTITION = "rollout_staging_test"

pytestmark = pytest.mark.nemo_gym


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
