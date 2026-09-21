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
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSink as TokenSinkProtocol,
)
from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSource as TokenSourceProtocol,
)

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    MEDIA_FLAG_FIELDS,
    MEDIA_STAGING_FIELDS,
    MEDIA_TENSOR_COLUMNS,
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


@pytest.mark.nemo_gym
def test_gym_staging_package_is_importable_in_the_nemo_gym_lane():
    """The --nemo-gym-only lane installs the extra; a missing staging package
    means the Gym pin moved off the capture branch and every capture test
    below has silently degraded to a skip."""
    import nemo_gym.token_id_capture.staging  # noqa: F401


class _RecordingClient:
    """Pass-through DataPlaneClient that records put/get/clear traffic."""

    def __init__(self, client, *, fail_put: Exception | None = None, fail_clear=None):
        self.client = client
        self.puts: list[list[str]] = []
        self.gets: list[list[str]] = []
        self.clears: list[list[str]] = []
        self.fail_put = fail_put
        self.fail_clear = fail_clear

    def register_partition(self, **kwargs):
        return self.client.register_partition(**kwargs)

    def put_samples(self, **kwargs):
        self.puts.append(list(kwargs["fields"].keys()))
        if self.fail_put is not None:
            raise self.fail_put
        return self.client.put_samples(**kwargs)

    def get_samples(self, **kwargs):
        self.gets.append(list(kwargs["select_fields"]))
        return self.client.get_samples(**kwargs)

    def clear_samples(self, **kwargs):
        self.clears.append(list(kwargs["sample_ids"] or []))
        if self.fail_clear is not None:
            raise self.fail_clear
        return self.client.clear_samples(**kwargs)


MEDIA_PARTITION = "rollout_staging_media_test"


@pytest.fixture()
def media_partition(tq_client):
    tq_client.register_partition(
        partition_id=MEDIA_PARTITION,
        fields=STAGING_FIELDS + list(MEDIA_STAGING_FIELDS),
        num_samples=64,
        consumer_tasks=["finalize"],
    )
    yield MEDIA_PARTITION
    tq_client.clear_samples(sample_ids=None, partition_id=MEDIA_PARTITION)


def _still(*, dtype=torch.bfloat16, sizes=((2, 4), (4, 2)), patch_size=2):
    """Packed-patch still images: ``sizes`` are (h, w) per image."""
    sizes_t = torch.tensor(sizes, dtype=torch.int32)
    patches = int((sizes_t[:, 0] * sizes_t[:, 1]).sum()) // patch_size**2
    feature = 3 * patch_size**2
    imgs = torch.arange(patches * feature, dtype=torch.float32).reshape(
        1, patches, feature
    )
    return {"imgs": imgs.to(dtype), "imgs_sizes": sizes_t}


def _video(*, frames=(1,), patch_size=2):
    per_frame = [(2, 2)] * sum(frames)
    bundle = _still(dtype=torch.float16, sizes=per_frame, patch_size=patch_size)
    bundle["num_frames"] = torch.tensor(frames, dtype=torch.int32)
    return bundle


def _assert_same_tensor(actual, expected):
    assert actual.dtype == expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_media_rows_round_trip_in_a_single_put(tq_client, media_partition):
    client = _RecordingClient(tq_client)
    sink = TQTokenSink(client, staging_partition=media_partition, capture_media=True)
    source = TQTokenSource(
        client, staging_partition=media_partition, capture_media=True
    )
    chain, _, _ = build_fixture_artifacts("worked_example")
    single, _, _ = build_fixture_artifacts("single_call")
    records = [*chain[:2], single[0]]
    still, video = _still(), _video(frames=(1,))
    bundles = [still, video, None]  # bf16 still, one-frame video, text call
    for record, bundle in zip(records, bundles, strict=True):
        assert sink.stage(record, attachments=bundle).ok
    # One put per call, each carrying token, flag, and tensor columns.
    assert len(client.puts) == 3
    for fields in client.puts:
        assert set(MEDIA_STAGING_FIELDS) <= set(fields)
        assert set(STAGING_FIELDS) <= set(fields)

    keys = [record.staging_key for record in records]
    fetched = source.fetch_for_finalization(keys)
    assert client.gets[-1] == STAGING_FIELDS + MEDIA_FLAG_FIELDS
    assert [item.media_present for item in fetched] == [True, True, False]
    assert [item.media_has_frames for item in fetched] == [False, True, False]

    [got_still] = source.fetch_media([fetched[0]])
    _assert_same_tensor(got_still.imgs, still["imgs"])
    _assert_same_tensor(got_still.imgs_sizes, still["imgs_sizes"])
    assert got_still.num_frames is None
    [got_video] = source.fetch_media([fetched[1]])
    _assert_same_tensor(got_video.imgs, video["imgs"])
    _assert_same_tensor(got_video.num_frames, video["num_frames"])
    assert client.gets[-1] == list(MEDIA_TENSOR_COLUMNS.values())
    # Rows that cannot share one nested column are rejected before the read.
    reads = len(client.gets)
    with pytest.raises(ValueError, match="uniform media_has_frames"):
        source.fetch_media([fetched[0], fetched[1]])
    with pytest.raises(ValueError, match="media_present=True"):
        source.fetch_media([fetched[2]])
    assert len(client.gets) == reads
    assert source.fetch_media([]) == []


def test_batched_media_read_returns_ragged_rows_in_request_order(
    tq_client, media_partition
):
    client = _RecordingClient(tq_client)
    sink = TQTokenSink(client, staging_partition=media_partition, capture_media=True)
    source = TQTokenSource(
        client, staging_partition=media_partition, capture_media=True
    )
    records, _, _ = build_fixture_artifacts("worked_example")
    small = _still(sizes=((2, 2),))
    large = _still(sizes=((4, 4), (2, 6)))
    assert sink.stage(records[0], attachments=small).ok
    assert sink.stage(records[1], attachments=large).ok
    fetched = source.fetch_for_finalization([r.staging_key for r in records[:2]])
    reads = len(client.gets)
    parts = source.fetch_media([fetched[1], fetched[0]])
    assert len(client.gets) == reads + 1  # one batched tensor read
    _assert_same_tensor(parts[0].imgs, large["imgs"])
    _assert_same_tensor(parts[1].imgs, small["imgs"])
    _assert_same_tensor(parts[0].imgs_sizes, large["imgs_sizes"])


def test_text_only_partition_never_touches_media_columns(tq_client, staging_partition):
    client = _RecordingClient(tq_client)
    sink = TQTokenSink(client, staging_partition=staging_partition)
    source = TQTokenSource(client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    assert sink.stage(records[0]).ok
    assert not (set(MEDIA_STAGING_FIELDS) & set(client.puts[0]))
    [item] = source.fetch_for_finalization([records[0].staging_key])
    assert client.gets[-1] == STAGING_FIELDS
    assert (item.media_present, item.media_has_frames) == (False, False)
    with pytest.raises(ValueError, match="media-enabled"):
        source.fetch_media([item])
    # A text-only sink must reject attachments rather than drop them.
    result = sink.stage(records[0], attachments=_still())
    assert not result.ok and "media-enabled" in (result.error or "")
    assert len(client.puts) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: {},
        lambda b: {"imgs": b["imgs"]},  # missing imgs_sizes
        lambda b: {**b, "extra": torch.ones(1)},
        lambda b: {**b, "imgs": b["imgs"].tolist()},
        lambda b: {**b, "imgs": b["imgs"][:, :0]},
        lambda b: {**b, "imgs": b["imgs"].to(torch.int64)},
        lambda b: {**b, "imgs": b["imgs"][0]},  # 2-D
        lambda b: {**b, "imgs": b["imgs"][:, :, :11]},  # not 3*P*P
        lambda b: {**b, "imgs_sizes": b["imgs_sizes"].to(torch.float32)},
        lambda b: {**b, "imgs_sizes": b["imgs_sizes"] + 1},  # not patch aligned
        lambda b: {**b, "imgs_sizes": b["imgs_sizes"][:1]},  # patch total mismatch
        lambda b: {**b, "imgs_sizes": b["imgs_sizes"] * 0},
        lambda b: {**b, "num_frames": torch.tensor([1], dtype=torch.int32)},  # sum != N
        lambda b: {**b, "num_frames": torch.tensor([[2]], dtype=torch.int32)},
        lambda b: {**b, "num_frames": torch.tensor([2.0])},
        lambda b: {**b, "num_frames": torch.tensor([2**31], dtype=torch.int64)},
        lambda b: "not a mapping",
    ],
)
def test_malformed_media_fails_before_any_put(tq_client, media_partition, mutate):
    client = _RecordingClient(tq_client)
    sink = TQTokenSink(client, staging_partition=media_partition, capture_media=True)
    records, _, _ = build_fixture_artifacts("single_call")
    result = sink.stage(records[0], attachments=mutate(_still()))
    assert not result.ok
    assert client.puts == []
    assert client.clears == []


def test_validate_media_tensors_preserves_dtypes():
    from nemo_rl.data_plane.tq_token_sink import validate_media_tensors

    assert validate_media_tensors(None) is None
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        media = validate_media_tensors(_still(dtype=dtype))
        assert media.imgs.dtype is dtype
        assert media.imgs_sizes.dtype is torch.int32
    video = validate_media_tensors(_video(frames=(2, 1)))
    assert video.num_frames.dtype is torch.int32
    assert video.patch_size == 2
    sizes64 = _still()
    sizes64["imgs_sizes"] = sizes64["imgs_sizes"].to(torch.int64)
    assert validate_media_tensors(sizes64).imgs_sizes.dtype is torch.int64


def test_failed_combined_write_discards_the_attempted_key(tq_client, media_partition):
    client = _RecordingClient(tq_client, fail_put=RuntimeError("storage down"))
    sink = TQTokenSink(client, staging_partition=media_partition, capture_media=True)
    records, _, _ = build_fixture_artifacts("single_call")
    result = sink.stage(records[0], attachments=_still())
    assert not result.ok and "storage down" in (result.error or "")
    assert client.clears == [[records[0].staging_key]]
    source = TQTokenSource(
        tq_client, staging_partition=media_partition, capture_media=True
    )
    with pytest.raises(KeyError):
        source.fetch([records[0].staging_key])


def test_cleanup_failure_still_reports_the_stage_failure(
    tq_client, media_partition, caplog
):
    client = _RecordingClient(
        tq_client,
        fail_put=RuntimeError("storage down"),
        fail_clear=RuntimeError("cleanup down"),
    )
    sink = TQTokenSink(client, staging_partition=media_partition, capture_media=True)
    records, _, _ = build_fixture_artifacts("single_call")
    with caplog.at_level("ERROR", logger="nemo_rl.data_plane.tq_token_sink"):
        result = sink.stage(records[0], attachments=_still())
    assert not result.ok and "storage down" in (result.error or "")
    assert client.clears == [[records[0].staging_key]]
    assert any("orphaned" in message for message in caplog.messages)


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
