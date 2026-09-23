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

import dataclasses
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSink as TokenSinkProtocol,
)
from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSource as TokenSourceProtocol,
)
from nemo_gym.token_id_capture.staging.digest import (  # noqa: E402
    compute_extras_digest,
    compute_staging_digest,
)
from nemo_gym.token_id_capture.staging.records import StagedCallRecord  # noqa: E402

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    COMPACT_LEN_FIELD,
    COMPACT_PREV_LEN_KEY,
    COMPACT_TOKEN_IDS_EXTRAS_KEY,
    COMPACT_TOKEN_IDS_FIELD,
    MEDIA_FLAG_FIELDS,
    MEDIA_PREV_COUNT_KEY,
    MEDIA_STAGING_FIELDS,
    MEDIA_TENSOR_COLUMNS,
    MINF_CAPTURE_PARAMS_FIELD,
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    STAGING_FIELDS,
    ChainPrefixCache,
    PrefixChains,
    TQMegatronPromptPreparer,
    TQMegatronTokenStager,
    TQStagingStore,
    TQTokenSink,
    TQTokenSource,
    _MegatronCapturePayload,
    resolve_admission_prefix_chains,
    slice_media_tensors,
)
from tests.unit.data_plane.token_capture_test_fixtures import (  # noqa: E402
    build_fixture_artifacts,
    fixture_names,
)

STAGING_PARTITION = "rollout_staging_test"

pytestmark = pytest.mark.nemo_gym


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


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


def _video(*, frames=(1,), patch_size=2, dtype=torch.float16):
    per_frame = [(2, 2)] * sum(frames)
    bundle = _still(dtype=dtype, sizes=per_frame, patch_size=patch_size)
    bundle["num_frames"] = torch.tensor(frames, dtype=torch.int32)
    return bundle


def _assert_same_tensor(actual, expected):
    assert actual.dtype == expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_media_rows_round_trip_in_a_single_put(tq_client, media_partition):
    client = _RecordingClient(tq_client)
    sink = TQTokenSink(
        client,
        staging_partition=media_partition,
        capture_media=True,
        media_pixel_dtype=torch.bfloat16,
    )
    source = TQTokenSource(
        client, staging_partition=media_partition, capture_media=True
    )
    chain, _, _ = build_fixture_artifacts("worked_example")
    single, _, _ = build_fixture_artifacts("single_call")
    records = [*chain[:2], single[0]]
    still, video = _still(), _video(frames=(1,), dtype=torch.bfloat16)
    bundles = [still, video, None]  # bf16 still, one-frame video, text call
    for record, bundle in zip(records, bundles, strict=True):
        assert sink.stage(record, attachments=bundle).ok
    # TQ keeps one dtype per field across live rows (a mismatch is swallowed by
    # the controller and drops the row's shape metadata), so the text call's
    # sentinels must share each column's dtype with the real rows.
    raw = tq_client.get_samples(
        partition_id=media_partition,
        sample_ids=[record.staging_key for record in records],
        select_fields=list(MEDIA_TENSOR_COLUMNS.values()),
    )
    for column in MEDIA_TENSOR_COLUMNS.values():
        dtypes = {raw[column][i].dtype for i in range(3)}
        assert len(dtypes) == 1, (column, dtypes)
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
    sink = TQTokenSink(
        client,
        staging_partition=media_partition,
        capture_media=True,
        media_pixel_dtype=torch.bfloat16,
    )
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
        lambda b: {**b, "imgs": b["imgs"].to(torch.float16)},  # not the column dtype
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
    sink = TQTokenSink(
        client,
        staging_partition=media_partition,
        capture_media=True,
        media_pixel_dtype=torch.bfloat16,
    )
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
    sink = TQTokenSink(
        client,
        staging_partition=media_partition,
        capture_media=True,
        media_pixel_dtype=torch.bfloat16,
    )
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
    sink = TQTokenSink(
        client,
        staging_partition=media_partition,
        capture_media=True,
        media_pixel_dtype=torch.bfloat16,
    )
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


def _with_extras(record: StagedCallRecord, extras: dict) -> StagedCallRecord:
    """Rebuild a fixture record carrying ``extras`` with both digests recomputed."""
    values = record.model_dump()
    values["extras"] = extras
    values["extras_digest"] = compute_extras_digest(extras)
    values["digest"] = compute_staging_digest(
        **{
            name: values[name]
            for name in (
                "schema_version",
                "digest_version",
                "extras_digest_version",
                "rollout_id",
                "model_call_id",
                "parent_call_id",
                "mode",
                "prev_len",
                "delta_len",
                "cum_len",
                "weight_version",
                "token_ids_delta",
                "token_mask_delta",
                "generation_log_probs_delta",
                "extras_digest",
                "chain_hash",
                "cumulative_hash",
            )
        }
    )
    return StagedCallRecord(**values)


@pytest.mark.parametrize(
    "compact_delta",
    ["not-a-list", [], [1, 1.5], [True]],
    ids=["str", "empty", "float", "bool"],
)
def test_stage_rejects_malformed_compact_delta_and_leaves_no_row(
    tq_client, staging_partition, compact_delta
):
    """A malformed ``compact_token_ids_delta`` fails the stage and stages nothing."""
    sink = TQTokenSink(tq_client, staging_partition=staging_partition)
    records, _, _ = build_fixture_artifacts("single_call")
    record = _with_extras(records[0], {COMPACT_TOKEN_IDS_EXTRAS_KEY: compact_delta})

    result = sink.stage(record)

    assert result.ok is False
    assert "compact_token_ids_delta" in (result.error or "")
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(KeyError):
        source.fetch_for_finalization([record.staging_key])


def test_fetch_prefix_chains_rejects_compact_len_mismatch(tq_client, staging_partition):
    """A row whose ``compact_len`` disagrees with its delta column is refused, not truncated."""
    store = TQStagingStore(tq_client, staging_partition=staging_partition)
    store.put(
        "r0/c1",
        {
            "token_ids_delta": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            COMPACT_TOKEN_IDS_FIELD: torch.tensor([[7, 8]], dtype=torch.int64),
            COMPACT_LEN_FIELD: torch.tensor([3], dtype=torch.int64),
        },
    )
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    with pytest.raises(ValueError, match="compact_len"):
        source.fetch_prefix_chains(["r0/c1"])


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


# ── Megatron stager / preparer ───────────────────────────────────────────────


def _minf_media_tensors():
    """Packed patches the toy encoder consumed: one 4x4 image, patch 2 -> 4 patches."""
    return {
        "imgs": torch.arange(4 * 12, dtype=torch.float32).reshape(1, 4, 12),
        "imgs_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
    }


def _minf_two_image_tensors():
    """Turn 2's engine tensors: image 1's four patches followed by image 2's four."""
    return {
        "imgs": torch.arange(8 * 12, dtype=torch.float32).reshape(1, 8, 12),
        "imgs_sizes": torch.tensor([[4, 4], [4, 4]], dtype=torch.int32),
    }


def _minf_payload(*, multimodal: bool, media_tensors=...):
    """A finished MInf payload. Multimodal: compact [80, 99, 81] expanded to three 99s."""
    if not multimodal:
        return SimpleNamespace(
            prompt_token_ids=[10, 11],
            generated_token_ids=[12, 13],
            generated_log_probs=[-0.25, -0.5],
        )
    return SimpleNamespace(
        prompt_token_ids=[80, 99, 99, 99, 81],
        generated_token_ids=[12, 2],
        generated_log_probs=[-0.25, -0.5],
        compact_prompt_token_ids=[80, 99, 81],
        media_tensors=_minf_media_tensors() if media_tensors is ... else media_tensors,
    )


def _megatron_sink(tq_client, partition, *, media: bool = False) -> TQTokenSink:
    """A sink matching ``partition``'s schema (the MInf pixels are float32 here)."""
    return TQTokenSink(
        tq_client,
        staging_partition=partition,
        capture_media=media,
        media_pixel_dtype=torch.float32 if media else None,
    )


def _stage_root(
    tq_client,
    partition,
    payload,
    *,
    rollout_id="minf-r0",
    model_call_id="c1",
    media: bool = False,
):
    stager = TQMegatronTokenStager(_megatron_sink(tq_client, partition, media=media))
    root = nemo_gym.CaptureAdmission(
        rollout_id=rollout_id, model_call_id=model_call_id, mode="text"
    )
    result = stager.stage(
        "minf-response-1",
        payload,
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={"ng_capture": root.model_dump(mode="json")},
    )
    assert result is not None
    return stager, result.response_metadata["ng_commit_coords"]


@pytest.mark.parametrize("multimodal", [False, True], ids=["text", "multimodal"])
def test_megatron_stager_writes_canonical_row_and_returns_coords(
    tq_client, request, multimodal
):
    """The expanded delta is the canonical row. A VLM call also stages the compact
    delta in its own column and the engine's media tensors in the shared media
    columns of the same key; a text call stages neither and its compact chain
    falls back to the expanded one."""
    partition = request.getfixturevalue(
        "media_partition" if multimodal else "staging_partition"
    )
    _, coords = _stage_root(
        tq_client, partition, _minf_payload(multimodal=multimodal), media=multimodal
    )
    assert coords["staging_key"] == "minf-r0/c1"
    assert coords["weight_version"] == 7
    assert coords["disposition"] == "staged"
    source = TQTokenSource(
        tq_client, staging_partition=partition, capture_media=multimodal
    )
    [fetched] = source.fetch_for_finalization(["minf-r0/c1"])
    chains = source.fetch_prefix_chains(["minf-r0/c1"])
    assert source.fetch_prefix_token_ids(["minf-r0/c1"]) == chains.expanded

    if not multimodal:
        assert fetched.snapshot.token_ids_delta == [10, 11, 12, 13]
        assert fetched.snapshot.token_mask_delta == [0.0, 0.0, 1.0, 1.0]
        assert fetched.snapshot.generation_log_probs_delta == [0.0, 0.0, -0.25, -0.5]
        assert fetched.extras is None
        assert fetched.media_present is False
        assert chains.compact == chains.expanded == [10, 11, 12, 13]
        assert chains.media_count == 0
        with pytest.raises(ValueError, match="media-enabled"):
            source.fetch_media([fetched])
        return

    assert fetched.snapshot.token_ids_delta == [80, 99, 99, 99, 81, 12, 2]
    # Mask and log probs are aligned with the *expanded* delta: every media
    # token is prompt (mask 0); only the two generated tokens train.
    assert fetched.snapshot.token_mask_delta == [0.0] * 5 + [1.0, 1.0]
    assert fetched.snapshot.generation_log_probs_delta == [0.0] * 5 + [-0.25, -0.5]
    assert (coords["prev_len"], coords["delta_len"]) == (0, 7)
    assert chains.expanded == [80, 99, 99, 99, 81, 12, 2]
    assert chains.compact == [80, 99, 81, 12, 2]
    assert chains.media_count == 1
    # The compact delta lives in its own column, not in the extras JSON, and
    # media rides the shared columns rather than a geometry summary.
    assert not fetched.extras
    assert fetched.media_present is True and fetched.media_has_frames is False
    [media] = source.fetch_media([fetched])
    assert torch.equal(media.imgs, _minf_media_tensors()["imgs"])
    assert media.imgs.dtype == torch.float32
    assert media.imgs_sizes.tolist() == [[4, 4]]
    assert media.num_frames is None


def test_megatron_stager_passes_media_tensors_as_attachments():
    """The stager hands MInf's media tensors to Gym as attachments; nothing is parked on the sink."""
    sink = MagicMock(spec=TQTokenSink)
    stager = TQMegatronTokenStager(sink)
    capture = MagicMock()
    capture.complete_call_from_response.return_value = MagicMock(
        model_dump=lambda mode: {"disposition": "staged"}
    )
    stager._capture = capture
    imgs = torch.zeros(1, 4, 768, dtype=torch.bfloat16)
    sizes = torch.tensor([[32, 32]], dtype=torch.int32)
    payload = SimpleNamespace(
        prompt_token_ids=[80, 99, 99, 99, 99, 81],
        generated_token_ids=[12, 2],
        generated_log_probs=[-0.1, -0.2],
        compact_prompt_token_ids=[80, 99, 81],
        media_tensors={"imgs": imgs, "imgs_sizes": sizes},
    )
    admission = nemo_gym.CaptureAdmission(
        rollout_id="r0", model_call_id="c1", mode="text"
    ).model_dump(mode="json")
    result = stager._stage_admitted(
        payload,
        capture_payload=admission,
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 0)]),
        minf_params={COMPACT_PREV_LEN_KEY: 0, MEDIA_PREV_COUNT_KEY: 0},
    )
    assert result.response_metadata == {"ng_commit_coords": {"disposition": "staged"}}
    kwargs = capture.complete_call_from_response.call_args.kwargs
    assert kwargs["attachments"]["imgs"] is imgs
    assert kwargs["attachments"]["imgs_sizes"] is sizes


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_minf_payload(multimodal=False), id="text"),
        pytest.param(
            _minf_payload(multimodal=True, media_tensors={}), id="empty-media-mapping"
        ),
    ],
)
def test_megatron_stager_passes_no_attachments_for_text_calls(payload):
    """A call without media tensors (absent or empty) reaches Gym with attachments=None."""
    sink = MagicMock(spec=TQTokenSink)
    stager = TQMegatronTokenStager(sink)
    capture = MagicMock()
    capture.complete_call_from_response.return_value = MagicMock(
        model_dump=lambda mode: {"disposition": "staged"}
    )
    stager._capture = capture
    admission = nemo_gym.CaptureAdmission(
        rollout_id="r0", model_call_id="c1", mode="text"
    ).model_dump(mode="json")
    stager._stage_admitted(
        payload,
        capture_payload=admission,
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 0)]),
    )
    assert capture.complete_call_from_response.call_args.kwargs["attachments"] is None


def test_fetch_prefix_chains_counts_media_items_from_columns(
    tq_client, media_partition
):
    """media_prev_count comes from the small media columns, not extras."""
    sink = TQTokenSink(
        tq_client,
        staging_partition=media_partition,
        capture_media=True,
        media_pixel_dtype=torch.bfloat16,
    )
    still = {
        "imgs": torch.zeros(1, 4, 768, dtype=torch.bfloat16),
        "imgs_sizes": torch.tensor([[32, 32]], dtype=torch.int32),
    }
    video = {
        "imgs": torch.zeros(1, 12, 768, dtype=torch.bfloat16),
        "imgs_sizes": torch.tensor([[32, 32]] * 3, dtype=torch.int32),
        "num_frames": torch.tensor([3], dtype=torch.int32),
    }
    # Stage three roots through Gym's capture core: one still, one text, one video.
    from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture

    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 0)
    for call_id, media in (("c1", still), ("c2", None), ("c3", video)):
        call = capture.begin_call(
            nemo_gym.CaptureAdmission(
                rollout_id="r0", model_call_id=call_id, mode="text"
            )
        )
        coords = capture.complete_call(
            call,
            prompt_token_ids=[1],
            generated_token_ids=[2],
            generated_logprobs=[-0.5],
            attachments=media,
        )
        assert coords.disposition == "staged", coords
    source = TQTokenSource(
        tq_client, staging_partition=media_partition, capture_media=True
    )
    chains = source.fetch_prefix_chains(["r0/c1", "r0/c2", "r0/c3"])
    assert chains.expanded == chains.compact == [1, 2] * 3
    assert chains.media_count == 2
    # The token-only reader never selects media columns.
    text_source = TQTokenSource(tq_client, staging_partition=media_partition)
    assert text_source.fetch_prefix_chains(["r0/c1", "r0/c3"]).media_count == 0


_PREPARER_CASES = {
    # (root payload or None for an inline prefix, prev_len, prompt, template prefix, eos,
    #  expected prompt, expected required prefix, expected compact_prev_len,
    #  expected media_prev_count)
    "staging_chain": (
        SimpleNamespace(
            prompt_token_ids=[10, 11],
            generated_token_ids=[12, 99],
            generated_log_probs=[-0.25, -0.5],
        ),
        4,
        [80, 81, 99, 20, 21],
        [80, 81, 99],
        99,
        [10, 11, 12, 99, 20, 21],
        [10, 11, 12, 99],
        4,
        0,
    ),
    "capture_admission": (
        None,
        4,
        [80, 81, 99, 20, 21],
        [80, 81, 99],
        99,
        [10, 11, 12, 99, 20, 21],
        [10, 11, 12, 99],
        4,
        0,
    ),
    # Turn 2 of a VLM rollout: the chat endpoint renders the history in compact form
    # (one media token per image, "a cat" retokenized as 13 not 12), so the preparer
    # splices the *compact* chain, hands Gym the *expanded* chain to verify the engine
    # prompt against, and tells the stager how much of the compact prompt and how many
    # media items the chain already covers. The new turn adds a second image.
    "multimodal_chain": (
        _minf_payload(multimodal=True),
        7,
        [80, 99, 81, 13, 2, 20, 99, 21],
        [80, 99, 81, 13, 2],
        2,
        [80, 99, 81, 12, 2, 20, 99, 21],
        [80, 99, 99, 99, 81, 12, 2],
        5,
        1,
    ),
}


@pytest.mark.parametrize("prefix_source", list(_PREPARER_CASES))
def test_megatron_prompt_preparer_splices_resolved_prefix(
    tq_client, request, prefix_source
):
    (
        root_payload,
        prev_len,
        prompt,
        template_prefix,
        eos,
        expected_prompt,
        expected_required,
        expected_compact_prev_len,
        expected_media_prev_count,
    ) = _PREPARER_CASES[prefix_source]
    media = prefix_source == "multimodal_chain"
    partition = request.getfixturevalue(
        "media_partition" if media else "staging_partition"
    )
    stager = None
    if root_payload is not None:
        stager, root_coords = _stage_root(
            tq_client, partition, root_payload, media=media
        )
        admission_kwargs = {
            "staging_chain": [root_coords["staging_key"]],
            "parent_chain_hash": root_coords["chain_hash"],
        }
    else:
        admission_kwargs = {
            "required_prefix_token_ids": expected_required,
            "parent_chain_hash": _digest("chain:c1"),
        }

    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=prev_len,
        mode="token_in",
        **admission_kwargs,
    )
    preparer = TQMegatronPromptPreparer(
        TQTokenSource(tq_client, staging_partition=partition, capture_media=media)
    )

    result = preparer.prepare_prompt(
        prompt,
        offload_params={
            "ng_capture": admission.model_dump(mode="json"),
            PREFIX_TEMPLATE_TOKEN_IDS_FIELD: template_prefix,
            PREFIX_EOS_TOKEN_ID_FIELD: eos,
        },
    )

    assert result.prompt == expected_prompt
    assert result.offload_params is not None
    assert (
        result.offload_params["ng_capture"]["required_prefix_token_ids"]
        == expected_required
    )
    assert result.offload_params[MINF_CAPTURE_PARAMS_FIELD] == {
        COMPACT_PREV_LEN_KEY: expected_compact_prev_len,
        MEDIA_PREV_COUNT_KEY: expected_media_prev_count,
    }

    if not media:
        return
    # The engine expands the spliced compact prompt (both images) and hands the
    # stager pixels for both; the stager cuts the compact delta at compact_prev_len,
    # keeps only image 2's pixels (media_prev_count), and Gym verifies the expanded prefix.
    two_images = _minf_two_image_tensors()
    turn2 = stager.stage(
        "minf-response-2",
        SimpleNamespace(
            prompt_token_ids=[80, 99, 99, 99, 81, 12, 2, 20, 99, 99, 99, 21],
            generated_token_ids=[30],
            generated_log_probs=[-0.1],
            compact_prompt_token_ids=result.prompt,
            media_tensors=two_images,
        ),
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params=result.offload_params,
    )
    coords2 = turn2.response_metadata["ng_commit_coords"]
    assert coords2["disposition"] == "staged"
    assert (coords2["prev_len"], coords2["delta_len"]) == (7, 6)
    source = TQTokenSource(tq_client, staging_partition=partition, capture_media=True)
    chains = source.fetch_prefix_chains(
        [root_coords["staging_key"], coords2["staging_key"]]
    )
    assert chains.expanded == [80, 99, 99, 99, 81, 12, 2, 20, 99, 99, 99, 21, 30]
    assert chains.compact == [80, 99, 81, 12, 2, 20, 99, 21, 30]
    assert chains.media_count == 2
    # Turn 2's row holds only the media new to it.
    [fetched2] = source.fetch_for_finalization([coords2["staging_key"]])
    assert fetched2.media_present is True
    [media2] = source.fetch_media([fetched2])
    assert torch.equal(media2.imgs, two_images["imgs"][:, 4:, :])
    assert media2.imgs_sizes.tolist() == [[4, 4]]


def test_megatron_stager_stamps_admission_epoch_when_request_spans_refit(
    tq_client, staging_partition, caplog
):
    """A request straddling a refit is stamped with its admission epoch, not masked.

    Mirrors vLLM, which freezes the version at begin_call. The engine stamps the
    admission epoch first and appends a boundary per refit, so epochs only grow.
    """
    stager = TQMegatronTokenStager(_megatron_sink(tq_client, staging_partition))
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )
    with caplog.at_level("WARNING", logger="nemo_rl.data_plane.tq_token_sink"):
        result = stager.stage(
            "minf-response-1",
            SimpleNamespace(
                prompt_token_ids=[10],
                generated_token_ids=[11, 12],
                generated_log_probs=[-0.1, -0.2],
            ),
            finished_metadata=SimpleNamespace(policy_epoch=[(0, 7), (1, 8), (2, 9)]),
            offload_params={"ng_capture": admission.model_dump(mode="json")},
        )
    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["disposition"] == "staged"
    assert coords["weight_version"] == 7
    assert stager.epoch_span_count == 1
    assert any("spans policy epochs [7, 8, 9]" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "missing_field",
    ["prompt_token_ids", "generated_token_ids", "generated_log_probs"],
)
def test_megatron_stager_poisons_malformed_payloads_with_capture_failed(
    tq_client, staging_partition, missing_field
):
    """Extraction errors return ``capture_failed`` coords, not ``None``.

    Gym maps returned failed coords to ``worker_capture_failed`` (as for
    vLLM); a ``None`` result would instead surface as
    ``worker_response_missing_commit_coordinates``.
    """
    stager = TQMegatronTokenStager(_megatron_sink(tq_client, staging_partition))
    fields = {
        "prompt_token_ids": [10, 11],
        "generated_token_ids": [12, 13],
        "generated_log_probs": [-0.25, -0.5],
    }
    del fields[missing_field]
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )

    result = stager.stage(
        "minf-response-1",
        SimpleNamespace(**fields),
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={"ng_capture": admission.model_dump(mode="json")},
    )

    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["disposition"] == "capture_failed"
    assert coords["weight_version"] == 7
    with pytest.raises(KeyError):
        TQTokenSource(tq_client, staging_partition=staging_partition).fetch(
            ["minf-r0/c1"]
        )


def test_megatron_stager_poisons_payload_view_failures_with_capture_failed(
    tq_client, staging_partition
):
    """Errors while deriving the media delta poison the call, not drop its coords.

    ``_MegatronCapturePayload.from_offloaded`` runs after ``begin_call`` and
    before Gym's extraction; a ``media_prev_count`` the engine's media cannot
    satisfy must still surface as ``capture_failed`` (``worker_capture_failed``
    in Gym), never as a ``None`` result, which Gym records as
    ``worker_response_missing_commit_coordinates``.
    """
    stager = TQMegatronTokenStager(_megatron_sink(tq_client, staging_partition))
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0", model_call_id="c1", mode="text"
    )

    result = stager.stage(
        "minf-response-1",
        _minf_payload(multimodal=True),  # the engine saw one image
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={
            "ng_capture": admission.model_dump(mode="json"),
            # The parent chain claims two images were already staged.
            MINF_CAPTURE_PARAMS_FIELD: {
                COMPACT_PREV_LEN_KEY: 0,
                MEDIA_PREV_COUNT_KEY: 2,
            },
        },
    )

    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["disposition"] == "capture_failed"
    assert coords["weight_version"] == 7
    with pytest.raises(KeyError):
        TQTokenSource(tq_client, staging_partition=staging_partition).fetch(
            ["minf-r0/c1"]
        )


@pytest.mark.parametrize(
    ("payload", "minf_params", "match"),
    [
        pytest.param(
            _minf_payload(multimodal=True, media_tensors=[torch.zeros(1, 4, 12)]),
            {COMPACT_PREV_LEN_KEY: 0, MEDIA_PREV_COUNT_KEY: 1},
            "media_tensors must be a mapping, got list",
            id="media-tensors-list",
        ),
        pytest.param(
            _minf_payload(multimodal=False),
            ["not", "a", "dict"],
            "capture params must be a dict, got list",
            id="minf-params-list",
        ),
    ],
)
def test_megatron_payload_view_rejects_non_mapping_inputs(payload, minf_params, match):
    """Structural payload errors surface as TypeError, which the stager's poison path catches."""
    with pytest.raises(TypeError, match=match):
        _MegatronCapturePayload.from_offloaded(payload, minf_params)


@pytest.mark.parametrize(
    ("payload", "minf_params"),
    [
        pytest.param(
            _minf_payload(multimodal=True, media_tensors=[torch.zeros(1, 4, 12)]),
            {COMPACT_PREV_LEN_KEY: 0, MEDIA_PREV_COUNT_KEY: 1},
            id="media-tensors-list",
        ),
        pytest.param(
            _minf_payload(multimodal=False), ["not", "a", "dict"], id="minf-params-list"
        ),
    ],
)
def test_megatron_stager_poisons_non_mapping_payload_inputs_with_capture_failed(
    tq_client, staging_partition, payload, minf_params
):
    """A non-mapping ``media_tensors`` or non-dict capture params poison the call.

    Without the explicit type checks these escape ``from_offloaded`` as
    ``AttributeError`` (``slice_media_tensors`` calls ``.get``) or are silently
    treated as empty, instead of returning ``capture_failed`` coordinates.
    """
    stager = TQMegatronTokenStager(_megatron_sink(tq_client, staging_partition))
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0", model_call_id="c1", mode="text"
    )

    result = stager.stage(
        "minf-response-1",
        payload,
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={
            "ng_capture": admission.model_dump(mode="json"),
            MINF_CAPTURE_PARAMS_FIELD: minf_params,
        },
    )

    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["disposition"] == "capture_failed"
    assert coords["weight_version"] == 7
    with pytest.raises(KeyError):
        TQTokenSource(tq_client, staging_partition=staging_partition).fetch(
            ["minf-r0/c1"]
        )


def test_megatron_stager_reports_media_on_text_partition_as_capture_failed(
    tq_client, staging_partition
):
    """Media attachments against a text-only partition poison the call, not the server."""
    stager = TQMegatronTokenStager(_megatron_sink(tq_client, staging_partition))
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0", model_call_id="c1", mode="text"
    )
    result = stager.stage(
        "minf-response-1",
        _minf_payload(multimodal=True),
        finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
        offload_params={"ng_capture": admission.model_dump(mode="json")},
    )
    assert result is not None
    coords = result.response_metadata["ng_commit_coords"]
    assert coords["disposition"] == "capture_failed"


@pytest.mark.parametrize(
    ("with_capture_metadata", "policy_epoch"),
    [
        pytest.param(False, [(0, 7)], id="missing-capture-metadata"),
        pytest.param(True, [], id="no-policy-epoch-boundaries"),
        pytest.param(True, [(0, "x")], id="invalid-policy-epoch"),
        pytest.param(True, [(0, -1)], id="negative-policy-epoch"),
    ],
)
def test_megatron_stager_declines_ineligible_requests(
    tq_client, staging_partition, with_capture_metadata, policy_epoch
):
    stager = TQMegatronTokenStager(_megatron_sink(tq_client, staging_partition))
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )
    result = stager.stage(
        "minf-response-1" if with_capture_metadata else "ordinary-request",
        SimpleNamespace(
            prompt_token_ids=[10],
            generated_token_ids=[11],
            generated_log_probs=[-0.1],
        ),
        finished_metadata=SimpleNamespace(policy_epoch=policy_epoch),
        offload_params=(
            {"ng_capture": admission.model_dump(mode="json")}
            if with_capture_metadata
            else None
        ),
    )
    assert result is None


class _RecordingSource:
    """Stand-in for TQTokenSource that records fetched keys.

    Serves both token spaces: two expanded tokens and one compact token per key.
    """

    def __init__(self):
        self.calls = []

    def fetch_prefix_token_ids(self, keys):
        self.calls.append(list(keys))
        return [int(k[1:]) * 10 + i for k in keys for i in range(2)]

    def fetch_prefix_chains(self, keys):
        return PrefixChains(
            expanded=self.fetch_prefix_token_ids(keys),
            compact=[int(k[1:]) * 10 for k in keys],
        )


def test_chain_prefix_cache_fetches_only_uncached_suffix():
    source = _RecordingSource()
    cache = ChainPrefixCache(source)

    assert cache.fetch_chains(["k1", "k2"]).expanded == [10, 11, 20, 21]
    assert cache.fetch_chains(["k1", "k2", "k3"]).expanded == [10, 11, 20, 21, 30, 31]
    assert cache.fetch_chains(["k1", "k2"]).expanded == [10, 11, 20, 21]
    assert source.calls == [["k1", "k2"], ["k3"]]
    assert cache.fetch_chains(["k1", "k2", "k3"]) == PrefixChains(
        expanded=[10, 11, 20, 21, 30, 31], compact=[10, 20, 30]
    )
    assert source.calls == [["k1", "k2"], ["k3"]]


def test_chain_prefix_cache_requires_an_installed_source():
    cache = ChainPrefixCache()
    with pytest.raises(RuntimeError, match="setup_token_capture"):
        cache.fetch_chains(["k1"])
    source = _RecordingSource()
    cache.install(source)
    assert cache.fetch_chains(["k1"]).expanded == [10, 11]


def test_chain_prefix_cache_evicts_oldest_insertion_past_256_entries():
    source = _RecordingSource()
    cache = ChainPrefixCache(source)
    for i in range(257):
        cache.fetch_chains([f"k{i}"])
    # k0 was the first insertion and is gone; k1 is still a hit.
    calls_before = len(source.calls)
    cache.fetch_chains(["k1"])
    assert len(source.calls) == calls_before
    cache.fetch_chains(["k0"])
    assert len(source.calls) == calls_before + 1


def test_resolve_admission_prefix_chains_dispatches_on_admission_shape():
    """Text -> empty; inline prefix -> same ids in both spaces; chain -> cache fetch."""
    source = _RecordingSource()
    cache = ChainPrefixCache(source)
    text = SimpleNamespace(mode="text", staging_chain=[], required_prefix_token_ids=[])
    inline = SimpleNamespace(
        mode="token_in", staging_chain=[], required_prefix_token_ids=[7, 8]
    )
    chained = SimpleNamespace(
        mode="token_in", staging_chain=["k1"], required_prefix_token_ids=[]
    )

    assert resolve_admission_prefix_chains(text, cache) == PrefixChains(
        expanded=[], compact=[]
    )
    assert resolve_admission_prefix_chains(inline, cache) == PrefixChains(
        expanded=[7, 8], compact=[7, 8]
    )
    assert resolve_admission_prefix_chains(chained, cache) == PrefixChains(
        expanded=[10, 11], compact=[10]
    )
    assert source.calls == [["k1"]]


def test_megatron_preparer_resolves_chains_through_the_shared_cache():
    source = _RecordingSource()
    preparer = TQMegatronPromptPreparer(source)
    assert isinstance(preparer._chain_prefix, ChainPrefixCache)

    child = nemo_gym.CaptureAdmission(
        rollout_id="r0",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=2,
        mode="token_in",
        staging_chain=["k1"],
        parent_chain_hash="a" * 64,
    )
    grandchild = nemo_gym.CaptureAdmission(
        rollout_id="r0",
        model_call_id="c3",
        parent_call_id="c2",
        prev_len=4,
        mode="token_in",
        staging_chain=["k1", "k2"],
        parent_chain_hash="b" * 64,
    )
    for admission, prompt, template_prefix in (
        (child, [80, 99, 5], [80, 99]),
        (grandchild, [80, 81, 82, 99, 6], [80, 81, 82, 99]),
    ):
        preparer.prepare_prompt(
            prompt,
            offload_params={
                "ng_capture": admission.model_dump(mode="json"),
                PREFIX_TEMPLATE_TOKEN_IDS_FIELD: template_prefix,
                PREFIX_EOS_TOKEN_ID_FIELD: 99,
            },
        )
    # k1 was cached by the child call; the grandchild fetched only k2.
    assert source.calls == [["k1"], ["k2"]]


def test_prefix_field_keys_match_megatron_constants():
    """The endpoint writes Megatron's constants; the preparer reads NeMo-RL's copies."""
    mcore = pytest.importorskip("megatron.core.inference.inference_request")
    if not hasattr(mcore, "PREFIX_TEMPLATE_TOKEN_IDS_FIELD"):
        pytest.skip(
            "pinned megatron-core predates MInf prefix-splice metadata (Megatron-LM #7015)"
        )
    assert PREFIX_TEMPLATE_TOKEN_IDS_FIELD == mcore.PREFIX_TEMPLATE_TOKEN_IDS_FIELD
    assert PREFIX_EOS_TOKEN_ID_FIELD == mcore.PREFIX_EOS_TOKEN_ID_FIELD


def test_extras_and_payload_keys_match_gym_constants():
    """Pin the Gym extras key and payload-view attribute names to Gym's constants.

    The sink pops Gym's compact extras key, and the stager's payload view feeds
    Gym's Megatron adapter by attribute name.
    """
    media = pytest.importorskip("nemo_gym.token_id_capture.staging.media")
    adapter = pytest.importorskip("nemo_gym.token_id_capture.adapters.megatron")
    if not hasattr(adapter, "COMPACT_PREV_LEN_FIELD"):
        pytest.skip(
            "pinned nemo_gym predates multimodal Megatron extras (NVIDIA-NeMo/Gym#2823)"
        )
    assert COMPACT_TOKEN_IDS_EXTRAS_KEY == media.COMPACT_TOKEN_IDS_DELTA_FIELD
    assert COMPACT_PREV_LEN_KEY == adapter.COMPACT_PREV_LEN_FIELD
    payload_fields = {f.name for f in dataclasses.fields(_MegatronCapturePayload)}
    assert payload_fields >= {
        adapter.PROMPT_IDS_FIELD,
        adapter.GENERATED_IDS_FIELD,
        adapter.GENERATED_LOGPROBS_FIELD,
        adapter.COMPACT_PROMPT_IDS_FIELD,
        adapter.COMPACT_PREV_LEN_FIELD,
    }
    # The media summary left the record with Gym #3513; pixels are attachments.
    assert not hasattr(adapter, "MEDIA_FIELD")
    assert "media" not in payload_fields


@pytest.mark.parametrize(
    ("media_tensors", "prev_count", "expected"),
    [
        # Packed patches, two 4x4 images of 4 patches each: drop image 1.
        (
            {
                "imgs": torch.arange(96.0).reshape(1, 8, 12),
                "imgs_sizes": torch.tensor([[4, 4], [4, 4]]),
            },
            1,
            {
                "imgs": torch.arange(48.0, 96.0).reshape(1, 4, 12),
                "imgs_sizes": torch.tensor([[4, 4]]),
            },
        ),
        # Nothing already staged: unchanged.
        (
            {
                "imgs": torch.arange(96.0).reshape(1, 8, 12),
                "imgs_sizes": torch.tensor([[4, 4], [4, 4]]),
            },
            0,
            {
                "imgs": torch.arange(96.0).reshape(1, 8, 12),
                "imgs_sizes": torch.tensor([[4, 4], [4, 4]]),
            },
        ),
        # Everything already staged: no media for this call.
        (
            {
                "imgs": torch.arange(96.0).reshape(1, 8, 12),
                "imgs_sizes": torch.tensor([[4, 4], [4, 4]]),
            },
            2,
            None,
        ),
        # Video: one 2-frame video already staged, one 1-frame video new.
        (
            {
                "imgs": torch.arange(144.0).reshape(1, 12, 12),
                "imgs_sizes": torch.tensor([[4, 4], [4, 4], [4, 4]]),
                "num_frames": torch.tensor([2, 1]),
            },
            1,
            {
                "imgs": torch.arange(96.0, 144.0).reshape(1, 4, 12),
                "imgs_sizes": torch.tensor([[4, 4]]),
                "num_frames": torch.tensor([1]),
            },
        ),
        # Padded pixels [N, C, H, W]: one row per image.
        (
            {
                "imgs": torch.arange(2 * 3 * 4 * 4.0).reshape(2, 3, 4, 4),
                "imgs_sizes": torch.tensor([[4, 4], [4, 4]]),
            },
            1,
            {
                "imgs": torch.arange(48.0, 96.0).reshape(1, 3, 4, 4),
                "imgs_sizes": torch.tensor([[4, 4]]),
            },
        ),
    ],
    ids=["patches", "none-staged", "all-staged", "video", "padded-pixels"],
)
def test_slice_media_tensors_keeps_only_new_items(media_tensors, prev_count, expected):
    sliced = slice_media_tensors(media_tensors, prev_count)
    if expected is None:
        assert sliced is None
        return
    assert set(sliced) == set(expected)
    for name, value in expected.items():
        assert torch.equal(sliced[name], value), name


@pytest.mark.parametrize(
    ("media_tensors", "prev_count", "error"),
    [
        (
            {"imgs": torch.ones(1, 4, 12), "imgs_sizes": torch.tensor([[4, 4]])},
            2,
            "exceeds",
        ),
        # No per-item geometry at all: nothing says where image 1 ends.
        ({"imgs": torch.ones(1, 4, 12)}, 1, "requires imgs_sizes"),
        # 5 patches cannot tile two 4x4 images (area 32).
        (
            {
                "imgs": torch.ones(1, 5, 12),
                "imgs_sizes": torch.tensor([[4, 4], [4, 4]]),
            },
            1,
            "do not divide",
        ),
        # Area 18 over 2 patches -> 9 per patch; image 1 (area 6) ends mid-patch.
        (
            {
                "imgs": torch.ones(1, 2, 12),
                "imgs_sizes": torch.tensor([[2, 3], [3, 4]]),
            },
            1,
            "patch boundary",
        ),
    ],
    ids=["exceeds", "no-geometry", "non-dividing", "mid-patch"],
)
def test_slice_media_tensors_rejects_inconsistent_geometry(
    media_tensors, prev_count, error
):
    with pytest.raises(ValueError, match=error):
        slice_media_tensors(media_tensors, prev_count)
