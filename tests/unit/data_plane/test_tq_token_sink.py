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

import hashlib
from types import SimpleNamespace

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

# The request-metadata keys the Megatron chat endpoint writes for the prompt
# preparer; the tests below play that endpoint.
from megatron.core.inference.inference_request import (  # noqa: E402
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
)
from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSink as TokenSinkProtocol,
)
from nemo_gym.token_id_capture.staging.protocols import (  # noqa: E402
    StagingSource as TokenSourceProtocol,
)

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    MEDIA_METADATA_FIELDS,
    MEDIA_STAGING_FIELDS,
    MEDIA_TENSOR_COLUMNS,
    STAGING_FIELDS,
    ChainPrefixCache,
    TQTokenSink,
    TQTokenSource,
    resolve_admission_prefix,
)
from nemo_rl.models.generation.megatron.token_capture import (  # noqa: E402
    TQMegatronPromptPreparer,
    TQMegatronTokenStager,
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
    assert client.gets[-1] == STAGING_FIELDS + MEDIA_METADATA_FIELDS
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


def _split_delta(record) -> tuple[list[int], list[int], list[float]]:
    """Split a fixture record's delta into (prompt ids, generated ids, logprobs)."""
    generated_start = record.token_mask_delta.index(1.0)
    return (
        record.token_ids_delta[:generated_start],
        record.token_ids_delta[generated_start:],
        record.generation_log_probs_delta[generated_start:],
    )


def _example_admission(record, parent_coords=None):
    """The Gym admission for a fixture record; a child chains off its parent's coords."""
    if parent_coords is None:
        return nemo_gym.CaptureAdmission(
            rollout_id=record.rollout_id,
            model_call_id=record.model_call_id,
            mode="text",
        )
    return nemo_gym.CaptureAdmission(
        rollout_id=record.rollout_id,
        model_call_id=record.model_call_id,
        parent_call_id=record.parent_call_id,
        prev_len=record.prev_len,
        mode="token_in",
        staging_chain=[parent_coords["staging_key"]],
        parent_chain_hash=parent_coords["chain_hash"],
    )


def _rerendered_turn2_template(root, child) -> tuple[list[int], list[int], int]:
    """Turn 2 as the chat template renders it: (template prefix, template, EOS).

    The template re-renders turn 1's assistant text with a different id (90 in
    place of the staged ids) and closes it with EOS; the staged ids must win.
    """
    turn1_prompt, _, _ = _split_delta(root)
    eos_token_id = root.token_ids_delta[-1]
    template_prefix = turn1_prompt + [90, eos_token_id]
    turn2_prompt, _, _ = _split_delta(child)
    return template_prefix, template_prefix + turn2_prompt, eos_token_id


def _stage_example_through_megatron(
    tq_client, partition, root, child, *, policy_epoch
) -> list[dict]:
    """Drive the two-turn rollout through the Megatron glue; return both coords.

    Turn 2 goes ``prepare_prompt`` -> ``stage`` through the ``offload_params``
    the preparer returned, the one Megatron-only handoff.
    """
    stager = TQMegatronTokenStager(TQTokenSink(tq_client, staging_partition=partition))
    preparer = TQMegatronPromptPreparer(
        TQTokenSource(tq_client, staging_partition=partition)
    )

    def stage(record, prompt, offload_params):
        _, generated, logprobs = _split_delta(record)
        result = stager.stage(
            f"minf-{record.model_call_id}",
            SimpleNamespace(
                prompt_token_ids=prompt,
                generated_token_ids=generated,
                generated_log_probs=logprobs,
            ),
            finished_metadata=SimpleNamespace(policy_epoch=policy_epoch),
            offload_params=offload_params,
        )
        assert result is not None
        return result.response_metadata["ng_commit_coords"]

    turn1_prompt, _, _ = _split_delta(root)
    root_coords = stage(
        root,
        turn1_prompt,
        {"ng_capture": _example_admission(root).model_dump(mode="json")},
    )
    template_prefix, template, eos_token_id = _rerendered_turn2_template(root, child)
    prepared = preparer.prepare_prompt(
        template,
        offload_params={
            "ng_capture": _example_admission(child, root_coords).model_dump(
                mode="json"
            ),
            PREFIX_TEMPLATE_TOKEN_IDS_FIELD: template_prefix,
            PREFIX_EOS_TOKEN_ID_FIELD: eos_token_id,
        },
    )
    return [root_coords, stage(child, prepared.prompt, prepared.offload_params)]


def _stage_example_through_vllm(
    tq_client, partition, root, child, *, weight_versions
) -> list[dict]:
    """Drive the two-turn rollout through the vLLM worker's glue; return both coords.

    ``weight_versions`` is (version at begin, version at finish); they differ
    when a refit lands while the call is in flight.
    """
    from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )
    from tests.unit.models.generation.test_vllm_token_capture_hosting import (
        _FakeRequest,
        _served_content,
        _worker_with_capture,
    )

    worker = _worker_with_capture(TQTokenSink(tq_client, staging_partition=partition))
    worker._chain_prefix.install(TQTokenSource(tq_client, staging_partition=partition))
    admitted_version, finished_version = weight_versions

    def call(record, request, prompt, **begin_kwargs):
        _, generated, logprobs = _split_delta(record)
        worker._rollout_weight_version = admitted_version
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, prompt, **begin_kwargs
        )
        worker._rollout_weight_version = finished_version
        content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
            worker, request, _served_content(generated, logprobs)
        )
        return content["ng_commit_coords"]

    turn1_prompt, _, _ = _split_delta(root)
    root_coords = call(
        root,
        _FakeRequest(
            ng_capture=_example_admission(root).model_dump(mode="json"), stream=False
        ),
        turn1_prompt,
    )
    request = _FakeRequest(
        ng_capture=_example_admission(child, root_coords).model_dump(mode="json"),
        stream=False,
    )
    admission = worker._capture_admission(request)
    prefix = worker._resolve_admission_prefix(admission)
    template_prefix, template, eos_token_id = _rerendered_turn2_template(root, child)
    # preprocess_chat's splice, with the prefix the worker resolved through TQ.
    prompt = replace_prefix_tokens(
        None, prefix, template_prefix, template, eos_token_id=eos_token_id
    )
    return [
        root_coords,
        call(child, request, prompt, admission=admission, prefix_token_ids=prefix),
    ]


@pytest.mark.parametrize("refit_mid_request", [False, True], ids=["steady", "refit"])
@pytest.mark.parametrize("backend", ["megatron", "vllm"])
def test_backend_capture_glue_reproduces_the_gym_worked_example(
    tq_client, staging_partition, backend, refit_mid_request
):
    """Both backends must stage the same rows for the same generation.

    Each backend's real glue is driven through Gym's ``worked_example`` as one
    two-turn rollout (text root, then a ``staging_chain`` continuation whose
    turn-1 tokens the chat template re-rendered) and must reproduce the
    fixture's staged records byte-for-byte, digests included, plus the matching
    commit coordinates. Reproducing one golden artifact is what makes the
    backends interchangeable, and a bug they share still fails.
    ``test_finalize_rollout_reproduces_the_golden_row`` finalizes exactly these
    records, so the finalizer is covered transitively.

    With ``refit`` the weights change while each call is in flight; both
    backends must still stamp the version the call was admitted under.
    """
    records, receipt, _ = build_fixture_artifacts("worked_example")
    root, child = records
    version = root.weight_version

    if backend == "megatron":
        # The engine reports the admission epoch plus one boundary per refit.
        policy_epoch = [(0, version)]
        if refit_mid_request:
            policy_epoch.append((2, version + 1))
        coords = _stage_example_through_megatron(
            tq_client, staging_partition, root, child, policy_epoch=policy_epoch
        )
    else:
        coords = _stage_example_through_vllm(
            tq_client,
            staging_partition,
            root,
            child,
            weight_versions=(version, version + int(refit_mid_request)),
        )

    # The manifest row also carries attribution-only fields (response_id,
    # admitted_at, the fingerprints) that commit coords never report, so
    # compare on the fields both models share.
    coord_fields = set(receipt.manifest[0].model_fields) & set(
        nemo_gym.CommitCoords.model_fields
    )
    assert [c["disposition"] for c in coords] == ["staged", "staged"]
    assert [{name: c[name] for name in coord_fields} for c in coords] == [
        manifest.model_dump(include=coord_fields) for manifest in receipt.manifest
    ]
    rows = TQTokenSource(tq_client, staging_partition=staging_partition).fetch(
        [record.staging_key for record in records]
    )
    assert [row.model_dump() for row in rows] == [
        record.model_dump(exclude={"extras"}) for record in records
    ]


@pytest.mark.parametrize("prefix_source", ["staging_chain", "capture_admission"])
def test_megatron_prompt_preparer_splices_resolved_prefix(
    tq_client, staging_partition, prefix_source
):
    admission_kwargs = {}
    if prefix_source == "staging_chain":
        stager = TQMegatronTokenStager(
            TQTokenSink(tq_client, staging_partition=staging_partition)
        )
        root = nemo_gym.CaptureAdmission(
            rollout_id="minf-r0", model_call_id="c1", mode="text"
        )
        root_result = stager.stage(
            "minf-response-1",
            SimpleNamespace(
                prompt_token_ids=[10, 11],
                generated_token_ids=[12, 99],
                generated_log_probs=[-0.25, -0.5],
            ),
            finished_metadata=SimpleNamespace(policy_epoch=[(0, 7)]),
            offload_params={"ng_capture": root.model_dump(mode="json")},
        )
        assert root_result is not None
        root_coords = root_result.response_metadata["ng_commit_coords"]
        admission_kwargs = {
            "staging_chain": [root_coords["staging_key"]],
            "parent_chain_hash": root_coords["chain_hash"],
        }
    else:
        admission_kwargs = {
            "required_prefix_token_ids": [10, 11, 12, 99],
            "parent_chain_hash": _digest("chain:c1"),
        }

    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=4,
        mode="token_in",
        **admission_kwargs,
    )
    preparer = TQMegatronPromptPreparer(
        TQTokenSource(tq_client, staging_partition=staging_partition)
    )

    result = preparer.prepare_prompt(
        [80, 81, 99, 20, 21],
        offload_params={
            "ng_capture": admission.model_dump(mode="json"),
            PREFIX_TEMPLATE_TOKEN_IDS_FIELD: [80, 81, 99],
            PREFIX_EOS_TOKEN_ID_FIELD: 99,
        },
    )

    assert result.prompt == [10, 11, 12, 99, 20, 21]
    assert result.offload_params is not None
    assert result.offload_params["ng_capture"]["required_prefix_token_ids"] == [
        10,
        11,
        12,
        99,
    ]


def test_megatron_stager_stamps_admission_epoch_when_request_spans_refit(
    tq_client, staging_partition, caplog
):
    """A request straddling a refit is stamped with its admission epoch, not masked.

    Mirrors vLLM, which freezes the version at begin_call. The engine stamps the
    admission epoch first and appends a boundary per refit, so epochs only grow.
    """
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
    admission = nemo_gym.CaptureAdmission(
        rollout_id="minf-r0",
        model_call_id="c1",
        mode="text",
    )
    with caplog.at_level(
        "WARNING", logger="nemo_rl.models.generation.megatron.token_capture"
    ):
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
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
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
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
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
    """Stand-in for TQTokenSource: records fetched keys, returns 2 tokens per key."""

    def __init__(self):
        self.calls = []

    def fetch_prefix_token_ids(self, keys):
        self.calls.append(list(keys))
        return [int(k[1:]) * 10 + i for k in keys for i in range(2)]


def test_chain_prefix_cache_fetches_only_uncached_suffix():
    source = _RecordingSource()
    cache = ChainPrefixCache(source)

    assert cache.fetch(["k1", "k2"]) == [10, 11, 20, 21]
    assert cache.fetch(["k1", "k2", "k3"]) == [10, 11, 20, 21, 30, 31]
    assert cache.fetch(["k1", "k2"]) == [10, 11, 20, 21]
    assert source.calls == [["k1", "k2"], ["k3"]]


def test_chain_prefix_cache_requires_an_installed_source():
    cache = ChainPrefixCache()
    with pytest.raises(RuntimeError, match="setup_token_capture"):
        cache.fetch(["k1"])
    source = _RecordingSource()
    cache.install(source)
    assert cache.fetch(["k1"]) == [10, 11]


def test_chain_prefix_cache_evicts_oldest_insertion_past_256_entries():
    source = _RecordingSource()
    cache = ChainPrefixCache(source)
    for i in range(257):
        cache.fetch([f"k{i}"])
    # k0 was the first insertion and is gone; k1 is still a hit.
    calls_before = len(source.calls)
    cache.fetch(["k1"])
    assert len(source.calls) == calls_before
    cache.fetch(["k0"])
    assert len(source.calls) == calls_before + 1


def test_resolve_admission_prefix_dispatches_like_the_vllm_worker():
    source = _RecordingSource()
    cache = ChainPrefixCache(source)
    text = SimpleNamespace(mode="text", staging_chain=[], required_prefix_token_ids=[])
    inline = SimpleNamespace(
        mode="token_in", staging_chain=[], required_prefix_token_ids=[7, 8]
    )
    chained = SimpleNamespace(
        mode="token_in", staging_chain=["k1"], required_prefix_token_ids=[]
    )

    assert resolve_admission_prefix(text, cache) == []
    assert resolve_admission_prefix(inline, cache) == [7, 8]
    assert resolve_admission_prefix(chained, cache) == [10, 11]
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
