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
    COMPACT_PREV_LEN_KEY,
    COMPACT_TOKEN_IDS_EXTRAS_KEY,
    MEDIA_EXTRAS_KEY,
    MEDIA_IMGS_FIELD,
    MEDIA_PREV_COUNT_KEY,
    MEDIA_STAGING_FIELDS,
    MINF_CAPTURE_PARAMS_FIELD,
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    STAGING_FIELDS,
    ChainPrefixCache,
    PrefixChains,
    TQMegatronPromptPreparer,
    TQMegatronTokenStager,
    TQTokenSink,
    TQTokenSource,
    _MegatronCapturePayload,
    media_field_dict,
    resolve_admission_prefix,
    row_to_media_tensors,
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


def _stage_root(tq_client, staging_partition, payload, *, rollout_id="minf-r0"):
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
    root = nemo_gym.CaptureAdmission(
        rollout_id=rollout_id, model_call_id="c1", mode="text"
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
    tq_client, staging_partition, multimodal
):
    """The expanded delta is the canonical row. A VLM call also stages the compact
    delta in its own column, the media geometry in the extras JSON, and the engine's
    media tensors as extra columns on the same key; a text call stages none of those
    and its compact chain falls back to the expanded one."""
    _, coords = _stage_root(
        tq_client, staging_partition, _minf_payload(multimodal=multimodal)
    )
    assert coords["staging_key"] == "minf-r0/c1"
    assert coords["weight_version"] == 7
    assert coords["disposition"] == "staged"
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    [fetched] = source.fetch_for_finalization(["minf-r0/c1"])
    chains = source.fetch_prefix_chains(["minf-r0/c1"])
    assert source.fetch_prefix_token_ids(["minf-r0/c1"]) == chains.expanded

    if not multimodal:
        assert fetched.snapshot.token_ids_delta == [10, 11, 12, 13]
        assert fetched.snapshot.token_mask_delta == [0.0, 0.0, 1.0, 1.0]
        assert fetched.snapshot.generation_log_probs_delta == [0.0, 0.0, -0.25, -0.5]
        assert fetched.extras is None
        assert chains.compact == chains.expanded == [10, 11, 12, 13]
        with pytest.raises(KeyError, match="media columns"):
            source.fetch_media("minf-r0/c1")
        return

    assert fetched.snapshot.token_ids_delta == [80, 99, 99, 99, 81, 12, 2]
    # Mask and log probs are aligned with the *expanded* delta: every media
    # token is prompt (mask 0); only the two generated tokens train.
    assert fetched.snapshot.token_mask_delta == [0.0] * 5 + [1.0, 1.0]
    assert fetched.snapshot.generation_log_probs_delta == [0.0] * 5 + [-0.25, -0.5]
    assert (coords["prev_len"], coords["delta_len"]) == (0, 7)
    assert chains.expanded == [80, 99, 99, 99, 81, 12, 2]
    assert chains.compact == [80, 99, 81, 12, 2]
    assert fetched.extras == {
        "media": {
            "modality": "image",
            "imgs_sizes": [[4, 4]],
            "num_frames": None,
            "num_tiles": None,
        }
    }
    media = source.fetch_media("minf-r0/c1")
    assert torch.equal(media.imgs, _minf_media_tensors()["imgs"])
    assert media.imgs.dtype == torch.float32
    assert media.imgs_sizes.tolist() == [[4, 4]]
    assert media.num_frames is None and media.num_tiles is None


@pytest.mark.parametrize(
    ("media_tensors", "error"),
    [
        ({"imgs_sizes": torch.tensor([[4, 4]])}, "non-empty imgs"),
        (
            {"imgs": torch.ones(1, 2, 3), "pixel_values": torch.ones(1)},
            "unsupported media tensors",
        ),
    ],
)
def test_media_field_dict_rejects_malformed_tensors(media_tensors, error):
    with pytest.raises(ValueError, match=error):
        media_field_dict(media_tensors)


@pytest.mark.parametrize(
    "media_tensors",
    [
        # Packed patches, one still image.
        {
            "imgs": torch.arange(48, dtype=torch.float32).reshape(1, 4, 12),
            "imgs_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
        },
        # Packed patches, bf16 as the engine may hand them over.
        {
            "imgs": torch.arange(48, dtype=torch.float32)
            .reshape(1, 4, 12)
            .to(torch.bfloat16),
            "imgs_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
        },
        # Video: per-frame sizes plus frames per video.
        {
            "imgs": torch.arange(96, dtype=torch.float32).reshape(1, 8, 12),
            "imgs_sizes": torch.tensor([[4, 4], [4, 4]], dtype=torch.int32),
            "num_frames": torch.tensor([2], dtype=torch.int32),
        },
        # Padded pixels with static tiling.
        {
            "imgs": torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(
                2, 3, 4, 4
            ),
            "num_tiles": torch.tensor([2], dtype=torch.int64),
        },
    ],
    ids=["packed-f32", "packed-bf16", "video", "tiled-pixels"],
)
def test_media_column_codec_round_trips_without_a_store(media_tensors):
    """``row_to_media_tensors`` inverts ``media_field_dict`` exactly, dtype included."""
    fields = media_field_dict(media_tensors)
    assert set(fields) == set(MEDIA_STAGING_FIELDS)
    assert all(tensor.shape[0] == 1 for tensor in fields.values())

    restored = row_to_media_tensors(fields)

    for name in ("imgs", "imgs_sizes", "num_frames", "num_tiles"):
        expected = media_tensors.get(name)
        actual = getattr(restored, name)
        if expected is None:
            assert actual is None, name
        else:
            assert actual.dtype == expected.dtype, name
            assert torch.equal(actual, expected), name


def test_row_to_media_tensors_rejects_geometry_column_disagreement():
    fields = media_field_dict(
        {
            "imgs": torch.ones(1, 4, 12),
            "imgs_sizes": torch.tensor([[4, 4]], dtype=torch.int32),
        }
    )
    # The columns are outside Gym's digest; a truncated column must not decode.
    fields[MEDIA_IMGS_FIELD] = fields[MEDIA_IMGS_FIELD][:, :-1]
    with pytest.raises(ValueError, match="geometry describes"):
        row_to_media_tensors(fields)


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
    tq_client, staging_partition, prefix_source
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
    stager = None
    if root_payload is not None:
        stager, root_coords = _stage_root(tq_client, staging_partition, root_payload)
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
        TQTokenSource(tq_client, staging_partition=staging_partition)
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

    if prefix_source != "multimodal_chain":
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
    source = TQTokenSource(tq_client, staging_partition=staging_partition)
    chains = source.fetch_prefix_chains(
        [root_coords["staging_key"], coords2["staging_key"]]
    )
    assert chains.expanded == [80, 99, 99, 99, 81, 12, 2, 20, 99, 99, 99, 21, 30]
    assert chains.compact == [80, 99, 81, 12, 2, 20, 99, 21, 30]
    assert chains.media_count == 2
    # Turn 2's row holds only the media new to it.
    media2 = source.fetch_media(coords2["staging_key"])
    assert torch.equal(media2.imgs, two_images["imgs"][:, 4:, :])
    assert media2.imgs_sizes.tolist() == [[4, 4]]
    [fetched2] = source.fetch_for_finalization([coords2["staging_key"]])
    assert fetched2.extras["media"]["imgs_sizes"] == [[4, 4]]


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
    stager = TQMegatronTokenStager(
        TQTokenSink(tq_client, staging_partition=staging_partition)
    )
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

    assert cache.fetch(["k1", "k2"]) == [10, 11, 20, 21]
    assert cache.fetch(["k1", "k2", "k3"]) == [10, 11, 20, 21, 30, 31]
    assert cache.fetch(["k1", "k2"]) == [10, 11, 20, 21]
    assert source.calls == [["k1", "k2"], ["k3"]]
    assert cache.fetch_chains(["k1", "k2", "k3"]) == PrefixChains(
        expanded=[10, 11, 20, 21, 30, 31], compact=[10, 20, 30]
    )
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
    """Pin the Gym extras keys and payload-view attribute names to Gym's constants.

    The sink pops Gym's extras keys, and the stager's payload view feeds Gym's
    Megatron adapter by attribute name.
    """
    media = pytest.importorskip("nemo_gym.token_id_capture.staging.media")
    adapter = pytest.importorskip("nemo_gym.token_id_capture.adapters.megatron")
    if not hasattr(adapter, "COMPACT_PREV_LEN_FIELD"):
        pytest.skip(
            "pinned nemo_gym predates multimodal Megatron extras (NVIDIA-NeMo/Gym#2823)"
        )
    assert COMPACT_TOKEN_IDS_EXTRAS_KEY == media.COMPACT_TOKEN_IDS_DELTA_FIELD
    assert MEDIA_EXTRAS_KEY == media.MEDIA_FIELD
    assert COMPACT_PREV_LEN_KEY == adapter.COMPACT_PREV_LEN_FIELD
    payload_fields = {f.name for f in dataclasses.fields(_MegatronCapturePayload)}
    assert payload_fields >= {
        adapter.PROMPT_IDS_FIELD,
        adapter.GENERATED_IDS_FIELD,
        adapter.GENERATED_LOGPROBS_FIELD,
        adapter.COMPACT_PROMPT_IDS_FIELD,
        adapter.COMPACT_PREV_LEN_FIELD,
        adapter.MEDIA_FIELD,
    }


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
        ({"imgs": torch.ones(1, 4, 12)}, 1, "requires imgs_sizes or num_tiles"),
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
