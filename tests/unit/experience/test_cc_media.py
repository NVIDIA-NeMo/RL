"""Processed-media storage composition; processor and storage boundaries doubled."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from nemo_gym.token_id_capture.staging.records import RolloutManifest, RolloutReceipt
from PIL import Image
from tensordict import TensorDict

from nemo_rl.data.multimodal_utils import (
    PACKED_MULTIMODAL_FIELDS,
    attach_image_model_inputs_to_message,
    image_to_data_url,
    reassemble_packed_multimodal,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.schema import ROUTED_EXTRAS_METADATA_FIELD
from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS, TQTokenSink, TQTokenSource
from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.experience.cc_media import fetch_segment_media, stage_segment_media
from nemo_rl.experience.rollout_reassembler import RolloutReassembler, SegmentReceipt
from nemo_rl.experience.rollout_reassembler_actor import assert_metadata_only
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from tests.unit.data_plane.token_capture_test_fixtures import (
    _manifest,
    build_fixture_artifacts,
)
from tests.unit.experience.test_cc_routes import _RecordingSink
from tests.unit.experience.test_logical_owner_finalization import finalize
from tests.unit.models.generation.test_cc_image_geometry import image_prompt
from tests.unit.models.generation.test_vllm_token_capture_hosting import (
    _FakeRequest,
    _served_content,
    _worker_with_capture,
)

pytestmark = pytest.mark.nemo_gym


class NemotronH_Nano_Omni_Reasoning_V3Processor:
    """Small deterministic processor with the existing dynamic-image API shape."""

    image_token = "<image>"
    model_input_names = ["input_ids", "pixel_values"]
    tokenizer = SimpleNamespace(model_input_names=["input_ids"])

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *, text, images, return_tensors):
        self.calls.append((text, [image.size for image in images], return_tensors))
        assert text == self.image_token * len(images)
        pixels = [
            torch.full(
                (3, image.height, image.width),
                image.getpixel((0, 0))[0],
                dtype=torch.float32,
            )
            for image in images
        ]
        return {
            "input_ids": torch.tensor([[99] * len(images)]),
            "pixel_values": pixels if return_tensors is None else torch.stack(pixels),
        }


class RecordingPlane(NoOpDataPlaneClient):
    def __init__(self):
        super().__init__()
        self.put_count = 0
        self.lose_ack = False

    def put_samples(self, *args, **kwargs):
        self.put_count += 1
        result = super().put_samples(*args, **kwargs)
        if self.lose_ack:
            raise OSError("media written but acknowledgement lost")
        return result


@pytest.fixture
def media_stack():
    plane = RecordingPlane()
    plane.register_partition(
        "staged", [*STAGING_FIELDS, *sorted(PACKED_MULTIMODAL_FIELDS)], 8, ["finalize"]
    )
    plane.register_partition("canonical", [], 8, ["train"])
    records, _, _ = build_fixture_artifacts("single_call", rollout_id="group_g0_s0")
    assert TQTokenSink(plane, staging_partition="staged").stage(records[0]).ok
    source = TQTokenSource(plane, staging_partition="staged")
    images = {
        "wide": Image.new("RGB", (4, 2), (11, 0, 0)),
        "tall": Image.new("RGB", (2, 4), (22, 0, 0)),
    }
    assets = {
        key: {
            "media_id": key,
            "source_part": {
                "type": "input_image",
                "image_url": image_to_data_url(image),
            },
        }
        for key, image in images.items()
    }
    yield plane, source, records[0], assets, images
    for image in images.values():
        image.close()


def _stage(stack, processor, occurrences):
    plane, _, record, assets, _ = stack
    return stage_segment_media(
        plane,
        staging_partition="staged",
        terminal_staging_key=record.staging_key,
        media_assets=assets,
        action_occurrences=occurrences,
        processor=processor,
        pad_dynamic_image_shapes=True,
    )


def _fetch(stack, descriptor):
    plane, _, record, _, _ = stack
    return fetch_segment_media(
        plane,
        staging_partition="staged",
        terminal_staging_key=record.staging_key,
        descriptor=descriptor,
    )


def test_media_uses_one_existing_key_and_preserves_tokens_and_true_geometry(
    media_stack,
):
    plane, source, record, _, images = media_stack
    before = source.fetch([record.staging_key])
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    descriptor = _stage(media_stack, processor, [["wide"], [], ["tall", "wide"]])
    assert descriptor.occurrence_counts == (1, 0, 2)
    assert_metadata_only(descriptor)
    assert plane.put_count == 2  # token capture, then one media-column update
    assert plane.list_sample_ids("staged") == [record.staging_key]
    assert source.fetch([record.staging_key]) == before
    loaded = _fetch(media_stack, descriptor)
    assert loaded["imgs_sizes"].as_tensor().tolist() == [[2, 4], [4, 2], [2, 4]]
    assert loaded["num_frames"].as_tensor().tolist() == [1, 1, 1]
    assert processor.calls == [
        ("<image>", [(4, 2)], "pt"),
        ("<image><image>", [(2, 4), (4, 2)], None),
    ]
    # Reference is #3910's existing per-action attach, not a second image algorithm.
    reference = []
    for group in ([images["wide"]], [images["tall"], images["wide"]]):
        message = {}
        attach_image_model_inputs_to_message(
            message, images=group, processor=processor, pad_dynamic_image_shapes=True
        )
        reference.append(message)
    for key, value in loaded.items():
        assert len(value) == 1
        expected_segments = [
            tensor for message in reference for tensor in message[key].tensors
        ]
        assert value.row_shapes() == [
            [list(tensor.shape) for tensor in expected_segments]
        ]
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(value.tensors, expected_segments, strict=True)
        )
    # Ordinary receipt ownership clears tokens and media together.
    TQTokenSink(plane, staging_partition="staged").clear([record.staging_key])
    assert plane.list_sample_ids("staged") == []


def test_media_empty_actions_do_not_process_or_write(media_stack):
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    assert _stage(media_stack, processor, [[], []]) is None
    assert processor.calls == [] and media_stack[0].put_count == 1


def test_unknown_media_fails_before_any_media_write(media_stack):
    with pytest.raises(KeyError):
        _stage(
            media_stack,
            NemotronH_Nano_Omni_Reasoning_V3Processor(),
            [["wide"], ["missing"]],
        )
    assert media_stack[0].put_count == 1


def test_uncertain_media_put_is_not_retried_or_cleaned_up(media_stack):
    plane, source, record, _, _ = media_stack
    plane.lose_ack = True
    with pytest.raises(OSError, match="acknowledgement lost"):
        _stage(media_stack, NemotronH_Nano_Omni_Reasoning_V3Processor(), [["wide"]])
    assert plane.put_count == 2
    assert source.fetch([record.staging_key])[0].digest == record.digest
    assert (
        plane.get_samples([record.staging_key], "staged", ["pixel_values"])[
            "pixel_values"
        ].numel()
        > 0
    )


@pytest.mark.parametrize("mutation", ["field", "duplicate", "shape"])
def test_media_read_requires_registered_columns_and_shape_metadata(
    media_stack, mutation
):
    descriptor = _stage(
        media_stack, NemotronH_Nano_Omni_Reasoning_V3Processor(), [["wide"]]
    )
    if mutation == "field":
        descriptor = replace(descriptor, field_names=("token_ids_delta",))
    elif mutation == "duplicate":
        descriptor = replace(descriptor, field_names=("pixel_values", "pixel_values"))
    else:
        descriptor = replace(descriptor, row_tags={})
    with pytest.raises(ValueError):
        _fetch(media_stack, descriptor)


@pytest.mark.parametrize("outcome", ["completed", "execution_failure"])
def test_env_stages_selected_media_and_returns_metadata_only(media_stack, outcome):
    plane, _, record, assets, _ = media_stack
    env = object.__new__(NemoGym.__ray_metadata__.modified_class)
    env.cfg = {"pad_dynamic_image_shapes": True}
    env._media_dp_client = plane
    env._media_staging_partition = "staged"
    env._processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    manifest = RolloutManifest(
        rollout_id=record.rollout_id, records=[_manifest(record)]
    )
    env._control = AsyncMock(return_value=manifest.model_dump())
    response_id = manifest.records[0].response_id
    result = {
        "reward": 1.0,
        "response": {"id": response_id},
        "context_compaction_result": {
            "logical_rollout_id": "group_g0",
            "outcome": outcome,
            "media_assets": assets,
            "segments": [
                {
                    "capture_rollout_id": "group_g0_s0",
                    "segment_index": 0,
                    "media_occurrence_refs": ["wide"],
                    "selected_actions": [
                        {
                            "response_id": response_id,
                            "finish_reason": "stop",
                            "last_output_item": None,
                            "new_media_occurrence_refs": ["wide"],
                        }
                    ],
                }
            ],
        },
    }
    processed = asyncio.run(
        env._postprocess_cc_receipt_mode({"_ng_rollout_id": "group_g0"}, result)
    )
    assert_metadata_only(processed)
    segment = processed["logical_segments"][0]
    assert (segment.media is None) == (outcome == "execution_failure")
    assert plane.put_count == (1 if outcome == "execution_failure" else 2)
    if segment.media is not None:
        assert _fetch(media_stack, segment.media)[
            "imgs_sizes"
        ].as_tensor().tolist() == [[2, 4]]
        finalizer = RolloutReassembler(
            plane,
            partition_id="canonical",
            staging_partition="staged",
            pad_token_id=0,
            max_seq_len=1024,
        )
        # This legacy fixture did not capture engine geometry: staging alone
        # must not qualify its image training row.
        with pytest.raises(ValueError, match="no verified input layout"):
            finalizer.finalize_group(
                "group",
                ["group_g0"],
                [None],
                [1.0],
                mask_sample=[False],
                fallback_weight_version=0,
                prompt_idx=0,
                logical_segments=[[segment]],
            )
        assert not plane.list_sample_ids("canonical")


def _captured_image_segment(stack, scope, actions, *, corrupt=None, routes=False):
    plane, _, _, assets, images = stack
    sink = _RecordingSink(plane)
    worker = _worker_with_capture(sink)
    worker._cc_capture_enabled = True
    prefix, spans, items = [], [], []
    parent_hash = None
    for index, occurrences in enumerate(actions):
        new = image_prompt(
            tuple((images[key].height, images[key].width) for key in occurrences)
        )
        for span in new["mm_placeholders"]["image"]:
            span.offset += len(prefix)
        spans += new["mm_placeholders"]["image"]
        items += new["mm_kwargs"]["image"]
        prompt = prefix + new["prompt_token_ids"]
        # Same embedding count/area, but a changed retained image at turn 2.
        if corrupt == "retained" and index == 1:
            data = items[0].get_data()
            data["imgs_sizes"] = (4, 2)
            data["pixel_values_flat"] = torch.zeros(3, 4, 2)
        admission = dict(
            rollout_id=scope,
            model_call_id=f"c{index}",
            mode="token_in" if index else "text",
        )
        if index:
            admission.update(
                parent_call_id=f"c{index - 1}",
                prev_len=len(prefix),
                required_prefix_token_ids=prefix,
                parent_chain_hash=parent_hash,
            )
        request = _FakeRequest(ng_capture=admission, stream=False)
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker,
            request,
            prompt,
            engine_prompt=dict(
                prompt_token_ids=prompt,
                mm_placeholders={"image": spans},
                mm_kwargs={"image": items},
            ),
        )
        if index == 0 and corrupt in ("generated_span", "wrong_token_span"):
            # Corrupt before commitment: these must fail alignment, not merely
            # extras authentication. Generated-token spans are never prompt media.
            geometry = worker._capture_calls[id(request)][2]
            geometry["images"][0].update(
                offset=len(prompt) if corrupt == "generated_span" else 0, length=1
            )
            if corrupt == "generated_span":
                geometry["images"][0]["token_id"] = 31
        payload = _served_content([31], [-0.25])
        if routes:
            payload["choices"][0]["message"]["routed_experts"] = [
                [[index * 100 + position]] for position in range(len(prompt))
            ] + [[[-1]]]
        response = VllmAsyncGenerationWorkerImpl._finish_request_capture(
            worker, request, payload
        )
        assert response["ng_commit_coords"]["disposition"] == "staged"
        parent_hash = response["ng_commit_coords"]["chain_hash"]
        prefix = prompt + [31]
    descriptor = stage_segment_media(
        plane,
        staging_partition="staged",
        terminal_staging_key=sink.records[-1].staging_key,
        media_assets=assets,
        action_occurrences=actions,
        processor=NemotronH_Nano_Omni_Reasoning_V3Processor(),
        pad_dynamic_image_shapes=True,
    )
    records = [
        _manifest(record).model_copy(
            update={"response_id": f"{scope}-{record.model_call_id}"}
        )
        for record in sink.records
    ]
    receipt = RolloutReceipt(
        rollout_id=scope,
        manifest=records,
        terminal_model_call_id=records[-1].model_call_id,
        terminal_selection="declared",
    )
    return SegmentReceipt(
        scope,
        receipt.model_dump(),
        tuple(record.response_id for record in records),
        media=descriptor,
    ), prefix


def _finalizer(stack, *, routes=False):
    return RolloutReassembler(
        stack[0],
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
        router_replay_enabled=routes,
    )


def test_real_capture_media_alignment_publication_and_owned_cleanup(media_stack):
    plane = media_stack[0]
    plane.clear_samples([media_stack[2].staging_key], "staged")
    first, first_tokens = _captured_image_segment(
        media_stack, "group_g0_s0", [["wide"], [], ["tall", "wide"]]
    )
    second, second_tokens = _captured_image_segment(
        media_stack, "group_g0_s1", [["tall"]]
    )
    # One sibling is text-only: canonical media must retain an empty row.
    sibling, _ = _captured_image_segment(media_stack, "group_g1_s0", [[]])
    for segment in (first, second, sibling):
        checked = _finalizer(media_stack).finalize_rollout(
            segment.capture_rollout_id,
            segment.receipt,
            reward=1.0,
            context_compaction=True,
            media=segment.media,
        )
        assert checked.valid, checked.rejection_reason
    result = finalize(
        _finalizer(media_stack), [[first, second], [sibling]], execution_row_multiple=4
    )
    assert result.valid_row_count == 3 and result.total_row_count == 3
    assert result.meta.sample_ids == [
        "group_g0_s0",
        "group_g0_s1",
        "group_g1_s0",
        "group_pad0",
    ]
    assert plane.list_sample_ids("staged") == []
    wire = plane.get_samples(
        result.meta.sample_ids, "canonical", ["input_ids", *first.media.field_names]
    )
    assert wire["input_ids"][0].tolist() == first_tokens
    assert wire["input_ids"][1].tolist() == second_tokens
    restored = {key: wire[key] for key in first.media.field_names}
    reassemble_packed_multimodal(restored, result.meta.tags)
    assert restored["imgs_sizes"].as_tensor().tolist() == [
        [2, 4],
        [4, 2],
        [2, 4],
        [4, 2],
        [2, 4],
        [4, 2],
        [2, 4],
    ]
    assert restored["pixel_values"].logical_segment_counts_by_row() == [2, 1, 0, 2]
    assert wire["input_ids"][3].tolist() == first_tokens


@pytest.mark.parametrize(
    "failure",
    [
        "retained",
        "count",
        "omitted",
        "geometry",
        "span",
        "digest",
        "missing",
        "generated_span",
        "wrong_token_span",
        "shape",
    ],
)
def test_bad_media_collapses_whole_owner_not_just_one_segment(media_stack, failure):
    plane = media_stack[0]
    plane.clear_samples([media_stack[2].staging_key], "staged")
    first, _ = _captured_image_segment(media_stack, "group_g0_s0", [["wide"]])
    bad, _ = _captured_image_segment(
        media_stack, "group_g0_s1", [["wide"], []], corrupt=failure
    )
    if failure == "count":
        bad = replace(bad, media=replace(bad.media, occurrence_counts=(0, 1)))
    elif failure == "omitted":
        bad = replace(bad, media=None)
    elif failure == "shape":
        tags = deepcopy(bad.media.row_tags)
        tags["pixel_values__row_shapes"]["shapes"][0][-1] = 5
        bad = replace(bad, media=replace(bad.media, row_tags=tags))
    elif failure in ("geometry", "span", "digest", "missing"):
        key = "group_g0_s1/c0"
        if failure == "geometry":
            plane.put_samples(
                ["group_g0_s1/c1"],
                "staged",
                TensorDict(
                    {"imgs_sizes": torch.tensor([[4, 2]], dtype=torch.int32)},
                    batch_size=[1],
                ),
                [{}],
            )
        else:
            raw = plane.get_samples([key], "staged", [ROUTED_EXTRAS_METADATA_FIELD])
            extra = json.loads(bytes(raw[ROUTED_EXTRAS_METADATA_FIELD][0].tolist()))
            if failure == "missing":
                extra.pop("cc_image_geometry")
            elif failure == "span":
                extra["cc_image_geometry"]["images"][0]["offset"] += 1
            else:
                extra["cc_image_geometry"]["prefix_digest"] = "0" * 64
            changed = torch.tensor(
                [list(json.dumps(extra).encode())], dtype=torch.uint8
            )
            plane.put_samples(
                [key],
                "staged",
                TensorDict({ROUTED_EXTRAS_METADATA_FIELD: changed}, batch_size=[1]),
                [{}],
            )
    sibling, _ = _captured_image_segment(media_stack, "group_g1_s0", [["tall"]])
    rejected = _finalizer(media_stack).finalize_rollout(
        bad.capture_rollout_id,
        bad.receipt,
        reward=1.0,
        context_compaction=True,
        media=bad.media,
    )
    assert not rejected.valid
    expected_reason = {
        "retained": "Retained engine image geometry changed",
        "count": "Worker/agent image occurrence counts disagree",
        "omitted": "Worker/agent image occurrence counts disagree",
        "geometry": "Training image geometry differs",
        "span": "extras commitment mismatch",
        "digest": "extras commitment mismatch",
        "missing": "extras commitment mismatch",
        "generated_span": "outside the new carried prompt",
        "wrong_token_span": "does not match exact captured tokens",
        "shape": "shape metadata does not match its payload",
    }[failure]
    assert expected_reason in rejected.rejection_reason
    for segment in (first, sibling):
        checked = _finalizer(media_stack).finalize_rollout(
            segment.capture_rollout_id,
            segment.receipt,
            reward=1.0,
            context_compaction=True,
            media=segment.media,
        )
        assert checked.valid, checked.rejection_reason
    result = finalize(_finalizer(media_stack), [[first, bad], [sibling]])
    assert result.valid_row_count == 1 and result.total_row_count == 2
    assert result.meta.sample_ids == ["group_g0_s0", "group_g1_s0"]
    masks = plane.get_samples(
        result.meta.sample_ids, "canonical", ["sample_mask", "token_mask"]
    )
    assert masks["sample_mask"].tolist() == [0, 1]
    assert not masks["token_mask"][0].any()
    wire = plane.get_samples(
        result.meta.sample_ids, "canonical", ["input_ids", *sibling.media.field_names]
    )
    torch.testing.assert_close(wire["input_ids"][0], wire["input_ids"][1])
    media = {key: wire[key] for key in sibling.media.field_names}
    # NoOp stacks equal-shaped rows; real TQ _from_wire deliberately retains
    # jagged packed-media fields, even for a batch of identical dummy images.
    media = {
        key: value
        if value.is_nested
        else torch.nested.as_nested_tensor(list(value.unbind()), layout=torch.jagged)
        for key, value in media.items()
    }
    reassemble_packed_multimodal(media, result.meta.tags)
    assert media["pixel_values"].logical_segment_counts_by_row() == [1, 1]
    torch.testing.assert_close(
        media["pixel_values"].slice([0]).as_tensor(),
        media["pixel_values"].slice([1]).as_tensor(),
    )
    assert plane.list_sample_ids("staged") == []


def test_image_and_route_extras_share_the_existing_commitment(media_stack):
    plane = media_stack[0]
    plane.clear_samples([media_stack[2].staging_key], "staged")
    segment, tokens = _captured_image_segment(
        media_stack, "group_g0_s0", [["wide"], ["tall"]], routes=True
    )
    result = finalize(_finalizer(media_stack, routes=True), [[segment]])
    assert result.valid_row_count == 1
    row = plane.get_samples(
        result.meta.sample_ids, "canonical", ["input_ids", "routed_experts"]
    )
    assert row["input_ids"][0].tolist() == tokens
    assert row["routed_experts"][0].shape == (len(tokens), 1, 1)
    # Root prompt has four tokens, then its generation. Turn 2 supplies the
    # previously missing route at exactly that generation's final token.
    assert row["routed_experts"][0][4, 0, 0].item() == 104
    assert plane.list_sample_ids("staged") == []


def test_uncertain_image_publication_retains_all_staging(media_stack):
    plane = media_stack[0]
    plane.clear_samples([media_stack[2].staging_key], "staged")
    segment, _ = _captured_image_segment(
        media_stack, "group_g0_s0", [["wide"], ["tall"]]
    )
    keys = plane.list_sample_ids("staged")
    put_count = plane.put_count
    plane.lose_ack = True
    with pytest.raises(OSError, match="acknowledgement lost"):
        finalize(_finalizer(media_stack), [[segment]], execution_row_multiple=4)
    assert plane.put_count == put_count + 1
    assert plane.list_sample_ids("staged") == keys
    assert plane.list_sample_ids("canonical") == [
        "group_g0_s0",
        "group_pad0",
        "group_pad1",
        "group_pad2",
    ]
