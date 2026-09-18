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
"""Worker-owned image capture through the real sink and ordinary finalizer."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureAdmission,
    RolloutReceipt,
)

from nemo_rl.data.captured_media import (
    IMAGE_CAPTURE_FIELD,
    attachment_pixels,
    capture_processed_images,
)
from nemo_rl.data.multimodal_utils import (
    extract_multimodal_model_inputs,
    reassemble_packed_multimodal,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS, ROUTED_EXTRAS_METADATA_FIELD
from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS, TQTokenSink, TQTokenSource
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)

pytestmark = pytest.mark.nemo_gym


@dataclass(frozen=True)
class Span:
    offset: int
    length: int
    is_embed: torch.Tensor | None = None


def engine_prompt(tokens, images=()):
    """Images are ordered (span, CHW pixels) pairs, like vLLM processor outputs."""
    items = []
    for span, pixels in images:
        data = {
            "pixel_values_flat": pixels,
            "imgs_sizes": list(pixels.shape[-2:]),
            "num_tokens_per_image": span.length
            if span.is_embed is None
            else int(span.is_embed.sum()),
        }
        items.append(SimpleNamespace(get_data=lambda data=data: data))
    return {
        "prompt_token_ids": tokens,
        "mm_placeholders": {"image": [span for span, _ in images]},
        "mm_kwargs": {"image": items},
    }


@pytest.fixture
def dp():
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="staging",
        fields=STAGING_FIELDS + ["pixel_values", "routed_experts"],
        num_samples=64,
        consumer_tasks=["finalize"],
    )
    client.register_partition(
        partition_id="train",
        fields=list(DP_TRAIN_FIELDS) + ["pixel_values", "imgs_sizes", "num_frames"],
        num_samples=64,
        consumer_tasks=["train"],
    )
    return client


def stage(
    dp, prompt, *, parent=None, retained=(), call_id="c1", rollout_id="r0", routes=False
):
    capture = RolloutTokenCapture(
        sink=TQTokenSink(dp, staging_partition="staging"),
        weight_version_fn=lambda: 3,
    )
    prev_len = parent.cum_len if parent is not None else 0
    descriptor, attachments = capture_processed_images(
        prompt, prev_len=prev_len, retained=retained
    )
    admission = CaptureAdmission(
        rollout_id=rollout_id,
        model_call_id=call_id,
        mode="text" if parent is None else "token_in",
        parent_call_id=parent.model_call_id if parent else None,
        prev_len=prev_len,
        required_prefix_token_ids=prompt["prompt_token_ids"][:prev_len],
        parent_chain_hash=parent.chain_hash if parent else None,
    )
    extras = {IMAGE_CAPTURE_FIELD: descriptor.to_dict()}
    if routes:
        extras["routed_experts"] = [[[0]]] * (
            len(prompt["prompt_token_ids"]) + 2 - prev_len
        )
    coords = capture.complete_call(
        capture.begin_call(admission),
        prompt_token_ids=prompt["prompt_token_ids"],
        generated_token_ids=[31, 2],
        generated_logprobs=[-0.25, -0.5],
        extras=extras,
        attachments=attachments,
    )
    assert coords.disposition == "staged"
    record = CallRecord(
        **coords.model_dump(exclude={"rollout_id", "disposition"}),
        mode=admission.mode,
        response_id=f"response-{call_id}",
    )
    return record, descriptor


def receipt(*records, rollout_id="r0", terminal=None):
    return RolloutReceipt(
        rollout_id=rollout_id,
        manifest=list(records),
        terminal_model_call_id=terminal or records[-1].model_call_id,
        terminal_selection="declared",
    ).model_dump()


def finalizer(dp, **kwargs):
    return RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=1000,
        capture_images=True,
        **kwargs,
    )


def test_two_turn_images_are_captured_once_and_survive_restart(dp, tmp_path):
    a = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    b = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    root, media = stage(dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]))
    prefix = [10, 18, 18, 11, 31, 2]
    child, _ = stage(
        dp,
        engine_prompt(prefix + [12, 18, 18, 11], [(Span(1, 2), a), (Span(7, 2), b)]),
        parent=root,
        retained=media.images,
        call_id="c2",
    )
    source = TQTokenSource(dp, staging_partition="staging")
    assert source.fetch_pixels(root.staging_key).numel() == a.numel()
    assert source.fetch_pixels(child.staging_key).numel() == b.numel()
    # Both the descriptor and pixels are restored with the ordinary call rows.
    dp.save_checkpoint(tmp_path / "checkpoint")
    restored = NoOpDataPlaneClient()
    restored.load_checkpoint(tmp_path / "checkpoint")
    row = finalizer(restored).finalize_rollout("r0", receipt(root, child), reward=1.0)
    assert row.valid, row.rejection_reason
    assert row.token_ids == prefix + [12, 18, 18, 11, 31, 2]
    assert row.token_mask == [0.0] * 4 + [1.0] * 2 + [0.0] * 4 + [1.0] * 2
    assert row.logprobs == [0.0] * 4 + [-0.25, -0.5] + [0.0] * 4 + [-0.25, -0.5]
    assert row.media["imgs_sizes"].as_tensor().tolist() == [[2, 3], [4, 2]]
    assert row.media["num_frames"].as_tensor().tolist() == [1, 1]
    pixels = row.media["pixel_values"].as_tensor()
    torch.testing.assert_close(pixels[0, :, :2, :3], a, rtol=0, atol=0)
    torch.testing.assert_close(pixels[1, :, :4, :2], b, rtol=0, atol=0)


def test_publication_packs_mixed_rows_and_cleans_all_call_media(dp):
    a = torch.ones(3, 2, 3)
    root, _ = stage(dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]))
    # An off-chain root is cleanup-owned but never contributes pixels.
    discarded, _ = stage(
        dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a * 9)]), call_id="discarded"
    )
    text, _ = stage(dp, engine_prompt([10, 11]), rollout_id="text")
    result = finalizer(dp).finalize_group(
        "g0",
        ["r0", "text", "bad"],
        [
            receipt(root, discarded, terminal="c1"),
            receipt(text, rollout_id="text"),
            None,
        ],
        [1.0, 0.0, 0.0],
        mask_sample=[False] * 3,
        fallback_weight_version=3,
        prompt_idx=0,
        canonical_sample_ids=["g0_g0", "g0_g1", "g0_g2"],
    )
    assert result.valid_row_count == 2
    fields = dp.get_samples(result.meta.sample_ids, "train", result.meta.fields)
    assert fields["sample_mask"].tolist() == [1.0, 1.0, 0.0]
    materialized = dict(fields)
    reassemble_packed_multimodal(materialized, result.meta.tags)
    assert materialized["pixel_values"].row_shapes() == [[[1, 3, 2, 3]], [], []]
    torch.testing.assert_close(materialized["pixel_values"].as_tensor(), a.unsqueeze(0))
    assert dp.list_sample_ids("staging") == []


@pytest.mark.parametrize("corruption", ["missing", "pixels", "descriptor"])
def test_missing_or_corrupt_media_rejects_rollout(dp, corruption):
    root, _ = stage(
        dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), torch.ones(3, 2, 3))])
    )
    stored = dp._partitions["staging"].rows[root.staging_key]
    if corruption == "missing":
        del stored["pixel_values"]
    elif corruption == "pixels":
        stored["pixel_values"][0] += 1
    else:
        stored[ROUTED_EXTRAS_METADATA_FIELD] = torch.tensor(
            list(b"null"), dtype=torch.uint8
        )
    row = finalizer(dp).finalize_rollout("r0", receipt(root), reward=1.0)
    assert not row.valid
    assert row.rejection_reason.startswith("image_assembly:")
    assert row.media == {}


def test_image_free_continuation_has_no_pixel_column(dp):
    a = torch.ones(3, 2, 3)
    root, media = stage(dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]))
    child, descriptor = stage(
        dp,
        engine_prompt([10, 18, 18, 11, 31, 2, 50], [(Span(1, 2), a)]),
        parent=root,
        retained=media.images,
        call_id="c2",
    )
    assert descriptor.images == ()
    assert "pixel_values" not in dp._partitions["staging"].rows[child.staging_key]
    row = finalizer(dp).finalize_rollout("r0", receipt(root, child), reward=1.0)
    assert row.valid, row.rejection_reason
    assert row.media["imgs_sizes"].as_tensor().tolist() == [[2, 3]]


def test_repeated_asset_occurrences_remain_distinct(dp):
    a = torch.ones(3, 2, 3)
    root, _ = stage(
        dp, engine_prompt([10, 18, 18, 11, 18, 18], [(Span(1, 2), a), (Span(4, 2), a)])
    )
    row = finalizer(dp).finalize_rollout("r0", receipt(root), reward=1.0)
    assert row.valid
    assert row.media["pixel_values"].as_tensor().shape == (2, 3, 2, 3)


def test_captured_inputs_match_existing_learner_conversion(dp):
    class NemotronH_Nano_Omni_Reasoning_V3Processor:
        model_input_names = ["input_ids", "pixel_values", "imgs_sizes"]
        tokenizer = SimpleNamespace(model_input_names=["input_ids"])

    pixels = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    root, _ = stage(dp, engine_prompt([18, 18], [(Span(0, 2), pixels)]))
    row = finalizer(dp).finalize_rollout("r0", receipt(root), reward=1.0)
    expected = extract_multimodal_model_inputs(
        NemotronH_Nano_Omni_Reasoning_V3Processor(),
        {
            "input_ids": torch.tensor([18, 18]),
            "pixel_values": pixels.unsqueeze(0),
            "imgs_sizes": torch.tensor([[2, 3]]),
        },
    )
    assert row.valid, row.rejection_reason
    assert set(row.media) == set(expected)
    for name in expected:
        torch.testing.assert_close(
            row.media[name].as_tensor(), expected[name].as_tensor(), rtol=0, atol=0
        )
        assert row.media[name].pad_to_max_shape == expected[name].pad_to_max_shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_snapshot_owns_storage_and_preserves_dtype(dtype):
    a = torch.arange(18, dtype=dtype).reshape(3, 2, 3)
    original = a.clone()
    descriptor, attachments = capture_processed_images(
        engine_prompt([18, 18], [(Span(0, 2), a)]), prev_len=0
    )
    a.zero_()
    restored = descriptor.decode_pixels(attachment_pixels(descriptor, attachments))[0]
    torch.testing.assert_close(restored, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "change", ["pixels", "length", "size", "dropped", "cache_reference"]
)
def test_changed_retained_images_fail_before_inference(change):
    a = torch.ones(3, 2, 3)
    first, _ = capture_processed_images(
        engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]), prev_len=0
    )
    image = (
        a + 1 if change == "pixels" else torch.ones(3, 3, 2) if change == "size" else a
    )
    span = Span(1, 1 if change == "length" else 2)
    prompt = engine_prompt(
        [10, 18, 18, 11, 31, 2, 50], [] if change == "dropped" else [(span, image)]
    )
    if change == "cache_reference":
        prompt["mm_kwargs"]["image"][0] = None
    with pytest.raises(ValueError):
        capture_processed_images(prompt, prev_len=6, retained=first.images)


def test_splice_coordinates_handle_reasoning_shift_and_repeated_pad_runs():
    a, b = torch.ones(3, 2, 3), torch.ones(3, 3, 2)
    original = [10, 18, 18, 11, 77, 78, 31, 2]
    first, _ = capture_processed_images(
        engine_prompt(original, [(Span(1, 2), a)]), prev_len=0
    )
    # The template drops old reasoning; the new image uses the same pad run.
    template = [10, 18, 18, 11, 31, 2, 12, 18, 18, 11]
    splice = splice_prefix_tokens(
        tokenizer=SimpleNamespace(eos_token_id=2),
        model_prefix_token_ids=original,
        template_prefix_token_ids=template[:6],
        template_token_ids=template,
    )
    prompt = engine_prompt(template, [(Span(1, 2), a), (Span(7, 2), b)])
    captured, _ = capture_processed_images(
        prompt, prev_len=len(original), retained=first.images, splice=splice
    )
    assert captured.images[0].offset == 9
    assert prompt["mm_placeholders"]["image"][1].offset == 9
    assert splice.token_ids == original + [12, 18, 18, 11]


def test_changed_retained_pad_run_cannot_steal_the_next_image():
    a = torch.ones(3, 2, 3)
    original = [10, 18, 18, 18, 18, 11, 2, 31, 2]
    first, _ = capture_processed_images(
        engine_prompt(original, [(Span(1, 4), a)]), prev_len=0
    )
    template_prefix = [10, 18, 18, 11, 2, 31, 2]
    template = template_prefix + [12, 18, 18, 11]
    splice = splice_prefix_tokens(
        tokenizer=SimpleNamespace(eos_token_id=2),
        model_prefix_token_ids=original,
        template_prefix_token_ids=template_prefix,
        template_token_ids=template,
    )
    with pytest.raises(ValueError, match="Retained image"):
        capture_processed_images(
            engine_prompt(template, [(Span(1, 2), a), (Span(8, 2), a)]),
            prev_len=len(original),
            retained=first.images,
            splice=splice,
        )


def test_noncontiguous_embedding_mask_is_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        capture_processed_images(
            engine_prompt(
                [18, 0, 18],
                [(Span(0, 3, torch.tensor([True, False, True])), torch.ones(3, 2, 3))],
            ),
            prev_len=0,
        )


def test_span_mask_is_preserved_on_remap():
    mask = torch.tensor([False, True, True, False])
    splice = splice_prefix_tokens(
        tokenizer=SimpleNamespace(eos_token_id=2),
        model_prefix_token_ids=[10, 77, 31, 2],
        template_prefix_token_ids=[10, 31, 2],
        template_token_ids=[10, 31, 2, 11, 18, 18, 12],
    )
    prompt = engine_prompt(
        [10, 31, 2, 11, 18, 18, 12], [(Span(3, 4, mask), torch.ones(3, 2, 3))]
    )
    captured, _ = capture_processed_images(prompt, prev_len=4, splice=splice)
    assert captured.images[0].offset == 5
    assert prompt["mm_placeholders"]["image"][0].is_embed is mask


def test_missing_attachment_is_not_a_successful_token_commit(dp):
    descriptor, _ = capture_processed_images(
        engine_prompt([18, 18], [(Span(0, 2), torch.ones(3, 2, 3))]), prev_len=0
    )
    capture = RolloutTokenCapture(
        sink=TQTokenSink(dp, staging_partition="staging"), weight_version_fn=lambda: 0
    )
    coords = capture.complete_call(
        capture.begin_call(
            CaptureAdmission(rollout_id="r0", model_call_id="c1", mode="text")
        ),
        prompt_token_ids=[18, 18],
        generated_token_ids=[2],
        generated_logprobs=[-0.5],
        extras={IMAGE_CAPTURE_FIELD: descriptor.to_dict()},
    )
    assert coords.disposition == "capture_failed"
    assert dp.list_sample_ids("staging") == []


def test_partial_put_never_returns_successful_coords(dp, monkeypatch):
    put = dp.put_samples

    def partial_put(*, fields, **kwargs):
        put(fields=fields.select("token_ids_delta"), **kwargs)
        raise RuntimeError("lost media write acknowledgement")

    monkeypatch.setattr(dp, "put_samples", partial_put)
    descriptor, attachments = capture_processed_images(
        engine_prompt([18, 18], [(Span(0, 2), torch.ones(3, 2, 3))]), prev_len=0
    )
    capture = RolloutTokenCapture(
        sink=TQTokenSink(dp, staging_partition="staging"), weight_version_fn=lambda: 0
    )
    coords = capture.complete_call(
        capture.begin_call(
            CaptureAdmission(rollout_id="r0", model_call_id="c1", mode="text")
        ),
        prompt_token_ids=[18, 18],
        generated_token_ids=[2],
        generated_logprobs=[-0.5],
        extras={IMAGE_CAPTURE_FIELD: descriptor.to_dict()},
        attachments=attachments,
    )
    assert coords.disposition == "capture_failed"
    assert dp.list_sample_ids("staging") == []


def test_media_and_routes_share_extras_integrity(dp):
    root, _ = stage(
        dp,
        engine_prompt([10, 18, 18, 11], [(Span(1, 2), torch.ones(3, 2, 3))]),
        routes=True,
    )
    row = finalizer(dp, router_replay_enabled=True).finalize_rollout(
        "r0", receipt(root), reward=1.0
    )
    assert row.valid, row.rejection_reason
    assert row.media and row.routed_experts is not None


def test_deferred_routes_and_image_capture_are_rejected(dp):
    with pytest.raises(ValueError, match="direct router"):
        finalizer(dp, router_replay_enabled=True, defer_routed_experts_to_policy=True)


@pytest.mark.parametrize("inline", [False, True])
def test_worker_restart_recovers_retained_geometry_without_fetching_pixels(
    dp, inline, monkeypatch
):
    a = torch.ones(3, 2, 3)
    root, _ = stage(dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]))
    source = TQTokenSource(dp, staging_partition="staging")
    monkeypatch.setattr(
        source, "fetch_pixels", lambda _: pytest.fail("prefix lookup fetched pixels")
    )
    worker = SimpleNamespace(_capture_images=True, _staging_source=source)
    prefix = [10, 18, 18, 11, 31, 2]
    admission = CaptureAdmission(
        rollout_id="r0",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=len(prefix),
        mode="token_in",
        parent_chain_hash=root.chain_hash,
        required_prefix_token_ids=prefix if inline else [],
        staging_chain=[] if inline else [root.staging_key],
    )
    descriptor, attachments = VllmAsyncGenerationWorkerImpl._capture_request_media(
        worker,
        engine_prompt(prefix + [50], [(Span(1, 2), a)]),
        admission=admission,
    )
    assert descriptor.images == ()
    assert attachments == ()
    with pytest.raises(ValueError, match="Retained image"):
        VllmAsyncGenerationWorkerImpl._capture_request_media(
            worker,
            engine_prompt(prefix + [50], [(Span(1, 2), a + 1)]),
            admission=admission,
        )


def test_worker_completion_stages_pixels_and_only_returns_capture_coordinates(dp):
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker._capture_calls = {}
    worker.token_capture = RolloutTokenCapture(
        sink=TQTokenSink(dp, staging_partition="staging"),
        weight_version_fn=lambda: 0,
        adapter=VLLMCaptureAdapter(),
    )
    request = SimpleNamespace(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
    )
    descriptor, attachments = capture_processed_images(
        engine_prompt([18, 18], [(Span(0, 2), torch.ones(3, 2, 3))]),
        prev_len=0,
    )
    worker._begin_request_capture(
        request, [18, 18], media=descriptor, attachments=attachments
    )
    content = {
        "choices": [
            {
                "message": {
                    "content": "done",
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.5],
                }
            }
        ]
    }
    response = worker._finish_request_capture(request, content)
    assert response["ng_commit_coords"]["disposition"] == "staged"
    assert IMAGE_CAPTURE_FIELD not in response and "pixel_values" not in response
    assert worker._capture_calls == {}
    torch.testing.assert_close(
        TQTokenSource(dp, staging_partition="staging").fetch_pixels("r0/c1"),
        torch.ones(18),
    )
