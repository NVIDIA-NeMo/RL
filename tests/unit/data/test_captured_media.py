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

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureAdmission,
    RolloutReceipt,
)

from nemo_rl.data.captured_media import (
    MEDIA_SPANS_FIELD,
    MediaCaptureRejected,
    pack_images,
)
from nemo_rl.data.captured_media import (
    capture_processed_media as _capture_processed_media,
)
from nemo_rl.data.multimodal_utils import reassemble_packed_multimodal
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.data_plane.tq_token_sink import (
    MEDIA_HAS_FRAMES_FIELD,
    MEDIA_IMGS_FIELD,
    MEDIA_IMGS_SIZES_FIELD,
    MEDIA_NUM_FRAMES_FIELD,
    MEDIA_PRESENT_FIELD,
    MEDIA_STAGING_FIELDS,
    MEDIA_TENSOR_COLUMNS,
    STAGING_FIELDS,
    TQTokenSink,
    TQTokenSource,
)
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)

pytestmark = pytest.mark.nemo_gym


def capture_processed_media(prompt, **kwargs):
    # Tiny fixtures use one-pixel patches; the worker supplies the model's size.
    kwargs.setdefault("patch_size", 1)
    kwargs.setdefault("image_token_id", 18)
    return _capture_processed_media(prompt, **kwargs)


def packed(pixels):
    frames = pixels.unsqueeze(0) if pixels.ndim == 3 else pixels
    return frames.permute(0, 2, 3, 1).reshape(-1, 3)


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
        fields=STAGING_FIELDS + list(MEDIA_STAGING_FIELDS) + ["routed_experts"],
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
    dp,
    prompt,
    *,
    parent=None,
    retained=(),
    call_id="c1",
    rollout_id="r0",
    routes=False,
    partition="staging",
    sink=None,
    expect="staged",
):
    sink = sink or TQTokenSink(
        dp,
        staging_partition=partition,
        capture_media=True,
        media_pixel_dtype=torch.float32,
    )
    capture = RolloutTokenCapture(
        sink=sink, weight_version_fn=lambda: 3, adapter=VLLMCaptureAdapter()
    )
    prev_len = parent.cum_len if parent is not None else 0
    media = capture_processed_media(prompt, prev_len=prev_len, retained=retained)
    admission = CaptureAdmission(
        rollout_id=rollout_id,
        model_call_id=call_id,
        mode="text" if parent is None else "token_in",
        parent_call_id=parent.model_call_id if parent else None,
        prev_len=prev_len,
        required_prefix_token_ids=prompt["prompt_token_ids"][:prev_len],
        parent_chain_hash=parent.chain_hash if parent else None,
    )
    message = {"generation_token_ids": [31, 2], "generation_log_probs": [-0.25, -0.5]}
    if routes:
        message["routed_experts"] = [[[0]]] * (
            len(prompt["prompt_token_ids"]) + 2 - prev_len
        )
    payload = {
        "prompt_token_ids": prompt["prompt_token_ids"],
        "choices": [{"message": message}],
        MEDIA_SPANS_FIELD: [item.to_dict() for item in media.items],
    }
    # Pixels ride beside the record as opaque attachments: one sink write.
    coords = capture.complete_call_from_response(
        capture.begin_call(admission), payload, attachments=media.tensors
    )
    assert coords.disposition == expect
    if expect != "staged":
        return coords, media
    record = CallRecord(
        **coords.model_dump(exclude={"rollout_id", "disposition"}),
        mode=admission.mode,
        response_id=f"response-{call_id}",
    )
    return record, media


def staged_media(dp, *keys, partition="staging"):
    """Read the media of the given call rows the way the finalizer does."""
    source = TQTokenSource(dp, staging_partition=partition, capture_media=True)
    items = source.fetch_for_finalization(list(keys))
    return source.fetch_media(items)


class RecordingClient:
    """NoOp client wrapper recording which columns each read selected."""

    def __init__(self, client):
        self.client = client
        self.gets = []

    def __getattr__(self, name):
        return getattr(self.client, name)

    def get_samples(self, *args, **kwargs):
        select = kwargs.get("select_fields", args[2] if len(args) > 2 else None)
        self.gets.append(list(select))
        return self.client.get_samples(*args, **kwargs)

    def tensor_reads(self):
        return [g for g in self.gets if set(g) & set(MEDIA_TENSOR_COLUMNS.values())]


def receipt(*records, rollout_id="r0", terminal=None):
    return RolloutReceipt(
        rollout_id=rollout_id,
        manifest=list(records),
        terminal_model_call_id=terminal or records[-1].model_call_id,
        terminal_selection="declared",
    ).model_dump()


def finalizer(dp, **kwargs):
    kwargs.setdefault("capture_media", True)
    return RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=1000,
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
        retained=media.items,
        call_id="c2",
    )
    [root_media, child_media] = staged_media(dp, root.staging_key, child.staging_key)
    assert root_media.imgs.numel() == a.numel()
    assert child_media.imgs.numel() == b.numel()
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
    torch.testing.assert_close(pixels[:6], packed(a), rtol=0, atol=0)
    torch.testing.assert_close(pixels[6:], packed(b), rtol=0, atol=0)


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
    # One of the two valid rows carries media; the rejected row does not count.
    assert result.metrics["finalize/media_row_rate"] == 0.5
    fields = dp.get_samples(result.meta.sample_ids, "train", result.meta.fields)
    assert fields["sample_mask"].tolist() == [1.0, 1.0, 0.0]
    materialized = dict(fields)
    reassemble_packed_multimodal(materialized, result.meta.tags)
    assert materialized["pixel_values"].row_shapes() == [[[6, 3]], [], []]
    torch.testing.assert_close(materialized["pixel_values"].as_tensor(), packed(a))
    assert dp.list_sample_ids("staging") == []


@pytest.mark.parametrize(
    "corruption, reason",
    [
        ("missing", "invalid_media_columns:"),
        ("geometry", "invalid_media_columns:"),
        ("dtype", "invalid_media_columns:"),
        ("frames_flag", "invalid_media_columns:"),
        ("orphan_frames_flag", "invalid_staging_row:"),
    ],
)
def test_missing_or_corrupt_media_rejects_rollout(dp, corruption, reason):
    root, _ = stage(
        dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), torch.ones(3, 2, 3))])
    )
    stored = dp._partitions["staging"].rows[root.staging_key]
    if corruption == "missing":
        del stored[MEDIA_IMGS_FIELD]
    elif corruption == "geometry":
        stored[MEDIA_IMGS_SIZES_FIELD][0] += 1
    elif corruption == "dtype":
        stored[MEDIA_IMGS_FIELD] = stored[MEDIA_IMGS_FIELD].to(torch.int64)
    elif corruption == "frames_flag":
        # A still flagged as video reads the all-zero num_frames sentinel.
        stored[MEDIA_HAS_FRAMES_FIELD] = torch.tensor(True)
    else:
        stored[MEDIA_PRESENT_FIELD] = torch.tensor(False)
        stored[MEDIA_HAS_FRAMES_FIELD] = torch.tensor(True)
    row = finalizer(dp).finalize_rollout("r0", receipt(root), reward=1.0)
    assert not row.valid
    assert row.rejection_reason.startswith(reason), row.rejection_reason
    assert row.media is None


def test_image_free_continuation_has_no_pixel_column(dp):
    a = torch.ones(3, 2, 3)
    root, media = stage(dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]))
    child, descriptor = stage(
        dp,
        engine_prompt([10, 18, 18, 11, 31, 2, 50], [(Span(1, 2), a)]),
        parent=root,
        retained=media.items,
        call_id="c2",
    )
    assert descriptor.items == ()
    child_row = dp._partitions["staging"].rows[child.staging_key]
    # Every row of a media partition carries the columns; this one is flagged
    # empty and holds sentinels, so the finalizer never reads its tensors. The
    # sentinels keep each column's dtype: TQ allows one dtype per field.
    assert child_row[MEDIA_PRESENT_FIELD].item() is False
    root_row = dp._partitions["staging"].rows[root.staging_key]
    for column in MEDIA_TENSOR_COLUMNS.values():
        assert child_row[column].dtype is root_row[column].dtype, column
    client = RecordingClient(dp)
    row = finalizer(client).finalize_rollout("r0", receipt(root, child), reward=1.0)
    assert row.valid, row.rejection_reason
    assert row.media["imgs_sizes"].as_tensor().tolist() == [[2, 3]]
    [tensor_read] = client.tensor_reads()
    assert tensor_read == list(MEDIA_TENSOR_COLUMNS.values())


def test_repeated_asset_occurrences_remain_distinct(dp):
    a = torch.ones(3, 2, 3)
    root, _ = stage(
        dp, engine_prompt([10, 18, 18, 11, 18, 18], [(Span(1, 2), a), (Span(4, 2), a)])
    )
    row = finalizer(dp).finalize_rollout("r0", receipt(root), reward=1.0)
    assert row.valid
    assert row.media["pixel_values"].as_tensor().shape == (12, 3)


def test_patch_order_matches_bridge_dynamic_images():
    frames = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4)
    result = pack_images(frames, torch.tensor([[4, 4], [4, 4]]), patch_size=2)
    expected = torch.stack(
        [
            frame[:, y : y + 2, x : x + 2].reshape(-1)
            for frame in frames
            for y in (0, 2)
            for x in (0, 2)
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_snapshot_owns_storage_and_preserves_dtype(dtype):
    a = torch.arange(18, dtype=dtype).reshape(3, 2, 3)
    original = packed(a.clone())
    media = capture_processed_media(
        engine_prompt([18, 18], [(Span(0, 2), a)]), prev_len=0
    )
    a.zero_()
    torch.testing.assert_close(media.tensors["imgs"][0], original, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["length", "size", "dropped", "cache_reference"])
def test_changed_retained_images_fail_before_inference(change):
    a = torch.ones(3, 2, 3)
    first = capture_processed_media(
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
    with pytest.raises(MediaCaptureRejected) as excinfo:
        capture_processed_media(prompt, prev_len=6, retained=first.items)
    # A changed retained image carries its own code so the worker's HTTP 400
    # (and its log line) distinguish it from other capture-time rejections.
    expected = (
        "retained_media_changed"
        if change in ("length", "size")
        else "media_capture_rejected"
    )
    assert excinfo.value.code == expected


def test_splice_coordinates_handle_reasoning_shift_and_repeated_pad_runs():
    a, b = torch.ones(3, 2, 3), torch.ones(3, 3, 2)
    original = [10, 18, 18, 11, 77, 78, 31, 2]
    first = capture_processed_media(
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
    captured = capture_processed_media(
        prompt, prev_len=len(original), retained=first.items, splice=splice
    )
    assert captured.items[0].embedding_spans[0][0] == 9
    assert prompt["mm_placeholders"]["image"][1].offset == 9
    assert splice.token_ids == original + [12, 18, 18, 11]


def test_changed_retained_pad_run_cannot_steal_the_next_image():
    a = torch.ones(3, 2, 3)
    original = [10, 18, 18, 18, 18, 11, 2, 31, 2]
    first = capture_processed_media(
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
    with pytest.raises(ValueError, match="Retained media"):
        capture_processed_media(
            engine_prompt(template, [(Span(1, 2), a), (Span(8, 2), a)]),
            prev_len=len(original),
            retained=first.items,
            splice=splice,
        )


def test_noncontiguous_embedding_mask_is_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        capture_processed_media(
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
    captured = capture_processed_media(prompt, prev_len=4, splice=splice)
    assert captured.items[0].embedding_spans[0][0] == 5
    assert prompt["mm_placeholders"]["image"][0].is_embed is mask


def test_failed_combined_write_is_capture_failed_and_leaves_no_row(dp, monkeypatch):
    """Tokens and pixels share one write: a failure is reported at call time
    (no ``staged`` coords for a row the finalizer would have to reject) and the
    attempted key is discarded."""
    sink = TQTokenSink(
        dp,
        staging_partition="staging",
        capture_media=True,
        media_pixel_dtype=torch.float32,
    )
    cleared = []

    def fail(*args, **kwargs):
        raise OSError("media write failed")

    monkeypatch.setattr(sink._store, "put", fail)
    monkeypatch.setattr(sink._store, "clear", lambda keys: cleared.append(list(keys)))
    coords, _ = stage(
        dp,
        engine_prompt([18, 18], [(Span(0, 2), torch.ones(3, 2, 3))]),
        sink=sink,
        expect="capture_failed",
    )
    assert coords.staging_key is None
    assert cleared == [["r0/c1"]]
    assert dp.list_sample_ids("staging") == []


def test_media_chain_reads_present_rows_once_in_chain_order(dp):
    """Media on calls one and two, none on three: one batched tensor read for
    the two present keys, in chain order, concatenated for the learner."""
    a, b = torch.ones(3, 2, 3), torch.full((3, 4, 2), 2.0)
    c1, media1 = stage(dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]))
    p1 = [10, 18, 18, 11, 31, 2]
    c2, media2 = stage(
        dp,
        engine_prompt(p1 + [12, 18, 18, 11], [(Span(1, 2), a), (Span(7, 2), b)]),
        parent=c1,
        retained=media1.items,
        call_id="c2",
    )
    p2 = p1 + [12, 18, 18, 11, 31, 2]
    c3, media3 = stage(
        dp,
        engine_prompt(p2 + [50], [(Span(1, 2), a), (Span(7, 2), b)]),
        parent=c2,
        retained=media1.items + media2.items,
        call_id="c3",
    )
    assert media3.tensors is None
    client = RecordingClient(dp)
    row = finalizer(client).finalize_rollout("r0", receipt(c1, c2, c3), reward=1.0)
    assert row.valid, row.rejection_reason
    assert len(client.tensor_reads()) == 1
    assert row.media["imgs_sizes"].as_tensor().tolist() == [[2, 3], [4, 2]]
    pixels = row.media["pixel_values"].as_tensor()
    torch.testing.assert_close(pixels[:6], packed(a), rtol=0, atol=0)
    torch.testing.assert_close(pixels[6:], packed(b), rtol=0, atol=0)


def test_text_rollout_in_media_partition_issues_no_tensor_read(dp):
    root, _ = stage(dp, engine_prompt([10, 11]))
    child, _ = stage(dp, engine_prompt([10, 11, 31, 2, 12]), parent=root, call_id="c2")
    client = RecordingClient(dp)
    row = finalizer(client).finalize_rollout("r0", receipt(root, child), reward=1.0)
    assert row.valid, row.rejection_reason
    assert row.media is None
    assert client.tensor_reads() == []
    # The base read still carried the presence flags.
    assert all(MEDIA_PRESENT_FIELD in read for read in client.gets[:1])


def test_mixed_image_and_video_chain_is_rejected_before_any_tensor_read(dp):
    a = torch.ones(3, 2, 3)
    root, media = stage(dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]))
    child, _ = stage(
        dp,
        engine_prompt(
            [10, 18, 18, 11, 31, 2, 18, 18], [(Span(1, 2), a), (Span(6, 2), a)]
        ),
        parent=root,
        retained=media.items,
        call_id="c2",
    )
    # Forge the child into a (well-formed) one-frame video row.
    child_row = dp._partitions["staging"].rows[child.staging_key]
    child_row[MEDIA_HAS_FRAMES_FIELD] = torch.tensor(True)
    child_row[MEDIA_NUM_FRAMES_FIELD] = torch.tensor([1], dtype=torch.int32)
    client = RecordingClient(dp)
    row = finalizer(client).finalize_rollout("r0", receipt(root, child), reward=1.0)
    assert not row.valid
    assert row.rejection_reason.startswith("media_chain_incompatible:mixed")
    assert client.tensor_reads() == []


def test_text_only_partition_finalizes_without_media_columns():
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="staging",
        fields=list(STAGING_FIELDS),
        num_samples=8,
        consumer_tasks=["finalize"],
    )
    client.register_partition(
        partition_id="train",
        fields=list(DP_TRAIN_FIELDS),
        num_samples=8,
        consumer_tasks=["train"],
    )
    sink = TQTokenSink(client, staging_partition="staging", capture_media=False)
    record, _ = stage(client, engine_prompt([10, 11]), sink=sink)
    assert not (
        set(MEDIA_STAGING_FIELDS) & set(client._partitions["staging"].rows["r0/c1"])
    )
    recording = RecordingClient(client)
    row = finalizer(recording, capture_media=False).finalize_rollout(
        "r0", receipt(record), reward=1.0
    )
    assert row.valid, row.rejection_reason
    assert row.media is None
    assert all(not (set(MEDIA_STAGING_FIELDS) & set(read)) for read in recording.gets)


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


@pytest.mark.parametrize("inline", [False, True])
def test_worker_restart_recovers_retained_geometry_without_fetching_pixels(
    dp, inline, monkeypatch
):
    a = torch.ones(3, 2, 3)
    root, _ = stage(dp, engine_prompt([10, 18, 18, 11], [(Span(1, 2), a)]))
    source = TQTokenSource(dp, staging_partition="staging", capture_media=True)
    monkeypatch.setattr(
        source, "fetch_media", lambda _: pytest.fail("prefix lookup fetched pixels")
    )
    worker = SimpleNamespace(
        _capture_media=True,
        _capture_image_token_id=18,
        _capture_patch_size=1,
        _staging_source=source,
    )
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
    descriptor = VllmAsyncGenerationWorkerImpl._capture_request_media(
        worker,
        engine_prompt(prefix + [50], [(Span(1, 2), a)]),
        admission=admission,
    )
    assert descriptor.items == ()
    assert descriptor.tensors is None
    with pytest.raises(ValueError, match="Retained media"):
        VllmAsyncGenerationWorkerImpl._capture_request_media(
            worker,
            engine_prompt(prefix + [50], [(Span(1, 2), torch.ones(3, 3, 2))]),
            admission=admission,
        )


def test_worker_completion_stages_pixels_and_only_returns_capture_coordinates(dp):
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker._capture_calls = {}
    worker.token_capture = RolloutTokenCapture(
        sink=TQTokenSink(
            dp,
            staging_partition="staging",
            capture_media=True,
            media_pixel_dtype=torch.float32,
        ),
        weight_version_fn=lambda: 0,
        adapter=VLLMCaptureAdapter(),
    )
    request = SimpleNamespace(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
    )
    descriptor = capture_processed_media(
        engine_prompt([18, 18], [(Span(0, 2), torch.ones(3, 2, 3))]),
        prev_len=0,
    )
    worker._begin_request_capture(request, [18, 18], media=descriptor)
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
    assert "media" not in response and MEDIA_SPANS_FIELD not in response
    assert worker._capture_calls == {}
    [media] = staged_media(dp, "r0/c1")
    torch.testing.assert_close(media.imgs, torch.ones(1, 6, 3))


def video_prompt(tokens, videos, images=()):
    """Use the exact vLLM 0.25.1 per-video processor field names."""
    prompt = engine_prompt(tokens, images)
    spans, items = [], []
    for span, frames in videos:
        data = {
            "pixel_values_flat_video": frames,
            "video_num_patches": torch.tensor(frames.shape[0]),
            "frames_indices": torch.arange(frames.shape[0]),
            "frame_duration_ms": torch.tensor(500),
        }
        spans.append(span)
        items.append(SimpleNamespace(get_data=lambda data=data: data))
    prompt["mm_placeholders"]["video"] = spans
    prompt["mm_kwargs"]["video"] = items
    return prompt


def test_native_video_keeps_frames_and_timestamp_separated_embeddings(dp):
    frames = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    # Two temporal tubelets: timestamp text separates their image-context runs.
    tokens = [10, 90, 18, 18, 91, 18, 18, 11]
    record, media = stage(dp, video_prompt(tokens, [(Span(1, 6), frames)]))
    assert media.items[0].modality == "video"
    assert media.items[0].embedding_spans == ((2, 2), (5, 2))
    assert media.items[0].placeholder_length == 6
    row = finalizer(dp).finalize_rollout("r0", receipt(record), reward=1.0)
    assert row.valid, row.rejection_reason
    torch.testing.assert_close(
        row.media["pixel_values"].as_tensor(), packed(frames), rtol=0, atol=0
    )
    assert row.media["imgs_sizes"].as_tensor().tolist() == [[2, 2]] * 4
    assert row.media["num_frames"].as_tensor().tolist() == [4]


def test_video_frame_groups_survive_checkpoint(dp, tmp_path):
    frames_a, frames_b = torch.ones(2, 3, 2, 2), torch.full((4, 3, 4, 2), 9.0)
    tokens = [10, 90, 18, 91, 18, 11]
    root, media = stage(dp, video_prompt(tokens, [(Span(1, 4), frames_a)]))
    prefix = tokens + [31, 2]
    child, _ = stage(
        dp,
        video_prompt(
            prefix + [90, 18, 91, 18],
            [(Span(1, 4), frames_a), (Span(len(prefix), 4), frames_b)],
        ),
        parent=root,
        retained=media.items,
        call_id="c2",
    )
    [child_media] = staged_media(dp, child.staging_key)
    assert child_media.imgs.numel() == frames_b.numel()
    assert child_media.num_frames.tolist() == [4]
    dp.save_checkpoint(tmp_path / "video")
    restored = NoOpDataPlaneClient()
    restored.load_checkpoint(tmp_path / "video")
    row = finalizer(restored).finalize_rollout("r0", receipt(root, child), reward=1.0)
    assert row.valid, row.rejection_reason
    assert row.media["num_frames"].as_tensor().tolist() == [2, 4]
    assert row.media["imgs_sizes"].as_tensor().tolist() == [[2, 2]] * 2 + [[4, 2]] * 4
    torch.testing.assert_close(
        row.media["pixel_values"].as_tensor(),
        torch.cat([packed(frames_a), packed(frames_b)]),
        rtol=0,
        atol=0,
    )


def test_mixed_image_video_capture_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="image-only or video-only"):
        capture_processed_media(
            video_prompt(
                [18, 90, 18],
                [(Span(1, 2), torch.ones(2, 3, 2, 2))],
                [(Span(0, 1), torch.ones(3, 2, 2))],
            ),
            prev_len=0,
        )


@pytest.mark.parametrize("change", ["frames", "timestamps"])
def test_changed_retained_video_is_rejected(change):
    frames = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    tokens = [90, 18, 91, 18]
    media = capture_processed_media(
        video_prompt(tokens, [(Span(0, 4), frames)]), prev_len=0, image_token_id=18
    )
    if change == "pixels":
        frames = frames + 1
    elif change == "order":
        frames = frames.flip(0)
    elif change == "frames":
        frames = frames[:2]
    else:
        tokens[0] = 92
    with pytest.raises(ValueError, match="Retained media"):
        capture_processed_media(
            video_prompt(tokens + [31, 2, 50], [(Span(0, 4), frames)]),
            prev_len=6,
            retained=media.items,
            image_token_id=18,
        )


def test_video_placeholder_remap_preserves_all_timestamp_tokens():
    frames = torch.ones(4, 3, 2, 2)
    original = [10, 77, 90, 18, 91, 18, 31, 2]
    first = capture_processed_media(
        video_prompt(original, [(Span(2, 4), frames)]), prev_len=0, image_token_id=18
    )
    template_prefix = [10, 90, 18, 91, 18, 31, 2]
    template = template_prefix + [92, 18, 93, 18]
    splice = splice_prefix_tokens(
        tokenizer=SimpleNamespace(eos_token_id=2),
        model_prefix_token_ids=original,
        template_prefix_token_ids=template_prefix,
        template_token_ids=template,
    )
    prompt = video_prompt(template, [(Span(1, 4), frames), (Span(7, 4), frames)])
    added = capture_processed_media(
        prompt,
        prev_len=len(original),
        retained=first.items,
        splice=splice,
        image_token_id=18,
    )
    assert [span.offset for span in prompt["mm_placeholders"]["video"]] == [2, 8]
    assert added.items[0].embedding_spans == ((9, 1), (11, 1))
    assert splice.token_ids == original + [92, 18, 93, 18]


@pytest.mark.parametrize(
    "column, reason",
    [
        (MEDIA_IMGS_FIELD, "invalid_media_columns:"),
        (MEDIA_IMGS_SIZES_FIELD, "invalid_media_columns:"),
        (MEDIA_NUM_FRAMES_FIELD, "invalid_media_columns:"),
        (MEDIA_PRESENT_FIELD, "missing_staging_row:"),
        (MEDIA_HAS_FRAMES_FIELD, "missing_staging_row:"),
    ],
)
def test_video_requires_every_committed_column(dp, column, reason):
    """A missing column in a media-enabled partition is an error, never
    evidence that media capture was off."""
    record, _ = stage(
        dp, video_prompt([90, 18, 91, 18], [(Span(0, 4), torch.ones(4, 3, 2, 2))])
    )
    del dp._partitions["staging"].rows[record.staging_key][column]
    row = finalizer(dp).finalize_rollout("r0", receipt(record), reward=1.0)
    assert not row.valid and row.rejection_reason.startswith(reason), (
        row.rejection_reason
    )


def test_native_video_rejects_inconsistent_frame_count():
    prompt = video_prompt([18, 18], [(Span(0, 2), torch.ones(4, 3, 2, 2))])
    prompt["mm_kwargs"]["video"][0].get_data()["video_num_patches"] = torch.tensor(2)
    with pytest.raises(ValueError, match="frame count"):
        capture_processed_media(prompt, prev_len=0, image_token_id=18)


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
def test_real_vllm_video_replacement_round_trips(dp, temporal_patch_size):
    # nemo_gym-marked tests run in a lane without the vllm extra.
    pytest.importorskip("vllm")
    from transformers import BatchFeature
    from vllm.model_executor.models.nano_nemotron_vl import (
        NanoNemotronVLMultiModalProcessor,
    )
    from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange
    from vllm.transformers_utils.processors.nano_nemotron_vl import (
        NanoNemotronVLProcessor,
    )

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[100 + ord(c) for c in text] for text in texts]}

    replacement = NanoNemotronVLProcessor.get_video_repl(
        tokens_per_frame=[2] * (4 // temporal_patch_size),
        frames_indices=[0, 3, 6, 9],
        frame_duration_ms=100,
        tokenizer=Tokenizer(),
        img_start_token_ids=[16],
        img_end_token_ids=[17],
        img_context_token_ids=[18],
        video_temporal_patch_size=temporal_patch_size,
    )
    tokens = replacement.full
    frames = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    processor = object.__new__(NanoNemotronVLMultiModalProcessor)
    hf_inputs = BatchFeature(
        data={
            "pixel_values_flat_video": frames,
            "video_num_patches": torch.tensor([4]),
            "frames_indices": torch.tensor([[0, 3, 6, 9]]),
            "frame_duration_ms": torch.tensor([100]),
        }
    )
    prompt = {
        "prompt_token_ids": tokens,
        "mm_placeholders": {"video": [PlaceholderRange(offset=0, length=len(tokens))]},
        "mm_kwargs": MultiModalKwargsItems.from_hf_inputs(
            hf_inputs, processor._get_video_fields_config(hf_inputs)
        ),
    }
    record, descriptor = stage(dp, prompt)
    assert len(descriptor.items[0].embedding_spans) == 4 // temporal_patch_size
    row = finalizer(dp).finalize_rollout("r0", receipt(record), reward=1.0)
    assert row.valid, row.rejection_reason
    assert row.token_ids[:-2] == tokens
    assert row.media["num_frames"].as_tensor().tolist() == [4]
    torch.testing.assert_close(
        row.media["pixel_values"].as_tensor(), packed(frames), rtol=0, atol=0
    )


@pytest.mark.parametrize("bad_count", [2.5, True, 2**32 + 4])
def test_video_geometry_is_never_silently_cast(bad_count):
    prompt = video_prompt([18, 18], [(Span(0, 2), torch.ones(4, 3, 2, 2))])
    prompt["mm_kwargs"]["video"][0].get_data()["video_num_patches"] = bad_count
    with pytest.raises(ValueError, match="geometry"):
        capture_processed_media(prompt, prev_len=0, image_token_id=18)


def test_video_publication_with_text_and_rejected_siblings(dp):
    frames = torch.ones(2, 3, 2, 2)
    video, _ = stage(dp, video_prompt([18, 90, 18], [(Span(0, 3), frames)]))
    text, _ = stage(dp, engine_prompt([10]), rollout_id="text")
    result = finalizer(dp).finalize_group(
        "g0",
        ["r0", "text", "bad"],
        [receipt(video), receipt(text, rollout_id="text"), None],
        [1.0, 0.0, 0.0],
        mask_sample=[False] * 3,
        fallback_weight_version=3,
        prompt_idx=0,
        canonical_sample_ids=["g0_g0", "g0_g1", "g0_g2"],
    )
    assert result.valid_row_count == 2
    assert result.metrics["finalize/media_row_rate"] == 0.5
    fields = dict(dp.get_samples(result.meta.sample_ids, "train", result.meta.fields))
    reassemble_packed_multimodal(fields, result.meta.tags)
    for name in ("pixel_values", "imgs_sizes", "num_frames"):
        assert fields[name].logical_segment_counts_by_row() == [1, 0, 0]
    assert fields["num_frames"].as_tensor().tolist() == [2]
    assert dp.list_sample_ids("staging") == []
