# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Captured media slots survive publication without consuming literal text IDs."""

import hashlib
import json

import pytest
import torch

from nemo_rl.data.captured_media import CapturedMediaItem
from nemo_rl.data.multimodal_utils import (
    MEDIA_ANCHOR_SPANS_TAG,
    PackedTensor,
    attach_captured_media_anchor_mask,
    attach_media_token_validity_mask,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink
from nemo_rl.data_plane.worker_mixin import _materialize_fetched
from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    RolloutReassembler,
    RolloutSelection,
)
from tests.unit.data_plane.token_capture_test_fixtures import _manifest, _record
from tests.unit.experience.test_finalized_call_views import _with_extras
from tests.unit.experience.test_logical_owner_finalization import PublicationDataPlane

pytestmark = pytest.mark.nemo_gym


def test_authenticated_media_slots_survive_publish_and_worker_materialization():
    plane = PublicationDataPlane()
    sink = TQTokenSink(
        plane,
        staging_partition="staged",
        capture_media=True,
        media_pixel_dtype=torch.bfloat16,
    )
    records = []
    prefix = []
    # A root prompt retains literal 18 from older reasoning; the first new
    # output writes another 18. Neither is an attached image feature.
    for index, (tokens, masks, anchor) in enumerate(
        (
            ([18, 17, 18, 80, 18], [0.0, 0.0, 0.0, 0.0, 1.0], 0),
            ([77, 18, 22], [0.0, 0.0, 1.0], 6),
        )
    ):
        record = _record(
            rollout_id="attempt",
            model_call_id=f"c{index}",
            parent_call_id=f"c{index - 1}" if index else None,
            prev_len=len(prefix),
            token_ids=tokens,
            token_mask=masks,
            logprobs=[-0.25 if mask else 0.0 for mask in masks],
            weight_version=3,
            parent_chain_hash=records[-1].chain_hash if records else None,
            cumulative_prefix=prefix,
        )
        occurrence = CapturedMediaItem(
            modality="image",
            placeholder_offset=anchor,
            placeholder_length=1,
            placeholder_digest=hashlib.sha256(b"[18]").hexdigest(),
            token_id=18,
            embedding_spans=((anchor, 1),),
            imgs_sizes=((2, 2),),
        )
        record = _with_extras(record, {"media_spans": [occurrence.to_dict()]})
        result = sink.stage(
            record,
            attachments={
                "imgs": torch.ones((1, 1, 12), dtype=torch.bfloat16),
                "imgs_sizes": torch.tensor([[2, 2]], dtype=torch.int32),
            },
        )
        assert result.ok, result.error
        records.append(record)
        prefix.extend(tokens)
    manifest = [
        _manifest(record).model_copy(update={"response_id": f"response-{i}"})
        for i, record in enumerate(records)
    ]
    receipt = dict(
        rollout_id="attempt",
        manifest=[item.model_dump() for item in manifest],
        terminal_model_call_id="c1",
        terminal_selection="declared",
    )
    finalizer = RolloutReassembler(
        plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
        capture_media=True,
    )
    row = finalizer.finalize_rollout(
        "attempt", receipt, reward=0.7, context_compaction=True
    )
    assert row.valid, row.rejection_reason
    assert row.media_anchor_spans == ((0, 1, 18), (6, 1, 18))
    assert row.token_ids == prefix and row.token_mask == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ]
    group = finalizer.finalize_group(
        "group",
        ["attempt"],
        [receipt],
        [0.7],
        mask_sample=[False],
        fallback_weight_version=3,
        prompt_idx=0,
        canonical_sample_ids=["group_g0"],
        logical_selections=[
            RolloutSelection(
                tuple(item.response_id for item in manifest),
                tuple(ActionOutputFlags(False, False) for _ in records),
            )
        ],
    )
    assert not group.dropped and group.meta is not None
    tags = json.loads(
        json.dumps(group.meta.tags)
    )  # Actual JSON-compatible wire metadata.
    assert tags[0][MEDIA_ANCHOR_SPANS_TAG] == [[0, 1, 18], [6, 1, 18]]
    wire = plane.get_samples(
        sample_ids=group.meta.sample_ids,
        partition_id="canonical",
        select_fields=[
            "input_ids",
            "input_lengths",
            "token_mask",
            "generation_logprobs",
        ],
    )
    batch = _materialize_fetched(
        wire,
        local_batch=False,
        layout="padded",
        pad_value_dict={},
        pad_to_seqlen=16,
        tags=tags,
    )
    expected = torch.zeros((1, 16), dtype=torch.bool)
    expected[0, [0, 6]] = True
    torch.testing.assert_close(batch["media_token_validity_mask"], expected)
    torch.testing.assert_close(batch["input_ids"][0, :8], torch.tensor(prefix))
    torch.testing.assert_close(batch["token_mask"][0, :8], torch.tensor(row.token_mask))
    torch.testing.assert_close(
        batch["generation_logprobs"][0, :8], torch.tensor(row.logprobs)
    )
    assert int((batch["input_ids"] == 18).sum()) == 4
    assert (
        int(((batch["input_ids"] == 18) & batch["media_token_validity_mask"]).sum())
        == 2
    )
    # The legacy helper must refine this mask rather than restore false slots.
    batch["pixel_values"] = PackedTensor([torch.ones((2, 12))], dim_to_pack=0)
    attach_media_token_validity_mask(batch, 18)
    torch.testing.assert_close(batch["media_token_validity_mask"], expected)


@pytest.mark.parametrize(
    "spans",
    (
        [[0, 2, 18]],  # Incorrect token in the claimed feature span.
        [[0, 5, 18]],  # Exceeds the valid input, even if padded storage exists.
        [[0, 1, 18], [0, 1, 18]],  # Overlapping anchors.
        [[True, 1, 18]],
        "not-spans",
    ),
)
def test_invalid_captured_anchors_fail_instead_of_guessing_features(spans):
    batch = {
        "input_ids": torch.tensor([[18, 90, 18, 0]]),
        "input_lengths": torch.tensor([3]),
    }
    with pytest.raises(ValueError, match="Captured media anchor"):
        attach_captured_media_anchor_mask(batch, [{MEDIA_ANCHOR_SPANS_TAG: spans}])


def test_mixed_old_checkpoint_and_new_anchor_tags_preserve_each_row():
    batch = {
        "input_ids": torch.tensor([[18, 18], [18, 91]]),
        "input_lengths": torch.tensor([2, 2]),
    }
    attach_captured_media_anchor_mask(
        batch, [{MEDIA_ANCHOR_SPANS_TAG: [[0, 1, 18]]}, {}]
    )
    torch.testing.assert_close(
        batch["media_token_validity_mask"], torch.tensor([[True, False], [True, True]])
    )


def test_absent_anchor_tags_keep_legacy_behavior():
    batch = {"input_ids": torch.tensor([[18, 90]]), "input_lengths": torch.tensor([2])}
    attach_captured_media_anchor_mask(batch, [{}])
    assert "media_token_validity_mask" not in batch
