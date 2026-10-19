# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authenticated staging becomes exact call inputs, without future media.

Storage is an in-memory transport; sink/source codecs, Gym verification, route
assembly, media geometry, target partitioning and finalization are production
code. These CPU tests do not qualify native model forwards or optimizer steps.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from nemo_gym.token_id_capture.staging.digest import (
    compute_extras_digest,
    compute_staging_digest,
)
from nemo_gym.token_id_capture.staging.records import RolloutReceipt, StagedCallRecord

from nemo_rl.data.captured_media import CapturedMediaItem
from nemo_rl.data_plane.schema import INVALID_TOOL_CALL_MASK, MALFORMED_THINKING_MASK
from nemo_rl.data_plane.tq_token_sink import MEDIA_IMGS_SIZES_FIELD, TQTokenSink
from nemo_rl.experience.rollout_reassembler import ActionOutputFlags, RolloutReassembler
from nemo_rl.utils.routed_experts_codec import encode_routed_experts
from tests.unit.data_plane.token_capture_test_fixtures import _manifest, _record
from tests.unit.experience.test_logical_owner_finalization import PublicationDataPlane

pytestmark = pytest.mark.nemo_gym


class _RecordingDataPlane(PublicationDataPlane):
    def __init__(self) -> None:
        super().__init__()
        self.reads: list[dict[str, Any]] = []

    def get_samples(self, **kwargs: Any) -> Any:
        self.reads.append(kwargs)
        return super().get_samples(**kwargs)


@dataclass(frozen=True)
class _Case:
    data_plane: _RecordingDataPlane
    finalizer: RolloutReassembler
    receipt: dict[str, Any]
    records: tuple[StagedCallRecord, ...]
    routes: tuple[torch.Tensor, ...]
    flags: tuple[ActionOutputFlags, ...]


def _with_extras(record: StagedCallRecord, extras: dict[str, Any]) -> StagedCallRecord:
    extras_digest = compute_extras_digest(extras)
    digest = compute_staging_digest(
        schema_version=record.schema_version,
        digest_version=record.digest_version,
        extras_digest_version=record.extras_digest_version,
        rollout_id=record.rollout_id,
        model_call_id=record.model_call_id,
        parent_call_id=record.parent_call_id,
        mode=record.mode,
        prev_len=record.prev_len,
        delta_len=record.delta_len,
        cum_len=record.cum_len,
        weight_version=record.weight_version,
        token_ids_delta=record.token_ids_delta,
        token_mask_delta=record.token_mask_delta,
        generation_log_probs_delta=record.generation_log_probs_delta,
        extras_digest=extras_digest,
        chain_hash=record.chain_hash,
        cumulative_hash=record.cumulative_hash,
    )
    return StagedCallRecord.model_validate(
        record.model_dump()
        | {"extras": extras, "extras_digest": extras_digest, "digest": digest}
    )


def _case(
    *,
    media: bool,
    zero_generation: bool = False,
    rollout_id: str = "owner",
    data_plane: _RecordingDataPlane | None = None,
) -> _Case:
    data_plane = data_plane if data_plane is not None else _RecordingDataPlane()
    sink = TQTokenSink(
        data_plane,
        staging_partition="staged",
        capture_media=media,
        media_pixel_dtype=torch.bfloat16 if media else None,
    )
    records = []
    routes = []
    prefix: list[int] = []
    for index in range(4):
        add_media = media and index in (1, 3)
        prompt = [18 if add_media else 10 + index]
        generation = [] if zero_generation and index == 2 else [30 + index]
        tokens = prompt + generation
        record = _record(
            rollout_id=rollout_id,
            model_call_id=f"c{index}",
            parent_call_id=f"c{index - 1}" if index else None,
            prev_len=len(prefix),
            token_ids=tokens,
            token_mask=[0.0] + [1.0] * len(generation),
            logprobs=[0.0] + [-0.125 * (index + 1)] * len(generation),
            weight_version=12 if index < 2 else 13,
            parent_chain_hash=records[-1].chain_hash if records else None,
            cumulative_prefix=prefix,
        )
        original_routes = torch.arange(
            index * 100, index * 100 + len(prefix) + len(tokens), dtype=torch.int16
        )[:, None, None]
        original_routes[-1] = -1
        extras = {
            "routed_experts": encode_routed_experts(original_routes),
            "routed_experts_prefix_len": len(prefix),
            "media_spans": [],
        }
        if index:
            extras["predecessor_tail_route"] = [[900 + index]]
        attachments = None
        if add_media:
            occurrence = CapturedMediaItem(
                modality="image",
                placeholder_offset=len(prefix),
                placeholder_length=1,
                placeholder_digest=hashlib.sha256(
                    json.dumps([18], separators=(",", ":")).encode()
                ).hexdigest(),
                token_id=18,
                embedding_spans=((len(prefix), 1),),
                imgs_sizes=((2, 2),),
            )
            extras["media_spans"] = [occurrence.to_dict()]
            attachments = {
                "imgs": torch.full((1, 1, 12), float(index), dtype=torch.bfloat16),
                "imgs_sizes": torch.tensor([[2, 2]], dtype=torch.int32),
            }
        record = _with_extras(record, extras)
        staged = sink.stage(record, attachments=attachments)
        assert staged.ok, staged.error
        records.append(record)
        routes.append(original_routes)
        prefix.extend(tokens)
    receipt = RolloutReceipt(
        rollout_id=rollout_id,
        manifest=[
            _manifest(record).model_copy(
                update={"response_id": f"{rollout_id}/{record.model_call_id}"}
            )
            for record in records
        ],
        terminal_model_call_id="c3",
        terminal_selection="declared",
    ).model_dump()
    finalizer = RolloutReassembler(
        data_plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
        router_replay_enabled=True,
        capture_media=media,
    )
    flags = tuple(ActionOutputFlags(i % 2 == 0, i == 3) for i in range(4))
    return _Case(data_plane, finalizer, receipt, tuple(records), tuple(routes), flags)


@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("zero_generation", [False, True])
def test_call_inputs_preserve_routes_context_targets_versions_and_flags(
    media: bool, zero_generation: bool
) -> None:
    case = _case(media=media, zero_generation=zero_generation)
    original = case.finalizer.finalize_rollout(
        "owner",
        case.receipt,
        reward=0.75,
        context_compaction=True,
        action_flags=case.flags,
    )
    if zero_generation:
        assert not original.valid
        assert "empty_generation" in original.rejection_reason
        rejected = case.finalizer.finalize_rollout_with_call_views(
            "owner", case.receipt, reward=0.75, action_flags=case.flags
        )
        assert not rejected.valid and rejected.call_views == ()
        assert rejected.rejection_reason == original.rejection_reason
        return
    assert original.valid and original.call_views == ()
    case.data_plane.reads.clear()
    row = case.finalizer.finalize_rollout_with_call_views(
        "owner", case.receipt, reward=0.75, action_flags=case.flags
    )
    assert row.valid, row.rejection_reason
    assert len(row.call_views) == 4
    assert row.token_ids == original.token_ids
    assert row.token_mask == original.token_mask
    assert row.logprobs == original.logprobs
    assert torch.equal(row.routed_experts, original.routed_experts)
    assert len(case.data_plane.reads) == (2 if media else 1)
    if media:
        assert case.data_plane.reads[-1]["sample_ids"] == ["owner/c1", "owner/c3"]
    covered = [0.0] * len(row.token_ids)
    for index, call in enumerate(row.call_views):
        current = call.row
        start, end = call.generation_start, call.generation_end
        assert current.valid and current.call_views == ()
        assert call.model_call_id == f"c{index}"
        assert call.generation_weight_version == (12 if index < 2 else 13)
        # Admission still sees the whole original segment's version range.
        assert (current.min_wv, current.max_wv) == (12, 13)
        assert current.reward == row.reward == 0.75
        assert current.prompt_len == start
        assert current.model_call_ids == (f"c{index}",)
        assert current.token_ids == row.token_ids[:end]
        assert current.logprobs == row.logprobs[:end]
        assert current.route_plan is None
        assert current.token_mask[:start] == [0.0] * start
        assert current.token_mask[start:] == row.token_mask[start:end]
        assert current.staging_keys == row.staging_keys
        expected_routes = case.routes[index].clone()
        if index:
            expected_routes[case.records[index].prev_len - 1] = 900 + index
        assert torch.equal(current.routed_experts, expected_routes)
        assert current.routed_experts.data_ptr() != row.routed_experts.data_ptr()
        for key in (INVALID_TOOL_CALL_MASK, MALFORMED_THINKING_MASK):
            assert (
                current.output_masks[key]
                == [False] * start + row.output_masks[key][start:end]
            )
        for position, mask in enumerate(current.token_mask):
            covered[position] += mask
    assert covered == row.token_mask
    assert case.data_plane.delete_count == 0
    assert case.data_plane.write_count == 4


def test_call_media_only_contains_occurrences_available_at_that_call() -> None:
    case = _case(media=True)
    row = case.finalizer.finalize_rollout_with_call_views(
        "owner", case.receipt, reward=1
    )
    assert row.valid, row.rejection_reason
    media = [call.row.media for call in row.call_views]
    assert media[0] is None
    assert media[1] is media[2]
    assert media[3] is row.media
    for index, expected in ((1, [1.0]), (2, [1.0]), (3, [1.0, 3.0])):
        assert media[index]["pixel_values"].tensors[0][:, 0].tolist() == expected
        assert media[index]["imgs_sizes"].tensors[0].tolist() == [[2, 2]] * len(
            expected
        )
        assert media[index]["num_frames"].tensors[0].tolist() == [1] * len(expected)


@pytest.mark.parametrize("corruption", ["routes", "tokens", "geometry", "flags"])
def test_late_corruption_returns_no_call_views(corruption: str) -> None:
    case = _case(media=True)
    stored = case.data_plane.rows["staged", "owner/c3"]
    flags = case.flags
    if corruption == "routes":
        stored["routed_experts"][0, 0, 0, 0] += 1
    elif corruption == "tokens":
        stored["token_ids_delta"][0, 0] += 1
    elif corruption == "geometry":
        stored[MEDIA_IMGS_SIZES_FIELD][0, 0, 0] = 4
    else:
        flags = flags[:-1]
    row = case.finalizer.finalize_rollout_with_call_views(
        "owner", case.receipt, reward=1, action_flags=flags
    )
    assert not row.valid and row.rejection_reason
    assert row.call_views == () and row.routed_experts is None
    assert case.data_plane.delete_count == 0


@pytest.mark.parametrize("enabled,deferred", [(False, False), (True, True)])
def test_call_views_reject_unsupported_assembly_modes(
    enabled: bool, deferred: bool
) -> None:
    finalizer = RolloutReassembler(
        _RecordingDataPlane(),
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
        router_replay_enabled=enabled,
        defer_routed_experts_to_policy=deferred,
    )
    with pytest.raises(ValueError, match="direct router replay"):
        finalizer.finalize_rollout_with_call_views("owner", None, reward=1)


def test_original_segment_length_cap_rejects_all_call_views() -> None:
    case = _case(media=False)
    row = case.finalizer.finalize_rollout_with_call_views(
        "owner", case.receipt, reward=1
    )
    assert row.valid and row.call_views
    limited = RolloutReassembler(
        case.data_plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
        router_replay_enabled=True,
        max_train_row_tokens=len(row.token_ids) - 1,
    )
    rejected = limited._cap_row_length(row)
    assert not rejected.valid and rejected.rejection_reason.startswith("row_too_long")
    assert rejected.call_views == ()
