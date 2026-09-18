# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner composition over real Gym capture, TQ codecs, and finalization.

Only generation and DataPlane storage are doubles; no verifier, custody, or
publication method is replaced. This does not qualify live TransferQueue/Ray.
"""

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    FinalizedGroup,
    RolloutReassembler,
    SegmentReceipt,
)
from tests.unit.experience.test_segment_capture_composition import (
    MemoryDataPlane,
    gym_harness,
    receipt_for,
)

pytestmark = pytest.mark.nemo_gym


class PublicationDataPlane(MemoryDataPlane):
    """Record actual published rows and exact cleanup ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str, list[str]]] = []

    def put_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict,
        tags: list[dict],
    ) -> None:
        self.events.append(("put", partition_id, list(sample_ids)))
        # Integer indexing works for both dense staging and jagged canonical
        # fields; batch slicing of a jagged tensor is not implemented by Torch.
        for index, key in enumerate(sample_ids):
            self.rows[partition_id, key] = TensorDict(
                {
                    name: value[index].unsqueeze(0).clone()
                    for name, value in fields.items()
                },
                batch_size=[1],
            )
            self.write_count += 1
        if self.lose_ack:
            raise OSError("write completed but acknowledgement lost")

    def clear_samples(self, *, sample_ids: list[str], partition_id: str) -> None:
        self.events.append(("clear", partition_id, list(sample_ids)))
        self.delete_count += 1
        for key in sample_ids:
            del self.rows[partition_id, key]


@pytest.fixture
def owner_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[Any, PublicationDataPlane, RolloutReassembler]]:
    data_plane = PublicationDataPlane()
    source = TQTokenSource(data_plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(data_plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
    )
    finalizer = RolloutReassembler(
        data_plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
    )
    yield harness, data_plane, finalizer
    harness.client.close()
    asyncio.run(harness.ledger.close())


def capture_segment(
    harness: Any, scope: str, *, retry: bool = False, child: bool = False
) -> SegmentReceipt:
    if retry:
        gym_harness.assert_clean(harness.post(scope, gym_harness.HISTORY, parent=None))
    selected = gym_harness.assert_clean(
        harness.post(scope, gym_harness.HISTORY, parent=None)
    )
    response_ids = [selected["id"]]
    if child:
        selected = gym_harness.assert_clean(
            harness.post(
                scope, gym_harness.HISTORY + selected["output"], parent=selected["id"]
            )
        )
        response_ids.append(selected["id"])
    return SegmentReceipt(
        capture_rollout_id=scope,
        receipt=receipt_for(harness, scope, selected["id"]),
        selected_response_ids=tuple(response_ids),
    )


def finalize(
    finalizer: RolloutReassembler,
    owners: list[list[SegmentReceipt]],
    *,
    mask_sample: list[bool] | None = None,
    execution_row_multiple: int = 1,
) -> FinalizedGroup:
    return finalizer.finalize_group(
        "group",
        [f"group_g{i}" for i in range(len(owners))],
        [None] * len(owners),
        [float(i + 1) for i in range(len(owners))],
        mask_sample=mask_sample if mask_sample is not None else [False] * len(owners),
        fallback_weight_version=7,
        prompt_idx=99,
        logical_segments=owners,
        execution_row_multiple=execution_row_multiple,
    )


def test_missing_action_flag_invalidates_owner_and_emits_no_output_penalty_mask(
    owner_stack,
):
    harness, plane, finalizer = owner_stack
    first = capture_segment(harness, "group_g0_s0", child=True)
    first = replace(first, action_flags=(ActionOutputFlags(True, True),))
    second = capture_segment(harness, "group_g0_s1")
    sibling = capture_segment(harness, "group_g1_s0")
    result = finalize(finalizer, [[first, second], [sibling]])
    assert result.total_row_count == 2 and result.valid_row_count == 1
    for key in ("invalid_tool_call_mask", "malformed_thinking_mask", "token_mask"):
        assert not plane.rows["canonical", "group_g0_s0"][key].any()
    assert sum(tag["num_assistant_messages"] for tag in result.meta.tags) == 4


@pytest.mark.parametrize("versions", [[7] * 5, [6, 7, 8, 9, 10]])
def test_unequal_segments_publish_once_with_owner_identity_and_cleanup(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, versions: list[int]
) -> None:
    capture_type = gym_harness.RolloutTokenCapture
    version_iter = iter(versions)
    monkeypatch.setattr(
        gym_harness,
        "RolloutTokenCapture",
        lambda **kwargs: capture_type(
            **(kwargs | {"weight_version_fn": lambda: next(version_iter)})
        ),
    )
    harness, data_plane, finalizer = request.getfixturevalue("owner_stack")
    owners = [
        [
            capture_segment(harness, "group_g0_s0", retry=True, child=True),
            capture_segment(harness, "group_g0_s1"),
        ],
        [capture_segment(harness, "group_g1_s0")],
    ]
    staged = {key for partition, key in data_plane.rows if partition == "staged"}
    result = finalize(finalizer, owners, mask_sample=[True, False])
    assert result.meta is not None and not result.dropped
    # Reuse ordinary capture's conservative range over the manifest, including retries.
    assert (result.group_min_wv, result.group_max_wv) == (
        min(versions),
        max(versions),
    )
    assert all(tag["weight_version"] == min(versions) for tag in result.meta.tags)
    assert result.metrics["finalize/logical_owner_count"] == 2
    assert result.metrics["finalize/valid_logical_owner_count"] == 2
    assert result.meta.sample_ids == ["group_g0_s0", "group_g0_s1", "group_g1_s0"]
    expected = [
        ([20, 21, 1002, 30, 31, 1003], [20, 21]),
        ([40, 41, 1004], [20, 21]),
        ([50, 51, 1005], [50, 51]),
    ]
    for index, (sample_id, (tokens, prompt)) in enumerate(
        zip(result.meta.sample_ids, expected)
    ):
        row = data_plane.rows["canonical", sample_id]
        assert row["input_ids"][0].tolist() == tokens
        assert row["token_mask"][0].tolist() == [0.0, 0.0, 1.0] * (len(tokens) // 3)
        assert row["generation_logprobs"][0].tolist() == [0.0, 0.0, -0.25] * (
            len(tokens) // 3
        )
        assert row["prompt_ids_for_adv"][0].tolist() == prompt
        assert row["sample_mask"].item() == 1
        assert row["mask_sample"].item() == (index < 2)
        assert row["total_reward"].item() == (1 if index < 2 else 2)
        tag = result.meta.tags[index]
        assert {
            key: tag[key]
            for key in (
                "dispatch_group_id",
                "logical_rollout_id",
                "logical_slot",
                "logical_group_size",
                "segment_index",
                "segment_count",
                "is_execution_padding",
            )
        } == {
            "dispatch_group_id": "group",
            "logical_rollout_id": f"group_g{int(index == 2)}",
            "logical_slot": int(index == 2),
            "logical_group_size": 2,
            "segment_index": index if index < 2 else 0,
            "segment_count": 2 if index < 2 else 1,
            "is_execution_padding": False,
        }
        assert tag["uses_borrowed_input"] is False
    assert data_plane.events[-2] == ("put", "canonical", result.meta.sample_ids)
    assert data_plane.events[-1][:2] == ("clear", "staged")
    assert set(data_plane.events[-1][2]) == staged and len(staged) == 5
    assert (
        len([event for event in data_plane.events if event[:2] == ("put", "canonical")])
        == 1
    )
    assert not any(partition == "staged" for partition, _ in data_plane.rows)


@pytest.mark.parametrize(
    "failure",
    [
        "later_digest",
        "all_invalid",
        "first_child_digest",
        "later_missing",
        "duplicate",
        "wrong_order",
        "omitted",
        "foreign",
    ],
)
def test_failed_segment_collapses_entire_owner_without_losing_initial_prompt(
    owner_stack: tuple, failure: str
) -> None:
    harness, data_plane, finalizer = owner_stack
    initial = capture_segment(harness, "group_g0_s0", child=True)
    later = capture_segment(harness, "group_g0_s1", child=True)
    sibling = capture_segment(harness, "group_g1_s0")
    if failure in {"later_digest", "first_child_digest", "all_invalid"}:
        target = initial if failure == "first_child_digest" else later
        key = target.receipt["manifest"][-1]["staging_key"]
        data_plane.rows["staged", key]["token_ids_delta"][0, -1] += 1
        if failure == "all_invalid":
            sibling = replace(
                sibling, receipt={**sibling.receipt, "capture_poisoned": True}
            )
    elif failure == "later_missing":
        later = replace(later, receipt=None)
    else:
        selected = {
            "duplicate": later.selected_response_ids * 2,
            "wrong_order": tuple(reversed(later.selected_response_ids)),
            "omitted": later.selected_response_ids[1:],
            "foreign": sibling.selected_response_ids,
        }[failure]
        later = replace(later, selected_response_ids=selected)
    if failure == "all_invalid":
        with pytest.raises(ValueError, match="no verified input layout"):
            finalize(finalizer, [[initial, later], [sibling]])
        assert not any(partition == "canonical" for partition, _ in data_plane.rows)
        assert any(partition == "staged" for partition, _ in data_plane.rows)
        return
    result = finalize(finalizer, [[initial, later], [sibling]])
    assert result.meta is not None and result.meta.sample_ids == [
        "group_g0_s0",
        "group_g1_s0",
    ]
    failed = data_plane.rows["canonical", "group_g0_s0"]
    assert (
        failed["sample_mask"].item() == 0 and failed["token_mask"][0].sum().item() == 0
    )
    assert failed["prompt_ids_for_adv"][0].tolist() == [10, 11]
    assert failed["total_reward"].item() == 1
    assert result.meta.tags[0]["segment_count"] == 1
    assert result.meta.tags[0]["uses_borrowed_input"] is True
    assert result.meta.tags[0]["is_execution_padding"] is False
    assert result.meta.tags[1]["uses_borrowed_input"] is False
    expected_valid = int(failure != "all_invalid")
    assert (
        data_plane.rows["canonical", "group_g1_s0"]["sample_mask"].item()
        == expected_valid
    )
    assert result.metrics["finalize/logical_owner_count"] == 2
    assert result.metrics["finalize/valid_logical_owner_count"] == expected_valid


@pytest.mark.parametrize("logprobs", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_failed_owner_checks_router_before_forward_without_execution_padding(
    owner_stack: tuple,
    monkeypatch: pytest.MonkeyPatch,
    logprobs: bool,
    supported: bool,
) -> None:
    # Load the heavier controller composition only for this integration test.
    from tests.unit.single_controller.test_cc_optimizer_batch import _setup

    harness, plane, finalizer = owner_stack
    first = capture_segment(harness, "group_g0_s0", child=True)
    first = replace(first, action_flags=(ActionOutputFlags(True, True),))
    sibling = capture_segment(harness, "group_g1_s0")
    result = finalize(finalizer, [[first], [sibling]], execution_row_multiple=2)
    meta = result.meta
    assert meta is not None and meta.size == 2
    assert result.metrics["finalize/execution_padding_rows"] == 0
    assert [tag["logical_rollout_id"] for tag in meta.tags] == ["group_g0", "group_g1"]
    assert not any(tag["is_execution_padding"] for tag in meta.tags)
    assert [tag["uses_borrowed_input"] for tag in meta.tags] == [True, False]
    data = plane.get_samples(
        sample_ids=meta.sample_ids,
        partition_id=meta.partition_id,
        select_fields=meta.fields,
    )
    torch.testing.assert_close(data["input_ids"][0], data["input_ids"][1])
    assert data["sample_mask"].tolist() == [0, 1]
    assert not data["token_mask"][0].any()
    assert data["prompt_ids_for_adv"][0].tolist() == [10, 11]
    assert data["total_reward"].tolist() == [1, 2]

    ctrl, _ = _setup(monkeypatch, batches=[(meta, data)])
    ctrl._policy_logprobs_required = ctrl._reference_logprobs_required = logprobs
    check = ctrl._trainer.worker_group.run_all_workers_single_data
    if not supported:
        check.side_effect = ValueError("unsafe router")
        with pytest.raises(ValueError, match="unsafe router"):
            asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=3))
        for method in (
            "prepare_for_lp_inference",
            "get_logprobs_from_meta",
            "get_reference_policy_logprobs_from_meta",
            "begin_train_step",
            "train_microbatches_from_meta",
            "finish_train_step",
        ):
            getattr(ctrl._trainer, method).assert_not_called()
        ctrl._sync_weights.assert_not_awaited()
    else:
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=3))
        submitted = ctrl._trainer.train_microbatches_from_meta.call_args.args[0]
        assert submitted.sample_ids == meta.sample_ids
        assert [tag["logical_rollout_id"] for tag in submitted.tags] == [
            "group_g0",
            "group_g1",
        ]
        ctrl._trainer.finish_train_step.assert_called_once()
    check.assert_called_once_with("validate_cc_execution_padding")


@pytest.mark.parametrize("multiple", [1, 4, 8])
def test_padding_preserves_real_rows_and_masks_ownerless_inputs(owner_stack, multiple):
    from nemo_rl.data_plane.preshard import shard_meta_for_dp

    harness, plane, finalizer = owner_stack
    owners = [
        [
            capture_segment(harness, "group_g0_s0"),
            capture_segment(harness, "group_g0_s1"),
        ],
        [capture_segment(harness, "group_g1_s0")],
    ]
    result = finalize(finalizer, owners, execution_row_multiple=multiple)
    count = (-3) % multiple
    assert result.meta.size == 3 + count
    assert result.total_row_count == result.valid_row_count == 3
    assert result.metrics["finalize/logical_owner_count"] == 2
    assert result.metrics["finalize/execution_padding_rows"] == count
    prototype = plane.rows["canonical", "group_g0_s0"]
    for index in range(count):
        row = plane.rows["canonical", f"group_pad{index}"]
        torch.testing.assert_close(row["input_ids"], prototype["input_ids"])
        for field in (
            "sample_mask",
            "token_mask",
            "generation_logprobs",
            "invalid_tool_call_mask",
            "malformed_thinking_mask",
            "total_reward",
        ):
            assert not row[field].any()
        assert row["mask_sample"].item()
        tag = result.meta.tags[3 + index]
        assert tag["uses_borrowed_input"] is True
        assert tag["is_execution_padding"] and tag["logical_rollout_id"] is None
        assert tag["segment_count"] == tag["num_assistant_messages"] == 0
    # Exercise the existing metadata sharder, including more ranks than real rows.
    shards, _ = shard_meta_for_dp(result.meta, dp_world=multiple, batch_size=None)
    assert len({shard.size for shard in shards}) == 1
    assert sorted(key for shard in shards for key in shard.sample_ids) == sorted(
        result.meta.sample_ids
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing_receipt",
        "missing_root",
        "corrupt_root",
        "foreign_scope",
        "foreign_key",
        "empty_owner",
    ],
)
def test_untrustworthy_original_prompt_or_foreign_custody_aborts_without_cleanup(
    owner_stack: tuple, failure: str
) -> None:
    harness, data_plane, finalizer = owner_stack
    segment = capture_segment(harness, "group_g0_s0")
    foreign = capture_segment(harness, "other_g0_s0")
    if failure == "missing_receipt":
        segment = replace(segment, receipt=None)
    elif failure == "missing_root":
        segment = replace(segment, selected_response_ids=("not-a-recorded-response",))
    elif failure == "corrupt_root":
        key = segment.receipt["manifest"][0]["staging_key"]
        data_plane.rows["staged", key]["token_ids_delta"][0, 0] += 1
    elif failure == "foreign_scope":
        segment = foreign
    elif failure == "foreign_key":
        segment.receipt["manifest"][0]["staging_key"] = foreign.receipt["manifest"][0][
            "staging_key"
        ]
    staged = set(data_plane.rows)
    with pytest.raises(ValueError):
        finalize(finalizer, [[] if failure == "empty_owner" else [segment]])
    assert set(data_plane.rows) == staged
    assert data_plane.delete_count == 0
    assert all(event[:2] != ("put", "canonical") for event in data_plane.events)


def test_canonical_write_ack_loss_is_fatal_and_preserves_all_staging(
    owner_stack: tuple,
) -> None:
    harness, data_plane, finalizer = owner_stack
    owners = [
        [
            capture_segment(harness, "group_g0_s0", retry=True),
            capture_segment(harness, "group_g0_s1"),
        ]
    ]
    staged = {key for partition, key in data_plane.rows if partition == "staged"}
    data_plane.lose_ack = True
    with pytest.raises(OSError, match="acknowledgement lost"):
        finalize(finalizer, owners)
    assert {
        key for partition, key in data_plane.rows if partition == "staged"
    } == staged
    assert data_plane.delete_count == 0
    assert data_plane.events[-1] == ("put", "canonical", ["group_g0_s0", "group_g0_s1"])
    assert all(
        ("canonical", key) in data_plane.rows for key in data_plane.events[-1][2]
    )
