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

from copy import deepcopy

import pytest
import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.experience.trace_batch_materialization import prepare_trace_batch
from nemo_rl.models.generation.interfaces import (
    ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
)
from tests.unit.trace_test_utils import trace_bundle

_PAD_TOKEN_ID = 999


def _fixture(*, compacted: bool) -> dict:
    return trace_bundle(compacted=compacted)


def _message_logs(
    bundle: dict,
    *,
    visual_trace_indices: set[int] | None = None,
    routed_experts: bool = False,
) -> list[list[dict]]:
    visual_trace_indices = visual_trace_indices or set()
    result = []
    for trace in bundle["physical_traces"]:
        messages = []
        for segment_index, segment in enumerate(trace["segments"]):
            token_ids = torch.tensor(segment["token_ids"], dtype=torch.int64)
            message = {
                "role": "user" if segment["kind"] == "prompt" else "assistant",
                "content": f"{segment['kind']}-{segment_index}",
                "token_ids": token_ids,
                "token_loss_mask": torch.tensor(
                    segment["loss_mask"], dtype=torch.int64
                ),
            }
            if segment["kind"] == "completion":
                message["generation_logprobs"] = torch.tensor(
                    segment["generation_logprobs"], dtype=torch.float32
                )
            if trace["trace_index"] in visual_trace_indices and segment_index == 0:
                message["pixel_values"] = PackedTensor(
                    torch.full(
                        (1, 3, 2, 2),
                        fill_value=trace["trace_index"] + 1,
                        dtype=torch.float32,
                    ),
                    dim_to_pack=0,
                )
            if routed_experts:
                message["routed_experts"] = (
                    torch.arange(4, dtype=torch.int32)
                    .view(1, 1, 4)
                    .expand(token_ids.numel(), 2, 4)
                    .clone()
                )
            messages.append(message)
        result.append(messages)
    return result


def _prepare(
    bundles: list[dict],
    *,
    advantages: list[float] | None = None,
    batch_quantum: int = 1,
    logs: list[list[list[dict]]] | None = None,
    prompt_ids: torch.Tensor | None = None,
    loss_multipliers: torch.Tensor | None = None,
    mask_sample: torch.Tensor | None = None,
    truncated: torch.Tensor | None = None,
    mask_truncated: bool = False,
    divisible_by: int = 1,
    expected_rollouts_per_group: int = 1,
    require_single_generation_policy_version: bool = True,
):
    rollout_count = len(bundles)
    physical_logs = logs or [_message_logs(bundle) for bundle in bundles]
    return prepare_trace_batch(
        {
            "message_log": [rollout_logs[0] for rollout_logs in physical_logs],
            "rollout_id": [bundle["rollout_id"] for bundle in bundles],
            "group_id": [bundle["group_id"] for bundle in bundles],
            "generation_policy_version": [
                bundle["generation_policy_version"] for bundle in bundles
            ],
            "physical_trace_ids": [
                [trace["trace_id"] for trace in bundle["physical_traces"]]
                for bundle in bundles
            ],
            "physical_message_logs": physical_logs,
            "total_reward": torch.tensor(
                [bundle["reward"] for bundle in bundles], dtype=torch.float32
            ),
            "loss_multiplier": (
                loss_multipliers
                if loss_multipliers is not None
                else torch.ones(rollout_count)
            ),
            "mask_sample": (
                mask_sample
                if mask_sample is not None
                else torch.zeros(rollout_count, dtype=torch.bool)
            ),
            "truncated": (
                truncated
                if truncated is not None
                else torch.zeros(rollout_count, dtype=torch.bool)
            ),
        },
        prompt_ids=(
            prompt_ids
            if prompt_ids is not None
            else torch.arange(rollout_count * 3).reshape(rollout_count, 3)
        ),
        logical_advantages=torch.tensor(
            advantages if advantages is not None else [2.5] * rollout_count,
            dtype=torch.float32,
        ).unsqueeze(-1),
        expected_rollouts_per_group=expected_rollouts_per_group,
        batch_quantum=batch_quantum,
        pad_token_id=_PAD_TOKEN_ID,
        mask_truncated=mask_truncated,
        make_sequence_length_divisible_by=divisible_by,
        require_single_generation_policy_version=(
            require_single_generation_policy_version
        ),
    )


def test_prepare_trace_batch_owns_projection_and_worker_inputs():
    prepared = _prepare([_fixture(compacted=True)], batch_quantum=4)

    assert prepared.logical_rollout_count == 1
    assert prepared.physical_trace_count == 3
    assert prepared.padding_row_count == 1
    assert prepared.train_data.size == 4
    assert prepared.logprob_data.size == 4
    assert prepared.project_logical_rows(["owner"]) == [
        "owner",
        "owner",
        "owner",
        None,
    ]
    assert prepared.train_overrides(micro_batch_size=1) == {
        "gbs": 4,
        "mbs": 1,
        "scheduler_step_increment": 1,
    }
    assert prepared.metrics()["physical_trace_training/physical_rows"] == 4


def test_distinct_groups_preserve_logical_to_physical_ownership():
    identity = _fixture(compacted=False)
    compacted = _fixture(compacted=True)
    prepared = _prepare(
        [identity, compacted],
        advantages=[1.0, -1.0],
        prompt_ids=torch.tensor([[1, 2, 3], [1, 2, 3]]),
    )

    assert prepared.logical_rollout_count == 2
    assert prepared.padding_row_count == 0
    assert prepared.parent_indices.tolist() == [0, 1, 1, 1]
    assert prepared.row_rollout_ids == [
        identity["rollout_id"],
        compacted["rollout_id"],
        compacted["rollout_id"],
        compacted["rollout_id"],
    ]
    assert prepared.train_data["advantages"][:, 0].tolist() == [
        1.0,
        -1.0,
        -1.0,
        -1.0,
    ]


def test_async_batch_allows_distinct_policy_versions_across_complete_groups():
    older_group = _fixture(compacted=False)
    newer_group = _fixture(compacted=True)
    older_group["generation_policy_version"] = "async-policy-weight-00000003"
    newer_group["generation_policy_version"] = "async-policy-weight-00000004"

    prepared = _prepare(
        [older_group, newer_group],
        require_single_generation_policy_version=False,
    )

    assert prepared.logical_rollout_count == 2


def test_sync_batch_rejects_distinct_policy_versions_across_groups():
    older_group = _fixture(compacted=False)
    newer_group = _fixture(compacted=True)
    older_group["generation_policy_version"] = "sync-policy-step-00000003"
    newer_group["generation_policy_version"] = "sync-policy-step-00000004"

    with pytest.raises(ValueError, match="one generation policy version"):
        _prepare([older_group, newer_group])


def test_one_group_cannot_mix_generation_policy_versions():
    first = _fixture(compacted=False)
    second = deepcopy(first)
    second["rollout_id"] = f"{first['rollout_id']}-replica"
    for trace in second["physical_traces"]:
        trace["trace_id"] = f"{trace['trace_id']}-replica"
    first["generation_policy_version"] = "async-policy-weight-00000003"
    second["generation_policy_version"] = "async-policy-weight-00000004"

    with pytest.raises(ValueError, match="mixes generation policy versions"):
        _prepare(
            [first, second],
            prompt_ids=torch.tensor([[1, 2, 3], [1, 2, 3]]),
            expected_rollouts_per_group=2,
            require_single_generation_policy_version=False,
        )


def test_physical_trace_rows_require_generation_policy_provenance():
    bundle = _fixture(compacted=True)
    bundle["generation_policy_version"] = None

    with pytest.raises(ValueError, match="invalid policy provenance"):
        _prepare(
            [bundle],
            require_single_generation_policy_version=False,
        )


def test_one_group_cannot_own_different_prompts():
    identity = _fixture(compacted=False)
    compacted = _fixture(compacted=True)
    compacted["group_id"] = identity["group_id"]

    with pytest.raises(ValueError, match="owns more than one tokenized prompt"):
        _prepare(
            [identity, compacted],
            prompt_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
            expected_rollouts_per_group=2,
        )


@pytest.mark.parametrize(
    ("mask_sample", "truncated", "mask_truncated", "expected_sample_mask"),
    [
        (False, False, False, 0.5),
        (True, False, False, 0.0),
        (False, True, False, 0.5),
        (False, True, True, 0.0),
    ],
)
def test_logical_sample_masks_project_to_every_physical_trace(
    mask_sample: bool,
    truncated: bool,
    mask_truncated: bool,
    expected_sample_mask: float,
):
    prepared = _prepare(
        [_fixture(compacted=True)],
        batch_quantum=4,
        loss_multipliers=torch.tensor([0.5]),
        mask_sample=torch.tensor([mask_sample]),
        truncated=torch.tensor([truncated]),
        mask_truncated=mask_truncated,
    )

    assert prepared.train_data["sample_mask"].tolist() == [
        expected_sample_mask,
        expected_sample_mask,
        expected_sample_mask,
        0.0,
    ]


def test_identity_trace_materializes_exact_canonical_tensors():
    bundle = _fixture(compacted=False)
    prepared = _prepare([bundle], advantages=[1.75])
    train_data = prepared.train_data
    trace = bundle["physical_traces"][0]
    expected_tokens = [
        token for segment in trace["segments"] for token in segment["token_ids"]
    ]
    expected_mask = [
        token for segment in trace["segments"] for token in segment["loss_mask"]
    ]

    assert (
        train_data["input_ids"][0, : len(expected_tokens)].tolist() == expected_tokens
    )
    assert train_data["token_mask"][0, : len(expected_mask)].tolist() == expected_mask
    assert torch.all(train_data["advantages"] == 1.75)
    assert prepared.eligible_action_token_count == sum(expected_mask)


def test_compacted_rows_share_advantage_and_padding_is_fully_masked():
    prepared = _prepare(
        [_fixture(compacted=True)],
        advantages=[-1.25],
        batch_quantum=4,
        divisible_by=8,
    )
    train_data = prepared.train_data

    assert train_data["input_ids"].shape == (4, 8)
    assert train_data["input_lengths"].tolist() == [5, 6, 3, 1]
    assert torch.all(train_data["advantages"][:3] == -1.25)
    assert torch.count_nonzero(train_data["advantages"][3]).item() == 0
    assert torch.count_nonzero(train_data["token_mask"][3]).item() == 0
    assert torch.count_nonzero(train_data["generation_logprobs"][3]).item() == 0
    assert torch.all(train_data["input_ids"][3] == _PAD_TOKEN_ID)
    assert train_data["sample_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert prepared.parent_indices.tolist() == [0, 0, 0, -1]


def test_materialization_does_not_mutate_source_message_logs():
    bundle = _fixture(compacted=True)
    source_logs = _message_logs(bundle)
    before = deepcopy(source_logs)

    _prepare([bundle], logs=[source_logs])

    for observed_trace, expected_trace in zip(source_logs, before, strict=True):
        for observed, expected in zip(observed_trace, expected_trace, strict=True):
            assert observed.keys() == expected.keys()
            for key in observed:
                if isinstance(observed[key], torch.Tensor):
                    assert torch.equal(observed[key], expected[key])
                else:
                    assert observed[key] == expected[key]


def test_text_first_rows_preserve_later_visual_ownership():
    compacted = _fixture(compacted=True)
    identity = _fixture(compacted=False)
    prepared = _prepare(
        [compacted, identity],
        batch_quantum=4,
        logs=[
            _message_logs(compacted),
            _message_logs(identity, visual_trace_indices={0}),
        ],
    )
    packed = prepared.train_data["pixel_values"]

    assert isinstance(packed, PackedTensor)
    assert len(packed) == 4
    assert packed.tensors[:3] == [None, None, None]
    assert torch.equal(packed.tensors[3], torch.ones((1, 3, 2, 2)))


def test_visual_rows_preserve_padding_as_empty_media():
    bundle = _fixture(compacted=False)
    prepared = _prepare(
        [bundle],
        batch_quantum=2,
        logs=[_message_logs(bundle, visual_trace_indices={0})],
    )
    packed = prepared.train_data["pixel_values"]

    assert len(packed) == 2
    assert torch.equal(packed.tensors[0], torch.ones((1, 3, 2, 2)))
    assert packed.tensors[1] is None


def test_routed_experts_follow_exact_tokens_and_padding_is_masked():
    bundle = _fixture(compacted=True)
    prepared = _prepare(
        [bundle],
        batch_quantum=4,
        logs=[_message_logs(bundle, routed_experts=True)],
    )
    train_data = prepared.train_data

    assert train_data["routed_experts"].shape[:2] == train_data["input_ids"].shape
    assert torch.all(
        train_data["routed_experts"][3, 0] == ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL
    )
    assert torch.count_nonzero(train_data["routed_experts"][3, 1:]).item() == 0
    assert train_data["sample_mask"][3].item() == 0.0


def test_partial_routed_expert_evidence_is_backfilled_on_physical_logs():
    bundle = _fixture(compacted=True)
    logs = _message_logs(bundle, routed_experts=True)
    del logs[1][0]["routed_experts"]

    prepared = _prepare([bundle], logs=[logs])

    missing_message_length = logs[1][0]["token_ids"].numel()
    physical_row = prepared.train_data["routed_experts"][1]
    assert torch.all(
        physical_row[:missing_message_length] == ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL
    )
    assert torch.all(physical_row[missing_message_length:] >= 0)


def test_misaligned_canonical_mask_fails_closed():
    bundle = _fixture(compacted=True)
    logs = _message_logs(bundle)
    logs[1][0]["token_loss_mask"] = torch.zeros(2, dtype=torch.int64)

    with pytest.raises(ValueError, match="unaligned token loss mask"):
        _prepare([bundle], logs=[logs])


def test_misaligned_canonical_logprobs_fail_closed():
    bundle = _fixture(compacted=True)
    logs = _message_logs(bundle)
    logs[0][1]["generation_logprobs"] = torch.zeros(2)

    with pytest.raises(ValueError, match="unaligned generation logprobs"):
        _prepare([bundle], logs=[logs])
