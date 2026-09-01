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
import json
from pathlib import Path

import pytest
import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.experience.rollout_traces import build_trace_batch_plan
from nemo_rl.experience.trace_batch_materialization import (
    materialize_trace_batch_plan,
    validate_trace_batch_materialization,
)
from nemo_rl.experience.trace_batch_scoring import _build_logprob_data


_FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "context_compaction_traces"
_PAD_TOKEN_ID = 999


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text())


def _plan(*bundles: dict, advantage: float = 2.5, batch_quantum: int = 1) -> dict:
    return build_trace_batch_plan(
        list(bundles),
        rollout_advantages={bundle["rollout_id"]: advantage for bundle in bundles},
        expected_rollouts_per_group=1,
        batch_quantum=batch_quantum,
        optimizer_step_id="materialization-test",
    )


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
                "role": ("user" if segment["kind"] == "prompt" else "assistant"),
                "content": f"{segment['kind']}-{segment_index}",
                "token_ids": token_ids,
                # Deliberately wrong: the exact trace segment must override this.
                "token_loss_mask": torch.full_like(token_ids, 7),
            }
            if segment["kind"] == "completion":
                message["generation_logprobs"] = torch.tensor(
                    segment["generation_logprobs"],
                    dtype=torch.float32,
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
                count = token_ids.numel()
                message["routed_experts"] = (
                    torch.arange(4, dtype=torch.int32)
                    .view(1, 1, 4)
                    .expand(count, 2, 4)
                    .clone()
                )
            messages.append(message)
        result.append(messages)
    return result


def _materialize(
    bundles: list[dict],
    *,
    advantage: float = 2.5,
    batch_quantum: int = 1,
    logs: dict[str, list[list[dict]]] | None = None,
    divisible_by: int = 1,
):
    plan = _plan(
        *bundles,
        advantage=advantage,
        batch_quantum=batch_quantum,
    )
    logs = logs or {bundle["rollout_id"]: _message_logs(bundle) for bundle in bundles}
    materialization = materialize_trace_batch_plan(
        plan,
        bundles=bundles,
        physical_message_logs_by_rollout=logs,
        pad_token_id=_PAD_TOKEN_ID,
        make_sequence_length_divisible_by=divisible_by,
    )
    return plan, materialization


def test_identity_trace_materializes_exact_pre_scoring_tensors():
    bundle = _fixture("without_compaction.json")

    plan, materialization = _materialize([bundle], advantage=1.75)
    train_data = materialization["train_data"]
    trace = bundle["physical_traces"][0]
    expected_tokens = [
        token for segment in trace["segments"] for token in segment["token_ids"]
    ]
    expected_mask = [
        token for segment in trace["segments"] for token in segment["loss_mask"]
    ]
    expected_logprobs = [
        logprob
        for segment in trace["segments"]
        for logprob in (
            segment["generation_logprobs"]
            if segment["kind"] == "completion"
            else [0.0] * len(segment["token_ids"])
        )
    ]

    assert train_data["input_ids"][0].tolist() == expected_tokens
    assert train_data["token_mask"][0].tolist() == expected_mask
    assert torch.allclose(
        train_data["generation_logprobs"][0],
        torch.tensor(expected_logprobs),
    )
    assert torch.all(train_data["advantages"] == 1.75)
    assert train_data["ordered_media_ids"] == [trace["ordered_media_ids"]]
    cache_keys = train_data["image_cache_keys"]
    assert isinstance(cache_keys, PackedTensor)
    assert cache_keys.tensors[0].shape == (len(trace["ordered_media_ids"]), 2)
    assert cache_keys.tensors[0].dtype == torch.int64
    assert plan["eligible_action_token_count"] == int(
        train_data["token_mask"].sum().item()
    )


def test_compacted_trace_rows_share_advantage_and_padding_is_fully_masked():
    bundle = _fixture("k2_compaction.json")

    plan, materialization = _materialize(
        [bundle],
        advantage=-1.25,
        batch_quantum=4,
        divisible_by=8,
    )
    train_data = materialization["train_data"]

    assert train_data["input_ids"].shape == (4, 8)
    assert train_data["input_lengths"].tolist() == [5, 6, 3, 1]
    assert torch.all(train_data["advantages"][:3] == -1.25)
    assert torch.count_nonzero(train_data["advantages"][3]).item() == 0
    assert torch.count_nonzero(train_data["token_mask"][3]).item() == 0
    assert torch.count_nonzero(train_data["generation_logprobs"][3]).item() == 0
    assert torch.all(train_data["input_ids"][3] == _PAD_TOKEN_ID)
    assert train_data["sample_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert plan["parent_indices"] == [0, 0, 0, -1]


def test_materialization_does_not_mutate_source_message_logs():
    bundle = _fixture("k2_compaction.json")
    source_logs = _message_logs(bundle)
    original_masks = [
        message["token_loss_mask"].clone()
        for trace_log in source_logs
        for message in trace_log
    ]

    _materialize(
        [bundle],
        logs={bundle["rollout_id"]: source_logs},
    )

    observed_masks = [
        message["token_loss_mask"] for trace_log in source_logs for message in trace_log
    ]
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(observed_masks, original_masks)
    )


def test_returned_message_logs_retain_only_metric_text_without_tensor_aliases():
    bundle = _fixture("k2_compaction.json")
    source_logs = _message_logs(
        bundle,
        visual_trace_indices={0},
        routed_experts=True,
    )

    _, materialization = _materialize(
        [bundle],
        batch_quantum=4,
        logs={bundle["rollout_id"]: source_logs},
    )

    assert materialization["materialized_message_logs_are_compact"] is True
    retained_logs = materialization["materialized_message_logs"]
    assert [row[0]["content"] for row in retained_logs[:3]] == [
        "".join(str(message["content"]) for message in trace_log)
        for trace_log in source_logs
    ]
    assert retained_logs[3] == [{"content": ""}]
    assert all(
        set(message) == {"content"}
        and not any(
            isinstance(value, (torch.Tensor, PackedTensor))
            for value in message.values()
        )
        for message_log in retained_logs
        for message in message_log
    )


def test_text_first_batch_preserves_later_visual_row_ownership():
    compacted = _fixture("k2_compaction.json")
    identity = _fixture("without_compaction.json")
    logs = {
        compacted["rollout_id"]: _message_logs(compacted),
        identity["rollout_id"]: _message_logs(
            identity,
            visual_trace_indices={0},
        ),
    }

    _, materialization = _materialize(
        [compacted, identity],
        batch_quantum=4,
        logs=logs,
    )
    packed = materialization["train_data"]["pixel_values"]

    assert isinstance(packed, PackedTensor)
    assert len(packed) == 4
    assert packed.tensors[:3] == [None, None, None]
    assert torch.equal(
        packed.tensors[3],
        torch.ones((1, 3, 2, 2)),
    )


def test_visual_first_batch_preserves_padding_as_empty_media_row():
    bundle = _fixture("without_compaction.json")
    logs = {
        bundle["rollout_id"]: _message_logs(
            bundle,
            visual_trace_indices={0},
        )
    }

    _, materialization = _materialize(
        [bundle],
        batch_quantum=2,
        logs=logs,
    )
    packed = materialization["train_data"]["pixel_values"]

    assert len(packed) == 2
    assert torch.equal(packed.tensors[0], torch.ones((1, 3, 2, 2)))
    assert packed.tensors[1] is None


def test_ordered_media_and_cache_keys_survive_logprob_projection():
    bundle = _fixture("k2_compaction.json")
    _, materialization = _materialize([bundle], batch_quantum=4)
    train_data = materialization["train_data"]
    logprob_data = _build_logprob_data(materialization)
    expected_order = [
        list(trace["ordered_media_ids"]) for trace in bundle["physical_traces"]
    ] + [[]]

    assert train_data["ordered_media_ids"] == expected_order
    assert logprob_data["ordered_media_ids"] == expected_order
    train_keys = train_data["image_cache_keys"]
    logprob_keys = logprob_data["image_cache_keys"]
    assert isinstance(train_keys, PackedTensor)
    assert isinstance(logprob_keys, PackedTensor)
    assert len(train_keys) == len(logprob_keys) == 4
    for train_row, logprob_row in zip(train_keys.tensors, logprob_keys.tensors):
        if train_row is None:
            assert logprob_row is None
        else:
            assert torch.equal(train_row, logprob_row)


def test_routed_experts_follow_exact_tokens_and_use_valid_padding_route():
    bundle = _fixture("k2_compaction.json")
    logs = {
        bundle["rollout_id"]: _message_logs(
            bundle,
            routed_experts=True,
        )
    }

    _, materialization = _materialize(
        [bundle],
        batch_quantum=4,
        logs=logs,
    )
    train_data = materialization["train_data"]

    assert train_data["routed_experts"].shape[:2] == (
        4,
        train_data["input_ids"].shape[1],
    )
    assert torch.equal(
        train_data["routed_experts"][3, 0],
        torch.arange(4, dtype=torch.int32).view(1, 4).expand(2, 4),
    )
    assert torch.count_nonzero(train_data["routed_experts"][3, 1:]).item() == 0


def test_partial_routed_expert_evidence_fails_closed():
    bundle = _fixture("k2_compaction.json")
    logs = _message_logs(bundle, routed_experts=True)
    del logs[1][0]["routed_experts"]

    with pytest.raises(ValueError, match="must cover every physical"):
        _materialize(
            [bundle],
            logs={bundle["rollout_id"]: logs},
        )


def test_token_mismatch_against_exact_trace_fails_closed():
    bundle = _fixture("k2_compaction.json")
    logs = _message_logs(bundle)
    logs[1][0]["token_ids"][0] = -1

    with pytest.raises(ValueError, match="tokens disagree"):
        _materialize(
            [bundle],
            logs={bundle["rollout_id"]: logs},
        )


def test_generation_logprob_mismatch_fails_closed():
    bundle = _fixture("k2_compaction.json")
    logs = _message_logs(bundle)
    logs[0][1]["generation_logprobs"][0] = -9.0

    with pytest.raises(ValueError, match="logprobs disagree"):
        _materialize(
            [bundle],
            logs={bundle["rollout_id"]: logs},
        )


def test_rollout_message_log_mapping_must_be_exact():
    bundle = _fixture("without_compaction.json")
    plan = _plan(bundle)

    with pytest.raises(ValueError, match="missing=.*extra="):
        materialize_trace_batch_plan(
            plan,
            bundles=[bundle],
            physical_message_logs_by_rollout={"wrong-rollout": []},
            pad_token_id=_PAD_TOKEN_ID,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda materialization: materialization["train_data"]["token_mask"][
                0, 0
            ].fill_(1),
            "token mask is corrupted",
        ),
        (
            lambda materialization: materialization["train_data"]["advantages"][
                0, 0
            ].fill_(123),
            "advantage is corrupted",
        ),
        (
            lambda materialization: materialization["parent_indices"][0].fill_(-1),
            "parent indices are corrupted",
        ),
    ],
)
def test_independent_validator_rejects_corruption(mutation, match):
    bundle = _fixture("without_compaction.json")
    plan, materialization = _materialize([bundle])
    corrupted = deepcopy(materialization)
    mutation(corrupted)

    with pytest.raises(ValueError, match=match):
        validate_trace_batch_materialization(
            corrupted,
            plan=plan,
            bundles=[bundle],
            pad_token_id=_PAD_TOKEN_ID,
        )
