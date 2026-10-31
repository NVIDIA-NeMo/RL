# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary-output selection and finalization over captured semantic fixtures.

Uses fake generation and in-memory storage, not live OpenCode or its gateway.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
)
from nemo_gym.token_id_capture.fingerprint import (
    FINGERPRINT_VERSION,
    assistant_fingerprint,
)
from nemo_gym.token_id_capture.staging.records import CallRecord

from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from tests.unit.experience.common_output_comparison import (
    selected_response_ids,
    selection_with_observed_metadata,
)
from tests.unit.experience.test_logical_owner_finalization import (
    PublicationDataPlane,
    decide_capture_input,
    finalize,
    gym_harness,
)

pytestmark = pytest.mark.nemo_gym


def message(text: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def record(index: int, output: list[dict], *, parent: int | None = None) -> CallRecord:
    return CallRecord(
        model_call_id=f"call-{index}",
        response_id=f"response-{index}",
        parent_call_id=None if parent is None else f"call-{parent}",
        prev_len=0 if parent is None else 3,
        delta_len=3,
        cum_len=3 if parent is None else 6,
        weight_version=7,
        digest=f"{index:064x}",
        extras_digest="0" * 64,
        staging_key=f"owner/call-{index}",
        mode="text" if parent is None else "token_in",
        chain_hash=f"{index:064x}",
        cumulative_hash=f"{index:064x}",
        output_fingerprint=assistant_fingerprint(output) or None,
        fingerprint_version=FINGERPRINT_VERSION,
    )


def test_multi_item_and_adjacent_calls_need_no_agent_boundaries() -> None:
    a, b, c = message("A"), message("B"), message("C")
    records = [record(1, [a, b]), record(2, [c], parent=1)]
    assert selected_response_ids(records, {"output": [a, b, c]}) == (
        "response-1",
        "response-2",
    )
    # A competing one-call interpretation must not silently win.
    records.append(record(3, [a, b, c]))
    with pytest.raises(ValueError, match="Ambiguous"):
        selected_response_ids(records, {"output": [a, b, c]})


@pytest.mark.parametrize("same_text", [False, True])
def test_rejected_attempts_are_not_automatically_training(same_text: bool) -> None:
    accepted, rejected, final = (
        message("A"),
        message("A" if same_text else "B"),
        message("C"),
    )
    records = [record(1, [accepted]), record(2, [rejected]), record(3, [final])]
    response = {"id": "response-3", "output": [accepted, final]}
    if same_text:
        with pytest.raises(ValueError, match="Ambiguous"):
            selected_response_ids(records, response)
    else:
        assert selected_response_ids(records, response) == ("response-1", "response-3")


def test_verified_parent_disambiguates_identical_retry_output() -> None:
    a, b = message("same"), message("next")
    records = [record(1, [a]), record(2, [a]), record(3, [b], parent=2)]
    assert selected_response_ids(records, {"output": [a, b]}) == (
        "response-2",
        "response-3",
    )


def test_parent_ancestry_is_not_an_implicit_loss_membership_rule() -> None:
    a, b = message("auxiliary"), message("accepted")
    records = [record(1, [a]), record(2, [b], parent=1)]
    with pytest.raises(ValueError, match="No complete"):
        selected_response_ids(records, {"output": [b]})


def test_erased_root_is_not_detectable_from_schema_alone() -> None:
    a, b, c = message("choice A"), message("choice B"), message("final C")
    records = [record(1, [a]), record(2, [b]), record(3, [c])]
    # This is valid for a final-action-only policy, but cannot prove whether A
    # or B was an earlier accepted action. The join cannot certify completeness.
    assert selected_response_ids(records, {"output": [c]}) == ("response-3",)


def test_opencode_shaped_outputs_ignore_producer_segment_labels() -> None:
    before, next_action, summary, after = map(
        message, ["before", "tool action", "summary", "after"]
    )
    records = [
        record(1, [before]),
        record(2, [next_action], parent=1),
        record(3, [summary]),
        record(4, [after]),
    ]
    # Mirrors the extension's post-prefix-slice responses[], not raw prompts.
    responses = [
        {"id": "synthetic-seg-0", "segment_id": 100, "output": [before, next_action]},
        {"id": "synthetic-seg-1", "segment_id": 200, "output": [summary]},
        {"id": "synthetic-seg-2", "segment_id": 300, "output": [after]},
    ]

    def join() -> tuple[str, ...]:
        return selected_response_ids(
            records,
            {
                "id": "synthetic-final",
                "output": [
                    item for response in responses for item in response["output"]
                ],
            },
        )

    expected = tuple(f"response-{i}" for i in range(1, 5))
    assert join() == expected
    for response in responses:
        response.pop("segment_id")
    assert join() == expected
    assert sum(r.parent_call_id is None for r in records) == 3


@pytest.mark.parametrize("include_summary", [False, True])
def test_opencode_shaped_selection_through_tq(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, include_summary: bool
) -> None:
    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=lambda keys: source.fetch_prefix_chains(keys).expanded,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
    )
    worker = NeMoGymAsyncOpenAI.create_chat_completion

    async def distinct_answers(client: Any, **body: Any) -> dict:
        payload = await worker(client, **body)
        payload["choices"][0]["message"]["content"] = (
            f"answer {len(harness.worker_calls)}"
        )
        return payload

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", distinct_answers)
    try:
        first = gym_harness.assert_clean(harness.post("owner", gym_harness.HISTORY))
        second = gym_harness.assert_clean(
            harness.post(
                "owner",
                gym_harness.HISTORY
                + first["output"]
                + [{"role": "user", "content": "next"}],
            )
        )
        # All five calls are captured. Only ordinary accepted output below
        # determines which of those generations receive the rollout's reward.
        gym_harness.assert_clean(
            harness.post("owner", [{"role": "user", "content": "rejected attempt"}])
        )
        summary = gym_harness.assert_clean(
            harness.post("owner", [{"role": "user", "content": "summarize"}])
        )
        after = gym_harness.assert_clean(
            harness.post(
                "owner",
                [{"role": "user", "content": "rewritten context"}]
                + summary["output"]
                + [{"role": "user", "content": "continue"}],
            )
        )
        manifest = harness.manifest("owner")
        assert [r.parent_call_id is None for r in manifest.records] == [
            True,
            False,
            True,
            True,
            True,
        ]
        responses = [
            {"id": "synthetic-before", "output": first["output"] + second["output"]},
            *(
                [{"id": "synthetic-summary", "output": summary["output"]}]
                if include_summary
                else []
            ),
            {"id": "synthetic-after", "output": after["output"]},
        ]
        ordinary = {
            "id": "synthetic-scored",
            "output": [item for response in responses for item in response["output"]],
        }
        candidate = selected_response_ids(manifest.records, ordinary)
        expected = (
            first["id"],
            second["id"],
            *([summary["id"]] if include_summary else []),
            after["id"],
        )
        assert candidate == expected
        results = []
        for ids in (expected, candidate):
            replay = PublicationDataPlane()
            replay.rows = {key: value.clone() for key, value in plane.rows.items()}
            finalizer = RolloutReassembler(
                replay,
                partition_id="canonical",
                staging_partition="staged",
                pad_token_id=0,
                max_seq_len=1024,
            )
            finalized = finalize(
                finalizer, [[{"scope": "owner", "manifest": manifest, "ids": ids}]]
            )
            meta = finalized.meta
            assert meta is not None and finalized.valid_row_count == (
                3 if include_summary else 2
            )
            assert not any(partition == "staged" for partition, _ in replay.rows)
            data = replay.get_samples(
                sample_ids=meta.sample_ids,
                partition_id="canonical",
                select_fields=meta.fields,
            )
            actual_tokens = [
                token
                for tokens, mask in zip(
                    data["input_ids"].unbind(), data["token_mask"].unbind(), strict=True
                )
                for token in tokens[mask.bool()].tolist()
            ]
            assert actual_tokens == [
                1001,
                1002,
                *([1004] if include_summary else []),
                1005,
            ]
            results.append((meta, data, replay.events))
        assert results[0][0] == results[1][0]
        assert results[0][2] == results[1][2]
        for field in results[0][1].keys():
            for actual, expected_tensor in zip(
                results[0][1][field].unbind(),
                results[1][1][field].unbind(),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected_tensor, rtol=0, atol=0)
    finally:
        harness.client.close()
        asyncio.run(harness.ledger.close())


def test_reasoning_only_capture_fails_visibly() -> None:
    reasoning = {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "thought"}],
    }
    with pytest.raises(ValueError, match="reasoning-only"):
        selected_response_ids([record(1, [reasoning])], {"output": [reasoning]})


def test_native_terminal_must_agree_with_output() -> None:
    a, b = message("A"), message("B")
    with pytest.raises(ValueError, match="No complete"):
        selected_response_ids(
            [record(1, [a]), record(2, [b])], {"id": "response-1", "output": [a, b]}
        )


def test_observed_length_and_output_flags_survive_join() -> None:
    item = message("<think>leaked thinking</think>")
    response = {"id": "response-1", "output": [item]}
    observed = response | {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    selection = selection_with_observed_metadata(
        [record(1, [item])], response, {"response-1": observed}
    )
    assert selection.truncated
    assert selection.action_flags[0].malformed_thinking
