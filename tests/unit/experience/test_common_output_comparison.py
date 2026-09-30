# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Same captured evidence, ordinary output versus LogicalCCResult selection.

Real shared client, HTTP capture, ledger, TQ codecs, finalizer and advantages;
fake generation and in-memory storage. OpenCode cases are semantic fixtures,
not execution of OpenCode or its gateway. The frozen baseline consumer is
executed unchanged as the migration oracle; generation is never repeated.
"""

import asyncio
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

import nemo_rl.environments.nemo_gym as gym_environment

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.config_types import ModelServerRef
from nemo_gym.context_management import (
    ContextHistoryConfig,
    ContextManagedResponsesClient,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.fingerprint import (
    FINGERPRINT_VERSION,
    assistant_fingerprint,
)
from nemo_gym.token_id_capture.staging.records import CallRecord, RolloutManifest
from responses_api_agents.simple_agent_with_compaction.tests.test_client import (
    http_response,
)

from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from tests.unit.experience.agent_exporter_probe import load_source_definitions
from tests.unit.experience.common_output_comparison import (
    selected_response_ids,
    selection_with_observed_metadata,
)
from tests.unit.experience.test_cc_dispatch import _env
from tests.unit.experience.test_logical_owner_finalization import (
    PublicationDataPlane,
    decide_capture_input,
    finalize,
    gym_harness,
)
from tests.unit.single_controller.test_logical_advantage import _controller

pytestmark = pytest.mark.nemo_gym


@pytest.mark.parametrize(
    "compact", [False, True], ids=["20_calls_one_row", "20_calls_two_rows"]
)
def test_shared_client_same_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, compact: bool
) -> None:
    baseline_root = os.environ.get("NEMO_RL_TEST_CC_BASELINE_ROOT")
    if not baseline_root:
        pytest.skip("set NEMO_RL_TEST_CC_BASELINE_ROOT to the pre-migration checkout")
    baseline_path = Path(baseline_root) / "nemo_rl/environments/nemo_gym.py"
    assert baseline_path.is_file(), baseline_path
    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        reasoning=True,
        decide_input=decide_capture_input,
    )
    env = _env(harness)
    baseline = load_source_definitions(
        baseline_path,
        {"NemoGym": {"_postprocess_cc_receipt_mode"}},
        {
            key: value
            for key, value in vars(gym_environment).items()
            if not key.startswith("__")
        },
    )
    # Load the schema from the same frozen checkout as the historical consumer.
    # The new production client no longer constructs or imports this object.
    schema_path = (
        Path(baseline_root)
        / "3rdparty/Gym-workspace/Gym/nemo_gym/context_management/result.py"
    )
    assert schema_path.is_file(), schema_path
    legacy_schema = load_source_definitions(
        schema_path,
        {"SelectedAction": None, "LogicalCCResult": None},
        {
            "Literal": Literal,
            "BaseModel": BaseModel,
            "ConfigDict": ConfigDict,
            "Field": Field,
            "model_validator": model_validator,
            "NeMoGymResponseOutputItem": NeMoGymResponseOutputItem,
        },
    )
    receipts, common_receipts, oracle, common = [], [], [], []
    owners = ["group_g0", "group_g1"]

    async def exercise() -> None:
        for owner in owners:
            observed = {}
            transport = MagicMock(spec=ServerClient)

            async def post(**kwargs: Any) -> Any:
                raw = gym_harness.assert_clean(
                    harness.client.post(
                        kwargs["url_path"],
                        json=kwargs["json"].model_dump(mode="json"),
                        headers=kwargs.get("headers"),
                    )
                )
                # Test tee of ordinary served responses. This includes every
                # served attempt, before the client makes its acceptance choice.
                observed[raw["id"]] = deepcopy(raw)
                return http_response(raw)

            transport.post = AsyncMock(side_effect=post)
            client = ContextManagedResponsesClient(
                server_client=transport,
                model_server=ModelServerRef(name="policy", type="responses_api_models"),
                logical_rollout_id=owner,
                initial_request=NeMoGymResponseCreateParamsNonStreaming(input="task"),
                config=ContextHistoryConfig.model_validate(
                    {
                        "enabled": compact,
                        "max_response_retries": 0,
                        "policy": {
                            "type": "recency",
                            "config": {
                                "reasoning": {"enabled": True, "keep_last_blocks": 0}
                            },
                        },
                        "schedule": {
                            "type": "turn_chunked_recency",
                            "actions_per_chunk": 10,
                        },
                    }
                ),
            )
            accepted_responses = []
            for turn in range(20):
                response = await client.create()
                accepted_responses.append(response.model_dump(mode="json"))
                if turn != 19:
                    client.append_observation(
                        [
                            {
                                "role": "user",
                                "type": "message",
                                "content": f"observation {turn}",
                            }
                        ]
                    )
            ordinary = response.model_dump(mode="json") | {
                "output": client.output_items
            }
            manifest = RolloutManifest.model_validate(
                await harness.ledger.manifest(owner)
            )
            # Compute the candidate independently of the historical fixture.
            candidate = selection_with_observed_metadata(
                manifest.records, ordinary, observed
            )
            common_receipts.append(
                env._assemble_receipt(
                    owner,
                    manifest.model_dump(mode="json"),
                    terminal_response_id=candidate.response_ids[-1],
                    reward=float(owners.index(owner) + 1),
                )
            )
            with pytest.raises(ValueError, match="completion metadata"):
                selection_with_observed_metadata(manifest.records, ordinary, {})
            client.finish(response)
            # This scenario contains only completed responses. Record membership
            # at the acceptance boundary, never infer the oracle from capture.
            assert all(not r["incomplete_details"] for r in accepted_responses)
            legacy_result = {
                "logical_rollout_id": owner,
                "selected_actions": [
                    {
                        "response_id": r["id"],
                        "finish_reason": "stop",
                        "last_output_item": r["output"][-1],
                    }
                    for r in accepted_responses
                ],
                "outcome": "completed",
            }
            processed = await env._postprocess_receipt_mode(
                {"_ng_rollout_id": owner},
                {
                    "response": ordinary,
                    "reward": float(owners.index(owner) + 1),
                },
            )
            assert candidate == processed["logical_selection"]
            # Only the unchanged historical consumer sees the old schema import.
            with monkeypatch.context() as legacy_import:
                legacy_import.setitem(
                    sys.modules, "nemo_gym.context_management.result", legacy_schema
                )
                previous = await baseline.NemoGym._postprocess_cc_receipt_mode(
                    env,
                    {"_ng_rollout_id": owner},
                    {
                        "response": ordinary,
                        "reward": float(owners.index(owner) + 1),
                        "context_compaction_result": legacy_result,
                    },
                )
            assert previous == processed
            receipts.append(previous["receipt"])
            oracle.append(previous["logical_selection"])
            common.append(candidate)

    try:
        asyncio.run(exercise())
        snapshots = []
        assert common_receipts == receipts
        for selections, selected_receipts in (
            (oracle, receipts),
            (common, common_receipts),
        ):
            replay = PublicationDataPlane()
            replay.rows = {key: value.clone() for key, value in plane.rows.items()}
            result = RolloutReassembler(
                replay,
                partition_id="canonical",
                staging_partition="staged",
                pad_token_id=0,
                max_seq_len=1024,
            ).finalize_group(
                "group",
                owners,
                deepcopy(selected_receipts),
                [1.0, 2.0],
                mask_sample=[False, False],
                fallback_weight_version=7,
                prompt_idx=99,
                canonical_sample_ids=owners,
                logical_selections=selections,
            )
            meta = result.meta
            assert meta is not None
            assert result.valid_row_count == (4 if compact else 2)
            assert not any(partition == "staged" for partition, _ in replay.rows)
            data = replay.get_samples(
                sample_ids=meta.sample_ids,
                partition_id="canonical",
                select_fields=meta.fields,
            )
            ctrl = _controller(meta, data, grpo={"baseline_population": "all_owners"})
            _, valid = asyncio.run(ctrl._advantage_stage(meta))
            assert valid
            generated = [
                token
                for tokens, mask in zip(
                    data["input_ids"].unbind(), data["token_mask"].unbind(), strict=True
                )
                for token in tokens[mask.bool()].tolist()
            ]
            assert sorted(generated) == list(range(1001, 1041))
            snapshots.append((meta, data, replay.events))
        left_meta, left, left_events = snapshots[0]
        right_meta, right, right_events = snapshots[1]
        assert left_meta == right_meta
        assert left_events == right_events
        assert set(left.keys()) == set(right.keys())
        for field in left.keys():
            for actual, expected in zip(
                left[field].unbind(), right[field].unbind(), strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        harness.client.close()
        asyncio.run(harness.ledger.close())


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
        fetch_prefix=source.fetch_prefix_token_ids,
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
