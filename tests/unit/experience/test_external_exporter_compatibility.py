# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual exporter methods with local artifacts; no external CLI or live model.

Historical OpenCode and Harbor dependencies are replaced only at orchestration
boundaries. Capture records here are semantic fixtures, not GPU/TQ evidence.
"""

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym import openai_utils, rollout_observability
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.opencode_agent.app import (
    _INTERNAL_TRAJECTORY_KEY,
    OpenCodeAgent,
    _parse_opencode_session,
    parse_opencode_session,
)
from responses_api_agents.opencode_agent.tests.test_app import _config, _session_db
from responses_api_agents.opencode_agent.tests.test_trajectory import _policy

from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    RolloutReassembler,
    RolloutSelection,
)
from tests.unit.experience.agent_exporter_probe import (
    GYM_ROOT,
    load_source_definitions,
    selected_chain_plan,
)
from tests.unit.experience.common_output_comparison import selected_response_ids
from tests.unit.experience.test_common_output_comparison import message, record

pytestmark = pytest.mark.nemo_gym


def response(index: int, text: str, **updates: Any) -> NeMoGymResponse:
    output = message(text) | {"id": f"native-{index}"}
    output["content"][0]["annotations"] = []
    return NeMoGymResponse.model_validate(
        {
            "id": f"response-{index}",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [output],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            **updates,
        }
    )


def namespace() -> dict:
    return {
        name: value
        for module in (openai_utils, rollout_observability)
        for name, value in vars(module).items()
        if not name.startswith("_")
    }


@pytest.fixture
def terminus() -> SimpleNamespace:
    dependencies = namespace() | {
        "dataclass": dataclass,
        "ModelServerRef": ModelServerRef,
        "LLMResponse": SimpleNamespace,
    }
    observations = load_source_definitions(
        GYM_ROOT / "responses_api_agents/terminus_2_sandboxed_agent/observability.py",
        {"ObservedResponse": None, "TerminusObservations": None},
        dependencies,
    )

    class ContextExceeded(Exception):
        pass

    llm = load_source_definitions(
        GYM_ROOT / "responses_api_agents/terminus_2_sandboxed_agent/app.py",
        {"NeMoGymLLM": None},
        dependencies
        | {
            "BaseLLM": object,
            "UsageInfo": SimpleNamespace,
            "ContextLengthExceededError": ContextExceeded,
            "TerminusObservations": observations.TerminusObservations,
            "asyncio": asyncio,
            "deepcopy": deepcopy,
            "uuid4": uuid4,
            "time": time.time,
            "perf_counter": time.perf_counter,
        },
    )
    observed = observations.TerminusObservations(
        "invocation",
        "task",
        "owner",
        ModelServerRef(name="policy", type="responses_api_models"),
    )
    client = SimpleNamespace(create_response=AsyncMock())
    adapter = llm.NeMoGymLLM(client, "test", 4096, 100, 10, observed)
    return SimpleNamespace(
        adapter=adapter, client=client, observed=observed, error=ContextExceeded
    )


@pytest.mark.parametrize("compact", [False, True])
def test_terminus_real_transcript_and_existing_decision_references(
    terminus: Any, compact: bool
) -> None:
    served = [response(1, "first")]
    if compact:
        served.extend(
            [
                response(
                    2, "rejected", incomplete_details={"reason": "max_output_tokens"}
                ),
                response(3, "summary"),
                response(4, "last"),
            ]
        )
    else:
        served.append(response(2, "last"))
    terminus.client.create_response.side_effect = [
        r.model_dump(mode="json") for r in served
    ]

    async def run() -> None:
        observed, llm = terminus.observed, terminus.adapter
        observed.begin_decision()
        observed.decision_response = await llm.call("task")
        observed.finish_decision(0)
        observed.begin_decision()
        if compact:
            with pytest.raises(terminus.error):
                await llm.call("too long")
            llm._is_compacting = True
            observed.compaction = rollout_observability.ContextCompactionObservation(
                invocation_id="invocation"
            )
            await llm.call("summarize")
            assert [ref.response_id for ref in observed.compaction.model_calls] == [
                "response-3"
            ]
            observed.compaction = None
            llm._is_compacting = False
        observed.decision_response = await llm.call(
            "continue",
            message_history=[
                {"role": "assistant", "content": "summary" if compact else "first"}
            ],
        )
        observed.finish_decision(1)

    asyncio.run(run())
    trajectory = terminus.observed.finish()
    selected = tuple(
        ref.response_id for turn in trajectory.turns for ref in turn.model_calls
    )
    assert selected == (
        ("response-1", "response-4") if compact else ("response-1", "response-2")
    )
    raw = [item.model_dump(mode="json") for item in terminus.adapter.trajectory]
    records = [
        record(
            i + 1,
            r.model_dump(mode="json")["output"],
            parent=1 if not compact and i else None,
        )
        for i, r in enumerate(served)
    ]
    if compact:
        assert any(
            item.get("id") == "native-2" for item in raw
        )  # rejected response was appended
        summary_id = terminus.client.create_response.call_args.kwargs["input"][0]["id"]
        assert summary_id
        assert any(
            item.get("id") == summary_id and "summary" in str(item) for item in raw
        )  # copied prompt retains the exporter's synthetic ID
        with pytest.raises(ValueError, match="No complete capture join"):
            selected_response_ids(records, {"output": raw})
        assert "model_response_rejected" in {gap.code for gap in trajectory.gaps}
    else:
        assert selected_response_ids(records, {"output": raw}) == selected
    receipt = {
        "capture_key": "owner",
        "manifest": [r.model_dump(mode="json") for r in records],
        "terminal_model_call_id": records[-1].model_call_id,
        "terminal_selection": "declared",
    }
    selection = RolloutSelection(
        selected, tuple(ActionOutputFlags(False, False) for _ in selected)
    )
    plan = RolloutReassembler._plan_selected_calls(["owner"], [receipt], [selection])[0]
    assert len(plan) == (2 if compact else 1)
    # Decision-only policy excludes the separately identified summary, not all
    # successful captures. Production policy must make this choice explicitly.
    assert (
        tuple(rid for segment in plan for rid in segment.selected_response_ids)
        == selected
    )


def test_interleaved_conversations_become_one_segment_each() -> None:
    records = [
        record(1, [message("m1")]),
        record(2, [message("s1")]),
        record(3, [message("s2")], parent=2),
        record(4, [message("m2")], parent=1),
    ]
    selected = ("response-1", "response-4", "response-2", "response-3")
    receipt = {
        "capture_key": "owner",
        "manifest": [r.model_dump(mode="json") for r in records],
        "terminal_model_call_id": "call-4",
        "terminal_selection": "declared",
    }
    selection = RolloutSelection(
        selected, tuple(ActionOutputFlags(False, False) for _ in selected)
    )
    plan = RolloutReassembler._plan_selected_calls(["owner"], [receipt], [selection])[0]
    assert [s.selected_response_ids for s in plan] == [
        ("response-1", "response-4"),
        ("response-2", "response-3"),
    ]


@pytest.mark.parametrize("compact", [False, True])
def test_gym_opencode_real_sqlite_export_and_episode(
    tmp_path: Path, compact: bool
) -> None:
    messages = [_policy({"type": "text", "text": "first"})]
    if compact:
        messages.append(_policy({"type": "text", "text": "summary"}, summary=True))
    messages.append(_policy({"type": "text", "text": "last"}))
    db = _session_db(tmp_path, messages)

    async def launch(*args: Any, **kwargs: Any) -> tuple:
        observations = _parse_opencode_session(db, "test", kwargs["trajectory"])
        items, usage = parse_opencode_session(db)
        return items, usage, "test", observations

    shell = SimpleNamespace(config=_config(), _run_opencode=launch)
    episode = asyncio.run(
        OpenCodeAgent._create_episode(
            shell,
            NeMoGymResponseCreateParamsNonStreaming(input="task"),
            rollout_id="owner",
        )
    )
    exported = episode.response.model_dump(mode="json")
    trajectory = getattr(episode.response, _INTERNAL_TRAJECTORY_KEY)
    assert len(trajectory["turns"]) == 2
    assert all(not turn["model_calls"] for turn in trajectory["turns"])
    assert exported["id"].startswith("resp_")
    texts = ["first", "summary", "last"] if compact else ["first", "last"]
    records = [
        record(i + 1, [message(text)], parent=1 if not compact and i else None)
        for i, text in enumerate(texts)
    ]
    selected, plan = selected_chain_plan(records, exported)
    assert len(selected) == len(texts)
    assert len(plan) == (3 if compact else 1)
    # Flat output includes the summary; existing decision trajectory excludes it.
    assert len(selected) - len(trajectory["turns"]) == int(compact)
    duplicate = record(99, [message("first")])
    if compact:
        with pytest.raises(ValueError, match="Ambiguous"):
            selected_response_ids([duplicate, *records], exported)
    else:
        assert selected_response_ids([duplicate, *records], exported) == selected


def test_gym_opencode_reasoning_only_is_omitted_from_flat_output(
    tmp_path: Path,
) -> None:
    db = _session_db(tmp_path, [_policy({"type": "reasoning", "text": "thinking"})])
    items, _ = parse_opencode_session(db)
    trajectory = rollout_observability.TrajectoryRecord(
        task_id="task", rollout_id="owner"
    )
    _parse_opencode_session(db, "test", trajectory)
    assert items == []
    assert trajectory.turns[0].reasoning_content[0]["text"] == "thinking"
    assert not trajectory.turns[0].model_calls
