# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual exporter methods with local artifacts; no external CLI or live model.

Historical OpenCode and Harbor dependencies are replaced only at orchestration
boundaries. Capture records here are semantic fixtures, not GPU/TQ evidence.
"""

import asyncio
import glob
import json
import os
import re
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, List, Optional
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym import openai_utils, rollout_observability
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from openai.types.responses.function_tool import FunctionTool
from pydantic import BaseModel, Field
from responses_api_agents.opencode_agent.app import (
    _INTERNAL_TRAJECTORY_KEY,
    OpenCodeAgent,
    _parse_opencode_session,
    parse_opencode_session,
)
from responses_api_agents.opencode_agent.tests.test_app import _session_db
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
        assert any(
            item.get("id") == "" and "summary" in str(item) for item in raw
        )  # copied prompt
        with pytest.raises(ValueError, match="No complete capture join"):
            selected_response_ids(records, {"output": raw})
        assert "model_response_rejected" in {gap.code for gap in trajectory.gaps}
    else:
        assert selected_response_ids(records, {"output": raw}) == selected
    receipt = {
        "rollout_id": "owner",
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

    shell = SimpleNamespace(
        config=SimpleNamespace(system_prompt=None), _run_opencode=launch
    )
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


class ForkInstanceConfig(BaseModel):
    """Only fields consumed by run(); no remote launch/config validation."""

    persistent_dir: Path
    instance_id: str


@pytest.fixture
def fork_exporter() -> Any:
    fork_root = os.environ.get("NEMO_RL_TEST_OPENCODE_FORK_ROOT")
    if not fork_root:
        pytest.skip("set NEMO_RL_TEST_OPENCODE_FORK_ROOT to the pinned fork snapshot")
    fork = Path(fork_root)
    assert fork.is_dir(), fork
    dependencies = namespace() | {
        "Path": Path,
        "Optional": Optional,
        "List": List,
        "ClassVar": ClassVar,
        "BaseModel": BaseModel,
        "Field": Field,
        "BaseVerifyResponse": BaseVerifyResponse,
        "SWEBenchWrapperInstanceConfig": ForkInstanceConfig,
        "json": json,
        "os": os,
        "glob": glob,
        "shutil": shutil,
        "time": time,
        "re": re,
        "uuid4": uuid4,
        "FunctionTool": FunctionTool,
    }
    converter = load_source_definitions(
        fork / "responses_api_models/vllm_model/app.py",
        {"VLLMConverter": None, "split_responses_input_output_items": None},
        dependencies,
    )
    source = load_source_definitions(
        fork / "responses_api_agents/swe_agents/app.py",
        {
            "RunOpenHandsAgent": {"_openhands_dir_copy_from_host"},
            "SWEBenchMetrics": None,
            "SWEBenchVerifyResponse": None,
            "SWEBenchWrapper": {
                "_materialize_trajectory",
                "get_all_session_trajectories_from_completions",
                "run",
            },
        },
        dependencies
        | {
            "split_responses_input_output_items": converter.split_responses_input_output_items
        },
    )
    wrapper = source.SWEBenchWrapper()
    wrapper._vllm_converter = converter.VLLMConverter(return_token_id_information=False)
    wrapper._sem = asyncio.Semaphore(1)
    return SimpleNamespace(source=source, wrapper=wrapper)


def fork_completion(
    index: int,
    segment: int,
    messages: list[dict],
    text: str,
    boundary: str | None = None,
) -> dict:
    return {
        "turn": index,
        "session_id": "main",
        "parent_session_id": None,
        "segment_index": segment,
        "segment_boundary_reason": boundary,
        "messages": messages,
        "response": {
            "id": f"response-{index}",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
        },
    }


@pytest.mark.parametrize("compact", [False, True])
def test_custom_opencode_real_collector_to_run_output(
    fork_exporter: Any, tmp_path: Path, compact: bool
) -> None:
    prompt = [{"role": "user", "content": "task"}]
    first = {"role": "assistant", "content": "first"}
    calls = [
        fork_completion(1, 0, prompt, "first"),
        fork_completion(
            2, 0, [*prompt, first, {"role": "user", "content": "continue"}], "second"
        ),
    ]
    if compact:
        copied = [*calls[-1]["messages"], {"role": "assistant", "content": "second"}]
        calls += [
            fork_completion(
                3,
                1,
                [*copied, {"role": "user", "content": "summarize"}],
                "summary",
                "compaction",
            ),
            fork_completion(
                4,
                2,
                [{"role": "user", "content": "summary"}],
                "last",
                "post_compaction",
            ),
        ]
    # The real collector deletes its source after harvesting. Every path is
    # inside this test's tmp_path, never the pinned source or an existing run.
    setup = tmp_path / "setup"
    dumps = setup / "opencode/eval/task/run/llm_completions/main"
    dumps.mkdir(parents=True)
    for call in calls:
        path = dumps / f"turn-{call['turn']:04d}.json"
        path.write_text(json.dumps(call))
        os.utime(path, (call["turn"], call["turn"]))
    collector = fork_exporter.source.RunOpenHandsAgent()
    collector.config = SimpleNamespace(
        problem_info={"instance_id": "task"},
        eval_dir_in_openhands="eval",
        agent_framework="opencode",
        openhands_config_file_path=tmp_path / "config",
        opencode_setup_dir=setup,
        trajectories_root=tmp_path / "trajectories/task",
    )
    collector._openhands_dir_copy_from_host(None)
    retained = list(
        (tmp_path / "trajectories/task/llm_completions/task").glob("*.json")
    )
    assert len(retained) == (3 if compact else 1)
    assert collector._opencode_turn_stats["num_model_calls"] == len(calls)
    assert not (setup / "opencode/eval").exists()
    wrapper = fork_exporter.wrapper
    entries = wrapper.get_all_session_trajectories_from_completions(
        tmp_path / "trajectories", "task"
    )
    assert entries[0]["prefix_message_count"] == 1
    if compact:
        assert entries[1]["prefix_message_count"] == len(calls[2]["messages"])
        assert entries[2]["segment_boundary_reason"] == "post_compaction"
    wrapper.responses = AsyncMock(
        return_value=response(
            len(calls),
            "public last",
            metadata={
                "input": json.dumps(prompt),
                "metrics": '{"resolved": true}',
                "instance_config": ForkInstanceConfig(
                    persistent_dir=tmp_path, instance_id="task"
                ).model_dump_json(),
            },
        )
    )
    exported = asyncio.run(
        wrapper.run(BaseRunRequest(responses_create_params={"input": "task"}))
    ).model_dump(mode="json")
    assert exported["reward"] == 1.0
    # Structural flattening only; no use of harness segment labels in RL plan.
    ordinary = {"output": [item for r in exported["responses"] for item in r["output"]]}
    texts = ["first", "second", "summary", "last"] if compact else ["first", "second"]
    records = [
        record(i + 1, [message(text)], parent=1 if i == 1 else None)
        for i, text in enumerate(texts)
    ]
    selected, plan = selected_chain_plan(records, ordinary)
    assert selected == tuple(f"response-{i + 1}" for i in range(len(calls)))
    assert len(plan) == (3 if compact else 1)
    # Copied pre-compaction assistant history did not become new actions.
    assert len(
        [item for item in ordinary["output"] if item.get("role") == "assistant"]
    ) == len(calls)
    assert all(
        r["id"].startswith("swebench-task-main-seg") for r in exported["responses"]
    )
    assert all("response_id" not in entry for entry in entries)
    with pytest.raises(ValueError, match="Ambiguous"):
        selected_response_ids(
            [records[0], record(99, [message("second")], parent=1), *records[1:]],
            ordinary,
        )


def test_custom_opencode_reasoning_only_and_identity_loss(fork_exporter: Any) -> None:
    data = fork_completion(1, 0, [{"role": "user", "content": "task"}], "")
    data["response"]["choices"][0]["message"]["reasoning_text"] = "thinking"
    messages, _ = fork_exporter.wrapper._materialize_trajectory(data)
    assert messages == data["messages"]  # reasoning-only final response omitted
    mixed = {"role": "assistant", "content": "answer", "reasoning_text": "thinking"}
    items = fork_exporter.wrapper._vllm_converter.chat_completions_messages_to_responses_items(
        [mixed]
    )
    assert [item.type for item in items] == ["message"]
    assert "thinking" not in str(items)
