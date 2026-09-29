# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual Gym exporters against captured evidence: compatibility, not live RL.

Responses agents execute against the existing HTTP capture/TQ-codec fixture.
Generation and resource services are doubles. Historical fork/Harbor probes
execute unchanged source methods with external orchestration omitted.
"""

import asyncio
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_gym.context_management import ContextManagedResponsesClient
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.aviary_agent.app import (
    AviaryAgent,
    AviaryAgentConfig,
    AviaryAgentRunRequest,
)
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent
from responses_api_agents.browsecomp_agent.tests.test_progress_tracking import (
    _make_config,
)
from responses_api_agents.simple_agent_with_compaction.tests.test_app import (
    make_agent,
    run_agent,
)
from responses_api_agents.simple_agent_with_compaction.tests.test_client import (
    http_response,
)

from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from tests.unit.experience.common_output_comparison import selected_response_ids
from tests.unit.experience.test_cc_dispatch import _env
from tests.unit.experience.test_logical_owner_finalization import (
    PublicationDataPlane,
    decide_capture_input,
    gym_harness,
)

pytestmark = pytest.mark.nemo_gym


def tool_message(index: int, name: str = "step", arguments: str = "{}") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"tool-{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


@pytest.fixture
def captured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[SimpleNamespace]:
    """Use the established HTTP capture and actual TQ serialization fixture."""
    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
        reasoning=True,
    )
    backend = NeMoGymAsyncOpenAI.create_chat_completion
    queue, served, requests = [], [], []

    async def worker(client: Any, **body: Any) -> dict:
        result = await backend(client, **body)
        result["choices"][0]["message"].update(deepcopy(queue.pop(0)))
        return result

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", worker)
    count = 0

    async def post(**kwargs: Any) -> Any:
        nonlocal count
        path = kwargs["url_path"]
        if path.endswith("/v1/responses"):
            body = kwargs["json"]
            body = body.model_dump(mode="json") if hasattr(body, "model_dump") else body
            requests.append(deepcopy(body))
            # In-process responses() probes get the same correlated URL a run supplies.
            if path == "/v1/responses":
                path = "/ng-rollout/owner/training-token-capture/v1/responses"
            result = gym_harness.assert_clean(
                harness.client.post(path, json=body, headers=kwargs.get("headers"))
            )
            served.append(deepcopy(result))
        elif path == "/seed_session":
            result = {
                "env_id": "env",
                "obs": [
                    {"role": "user", "content": "state zero", "is_env_state": True}
                ],
                "tools": [],
            }
        elif path == "/verify":
            result = kwargs["json"] | {"reward": 0.75}
        elif path == "/close":
            result = {}
        else:
            count += 1
            result = {
                "obs": [
                    {
                        "type": "function_call_output",
                        "call_id": f"tool-{count}",
                        "output": f"result {count}",
                    },
                    {"role": "user", "content": f"state {count}", "is_env_state": True},
                ],
                "reward": 0.0,
                "done": False,
                "results_string": f"tool result {count}",
            }
        response = http_response(result)
        response.json = AsyncMock(return_value=result)
        response.raise_for_status = MagicMock()
        return response

    transport = MagicMock(
        spec=ServerClient, global_config_dict={"token_id_capture": {"enabled": True}}
    )
    transport.post = AsyncMock(side_effect=post)
    yield SimpleNamespace(
        harness=harness,
        plane=plane,
        queue=queue,
        served=served,
        requests=requests,
        transport=transport,
    )
    harness.client.close()
    asyncio.run(harness.ledger.close())


def assert_training_rows(
    captured: Any, response: dict, *, owner: str, rows: int
) -> None:
    selected = tuple(item["id"] for item in captured.served)
    ordinary = asyncio.run(
        _env(captured.harness)._postprocess_receipt_mode(
            {"_ng_rollout_id": owner},
            {"response": response, "reward": 0.75},
        )
    )
    assert ordinary["logical_selection"].response_ids == selected
    receipt = ordinary["receipt"]
    finalizer = RolloutReassembler(
        captured.plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
    )
    # Ordinary reconstruction is the negative control after a rewrite.
    terminal_only = finalizer.finalize_rollout(owner, receipt, reward=0.75)
    assert terminal_only.valid
    terminal_tokens = [
        token
        for token, mask in zip(
            terminal_only.token_ids, terminal_only.token_mask, strict=True
        )
        if mask
    ]
    if rows > 1:
        assert len(terminal_tokens) < len(selected)
    result = finalizer.finalize_group(
        "group",
        [owner],
        [receipt],
        [0.75],
        mask_sample=[False],
        fallback_weight_version=7,
        prompt_idx=0,
        canonical_sample_ids=["group_g0"],
        logical_selections=[ordinary["logical_selection"]],
    )
    assert result.valid_row_count == rows
    assert result.meta is not None
    data = captured.plane.get_samples(
        sample_ids=result.meta.sample_ids,
        partition_id="canonical",
        select_fields=result.meta.fields,
    )
    generated = [
        token
        for tokens, mask in zip(
            data["input_ids"].unbind(), data["token_mask"].unbind(), strict=True
        )
        for token in tokens[mask.bool()].tolist()
    ]
    assert generated == list(range(1001, 1001 + len(selected)))
    assert not any(partition == "staged" for partition, _ in captured.plane.rows)


@pytest.mark.parametrize("compact", [False, True])
def test_simple_actual_run_returns_common_history(
    captured: Any, compact: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = []
    original_finish = ContextManagedResponsesClient.finish

    def observe_finish(client: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_finish(client, *args, **kwargs)
        oracle.append(result)
        return result

    monkeypatch.setattr(ContextManagedResponsesClient, "finish", observe_finish)
    captured.queue.extend(
        [tool_message(1), tool_message(2), {"role": "assistant", "content": "done"}]
    )
    agent, _ = make_agent(
        config={
            "context_history": {
                "enabled": compact,
                "max_response_retries": 0,
                "policy": {
                    "type": "recency",
                    "config": {"reasoning": {"enabled": True, "keep_last_blocks": 0}},
                },
                "schedule": {"type": "turn_chunked_recency", "actions_per_chunk": 2},
            },
        }
    )
    agent.server_client = captured.transport
    result = asyncio.run(run_agent(agent)).model_dump(mode="json")
    assert result["reward"] == 0.75
    ordinary = deepcopy(result)
    assert "context_compaction_result" not in ordinary
    manifest = captured.harness.manifest("dispatch_g0")
    # Membership is computed before consulting the old dedicated result oracle.
    selected = selected_response_ids(manifest.records, ordinary["response"])
    assert selected == tuple(
        action.response_id for action in oracle[0].selected_actions
    )
    assert sum(r.parent_call_id is None for r in manifest.records) == (
        2 if compact else 1
    )
    assert_training_rows(
        captured, ordinary["response"], owner="dispatch_g0", rows=2 if compact else 1
    )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "snapshots", [False, True], ids=["flat", "transition_snapshots"]
)
def test_aviary_actual_run_modes(captured: Any, compact: bool, snapshots: bool) -> None:
    captured.queue.extend(
        [tool_message(1), tool_message(2), {"role": "assistant", "content": "done"}]
    )
    agent = AviaryAgent(
        config=AviaryAgentConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="aviary",
            model_server={"type": "responses_api_models", "name": "policy"},
            resources_server={"type": "resources_servers", "name": "resources"},
            max_steps=3,
            return_transitions=snapshots,
            collapse_old_env_states=compact,
        ),
        server_client=captured.transport,
    )
    body = AviaryAgentRunRequest.model_validate(
        {
            "task_idx": 0,
            "_ng_rollout_id": "owner",
            "responses_create_params": {"input": [{"role": "user", "content": "task"}]},
        }
    )
    result = asyncio.run(agent.run(body)).model_dump(mode="json")
    assert result["reward"] == 0.75
    response = result["response"]
    if snapshots:
        assert response["contains_transitions"] is True
        assert len(response["output"]) == 3 and all(
            isinstance(snapshot, list) for snapshot in response["output"]
        )
        with pytest.raises(ValueError, match="Malformed ordinary output item"):
            selected_response_ids(captured.harness.manifest("owner").records, response)
        # Actual accumulated snapshots retain the generated identities. A shared
        # normalization may recover them; the flat-only probe cannot consume them.
        served_ids = {
            item["id"] for served in captured.served for item in served["output"]
        }
        snapshot_ids = {
            item.get("id") for snapshot in response["output"] for item in snapshot
        }
        assert served_ids <= snapshot_ids
    assert_training_rows(captured, response, owner="owner", rows=3 if compact else 1)


@pytest.mark.parametrize("compact", [False, True])
def test_browsecomp_real_export_preserves_history_across_tool_hiding(
    captured: Any, compact: bool
) -> None:
    captured.queue.extend(
        [
            tool_message(1, "search"),
            tool_message(2, "search"),
            {"role": "assistant", "content": "Exact Answer: done"},
        ]
    )
    agent = BrowsecompAgent(
        config=_make_config(keep_rounds=1 if compact else 9999),
        server_client=captured.transport,
    )
    response = asyncio.run(
        agent.responses(
            MagicMock(cookies={}),
            MagicMock(),
            NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "task"}]
            ),
        )
    ).model_dump(mode="json")
    assert_training_rows(captured, response, owner="owner", rows=2 if compact else 1)


@pytest.mark.parametrize(
    "synthesize", [False, True], ids=["edit_existing_answer", "synthesize_answer"]
)
def test_browsecomp_progress_recovery_is_not_sampled_output(
    captured: Any, synthesize: bool
) -> None:
    captured.queue.append(
        tool_message(1, "update_progress", '{"board":"Exact Answer: board-only"}')
    )
    if not synthesize:
        captured.queue.append({"role": "assistant", "content": "No final answer"})
    agent = BrowsecompAgent(
        config=_make_config(progress=True, max_steps=1 if synthesize else 2),
        server_client=captured.transport,
    )
    response = asyncio.run(
        agent.responses(
            MagicMock(cookies={}),
            MagicMock(),
            NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                ]
            ),
        )
    ).model_dump(mode="json")
    assert "Recovered from progress board" in str(response["output"])
    assert "Recovered from progress board" not in str(captured.served)
    if synthesize:
        assert response["output"][-1]["id"] == "msg_progress_board_recovery"
    else:
        assert response["output"][-1]["id"] == captured.served[-1]["output"][-1]["id"]
    with pytest.raises(ValueError, match="No complete capture join"):
        selected_response_ids(captured.harness.manifest("owner").records, response)
    for with_references in (False, True):
        result = {"response": response, "reward": 0.75}
        if with_references:
            result["ng_trajectory"] = {
                "turns": [
                    {
                        "model_calls": [
                            {"response_id": record.response_id}
                            for record in captured.harness.manifest("owner").records
                        ]
                    }
                ]
            }
        with pytest.raises(ValueError, match="Rollout 'owner': Missing or ambiguous"):
            asyncio.run(
                _env(captured.harness)._postprocess_receipt_mode(
                    {"_ng_rollout_id": "owner"}, result
                )
            )


@pytest.mark.parametrize(
    "path,body",
    [
        (
            "chat/completions",
            {"messages": [{"role": "user", "content": "task"}], "stream": True},
        ),
        (
            "messages",
            {"messages": [{"role": "user", "content": "task"}], "max_tokens": 10},
        ),
        ("responses", {"input": "task", "stream": True}),
    ],
)
def test_existing_framework_capture_protocol_gap(
    captured: Any, path: str, body: dict
) -> None:
    response = captured.harness.client.post(
        f"/ng-rollout/owner/training-token-capture/v1/{path}", json=body
    )
    assert response.status_code == 422
    assert "Framework" in response.text
    assert not captured.served and not captured.harness.worker_calls
