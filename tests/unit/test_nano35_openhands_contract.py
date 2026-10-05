# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the prebuilt OpenHands client against the installed Gym schema."""

import json
import os
import subprocess
from pathlib import Path

import pytest

G_OPENHANDS = Path(
    "/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/"
    "swe_openhands_setup/OpenHands"
)
pytestmark = pytest.mark.skipif(
    not (G_OPENHANDS / ".venv/bin/python").is_file(),
    reason="requires the prebuilt Nano SWE runtime",
)


@pytest.fixture(scope="module")
def requests(tmp_path_factory):
    output = tmp_path_factory.mktemp("openhands-contract") / "requests.json"
    script = r"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch
from litellm import ChatCompletionMessageToolCall
from openhands.core.config import LLMConfig
from openhands.core.message import Message, TextContent
from openhands.llm.llm import LLM
from openhands.agenthub.nemo_gym_client import NemoGymClient

requests = []

class Captured(Exception):
    pass

class Transport:
    async def post(self, **kwargs):
        requests.append(kwargs["json"])
        raise Captured

async def capture():
    for _ in range(2):
        # Keep request construction real; suppress only provider discovery.
        with patch.object(LLM, "init_model_info"):
            llm = LLM(LLMConfig(model="openai/nano35", max_output_tokens=196608,
                                max_input_tokens=196608, temperature=1.0, top_p=1.0),
                      service_id="contract-test")
        client = NemoGymClient.__new__(NemoGymClient)
        client.llm = llm
        client.ng_server_client = Transport()
        client.model_server_cookies = None
        for turn in range(2):
            messages = [Message(role="user", content=[TextContent(text=f"Turn {turn}")])]
            if turn:
                messages += [
                    Message(role="assistant", content=[TextContent(text="Inspecting the workspace.")],
                            tool_calls=[ChatCompletionMessageToolCall(
                                id="call_1", type="function",
                                function={"name": "execute_bash", "arguments": '{"command":"pwd"}'})],
                            prompt_token_ids=[1, 2], generation_token_ids=[3],
                            generation_log_probs=[-0.1]),
                    Message(role="tool", content=[TextContent(text="/workspace")],
                            tool_call_id="call_1", name="execute_bash"),
                ]
            try:
                await client._post_completion(messages)
            except Captured:
                pass
    llm._nemo_gym_llm_kwargs["unexpected_extension"] = True
    try:
        await client._post_completion(messages)
    except Captured:
        pass

asyncio.run(capture())
Path(sys.argv[1]).write_text(json.dumps(requests))
"""
    subprocess.run(
        [
            "uv",
            "run",
            "--no-config",
            "--no-project",
            "--python",
            str(G_OPENHANDS / ".venv/bin/python"),
            "python",
            "-c",
            script,
            str(output),
        ],
        cwd=G_OPENHANDS,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        check=True,
        timeout=60,
    )
    return json.loads(output.read_text())


def test_openhands_requests_match_gym_and_keep_session_identity(requests):
    from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming

    validated = [
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body)
        for body in requests[:4]
    ]
    assert validated[0].user == validated[1].user
    assert validated[2].user == validated[3].user
    assert validated[0].user and validated[0].user != validated[2].user
    for body in validated:
        assert body.temperature == 1.0
        assert body.top_p == 1.0
        assert body.max_completion_tokens == 196608
        assert body.messages
    for body in validated[1::2]:
        assistant, tool = body.messages[-2:]
        assert assistant["tool_calls"][0]["function"]["name"] == "execute_bash"
        assert assistant["generation_token_ids"] == [3]
        assert assistant["generation_log_probs"] == [-0.1]
        assert tool["tool_call_id"] == "call_1"
        assert tool["content"] == "/workspace"


def test_unknown_extensions_are_not_silently_dropped(requests):
    from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
    from pydantic import ValidationError

    assert requests[4]["unexpected_extension"] is True
    with pytest.raises(ValidationError, match="unexpected_extension"):
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(requests[4])
