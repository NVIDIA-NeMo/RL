# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Search-R1-compatible NeMo Gym agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.search_r1_agent.app import (
    INVALID_ACTION_OBSERVATION,
    SearchR1Agent,
    SearchR1AgentConfig,
    _assistant_text,
    format_search_observation,
    parse_action,
    truncate_observation,
)


def _http_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.ok = True
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    response.cookies = {}
    return response


def _model_response(text: str, response_id: str) -> dict:
    return {
        "id": response_id,
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "output": [
            {
                "id": f"message-{response_id}",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _agent(max_turns: int = 4) -> SearchR1Agent:
    config = SearchR1AgentConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="search_r1_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="ai_search"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=max_turns,
    )
    agent = SearchR1Agent(config=config, server_client=MagicMock(spec=ServerClient))
    agent._tokenizer = _CharacterTokenizer()
    return agent


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


def _request() -> MagicMock:
    request = MagicMock(spec=Request)
    request.cookies = {}
    request.path_params = {}
    return request


def test_parse_action_matches_search_r1_case_sensitive_first_match() -> None:
    assert parse_action("<search>  Ada Lovelace </search>") == (
        "search",
        "Ada Lovelace",
    )
    assert parse_action("<answer>first</answer><search>later</search>") == (
        "answer",
        "first",
    )
    assert parse_action("<SEARCH>ignored</SEARCH>") == (None, "")


def test_format_search_observation_matches_search_r1() -> None:
    assert format_search_observation(
        {
            "results": [
                {"title": "First", "text": "Alpha"},
                {"title": "Second", "text": "Beta"},
            ]
        }
    ) == (
        "\n\n<information>Doc 1(Title: First) Alpha\n"
        "Doc 2(Title: Second) Beta</information>\n\n"
    )


def test_observation_is_truncated_at_token_boundary() -> None:
    tokenizer = _CharacterTokenizer()
    assert truncate_observation("abcdef", tokenizer, 4) == "abcd"
    assert truncate_observation("abc", tokenizer, 4) == "abc"


@pytest.mark.asyncio
async def test_search_round_uses_fixed_top_three_then_stops_on_answer() -> None:
    server = _agent()
    server.server_client.post = AsyncMock(
        side_effect=[
            _http_response(
                _model_response("<think>x</think><search>Ada</search>", "1")
            ),
            _http_response(
                {
                    "query": "Ada",
                    "results": [
                        {
                            "rank": 1,
                            "doc_id": "d1",
                            "title": "Ada",
                            "text": "Lovelace",
                            "score": 1.0,
                        }
                    ],
                    "error": None,
                }
            ),
            _http_response(_model_response("<answer>Lovelace</answer>", "2")),
        ]
    )

    result = await server.responses(
        _request(),
        Response(),
        NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "question"}],
            parallel_tool_calls=False,
        ),
    )

    calls = server.server_client.post.call_args_list
    assert [call.kwargs["server_name"] for call in calls] == [
        "policy_model",
        "ai_search",
        "policy_model",
    ]
    assert calls[1].kwargs["json"] == {"query": "Ada", "top_k": 3}
    second_model_input = calls[2].kwargs["json"].input
    assert second_model_input[-1].content == (
        "\n\n<information>Doc 1(Title: Ada) Lovelace</information>\n\n"
    )
    assert _assistant_text(result.output) == (
        "<think>x</think><search>Ada</search><answer>Lovelace</answer>"
    )


@pytest.mark.asyncio
async def test_four_executable_turns_are_followed_by_one_terminal_call() -> None:
    server = _agent(max_turns=4)
    responses = []
    for index in range(4):
        responses.extend(
            [
                _http_response(
                    _model_response(f"<search>query-{index}</search>", str(index))
                ),
                _http_response(
                    {"query": f"query-{index}", "results": [], "error": None}
                ),
            ]
        )
    responses.append(_http_response(_model_response("<answer>done</answer>", "last")))
    server.server_client.post = AsyncMock(side_effect=responses)

    await server.responses(
        _request(),
        Response(),
        NeMoGymResponseCreateParamsNonStreaming(input="question"),
    )

    calls = server.server_client.post.call_args_list
    assert sum(call.kwargs["server_name"] == "policy_model" for call in calls) == 5
    assert sum(call.kwargs["server_name"] == "ai_search" for call in calls) == 4


@pytest.mark.asyncio
async def test_invalid_action_reuses_search_r1_feedback() -> None:
    server = _agent()
    server.server_client.post = AsyncMock(
        side_effect=[
            _http_response(_model_response("not tagged", "1")),
            _http_response(_model_response("<answer>done</answer>", "2")),
        ]
    )

    await server.responses(
        _request(),
        Response(),
        NeMoGymResponseCreateParamsNonStreaming(input="question"),
    )

    second_model_input = (
        server.server_client.post.call_args_list[1].kwargs["json"].input
    )
    assert second_model_input[-1].content == INVALID_ACTION_OBSERVATION
