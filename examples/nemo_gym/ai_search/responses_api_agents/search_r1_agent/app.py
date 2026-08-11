# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Gym agent that reproduces Search-R1's text-action rollout loop."""

import json
import re
from typing import Any, Literal

from fastapi import Request, Response
from pydantic import ConfigDict, Field, PrivateAttr, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


_ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", flags=re.DOTALL)
INVALID_ACTION_OBSERVATION = (
    "\nMy previous action is invalid. If I want to search, I should put the "
    "query between <search> and </search>. If I want to give the final answer, "
    "I should put the answer between <answer> and </answer>. Let me try again.\n"
)


class SearchR1AgentConfig(BaseResponsesAPIAgentConfig):
    """Services and fixed rollout limits used by the Search-R1 protocol."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_turns: int = Field(default=4, ge=1)
    top_k: Literal[3] = 3
    max_observation_tokens: Literal[500] = 500
    tokenizer_name: Literal["Qwen/Qwen2.5-7B"] = "Qwen/Qwen2.5-7B"


class SearchR1AgentRunRequest(BaseRunRequest):
    """Run payload with AI-search task fields supplied by the dataset."""

    model_config = ConfigDict(extra="allow")


class SearchR1AgentVerifyRequest(BaseVerifyRequest):
    """Verification payload forwarded to the AI-search resource server."""

    model_config = ConfigDict(extra="allow")


class SearchR1AgentVerifyResponse(BaseVerifyResponse):
    """Verification response returned by the AI-search resource server."""

    model_config = ConfigDict(extra="allow")


def parse_action(text: str) -> tuple[str | None, str]:
    """Return Search-R1's first case-sensitive tagged action and its content."""
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return None, ""
    return match.group(1), match.group(2).strip()


def format_search_observation(payload: dict[str, Any]) -> str:
    """Render local search results exactly like Search-R1's observation text."""
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("AI-search response must contain a list-valued 'results'")

    passages: list[str] = []
    for index, result in enumerate(raw_results, start=1):
        if not isinstance(result, dict):
            raise ValueError("AI-search result entries must be objects")
        title = str(result.get("title", ""))
        text = str(result.get("text", ""))
        passages.append(f"Doc {index}(Title: {title}) {text}\n")

    joined = "".join(passages)
    return f"\n\n<information>{joined.strip()}</information>\n\n"


def truncate_observation(text: str, tokenizer: Any, max_tokens: int) -> str:
    """Truncate an observation at the same token boundary as Search-R1."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(
        token_ids[:max_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _assistant_text(output: list[Any]) -> str:
    """Join assistant text from one model call without reading earlier turns."""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, NeMoGymResponseOutputMessage):
            continue
        for content in item.content:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def _merge_usage(total: Any, current: Any) -> Any:
    if total is None:
        return current
    if current is None:
        return total
    total.input_tokens += current.input_tokens
    total.output_tokens += current.output_tokens
    total.total_tokens += current.total_tokens
    total.input_tokens_details.cached_tokens = 0
    total.output_tokens_details.reasoning_tokens = 0
    return total


class SearchR1Agent(SimpleResponsesAPIAgent):
    """Execute four Search-R1 rounds followed by one answer-only generation."""

    config: SearchR1AgentConfig
    _tokenizer: Any = PrivateAttr(default=None)

    def _truncate_observation(self, observation: str) -> str:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        return truncate_observation(
            observation,
            self._tokenizer,
            self.config.max_observation_tokens,
        )

    async def _call_model(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming,
        new_outputs: list[Any],
        cookies: Any,
    ) -> tuple[NeMoGymResponse, Any]:
        model_body = body.model_copy(update={"input": body.input + new_outputs})
        raw_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=model_body,
            cookies=cookies,
        )
        await raise_for_status(raw_response)
        response_json = await get_response_json(raw_response)
        try:
            model_response = NeMoGymResponse.model_validate(response_json)
        except ValidationError as error:
            raise RuntimeError(
                "Received an invalid response from model server: "
                f"{json.dumps(response_json)}"
            ) from error
        return model_response, raw_response.cookies

    async def _search(self, query: str, cookies: Any) -> tuple[str, Any]:
        raw_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/search",
            json={"query": query, "top_k": self.config.top_k},
            cookies=cookies,
        )
        await raise_for_status(raw_response)
        payload = await get_response_json(raw_response)
        if not isinstance(payload, dict):
            raise ValueError("AI-search response must be a JSON object")
        observation = self._truncate_observation(format_search_observation(payload))
        return observation, raw_response.cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Run Search-R1's four executable turns and terminal generation."""
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs: list[Any] = []
        usage = None
        model_server_cookies = None
        resources_server_cookies = request.cookies
        last_response: NeMoGymResponse | None = None
        ended = False

        for _ in range(self.config.max_turns):
            last_response, model_server_cookies = await self._call_model(
                request, body, new_outputs, model_server_cookies
            )
            new_outputs.extend(last_response.output)
            usage = _merge_usage(usage, last_response.usage)

            if last_response.incomplete_details:
                ended = True
                break

            action, content = parse_action(_assistant_text(last_response.output))
            if action == "answer":
                ended = True
                break
            if action == "search":
                observation, resources_server_cookies = await self._search(
                    content, resources_server_cookies
                )
            else:
                observation = INVALID_ACTION_OBSERVATION
            new_outputs.append(
                NeMoGymEasyInputMessage(role="user", content=observation)
            )

        if not ended:
            last_response, model_server_cookies = await self._call_model(
                request, body, new_outputs, model_server_cookies
            )
            new_outputs.extend(last_response.output)
            usage = _merge_usage(usage, last_response.usage)

        if last_response is None:  # pragma: no cover - max_turns is validated >= 1
            raise RuntimeError("Search-R1 agent made no model calls")

        for cookie_jar in (resources_server_cookies, model_server_cookies):
            if cookie_jar is None:
                continue
            for key, value in cookie_jar.items():
                response.set_cookie(key, value)

        last_response.output = new_outputs
        last_response.usage = usage
        return last_response

    async def run(
        self, request: Request, body: SearchR1AgentRunRequest
    ) -> SearchR1AgentVerifyResponse:
        """Seed the search session, collect one rollout, and verify its answer."""
        cookies = request.cookies
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        rollout_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(rollout_response)
        cookies = rollout_response.cookies

        verify_request = SearchR1AgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(rollout_response)}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return SearchR1AgentVerifyResponse.model_validate(
            await get_response_json(verify_response)
        )

    async def aggregate_metrics(
        self, body: AggregateMetricsRequest = Body()
    ) -> AggregateMetrics:
        """Proxy aggregate metrics to the AI-search resource server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SearchR1Agent.run_webserver()
