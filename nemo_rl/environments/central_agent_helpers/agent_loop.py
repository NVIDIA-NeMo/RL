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

"""The central agent loop: one rollout, driven by NeMo-RL instead of a Gym agent server."""

import json
from typing import Any, Optional

from nemo_rl.environments.central_agent_helpers.rollout_tree import build_rollout_tree
from nemo_rl.environments.central_agent_helpers.steering_manager import (
    build_steering_manager,
)
from nemo_rl.environments.central_agent_helpers.termination_manager import (
    TurnState,
    build_termination_manager,
)
from nemo_rl.environments.central_agent_helpers.tool_call_manager import (
    build_tool_call_manager,
)


async def _read(response: Any) -> str:
    return (await response.content.read()).decode()


def _raise_for_status(response: Any, what: str, body: Optional[str] = None) -> None:
    if response.status >= 400:
        raise RuntimeError(
            f"Central agent {what} failed with status {response.status}: {body}"
        )


def _merge_usage(
    accumulated: Optional[dict[str, Any]], new: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Sum usage across the turns of one rollout, as simple_agent does."""
    if not new:
        return accumulated
    if not accumulated:
        return dict(new)
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in new:
            accumulated[key] = accumulated.get(key, 0) + new[key]
    # TODO support more advanced token details
    if isinstance(accumulated.get("input_tokens_details"), dict):
        accumulated["input_tokens_details"]["cached_tokens"] = 0
    if isinstance(accumulated.get("output_tokens_details"), dict):
        accumulated["output_tokens_details"]["reasoning_tokens"] = 0
    return accumulated


class CentralAgent:
    """Runs one NeMo-Gym rollout end to end: seed_session, the turn loop, verify.

    Replaces ``responses_api_agents/simple_agent/app.py`` and other similar agentic loops, whose ``/run`` used to
    own this. The returned dict is the resources server's verify response, i.e.
    exactly what ``_postprocess_nemo_gym_to_nemo_rl_result`` already consumes.

    ``client`` is a NeMo-Gym ``ServerClient``; only ``post(server_name, url_path,
    json, cookies)`` is used, and its response only needs ``.status``,
    ``.cookies``, and ``await .content.read()``. Model output is kept as raw JSON
    end to end so the ``prompt_token_ids`` / ``generation_token_ids`` /
    ``generation_log_probs`` fields survive into verify untouched.
    """

    def __init__(
        self, client: Any, agent_entry: dict[str, Any], cfg: Optional[dict] = None
    ) -> None:
        self._client = client
        self._resources_server = agent_entry["resources_server"]
        self._model_server = agent_entry["model_server"]
        self._cfg = cfg or agent_entry.get("central_agent") or {}
        self._resources_cookies: Any = None
        self._model_cookies: Any = None

    async def run(self, row: dict[str, Any]) -> dict[str, Any]:
        response = await self._post(self._resources_server, "/seed_session", row)
        _raise_for_status(response, "seed_session", await _read(response))
        self._resources_cookies = response.cookies

        create_params = row["responses_create_params"]
        prompt = create_params.get("input") or []
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        tree = build_rollout_tree(
            self._cfg.get("rollout_tree"), prompt, self._call_model
        )
        tool_manager = build_tool_call_manager(
            self._cfg.get("tool_calls"), self._call_tool
        )
        steering = build_steering_manager(
            self._cfg.get("steering_message"), self._post_steering
        )
        termination = build_termination_manager(self._cfg.get("termination"))

        last_response: Optional[dict[str, Any]] = None
        usage: Optional[dict[str, Any]] = None

        # core agent loop
        while True:
            model_response = await self._call_model(
                {**create_params, "input": tree.get_active_branch()}
            )
            last_response = model_response
            usage = _merge_usage(usage, model_response.get("usage"))
            output = model_response.get("output") or []
            tree.append(output)

            tool_calls = [o for o in output if o.get("type") == "function_call"]
            # A length-truncated response can carry truncated arguments, so its
            # calls are never executed (simple_agent breaks here; Hermes, Pi and
            # nanobot all refuse the same way). Termination still sees them.
            executable = [] if model_response.get("incomplete_details") else tool_calls
            tool_results = await tool_manager.submit_and_collect(executable)
            tree.append(tool_results)

            steering_messages = await steering.get_steering_messages(output)
            tree.append(steering_messages)

            terminated = termination.is_termination_criteria_met(
                TurnState(
                    response=model_response,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    steering_messages=steering_messages,
                    tool_stats=tool_manager.stats(),
                    has_output_message=any(
                        o.get("type") == "message" and o.get("role") == "assistant"
                        for o in output
                    ),
                )
            )
            if terminated:
                tree.append(await tool_manager.drain())
            tree.branch_and_compact_context()
            if terminated:
                break

        # /verify sees the final active branch only. return_all_rollouts() is what
        # a compacting tree would fan out for training; LinearRolloutTree returns
        # this same single branch.
        assert last_response is not None, "the turn loop always runs at least once"
        verify_body = {
            **row,
            "response": {
                **last_response,
                "output": tree.get_active_outputs(),
                "usage": usage,
            },
        }
        response = await self._post(
            self._resources_server, "/verify", verify_body, self._resources_cookies
        )
        body = await _read(response)
        # TODO(@rohitrango): change this to return the entire compacted response tree
        _raise_for_status(response, "verify", body)
        return {
            **json.loads(body),
            **termination.termination_stats(),
            **tool_manager.stats(),
        }

    async def _post(
        self,
        server_name: str,
        url_path: str,
        body: Any,
        cookies: Any = None,
    ) -> Any:
        return await self._client.post(
            server_name=server_name, url_path=url_path, json=body, cookies=cookies
        )

    async def _call_model(self, body: dict[str, Any]) -> dict[str, Any]:
        """One model turn. Model calls are expected to always work, so this raises."""
        response = await self._post(
            self._model_server, "/v1/responses", body, self._model_cookies
        )
        raw = await _read(response)
        _raise_for_status(response, "model call", raw)
        self._model_cookies = response.cookies
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Received an invalid response from model server: {raw}"
            ) from e

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call one tool. Non-2xx is a valid outcome; its body is the tool output."""
        response = await self._post(
            self._resources_server, f"/{name}", arguments, self._resources_cookies
        )
        # Mirrors simple_agent: the resources server's session cookies are carried
        # forward. Under parallel tool calls this is last-writer-wins.
        self._resources_cookies = response.cookies
        return await _read(response)

    async def _post_steering(
        self, url_path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Ask this agent's own resources server what to inject next."""
        response = await self._post(
            self._resources_server, url_path, body, self._resources_cookies
        )
        raw = await _read(response)
        _raise_for_status(response, "steering", raw)
        return json.loads(raw)
