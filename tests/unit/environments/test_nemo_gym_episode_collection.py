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
"""Exercise Gym's real HTTP collector through RL's streaming rollout adapter.

The local endpoints return deterministic protocol-valid results. This covers
collector/adapter integration without starting an agent or a policy model.
"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from omegaconf import OmegaConf

from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.experience.rollouts import _prepare_nemo_gym_rows

pytest.importorskip("nemo_gym.episode_types")
gym_config = pytest.importorskip("nemo_gym.global_config")
gym_resources = pytest.importorskip("nemo_gym.base_resources_server")
gym_rollouts = pytest.importorskip("nemo_gym.rollout_collection")
gym_server = pytest.importorskip("nemo_gym.server_utils")
gym_openai = pytest.importorskip("nemo_gym.openai_utils")
gym_protocol = pytest.importorskip("nemo_gym.single_agent_turn_types")

pytestmark = pytest.mark.nemo_gym


class _TokenDecoder:
    def batch_decode(self, batch: list[list[int]]) -> list[str]:
        return [" ".join(map(str, token_ids)) for token_ids in batch]


def _model_response(*, episode_protocol: bool) -> gym_openai.NeMoGymResponse:
    return gym_openai.NeMoGymResponse(
        id="response-episode" if episode_protocol else "response-flat",
        created_at=0,
        model="deterministic-test-model",
        object="response",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            gym_openai.NeMoGymResponseOutputMessageForTraining(
                id="message-episode" if episode_protocol else "message-flat",
                content=[
                    gym_openai.NeMoGymResponseOutputText(
                        text="The weather is sunny.", annotations=[]
                    )
                ],
                prompt_token_ids=[1, 2, 3] if episode_protocol else [4, 5],
                generation_token_ids=[11, 12] if episode_protocol else [21],
                generation_log_probs=[-0.1, -0.2] if episode_protocol else [-0.4],
            )
        ],
    )


@pytest.mark.parametrize(
    ("token_capture", "group_attempt", "compatibility_response"),
    [
        (False, 0, False),
        (False, 2, False),
        (True, 3, False),
        (False, 0, True),
        (True, 3, True),
    ],
)
def test_real_collector_streams_episode_and_flat_http_results(
    monkeypatch: pytest.MonkeyPatch,
    token_capture: bool,
    group_attempt: int,
    compatibility_response: bool,
) -> None:
    async def run() -> None:
        received_episode = []
        received_flat = []

        async def episode_run(request: web.Request) -> web.Response:
            body = await request.json()
            episode = gym_protocol.SingleAgentTurnRequest.model_validate(body)
            received_episode.append(body)
            reply = gym_protocol.SingleAgentTurnResponse(
                episode_id=episode.episode_id,
                task_id=episode.task.task_id,
                result=gym_protocol.SingleAgentTurnResult(
                    responses_create_params=episode.task.task_input.responses_create_params,
                    response=_model_response(episode_protocol=True),
                    reward=0.75,
                ),
            )
            if compatibility_response:
                return web.json_response(
                    reply.result.model_dump(mode="json")
                    | {"agent_ref": {"name": "episode_agent"}}
                )
            return web.json_response(reply.model_dump(mode="json"))

        async def flat_run(request: web.Request) -> web.Response:
            body = await request.json()
            flat_request = gym_resources.BaseRunRequest.model_validate(body)
            received_flat.append(body)
            reply = gym_resources.BaseVerifyResponse(
                responses_create_params=flat_request.responses_create_params,
                response=_model_response(episode_protocol=False),
                reward=0.25,
            )
            return web.json_response(reply.model_dump(mode="json"))

        episode_app = web.Application()
        episode_app.router.add_post("/run", episode_run)
        flat_app = web.Application()
        flat_app.router.add_post("/run", flat_run)
        async with (
            TestServer(episode_app) as episode_server,
            TestServer(flat_app) as flat_server,
            ClientSession() as http_client,
        ):
            global_config = OmegaConf.create(
                {
                    "environment_server_routes": {
                        "weather:train": "episode_environment"
                    },
                    "episode_agent": {"responses_api_agents": {"simple_agent": {}}},
                    "legacy_agent": {"responses_api_agents": {"simple_agent": {}}},
                    "episode_environment": {
                        "environment_servers": {
                            (
                                "single_agent_turn_legacy"
                                if compatibility_response
                                else "single_agent_turn"
                            ): {
                                "host": episode_server.host,
                                "port": episode_server.port,
                                "agent_server": {
                                    "type": "responses_api_agents",
                                    "name": "episode_agent",
                                },
                            }
                        }
                    },
                    "flat_environment": {
                        "environment_servers": {
                            "legacy_agent": {
                                "host": flat_server.host,
                                "port": flat_server.port,
                                "agent_server": {
                                    "type": "responses_api_agents",
                                    "name": "legacy_agent",
                                },
                            }
                        }
                    },
                }
            )
            # Use Gym's normal injected-config bootstrap and actual ServerClient.
            # Isolate only the process globals so other tests keep their clients.
            monkeypatch.setenv(
                gym_config.NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
                OmegaConf.to_yaml(global_config),
            )
            monkeypatch.setattr(gym_config, "_GLOBAL_CONFIG_DICT", global_config)
            monkeypatch.setattr(gym_server, "_GLOBAL_AIOHTTP_CLIENT", http_client)

            materialized_task = {
                "task_id": {"taskset": "weather:train", "task_id": "weather-17"},
                "task_input": {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "What is the weather?"}],
                        "temperature": 0.7,
                        "max_output_tokens": 16,
                    },
                    "task_data": {
                        "location": "Paris",
                        "opaque_state": {"days": [1, 2]},
                    },
                },
            }
            rows = [deepcopy(materialized_task) for _ in range(2)]
            rows.append(
                {
                    "agent_ref": {"name": "legacy_agent"},
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "Flat weather request"}]
                    },
                }
            )
            expected_task = deepcopy(materialized_task)
            if token_capture:
                for index, row in enumerate(rows[:2]):
                    row.update(
                        {
                            "_rowidx": index,
                            "_ng_group_id": "episode-group",
                            "_ng_group_attempt": group_attempt,
                            "_ng_rollout_index": index,
                            "_ng_rollout_id": f"capture-episode-{index}",
                            "_ng_attempt_index": index * 2,
                        }
                    )
                rows[2].update({"_rowidx": 2, "_ng_rollout_id": "capture-flat"})
                expected_episode_ids = [
                    {"rollout_id": f"capture-episode-{index}", "attempt": index * 2}
                    for index in range(2)
                ]
            else:
                # Dataset rows have no rollout identity. Exercise the driver's
                # real preparation step with complete materialized/flat groups.
                rows.append(deepcopy(rows[2]))
                if group_attempt:
                    for row in rows:
                        row["_ng_group_attempt"] = group_attempt
                _prepare_nemo_gym_rows(
                    rows,
                    generation_config={"max_new_tokens": 64},
                    sampling_params=SimpleNamespace(temperature=0.7, top_p=0.9),
                    num_generations=2,
                )
                assert all("_ng_rollout_id" not in row for row in rows)
                expected_task["task_input"]["responses_create_params"]["top_p"] = 0.9
                expected_episode_ids = [
                    {
                        "rollout_id": f"{row['_ng_group_id']}-g{index}-t{group_attempt}",
                        "attempt": 0,
                    }
                    for index, row in enumerate(rows[:2])
                ]

            actor = NemoGym.__ray_metadata__.modified_class({})
            actor.rh = object()
            actor.rch = gym_rollouts.RolloutCollectionHelper()
            actor.head_server_config = gym_server.BaseServerConfig(
                host=episode_server.host, port=episode_server.port
            )
            actor._tokenizer = _TokenDecoder()
            actor._token_capture_enabled = token_capture

            async def manifest(method: str, path: str) -> dict:
                capture_key = path.split("/")[-2]
                response_id = (
                    "response-flat"
                    if capture_key == "capture-flat"
                    else "response-episode"
                )
                return {
                    "rollout_id": capture_key,
                    "records": [
                        {
                            "model_call_id": "call-1",
                            "parent_call_id": None,
                            "prev_len": 0,
                            "delta_len": 5,
                            "cum_len": 5,
                            "weight_version": 3,
                            "digest": "a" * 64,
                            "extras_digest": "b" * 64,
                            "staging_key": f"{capture_key}/call-1",
                            "mode": "text",
                            "response_id": response_id,
                            "chain_hash": "c" * 64,
                            "cumulative_hash": "d" * 64,
                        }
                    ],
                    "failures": [],
                }

            actor._control = AsyncMock(side_effect=manifest)
            streamed = [
                item async for item in actor.run_rollouts(rows, "timing/integration")
            ]

        assert len(received_episode) == 2
        assert len(received_flat) == len(rows) - 2
        assert {body["episode_id"]["rollout_id"] for body in received_episode} == {
            episode_id["rollout_id"] for episode_id in expected_episode_ids
        }
        for body in received_episode:
            assert set(body) == {"episode_id", "task"}
            assert body["episode_id"] in expected_episode_ids
            assert body["task"] == expected_task
        for body in received_flat:
            assert "episode_id" not in body
            assert body["agent_ref"]["name"] == "legacy_agent"

        by_index = {
            row_index: (agent_ref, result)
            for row_index, agent_ref, result, _ in streamed
        }
        assert set(by_index) == set(range(len(rows)))
        assert sum(timing is not None for _, _, _, timing in streamed) == 1
        for index in (0, 1):
            agent_ref, result = by_index[index]
            assert agent_ref == {
                "type": "responses_api_agents",
                "name": "episode_agent",
            }
            assert result["full_result"]["reward"] == 0.75
            if not compatibility_response:
                assert (
                    result["full_result"]["_ng_episode_id"]
                    == expected_episode_ids[index]
                )
                assert (
                    result["full_result"]["_ng_task_id"] == materialized_task["task_id"]
                )
                assert (
                    result["full_result"]["_ng_environment_server"]
                    == "episode_environment"
                )
            if not token_capture:
                assert result["message_log"][0]["token_ids"].tolist() == [1, 2, 3]
                assert result["message_log"][1]["token_ids"].tolist() == [11, 12]
                assert result["message_log"][1][
                    "generation_logprobs"
                ].tolist() == pytest.approx([-0.1, -0.2])
                assert result["message_log"][1]["role"] == "assistant"
        for index in range(2, len(rows)):
            flat_ref, flat_result = by_index[index]
            assert flat_ref["name"] == "legacy_agent"
            assert flat_result["full_result"]["reward"] == 0.25
            if not token_capture:
                assert flat_result["message_log"][1]["token_ids"].tolist() == [21]
                assert flat_result["message_log"][1][
                    "generation_logprobs"
                ].tolist() == pytest.approx([-0.4])
        if token_capture:
            expected_captures = [
                ("capture-episode-0", 0.75),
                ("capture-episode-1-a2", 0.75),
                ("capture-flat", 0.25),
            ]
            actor._control.assert_has_awaits(
                [
                    call(
                        "GET",
                        f"/training-token-capture/control/rollouts/{capture_key}/manifest",
                    )
                    for capture_key, _ in expected_captures
                ],
                any_order=True,
            )
            assert actor._control.await_count == 3
            for index, (capture_key, reward) in enumerate(expected_captures):
                result = by_index[index][1]
                assert result["message_log"] == []
                assert result["rollout_id"] == capture_key
                assert result["receipt"]["rollout_id"] == capture_key
                assert result["receipt"]["reward"] == reward
                assert result["receipt"]["terminal_model_call_id"] == "call-1"
                assert result["receipt"]["terminal_selection"] == "response_id"
                assert result["receipt"]["capture_poisoned"] is False
        else:
            actor._control.assert_not_awaited()

    asyncio.run(run())
