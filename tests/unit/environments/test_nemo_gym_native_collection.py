"""Exercise Gym's real HTTP collector through RL's streaming rollout adapter.

The local endpoints return deterministic protocol-valid results. This covers
collector/adapter integration without starting an agent or a policy model.
"""

import asyncio
from copy import deepcopy

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from omegaconf import OmegaConf

from nemo_rl.environments.nemo_gym import NemoGym

# The draft can run against the old Gym pin until its native dependency merges.
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


def _model_response(*, native: bool) -> gym_openai.NeMoGymResponse:
    return gym_openai.NeMoGymResponse(
        id="response-native" if native else "response-legacy",
        created_at=0,
        model="deterministic-test-model",
        object="response",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            gym_openai.NeMoGymResponseOutputMessageForTraining(
                id="message-native" if native else "message-legacy",
                content=[
                    gym_openai.NeMoGymResponseOutputText(
                        text="The weather is sunny.", annotations=[]
                    )
                ],
                prompt_token_ids=[1, 2, 3] if native else [4, 5],
                generation_token_ids=[11, 12] if native else [21],
                generation_log_probs=[-0.1, -0.2] if native else [-0.4],
            )
        ],
    )


def test_real_collector_streams_native_and_legacy_http_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        received_native = []
        received_legacy = []

        async def native_run(request: web.Request) -> web.Response:
            body = await request.json()
            episode = gym_protocol.SingleAgentTurnRequest.model_validate(body)
            received_native.append(body)
            reply = gym_protocol.SingleAgentTurnResponse(
                episode_id=episode.episode_id,
                task_id=episode.task.task_id,
                result=gym_protocol.SingleAgentTurnResult(
                    responses_create_params=episode.task.task_input.responses_create_params,
                    response=_model_response(native=True),
                    reward=0.75,
                ),
            )
            return web.json_response(reply.model_dump(mode="json"))

        async def legacy_run(request: web.Request) -> web.Response:
            body = await request.json()
            legacy = gym_resources.BaseRunRequest.model_validate(body)
            received_legacy.append(body)
            reply = gym_resources.BaseVerifyResponse(
                responses_create_params=legacy.responses_create_params,
                response=_model_response(native=False),
                reward=0.25,
            )
            return web.json_response(reply.model_dump(mode="json"))

        native_app = web.Application()
        native_app.router.add_post("/run", native_run)
        legacy_app = web.Application()
        legacy_app.router.add_post("/run", legacy_run)
        async with (
            TestServer(native_app) as native_server,
            TestServer(legacy_app) as legacy_server,
            ClientSession() as http_client,
        ):
            global_config = OmegaConf.create(
                {
                    "environment_server_routes": {
                        "weather:train": "native_environment"
                    },
                    "native_agent": {"responses_api_agents": {"simple_agent": {}}},
                    "legacy_agent": {"responses_api_agents": {"simple_agent": {}}},
                    "native_environment": {
                        "environment_servers": {
                            "single_agent_turn": {
                                "host": native_server.host,
                                "port": native_server.port,
                                "agent_server": {
                                    "type": "responses_api_agents",
                                    "name": "native_agent",
                                },
                            }
                        }
                    },
                    "legacy_environment": {
                        "environment_servers": {
                            "legacy_agent": {
                                "host": legacy_server.host,
                                "port": legacy_server.port,
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

            native_task = {
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
            rows = [
                {
                    **deepcopy(native_task),
                    "_rowidx": index,
                    "_ng_group_id": "native-group",
                    "_ng_group_attempt": 2,
                    "_ng_rollout_index": index,
                }
                for index in range(2)
            ]
            rows.append(
                {
                    "_rowidx": 2,
                    "agent_ref": {"name": "legacy_agent"},
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "Legacy weather request"}]
                    },
                }
            )

            actor = NemoGym.__ray_metadata__.modified_class({})
            actor.rh = object()
            actor.rch = gym_rollouts.RolloutCollectionHelper()
            actor.head_server_config = gym_server.BaseServerConfig(
                host=native_server.host, port=native_server.port
            )
            actor._tokenizer = _TokenDecoder()
            streamed = [
                item async for item in actor.run_rollouts(rows, "timing/integration")
            ]

        assert len(received_native) == 2
        assert len(received_legacy) == 1
        assert {body["episode_id"]["rollout_id"] for body in received_native} == {
            "native-group-0",
            "native-group-1",
        }
        for body in received_native:
            assert set(body) == {"episode_id", "task"}
            assert body["episode_id"]["attempt"] == 2
            expected_task = deepcopy(native_task)
            expected_task["task_input"]["task_data"].update(
                _ng_group_id="native-group",
                _ng_group_attempt=2,
                _ng_rollout_index=int(
                    body["episode_id"]["rollout_id"].rsplit("-", 1)[1]
                ),
            )
            assert body["task"] == expected_task
        assert "episode_id" not in received_legacy[0]
        assert received_legacy[0]["agent_ref"]["name"] == "legacy_agent"

        by_index = {
            row_index: (agent_ref, result)
            for row_index, agent_ref, result, _ in streamed
        }
        assert set(by_index) == {0, 1, 2}
        assert sum(timing is not None for _, _, _, timing in streamed) == 1
        for index in (0, 1):
            agent_ref, result = by_index[index]
            assert agent_ref == {"type": "responses_api_agents", "name": "native_agent"}
            assert result["full_result"]["reward"] == 0.75
            assert result["full_result"]["_ng_episode_id"] == {
                "rollout_id": f"native-group-{index}",
                "attempt": 2,
            }
            assert result["full_result"]["_ng_task_id"] == native_task["task_id"]
            assert (
                result["full_result"]["_ng_environment_server"] == "native_environment"
            )
            assert result["message_log"][0]["token_ids"].tolist() == [1, 2, 3]
            assert result["message_log"][1]["token_ids"].tolist() == [11, 12]
            assert result["message_log"][1][
                "generation_logprobs"
            ].tolist() == pytest.approx([-0.1, -0.2])
            assert result["message_log"][1]["role"] == "assistant"
        legacy_ref, legacy_result = by_index[2]
        assert legacy_ref["name"] == "legacy_agent"
        assert legacy_result["full_result"]["reward"] == 0.25
        assert legacy_result["message_log"][1]["token_ids"].tolist() == [21]
        assert legacy_result["message_log"][1][
            "generation_logprobs"
        ].tolist() == pytest.approx([-0.4])

    asyncio.run(run())
