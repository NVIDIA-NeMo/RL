"""Exercise Gym's real HTTP collector through RL's streaming rollout adapter.

The local endpoints return deterministic protocol-valid results. This covers
collector/adapter integration without starting an agent or a policy model.
"""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, call

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


@pytest.mark.parametrize("token_capture", [False, True])
def test_real_collector_streams_native_and_legacy_http_results(
    monkeypatch: pytest.MonkeyPatch,
    token_capture: bool,
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
                    "_ng_group_attempt": 3,
                    "_ng_rollout_index": index,
                    "_ng_rollout_id": f"capture-native-{index}",
                    "_ng_attempt_index": index * 2,
                }
                for index in range(2)
            ]
            rows.append(
                {
                    "_rowidx": 2,
                    "_ng_rollout_id": "capture-legacy",
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
            actor._token_capture_enabled = token_capture

            async def manifest(method: str, path: str) -> dict:
                capture_key = path.split("/")[-2]
                response_id = (
                    "response-legacy"
                    if capture_key == "capture-legacy"
                    else "response-native"
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

        assert len(received_native) == 2
        assert len(received_legacy) == 1
        assert {body["episode_id"]["rollout_id"] for body in received_native} == {
            "capture-native-0",
            "capture-native-1",
        }
        for body in received_native:
            assert set(body) == {"episode_id", "task"}
            index = int(body["episode_id"]["rollout_id"].rsplit("-", 1)[1])
            assert body["episode_id"]["attempt"] == index * 2
            assert body["task"] == native_task
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
                "rollout_id": f"capture-native-{index}",
                "attempt": index * 2,
            }
            assert result["full_result"]["_ng_task_id"] == native_task["task_id"]
            assert (
                result["full_result"]["_ng_environment_server"] == "native_environment"
            )
            if not token_capture:
                assert result["message_log"][0]["token_ids"].tolist() == [1, 2, 3]
                assert result["message_log"][1]["token_ids"].tolist() == [11, 12]
                assert result["message_log"][1][
                    "generation_logprobs"
                ].tolist() == pytest.approx([-0.1, -0.2])
                assert result["message_log"][1]["role"] == "assistant"
        legacy_ref, legacy_result = by_index[2]
        assert legacy_ref["name"] == "legacy_agent"
        assert legacy_result["full_result"]["reward"] == 0.25
        if token_capture:
            expected_captures = [
                ("capture-native-0", 0.75),
                ("capture-native-1-a2", 0.75),
                ("capture-legacy", 0.25),
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
            assert legacy_result["message_log"][1]["token_ids"].tolist() == [21]
            assert legacy_result["message_log"][1][
                "generation_logprobs"
            ].tolist() == pytest.approx([-0.4])

    asyncio.run(run())
