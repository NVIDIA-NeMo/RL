import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from nemo_rl.environments.nemo_gym import (
    NemoGym,
    _normalize_nemo_gym_episode_result,
    _prepare_native_nemo_gym_rows,
)
from nemo_rl.experience.failures import (
    FailureClass,
    GymTerminalEpisodeFailure,
    RolloutDataFailure,
    classify_rollout_failure,
)

pytest.importorskip(
    "nemo_gym.episode_types", reason="Requires Gym Environment Server contracts"
)
gym_rollouts = pytest.importorskip("nemo_gym.rollout_collection")
pytestmark = pytest.mark.nemo_gym


def _row() -> dict:
    return {
        "task_id": {"taskset": "weather:train", "task_id": "first"},
        "task_input": {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Weather?"}]
            },
            "task_data": {"opaque": [1, {"city": "Paris"}]},
        },
        "_rowidx": 0,
        "_ng_group_id": "group-123",
        "_ng_rollout_index": 0,
        "_ng_group_attempt": 0,
        "_ng_rollout_id": "caller-owned-rollout",
    }


def _config():
    return OmegaConf.create(
        {
            "weather_environment": {
                "environment_servers": {
                    "single_agent_turn": {
                        "entrypoint": "app.py",
                        "agent_server": {
                            "type": "responses_api_agents",
                            "name": "weather_agent",
                        },
                    }
                }
            },
            "weather_agent": {
                "responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}
            },
        }
    )


def _prepare(rows):
    _prepare_native_nemo_gym_rows(
        rows, _config(), {"weather:train": "weather_environment"}
    )


def _reply(row: dict) -> dict:
    return {
        "episode_id": gym_rollouts._native_episode_request_body(row)["episode_id"],
        "task_id": deepcopy(row["task_id"]),
        "result": {
            "responses_create_params": deepcopy(
                row["task_input"]["responses_create_params"]
            ),
            "response": {"id": "resp-1", "output": []},
            "reward": 0.75,
            "reward_components": {"accuracy": 0.75},
            "ng_agent_observations": {"custom": "kept"},
            "mask_sample": False,
        },
        "failure": None,
    }


def test_native_preparation_preserves_payload_and_resolves_actual_agent():
    row = _row()
    original_payload = deepcopy(row["task_input"])
    row["agent_ref"] = {"name": "stale_dataset_agent"}
    row["_ng_group_attempt"] = 3
    _prepare([row])
    assert row["_ng_environment_server"] == "weather_environment"
    assert row["agent_ref"] == {"type": "responses_api_agents", "name": "weather_agent"}
    assert row["task_input"] == original_payload
    assert row["_ng_rollout_id"] == "caller-owned-rollout"
    assert "_ng_attempt_index" not in row
    assert "responses_create_params" not in row


def test_native_preparation_does_not_create_episode_identity():
    row = _row()
    del row["_ng_rollout_id"]
    row["_ng_group_attempt"] = 2
    original = deepcopy(row)
    _prepare([row])
    assert row == {
        **original,
        "_ng_environment_server": "weather_environment",
        "agent_ref": {"type": "responses_api_agents", "name": "weather_agent"},
    }


def test_legacy_preparation_and_result_are_opaque():
    row = {"agent_ref": {"name": "legacy"}, "responses_create_params": {"input": "hi"}}
    original = deepcopy(row)
    _prepare([row])
    reply = {"result": {"benchmark_specific": True}, "reward": 1}
    assert row == original
    assert _normalize_nemo_gym_episode_result(row, reply) is reply


def test_native_preparation_rejects_bad_destination_and_agent_reference():
    with pytest.raises(ValueError, match="No environment server route"):
        _prepare_native_nemo_gym_rows([_row()], _config(), {})
    with pytest.raises(ValueError, match="exactly one Environment Server"):
        _prepare_native_nemo_gym_rows(
            [_row()], _config(), {"weather:train": "weather_agent"}
        )
    config = _config()
    del config.weather_environment.environment_servers.single_agent_turn.agent_server
    with pytest.raises(ValueError, match="agent_server reference"):
        _prepare_native_nemo_gym_rows(
            [_row()], config, {"weather:train": "weather_environment"}
        )


def test_native_success_retains_reward_components_observations_and_identity():
    row = _row()
    _prepare([row])
    reply = _reply(row)
    original = deepcopy(reply)
    result = _normalize_nemo_gym_episode_result(row, reply)
    for key, value in reply["result"].items():
        assert result[key] == value
    assert result["_ng_task_id"] == row["task_id"]
    assert result["_ng_episode_id"] == reply["episode_id"]
    assert result["_ng_environment_server"] == "weather_environment"
    assert reply == original


@pytest.mark.parametrize("key", ["task_id", "episode_id"])
def test_native_reply_identity_must_match_request(key):
    row = _row()
    _prepare([row])
    reply = _reply(row)
    if key == "task_id":
        reply[key]["task_id"] = "different-task"
    else:
        reply[key]["attempt"] = 1
    with pytest.raises(RolloutDataFailure, match=f"mismatched {key}"):
        _normalize_nemo_gym_episode_result(row, reply)


@pytest.mark.parametrize(
    "terminal,error_type",
    [(True, GymTerminalEpisodeFailure), (False, RuntimeError)],
)
def test_handled_native_failure_is_never_a_scored_result(terminal, error_type):
    row = _row()
    _prepare([row])
    reply = _reply(row)
    reply["result"] = None
    reply["failure"] = {
        "message": "verification failed",
        "terminal": terminal,
        "stage": "verification",
        "partial_response": {"id": "partial"},
    }
    with pytest.raises(
        error_type, match="verification failed.*stage=verification"
    ) as error:
        _normalize_nemo_gym_episode_result(row, reply)
    assert classify_rollout_failure(error.value) is FailureClass.DATA


@pytest.mark.parametrize("reward", [None, "1.0", True, float("nan"), float("inf")])
def test_native_training_requires_a_finite_scalar_reward(reward):
    row = _row()
    _prepare([row])
    reply = _reply(row)
    reply["result"]["reward"] = reward
    with pytest.raises(RolloutDataFailure, match="finite scalar reward"):
        _normalize_nemo_gym_episode_result(row, reply)


@pytest.mark.parametrize(
    "explicit_id,attempt,capture_key",
    [
        (True, 0, "caller-owned-rollout"),
        (True, 2, "caller-owned-rollout-a2"),
        (False, 0, "17-0"),
        (False, 2, "17-0-a2"),
    ],
)
def test_native_receipt_reads_attempt_qualified_capture_manifest(
    explicit_id, attempt, capture_key
):
    row = _row()
    row["_ng_attempt_index"] = attempt
    if not explicit_id:
        del row["_ng_rollout_id"]
        row["_ng_task_index"] = 17
    _prepare([row])
    reply = _reply(row)
    del reply["result"]["response"]
    del reply["result"]["responses_create_params"]
    result = _normalize_nemo_gym_episode_result(row, reply)
    env_class = NemoGym.__ray_metadata__.modified_class
    env = object.__new__(env_class)
    env._control = AsyncMock(return_value={"records": [], "failures": []})
    processed = asyncio.run(env._postprocess_receipt_mode(row, result))
    env._control.assert_awaited_once_with(
        "GET", f"/training-token-capture/control/rollouts/{capture_key}/manifest"
    )
    assert processed["rollout_id"] == capture_key
    assert processed["receipt"]["reward"] == 0.75
    assert processed["full_result"]["ng_agent_observations"] == {"custom": "kept"}
