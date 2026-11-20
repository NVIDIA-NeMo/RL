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
import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.nemo_gym import (
    NemoGym,
    _normalize_nemo_gym_episode_result,
    _prepare_materialized_nemo_gym_rows,
)
from nemo_rl.environments.nemo_gym_multi_harness import MultiHarnessRoutes
from nemo_rl.experience.failures import (
    FailureClass,
    GymTerminalEpisodeFailure,
    GymTransportError,
    RolloutDataFailure,
    classify_rollout_failure,
    is_terminal_gym_episode_failure,
)

pytest.importorskip(
    "nemo_gym.episode_types", reason="Requires Gym Environment Server contracts"
)
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


def _prepare(rows):
    _prepare_materialized_nemo_gym_rows(rows, {"weather:train": "weather_environment"})


def _reply(row: dict) -> dict:
    return {
        "episode_id": {"rollout_id": "caller-owned-rollout", "attempt": 0},
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


def test_materialized_task_preparation_preserves_payload_without_adding_agent_ref():
    row = _row()
    original_payload = deepcopy(row["task_input"])
    row["_ng_group_attempt"] = 3
    _prepare([row])
    assert row["_ng_environment_server"] == "weather_environment"
    assert "agent_ref" not in row
    assert row["task_input"] == original_payload
    assert row["_ng_rollout_id"] == "caller-owned-rollout"
    assert "_ng_attempt_index" not in row
    assert "responses_create_params" not in row


def test_flat_row_preparation_and_result_are_opaque():
    row = {
        "agent_ref": {"name": "flat_row_agent"},
        "responses_create_params": {"input": "hi"},
    }
    original = deepcopy(row)
    _prepare([row])
    reply = {"result": {"benchmark_specific": True}, "reward": 1}
    assert row == original
    assert _normalize_nemo_gym_episode_result(row, reply) is reply


def test_materialized_task_keeps_flat_compatibility_results_opaque() -> None:
    row = _row()
    _prepare([row])
    reply = {"reward": 0.75, "response": {}, "result": {"benchmark_specific": True}}
    assert _normalize_nemo_gym_episode_result(row, reply) is reply


def test_compatibility_judge_failure_stays_a_masked_result() -> None:
    row = _row()
    _prepare([row])
    reply = {
        "reward": 0.0,
        "response": {},
        "mask_sample": True,
        "failure_kind": "judge_failed",
        "_ng_failure_class": "judge_failed",
    }
    assert _normalize_nemo_gym_episode_result(row, reply) is reply


_WEATHER_ROUTES = MultiHarnessRoutes(
    candidates=(
        (
            "weather:train",
            (
                ("careful_agent", "weather_careful", 1.0),
                ("fast_agent", "weather_fast", 1.0),
            ),
        ),
    )
)


def _group_rows(groups, generations):
    rows = []
    for group in groups:
        for index in range(generations):
            row = _row()
            row["_rowidx"] = len(rows)
            row["_ng_group_id"] = group
            row["_ng_rollout_index"] = index
            row["_ng_rollout_id"] = f"rollout-{len(rows)}"
            rows.append(row)
    return rows


def test_multi_harness_stamps_one_server_per_group():
    rows = _group_rows([f"group-{i}" for i in range(8)], generations=3)

    _prepare_materialized_nemo_gym_rows(rows, {}, _WEATHER_ROUTES)

    for row in rows:
        assert (
            row["_ng_environment_server"]
            == _WEATHER_ROUTES.select(
                "weather:train", row["_ng_group_id"]
            ).environment_server
        )
    assert {row["_ng_environment_server"] for row in rows} == {
        "weather_fast",
        "weather_careful",
    }


def test_stream_runs_each_group_on_its_harness_and_reports_it():
    rows = _group_rows(["group-1", "group-2", "group-3", "group-4"], generations=2)
    sent = []

    class _Collector:
        def run_examples(self, examples, head_server_config, route_failures_to_sidecar):
            del head_server_config, route_failures_to_sidecar
            sent.extend(deepcopy(examples))

            async def _done(row):
                reply = _reply(row)
                reply["episode_id"]["rollout_id"] = row["_ng_rollout_id"]
                return row, reply

            return [_done(row) for row in examples]

    cls = NemoGym.__ray_metadata__.modified_class
    env = cls.__new__(cls)
    env.__init__({})
    env.set_multi_harness_routes(_WEATHER_ROUTES)
    env.rh = object()
    env.rch = _Collector()
    env.head_server_config = object()
    env._tokenizer = object()
    env._environment_server_routes = {}
    env._postprocess_nemo_gym_to_nemo_rl_result = (
        lambda row, result, tokenizer, include_initial_multimodal_data: {
            "message_log": [],
            "full_result": result,
        }
    )

    async def _collect():
        return [item async for item in env._stream_rollouts(deepcopy(rows), "test")]

    streamed = asyncio.run(_collect())

    assert len(streamed) == len(rows)
    for rowidx, agent_ref, result, _ in streamed:
        choice = _WEATHER_ROUTES.select("weather:train", rows[rowidx]["_ng_group_id"])
        assert agent_ref is None
        assert result["attribution_key"] == "weather:train"
        assert (
            result["full_result"]["_ng_environment_server"] == choice.environment_server
        )
        assert result["full_result"]["_ng_harness"] == choice.harness
    # Gym receives the chosen server, not RL's harness label.
    assert all("_ng_harness" not in row for row in sent)
    assert {row["_ng_environment_server"] for row in sent} == {
        "weather_fast",
        "weather_careful",
    }


def test_materialized_task_preparation_rejects_missing_destination():
    with pytest.raises(ValueError, match="No environment server route"):
        _prepare_materialized_nemo_gym_rows([_row()], {})


def test_episode_success_retains_reward_components_observations_and_identity():
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
def test_episode_reply_identity_must_match_request(key):
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
    "terminal,error_type,failure_class",
    [
        (True, GymTerminalEpisodeFailure, FailureClass.DATA),
        (False, GymTransportError, FailureClass.INFRA),
    ],
)
@pytest.mark.parametrize("compatibility_response", [False, True])
def test_handled_episode_failure_is_never_a_scored_result(
    terminal, error_type, failure_class, compatibility_response
):
    row = _row()
    _prepare([row])
    reply = _reply(row)
    reply["result"] = None
    reply["failure"] = {
        "failure_reason": "verification failed",
        "terminal": terminal,
        "stage": "verification",
        "partial_response": {"id": "partial"},
    }
    if compatibility_response:
        reply = {
            "_ng_failure_class": "environment_server_failed",
            "_ng_failure_terminal": terminal,
            "_ng_failure_message": "verification failed",
            "_ng_failure_stage": "verification",
        }
    with pytest.raises(
        error_type, match="verification failed.*stage=verification"
    ) as error:
        _normalize_nemo_gym_episode_result(row, reply)
    assert classify_rollout_failure(error.value) is failure_class


@pytest.mark.parametrize("reward", [None, "1.0", True, float("nan"), float("inf")])
def test_episode_training_requires_a_finite_scalar_reward(reward):
    row = _row()
    _prepare([row])
    reply = _reply(row)
    reply["result"]["reward"] = reward
    with pytest.raises(RolloutDataFailure, match="finite scalar reward"):
        _normalize_nemo_gym_episode_result(row, reply)


@pytest.mark.parametrize("compatibility_response", [False, True])
def test_response_free_result_requires_token_capture(compatibility_response):
    row = _row()
    _prepare([row])
    reply = _reply(row)
    del reply["result"]["response"]
    if compatibility_response:
        reply = reply["result"]
    with pytest.raises(
        GymTerminalEpisodeFailure,
        match="weather:train.*weather_environment.*token_capture.enabled: true",
    ) as error:
        _normalize_nemo_gym_episode_result(row, reply)
    assert is_terminal_gym_episode_failure(error.value)
    assert (
        _normalize_nemo_gym_episode_result(row, reply, token_capture_enabled=True)[
            "reward"
        ]
        == 0.75
    )


@pytest.mark.parametrize("attempt,capture_key", [(0, "17-0"), (2, "17-0-a2")])
def test_episode_receipt_uses_fallback_identity_without_response(attempt, capture_key):
    row = _row()
    row["_ng_attempt_index"] = attempt
    del row["_ng_rollout_id"]
    row["_ng_task_index"] = 17
    _prepare([row])
    assert "_ng_rollout_id" not in row
    reply = _reply(row)
    reply["episode_id"] = {"rollout_id": "17-0", "attempt": attempt}
    del reply["result"]["response"]
    del reply["result"]["responses_create_params"]
    result = _normalize_nemo_gym_episode_result(row, reply, token_capture_enabled=True)
    env_class = NemoGym.__ray_metadata__.modified_class
    env = env_class({})
    env._control = AsyncMock(return_value={"records": [], "failures": []})
    processed = asyncio.run(env._postprocess_receipt_mode(row, result))
    env._control.assert_awaited_once_with(
        "GET", f"/training-token-capture/control/rollouts/{capture_key}/manifest"
    )
    assert processed["rollout_id"] == capture_key
    assert processed["receipt"]["capture_key"] == capture_key
    assert processed["receipt"]["reward"] == 0.75
    assert processed["full_result"]["ng_agent_observations"] == {"custom": "kept"}
