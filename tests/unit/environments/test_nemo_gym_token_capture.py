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

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from nemo_rl.environments.nemo_gym import NemoGym


def _capture_env() -> NemoGym:
    env_cls = NemoGym.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env._control_owner_id = "owner-1"
    return env


def test_register_rollouts_returns_capabilities_and_uses_idempotent_ownership() -> None:
    env = _capture_env()
    env._control = AsyncMock(
        side_effect=[
            {"data_capability": "cap-r0"},
            {"data_capability": "cap-r1"},
        ]
    )

    capabilities = asyncio.run(env.register_rollouts(["r0", "r1"]))

    assert capabilities == {"r0": "cap-r0", "r1": "cap-r1"}
    for rollout_id, call in zip(("r0", "r1"), env._control.await_args_list):
        assert call.args == (
            "PUT",
            f"/training-token-capture/control/rollouts/{rollout_id}",
        )
        assert call.kwargs["json"]["owner_id"] == "owner-1"
        assert call.kwargs["json"]["operation_id"].startswith("register-")


def test_receipt_postprocess_fails_closed_without_a_terminal_logical_id() -> None:
    env = _capture_env()
    env.fail_rollouts = AsyncMock()

    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"},
            {"reward": 1.0},
        )
    )

    env.fail_rollouts.assert_awaited_once_with(
        ["r0"], reason="missing_terminal_logical_request_id"
    )
    assert result["receipt"] is None


def test_receipt_postprocess_seals_the_trusted_logical_terminal() -> None:
    env = _capture_env()
    env._control = AsyncMock(return_value={"rollout_id": "r0", "manifest": []})

    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"},
            {
                "reward": 1.0,
                "terminal_logical_request_id": "logical-terminal",
            },
        )
    )

    call = env._control.await_args
    assert call.args == (
        "POST",
        "/training-token-capture/control/rollouts/r0/seal",
    )
    assert call.kwargs["json"]["owner_id"] == "owner-1"
    assert call.kwargs["json"]["operation_id"].startswith("seal-")
    assert call.kwargs["json"]["terminal_logical_request_id"] == ("logical-terminal")
    assert result["receipt"] == {"rollout_id": "r0", "manifest": []}
