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
import hashlib
import time
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.nemo_gym import (
    NemoGym,
    _adapt_execution_identity_for_installed_gym,
)


def _capability(component: str, name: str, **overrides):
    payload = {
        "component": component,
        "name": name,
        "schema_version": 1,
        "admission_states": ["accepting"],
        "checkpoint_mode": "export_restore",
        "concurrency_contract": "serialized_per_session",
        "multi_process": {"mode": "single_worker", "num_workers": 1},
        "instance_role": None,
        "phase": "idle",
        "active_checkpoint_id": None,
        "deadline_ts": None,
    }
    payload.update(overrides)
    return payload


def _checkpoint_env():
    env_cls = NemoGym.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.rh = object()
    env._gym_checkpoint_participants = ()
    env._control_timeout_s = 60.0
    return env


@pytest.mark.parametrize(
    ("stable_identity_enabled", "expected"),
    [
        (
            True,
            {"_ng_rollout_id": "rollout-1", "_ng_attempt_index": 2},
        ),
        (False, {"_ng_rollout_id": "rollout-1-a2"}),
    ],
)
def test_execution_identity_adapts_only_for_legacy_gym(
    stable_identity_enabled: bool,
    expected: dict,
) -> None:
    row = {"_ng_rollout_id": "rollout-1", "_ng_attempt_index": 2}

    _adapt_execution_identity_for_installed_gym(
        row,
        stable_execution_identity_enabled=stable_identity_enabled,
    )

    assert row == expected


def test_checkpoint_capability_discovery_validates_and_caches_participants() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent": _capability("responses_api_agents", "agent"),
        "tools": _capability("resources_servers", "tools"),
    }

    async def control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=control)

    discovered = asyncio.run(
        env.discover_checkpoint_capabilities(["tools", "policy", "agent"])
    )

    assert [
        item["participant"]["server_name"] for item in discovered["participants"]
    ] == [
        "agent",
        "policy",
        "tools",
    ]
    assert len(env._gym_checkpoint_participants) == 3


def test_checkpoint_capability_discovery_rejects_unmanaged_workers() -> None:
    env = _checkpoint_env()
    env._control = AsyncMock(
        return_value=_capability(
            "resources_servers",
            "tools",
            multi_process={"mode": "unmanaged", "num_workers": 4},
        )
    )

    with pytest.raises(RuntimeError, match="4 unmanaged workers"):
        asyncio.run(env.discover_checkpoint_capabilities(["tools"]))


def test_checkpoint_capability_discovery_requires_policy_model() -> None:
    env = _checkpoint_env()
    env._control = AsyncMock(return_value=_capability("resources_servers", "tools"))

    with pytest.raises(RuntimeError, match="no policy model participant"):
        asyncio.run(env.discover_checkpoint_capabilities(["tools"]))


def test_checkpoint_prepare_fans_out_using_component_routes() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent": _capability("responses_api_agents", "agent"),
        "tools": _capability("resources_servers", "tools"),
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))

    responses = {
        "policy": {
            "state": "paused",
            "workers": {"acknowledged": 1, "expected": 1},
            "inflight_total": 0,
            "waiters_total": 0,
        },
        "agent": {
            "state": "preparing",
            "ready_to_commit": True,
            "running": 0,
            "parked": 1,
            "parked_with_boundary": 1,
            "parked_without_boundary": 0,
            "completed_unacknowledged": 0,
            "active": 1,
            "blocking_attempts": [],
            "completed_unacknowledged_attempts": [],
            "executions": [],
        },
        "tools": {"sessions": 1, "state": "prepared"},
    }

    async def prepare_control(method, path, *, server_name, timeout_s, json):
        assert method == "POST"
        assert path.endswith("/prepare") or path.endswith("/pause")
        assert timeout_s > 0
        assert json == {
            "schema_version": 1,
            "checkpoint_id": "snapshot-7",
            "deadline_ts": 123.0,
        }
        return responses[server_name]

    env._control = AsyncMock(side_effect=prepare_control)

    result = asyncio.run(env.prepare_checkpoint("snapshot-7", 123.0))

    assert result["ready"] is True
    assert {item["participant"]["server_name"] for item in result["participants"]} == {
        "policy",
        "agent",
        "tools",
    }


def test_checkpoint_prepare_waits_for_draining_policy_model() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent": _capability("responses_api_agents", "agent"),
        "tools": _capability("resources_servers", "tools"),
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))
    calls = []

    async def prepare_control(method, path, *, server_name, **_kwargs):
        calls.append((method, path, server_name))
        if server_name == "policy" and path.endswith("/pause"):
            return {
                "state": "draining",
                "workers": {"acknowledged": 1, "expected": 1},
                "inflight_total": 1,
                "waiters_total": 0,
            }
        if server_name == "policy" and path.endswith("/status"):
            return {
                "checkpoint_id": "snapshot-8",
                "state": "paused",
                "per_worker": {"0": {"state": "paused", "inflight": 0}},
                "inflight_total": 0,
                "waiters_total": 0,
                "inflight": [],
                "tombstones": [],
            }
        if server_name == "agent":
            return {
                "state": "preparing",
                "ready_to_commit": True,
                "running": 0,
                "parked": 0,
                "parked_with_boundary": 0,
                "parked_without_boundary": 0,
                "completed_unacknowledged": 0,
                "active": 0,
                "blocking_attempts": [],
                "completed_unacknowledged_attempts": [],
                "executions": [],
            }
        return {"sessions": 0, "state": "prepared"}

    env._control = AsyncMock(side_effect=prepare_control)

    result = asyncio.run(env.prepare_checkpoint("snapshot-8", time.time() + 10.0))

    assert result["ready"] is True
    assert any(path.endswith("/status") for _method, path, _server in calls)


def test_checkpoint_prepare_timeout_resumes_touched_participants() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent": _capability("responses_api_agents", "agent"),
        "tools": _capability("resources_servers", "tools"),
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))
    resume_order = []

    async def prepare_control(_method, path, *, server_name, **_kwargs):
        if path.endswith("/pause"):
            return {
                "state": "draining",
                "workers": {"acknowledged": 1, "expected": 1},
                "inflight_total": 1,
                "waiters_total": 0,
            }
        if path.endswith("/status"):
            return {
                "checkpoint_id": "snapshot-9",
                "state": "draining",
                "per_worker": {"0": {"state": "draining", "inflight": 1}},
                "inflight_total": 1,
                "waiters_total": 0,
                "inflight": [
                    {
                        "rollout_id": "rollout-1",
                        "attempt_index": 0,
                        "plane": "policy",
                        "age_seconds": 1.0,
                    }
                ],
                "tombstones": [],
            }
        if path.endswith("/prepare") and server_name == "agent":
            return {
                "state": "preparing",
                "ready_to_commit": True,
                "running": 0,
                "parked": 0,
                "parked_with_boundary": 0,
                "parked_without_boundary": 0,
                "completed_unacknowledged": 0,
                "active": 0,
                "blocking_attempts": [],
                "completed_unacknowledged_attempts": [],
                "executions": [],
            }
        if path.endswith("/prepare"):
            return {"sessions": 0, "state": "prepared"}
        resume_order.append(server_name)
        if server_name == "policy":
            return {
                "state": "accepting",
                "workers": {"acknowledged": 1, "expected": 1},
                "released_waiters": 0,
            }
        if server_name == "agent":
            return {"state": "accepting", "released": 0}
        return {"state": "accepting"}

    env._control = AsyncMock(side_effect=prepare_control)

    with pytest.raises(TimeoutError, match="remained 'draining'"):
        asyncio.run(env.prepare_checkpoint("snapshot-9", time.time() + 10.0))

    assert resume_order == ["tools", "agent", "policy"]


def test_checkpoint_commit_restore_and_resume_fan_out() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent": _capability("responses_api_agents", "agent"),
        "tools": _capability("resources_servers", "tools"),
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))

    responses = {
        ("policy", "commit"): {
            "rollouts": 2,
            "rows": 4,
            "excluded_tombstoned": 0,
            "manifest_digest": "a" * 64,
        },
        ("agent", "commit"): {
            "records": 2,
            "manifest_digest": "b" * 64,
        },
        ("tools", "commit"): {
            "sessions": 2,
            "manifest_digest": "c" * 64,
        },
        ("policy", "restore"): {
            "rollouts": 2,
            "rows": 4,
            "checkpoint_id": "snapshot-7",
            "tombstones": [],
            "source_attempts": [{"rollout_id": "rollout-1", "attempt_index": 0}],
        },
        ("agent", "restore"): {
            "records": 2,
            "source_checkpoint_id": "snapshot-7",
        },
        ("tools", "restore"): {
            "sessions": 2,
            "source_checkpoint_id": "snapshot-7",
        },
        ("policy", "resume"): {
            "state": "accepting",
            "workers": {"acknowledged": 1, "expected": 1},
            "released_waiters": 0,
        },
        ("agent", "resume"): {"state": "accepting", "released": 2},
        ("tools", "resume"): {"state": "accepting"},
    }
    calls = []

    async def lifecycle_control(method, path, *, server_name, timeout_s, json):
        assert method == "POST"
        assert timeout_s > 0
        calls.append((server_name, path, json))
        return responses[(server_name, path.rsplit("/", 1)[-1])]

    env._control = AsyncMock(side_effect=lifecycle_control)
    committed = asyncio.run(
        env.commit_checkpoint("snapshot-7", 123.0, "/tmp/snapshot-7")
    )
    restored = asyncio.run(
        env.restore_checkpoint("snapshot-7", 123.0, "/tmp/snapshot-7")
    )
    resumed = asyncio.run(env.resume_checkpoint("snapshot-7", 123.0))

    assert {
        item["manifest"]["relative_path"] for item in committed["participants"]
    } == {
        "model-ledger/policy/manifest.json",
        f"agent/instance-{hashlib.sha256(b'agent').hexdigest()}/manifest.json",
        "resources/tools/manifest.json",
    }
    assert len(restored["participants"]) == 3
    assert len(resumed["participants"]) == 3
    assert len(calls) == 9
    assert [server_name for server_name, _path, _json in calls] == [
        "policy",
        "agent",
        "tools",
        "policy",
        "agent",
        "tools",
        "tools",
        "agent",
        "policy",
    ]


def test_abort_checkpoint_uses_idempotent_resume_routes() -> None:
    env = _checkpoint_env()
    env.resume_checkpoint = AsyncMock(return_value={"checkpoint_id": "snapshot-7"})

    result = asyncio.run(env.abort_checkpoint("snapshot-7", 123.0))

    assert result == {"checkpoint_id": "snapshot-7"}
    env.resume_checkpoint.assert_awaited_once_with("snapshot-7", 123.0)
