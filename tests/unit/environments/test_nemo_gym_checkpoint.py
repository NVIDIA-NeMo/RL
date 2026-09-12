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
    GymControlRequestError,
    NemoGym,
    _adapt_execution_identity_for_installed_gym,
)
from nemo_rl.environments.gym_checkpoint import (
    GymActorExecutionRegistry,
    GymExecutionIdentity,
    GymResourcesPrepareResponse,
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
    env._active_gym_checkpoint_id = None
    env._gym_execution_registry = GymActorExecutionRegistry()
    return env


def _completion_receipt(
    rollout_id: str = "group-7_g0",
    attempt_index: int = 2,
) -> dict:
    return {
        "rollout_id": rollout_id,
        "attempt_index": attempt_index,
        "execution_generation": attempt_index + 1,
        "result_identity": f"result-{rollout_id}-{attempt_index}",
        "result_digest": f"{attempt_index + 1:064x}",
    }


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
        "tools",
        "agent",
        "policy",
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


def test_discard_restored_agent_continuations_fans_out_before_resume() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent-route": _capability("responses_api_agents", "agent"),
        "tools": _capability(
            "resources_servers",
            "tools",
            checkpoint_mode="restart_only",
            concurrency_contract="stateless",
        ),
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))
    env._active_gym_checkpoint_id = "restore-1"
    env._control = AsyncMock(return_value={"discarded": True})

    result = asyncio.run(
        env.discard_restored_agent_continuations(
            "restore-1",
            time.time() + 60,
            [{"rollout_id": "group_g0", "attempt_index": 1}],
        )
    )

    assert result == {
        "executions": 1,
        "agent_participants": 1,
        "discarded": 1,
    }
    call = env._control.await_args
    assert call.args[:2] == (
        "POST",
        "/ng-control/v1/agent-checkpoint/discard-restored-continuation",
    )
    assert call.kwargs["server_name"] == "agent-route"
    assert call.kwargs["json"]["rollout_id"] == "group_g0"
    assert call.kwargs["json"]["attempt_index"] == 1


def test_checkpoint_prepare_fans_out_using_component_routes() -> None:
    env = _checkpoint_env()
    deadline_ts = time.time() + 60.0
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
            "acknowledged_completed": 0,
            "active": 1,
            "blocking_attempts": [],
            "completed_unacknowledged_attempts": [],
            "selected_boundaries": [],
            "executions": [],
        },
        "tools": {
            "sessions": 1,
            "state": "prepared",
            "inventory": [
                {
                    "rollout_id": "rollout-1",
                    "attempt_index": 0,
                    "revision": 2,
                    "mutation_receipts": 2,
                }
            ],
        },
    }

    async def prepare_control(method, path, *, server_name, timeout_s, json):
        assert method == "POST"
        assert path.endswith("/prepare") or path.endswith("/pause")
        assert timeout_s > 0
        assert json == {
            "schema_version": 1,
            "checkpoint_id": "snapshot-7",
            "deadline_ts": deadline_ts,
        }
        return responses[server_name]

    env._control = AsyncMock(side_effect=prepare_control)

    result = asyncio.run(env.prepare_checkpoint("snapshot-7", deadline_ts))

    assert result["ready"] is True
    assert {item["participant"]["server_name"] for item in result["participants"]} == {
        "policy",
        "agent",
        "tools",
    }


def test_resources_prepare_inventory_count_must_match_sessions() -> None:
    with pytest.raises(ValueError, match="session count does not match inventory"):
        GymResourcesPrepareResponse.model_validate(
            {
                "sessions": 1,
                "state": "prepared",
                "inventory": [],
            }
        )


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
                "per_worker": {
                    "0": {
                        "state": "paused",
                        "inflight": 0,
                        "future_worker_metric": 7,
                    }
                },
                "inflight_total": 0,
                "response_inflight_total": 0,
                "generation_pending_total": 0,
                "waiters_total": 0,
                "inflight": [],
                "tombstones": [],
                "future_status_metric": 9,
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
                "acknowledged_completed": 0,
                "active": 0,
                "blocking_attempts": [],
                "completed_unacknowledged_attempts": [],
                "selected_boundaries": [],
                "executions": [],
            }
        return {"sessions": 0, "state": "prepared", "inventory": []}

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
                "acknowledged_completed": 0,
                "active": 0,
                "blocking_attempts": [],
                "completed_unacknowledged_attempts": [],
                "selected_boundaries": [],
                "executions": [],
            }
        if path.endswith("/prepare"):
            return {"sessions": 0, "state": "prepared", "inventory": []}
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
    continuation_index = {
        "schema_version": 1,
        "relative_path": "agent/instance-test/continuations.jsonl",
        "sha256": "d" * 64,
        "records": 2,
        "bytes": 256,
    }
    storage_reference_index = {
        "schema_version": 1,
        "relative_path": "model-ledger/policy/storage-references.jsonl",
        "sha256": "e" * 64,
        "records": 4,
        "bytes": 512,
    }
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
            features=["external_storage_reference_index_v1"],
        ),
        "agent": _capability(
            "responses_api_agents",
            "agent",
            features=["agent_continuation_index_v1"],
        ),
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
            "excluded_inactive": 1,
            "manifest_digest": "a" * 64,
            "storage_reference_index": storage_reference_index,
        },
        ("agent", "commit"): {
            "records": 2,
            "manifest_digest": "b" * 64,
            "continuation_index": continuation_index,
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
            "storage_reference_index": storage_reference_index,
        },
        ("agent", "restore"): {
            "records": 2,
            "source_checkpoint_id": "snapshot-7",
            "continuation_index": continuation_index,
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
        env.restore_checkpoint(
            "restore-7",
            123.0,
            "/tmp/snapshot-7",
            source_checkpoint_id="snapshot-7",
        )
    )
    resumed = asyncio.run(env.resume_checkpoint("restore-7", 123.0))

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
        "agent",
        "policy",
        "tools",
        "policy",
        "agent",
        "tools",
        "tools",
        "agent",
        "policy",
    ]
    commit_calls = calls[:3]
    assert "include_continuation_index" not in commit_calls[0][2]
    assert commit_calls[1][2]["continuation_indexes"] == [continuation_index]
    restore_calls = calls[3:6]
    assert "include_storage_reference_index" not in restore_calls[0][2]
    assert "include_continuation_index" not in restore_calls[1][2]


def test_abort_checkpoint_uses_idempotent_resume_routes() -> None:
    env = _checkpoint_env()
    env.resume_checkpoint = AsyncMock(return_value={"checkpoint_id": "snapshot-7"})

    result = asyncio.run(env.abort_checkpoint("snapshot-7", 123.0))

    assert result == {"checkpoint_id": "snapshot-7"}
    env.resume_checkpoint.assert_awaited_once_with("snapshot-7", 123.0)


def test_completed_results_are_acknowledged_by_resolved_agent() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent-route": _capability(
            "responses_api_agents",
            "resolved-agent",
            features=["completed_result_acknowledgement"],
        ),
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))

    receipt = _completion_receipt()

    async def acknowledge_control(method, path, *, server_name, json, **_kwargs):
        assert method == "POST"
        assert path.endswith("/acknowledge-completed")
        assert server_name == "agent-route"
        assert json == {
            "schema_version": 1,
            "executions": [receipt],
        }
        return {"acknowledged": json["executions"]}

    env._control = AsyncMock(side_effect=acknowledge_control)
    result = asyncio.run(
        env.acknowledge_completed_executions(
            [
                {
                    "receipt": receipt,
                    "agent_name": "resolved-agent",
                }
            ]
        )
    )

    assert result["acknowledged"] == [
        {
            "receipt": receipt,
            "agent_name": "resolved-agent",
        }
    ]


def test_completed_result_acknowledgement_rejects_a_different_receipt() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent-route": _capability(
            "responses_api_agents",
            "resolved-agent",
            features=["completed_result_acknowledgement"],
        ),
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))
    receipt = _completion_receipt()

    async def acknowledge_control(_method, _path, **_kwargs):
        return {
            "acknowledged": [{**receipt, "result_digest": "f" * 64}],
        }

    env._control = AsyncMock(side_effect=acknowledge_control)
    with pytest.raises(RuntimeError, match="did not cover"):
        asyncio.run(
            env.acknowledge_completed_executions(
                [{"receipt": receipt, "agent_name": "resolved-agent"}]
            )
        )


def test_completion_receipt_uses_exact_participant_lookup() -> None:
    env = _checkpoint_env()
    capabilities = {
        "policy": _capability(
            "responses_api_models",
            "policy",
            admission_states=["accepting", "draining", "paused"],
            concurrency_contract="stateless",
            instance_role="policy",
        ),
        "agent-route": _capability(
            "responses_api_agents",
            "resolved-agent",
            features=["completed_result_acknowledgement"],
        ),
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))
    receipt = _completion_receipt()

    async def receipt_control(method, path, *, server_name, params, **_kwargs):
        assert method == "GET"
        assert path.endswith("/completion-receipt")
        assert server_name == "agent-route"
        assert params == {
            "rollout_id": "group-7_g0",
            "attempt_index": 2,
        }
        return receipt

    env._control = AsyncMock(side_effect=receipt_control)
    resolved = asyncio.run(
        env._completion_receipt_for(
            GymExecutionIdentity(rollout_id="group-7_g0", attempt_index=2),
            agent_name="resolved-agent",
        )
    )

    assert resolved.model_dump(mode="json") == receipt


def test_actor_registry_fences_dispatch_and_tracks_frozen_membership() -> None:
    registry = GymActorExecutionRegistry()
    execution = GymExecutionIdentity(rollout_id="rollout-1", attempt_index=1)
    registry.register(execution)
    registry.mark_terminal(execution)

    frozen = registry.freeze("snapshot-1")

    assert frozen[0].identity == execution
    assert registry.status() == {
        "frozen_checkpoint_id": "snapshot-1",
        "live": 1,
        "running": 0,
        "terminal_unreleased": 1,
        "frozen_membership": 1,
    }
    with pytest.raises(RuntimeError, match="dispatch is frozen"):
        registry.register(GymExecutionIdentity(rollout_id="rollout-2", attempt_index=0))

    registry.release(execution)
    registry.unfreeze("snapshot-1")
    registry.unfreeze("snapshot-1")
    assert registry.status()["frozen_checkpoint_id"] is None
    with pytest.raises(RuntimeError, match="already retired"):
        registry.freeze("snapshot-1")


def test_agent_prepare_retries_completed_result_blocker() -> None:
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
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))
    calls = [0]

    async def prepare_control(_method, path, *, server_name, **_kwargs):
        if server_name == "policy":
            return {
                "state": "paused",
                "workers": {"acknowledged": 1, "expected": 1},
                "inflight_total": 0,
                "waiters_total": 0,
            }
        assert path.endswith("/prepare")
        calls[0] += 1
        if calls[0] == 1:
            raise GymControlRequestError(
                "not ready",
                status=409,
                error_code="agent_prepare_incomplete",
            )
        return {
            "state": "preparing",
            "ready_to_commit": True,
            "running": 0,
            "parked": 0,
            "parked_with_boundary": 0,
            "parked_without_boundary": 0,
            "completed_unacknowledged": 0,
            "acknowledged_completed": 0,
            "active": 0,
            "blocking_attempts": [],
            "completed_unacknowledged_attempts": [],
            "selected_boundaries": [],
            "executions": [],
        }

    env._control = AsyncMock(side_effect=prepare_control)
    result = asyncio.run(env.prepare_checkpoint("snapshot-ack", time.time() + 2.0))

    assert result["ready"] is True
    assert calls == [2]


def test_agent_prepare_timeout_resumes_after_completed_result_blocker() -> None:
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
    }

    async def discover_control(_method, _path, *, server_name, **_kwargs):
        return capabilities[server_name]

    env._control = AsyncMock(side_effect=discover_control)
    asyncio.run(env.discover_checkpoint_capabilities(list(capabilities)))
    resume_order: list[str] = []

    async def prepare_control(_method, path, *, server_name, **_kwargs):
        if path.endswith("/pause"):
            return {
                "state": "paused",
                "workers": {"acknowledged": 1, "expected": 1},
                "inflight_total": 0,
                "waiters_total": 0,
            }
        if path.endswith("/prepare"):
            raise GymControlRequestError(
                "completed prompt-group result still awaits acknowledgement",
                status=409,
                error_code="agent_prepare_incomplete",
            )
        resume_order.append(server_name)
        if server_name == "policy":
            return {
                "state": "accepting",
                "workers": {"acknowledged": 1, "expected": 1},
                "released_waiters": 0,
            }
        return {"state": "accepting", "released": 0}

    env._control = AsyncMock(side_effect=prepare_control)

    with pytest.raises(TimeoutError, match="participants were resumed"):
        asyncio.run(env.prepare_checkpoint("snapshot-prompt-group", time.time() + 0.01))

    assert resume_order == ["agent", "policy"]
    assert env._active_gym_checkpoint_id is None
    assert env._gym_execution_registry.status()["frozen_checkpoint_id"] is None
