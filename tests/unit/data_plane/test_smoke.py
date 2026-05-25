# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Tier-0 smoke tests — pre-commit gates.

Cheapest tier: catches drift in module paths, registry keys, and the
public ABC surface. Each test runs in milliseconds and never touches
real Ray / vLLM / TQ.
"""

from __future__ import annotations

import inspect


def test_sync_utils_module_imports() -> None:
    """Catches FQN drift after the algorithms.sync_utils consolidation."""
    from nemo_rl.experience.sync_rollout_actor import (
        SyncRolloutActor,
        kv_first_write,
    )

    # ``SyncRolloutActor`` is wrapped by ``@ray.remote`` into
    # ``ActorClass(SyncRolloutActor)`` — the wrapper has no
    # ``__name__`` attribute. Check via ``repr`` instead.
    assert "SyncRolloutActor" in repr(SyncRolloutActor)
    assert callable(kv_first_write)


def test_sync_rollout_actor_registered_under_vllm_tier() -> None:
    """Multinode runs depend on this — without it, tensordict missing on
    worker nodes (real bug seen in job 11614968)."""
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    fqn = "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
    env = get_actor_python_env(fqn)
    # Same tier as vLLM workers / AsyncTrajectoryCollector / ReplayBuffer.
    # Allow either the resolved exec path or the SYSTEM-override sentinel.
    assert env in (PY_EXECUTABLES.VLLM, PY_EXECUTABLES.SYSTEM), (
        f"unexpected env tier for {fqn}: {env!r}"
    )


def test_kvbatchmeta_schema_unchanged() -> None:
    """Schema break check — KVBatchMeta is the cross-process boundary;
    adding/removing a field silently would break adapters that pickle it."""
    from nemo_rl.data_plane.interfaces import KVBatchMeta

    expected_fields = {
        "partition_id",
        "task_name",
        "sample_ids",
        "fields",
        "sequence_lengths",
        "extra_info",
        "tags",
    }
    actual_fields = {f.name for f in KVBatchMeta.__dataclass_fields__.values()}
    assert actual_fields == expected_fields, (
        f"KVBatchMeta schema drifted. expected={expected_fields}, "
        f"actual={actual_fields}"
    )


def test_dataplane_client_abc_surface() -> None:
    """Catches accidental ABC method removal / rename — e.g. dropping
    ``clear_samples`` would break step-end teardown silently."""
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    expected_methods = {
        # task-mediated
        "register_partition",
        "claim_meta",
        "get_data",
        "check_consumption_status",
        # direct-by-key
        "put_samples",
        "get_samples",
        "clear_samples",
        # lifecycle
        "close",
    }
    actual_methods = {
        name
        for name, member in inspect.getmembers(DataPlaneClient, callable)
        if not name.startswith("_") and getattr(member, "__isabstractmethod__", False)
    }
    assert expected_methods.issubset(actual_methods), (
        f"DataPlaneClient ABC missing methods: {expected_methods - actual_methods}"
    )


def test_async_and_sync_actors_share_env_tier() -> None:
    """Sync should mirror async's env tier — both drive vLLM and write
    tensordict to TQ, so they need the same VLLM venv."""
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )

    sync_env = get_actor_python_env(
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
    )
    async_env = get_actor_python_env(
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
    )
    assert sync_env == async_env, (
        f"Sync vs async env tier drift: sync={sync_env!r}, async={async_env!r}"
    )
