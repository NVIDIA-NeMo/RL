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

import importlib

import nemo_rl.distributed.ray_actor_environment_registry as registry

DEFAULT_EXECUTABLES = {
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": registry.PY_EXECUTABLES.FSDP,
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": registry.PY_EXECUTABLES.AUTOMODEL,
    "nemo_rl.models.value.workers.dtensor_value_worker_v2.DTensorValueWorkerV2": registry.PY_EXECUTABLES.AUTOMODEL,
    "nemo_rl.environments.nemo_gym.NemoGym": registry.PY_EXECUTABLES.NEMO_GYM,
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector": registry.PY_EXECUTABLES.VLLM,
    "nemo_rl.algorithms.async_utils.ReplayBuffer": registry.PY_EXECUTABLES.VLLM,
    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": registry.PY_EXECUTABLES.VLLM,
}


def _restore_actor_environment_registry(original_registry: dict[str, str]) -> None:
    restored_registry = importlib.reload(registry).ACTOR_ENVIRONMENT_REGISTRY
    original_registry.clear()
    original_registry.update(restored_registry)
    registry.ACTOR_ENVIRONMENT_REGISTRY = original_registry


def test_actor_environments_use_default_executables(monkeypatch):
    original_registry = registry.ACTOR_ENVIRONMENT_REGISTRY

    try:
        with monkeypatch.context() as context:
            context.delenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", raising=False)
            importlib.reload(registry)

            assert {
                actor: registry.get_actor_python_env(actor)
                for actor in DEFAULT_EXECUTABLES
            } == DEFAULT_EXECUTABLES
    finally:
        _restore_actor_environment_registry(original_registry)


def test_actor_environments_use_system_executable(monkeypatch):
    original_registry = registry.ACTOR_ENVIRONMENT_REGISTRY

    try:
        with monkeypatch.context() as context:
            context.setenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")
            importlib.reload(registry)

            assert {
                actor: registry.get_actor_python_env(actor)
                for actor in DEFAULT_EXECUTABLES
            } == dict.fromkeys(DEFAULT_EXECUTABLES, registry.PY_EXECUTABLES.SYSTEM)
    finally:
        _restore_actor_environment_registry(original_registry)
