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

import shlex
from pathlib import Path

from flow_grpo.actor_environments import (
    POLICY_WORKER,
    REWARD_WORKER,
    register_actor_environments,
)
from flow_grpo.models.policy.workers.flow_grpo_worker import FlowGRPOPolicyWorker

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES


def test_policy_worker_accepts_builder_topology():
    # RayWorkerBuilder passes all three keywords when creating a policy actor.
    resources, _, _, _ = FlowGRPOPolicyWorker.configure_worker(
        num_gpus=1, bundle_indices=(0, [0]), num_gpus_per_node=8
    )
    assert resources["num_gpus"] == 1


def test_actors_select_research_project(monkeypatch):
    monkeypatch.delenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", raising=False)
    for worker in (POLICY_WORKER, REWARD_WORKER):
        monkeypatch.setitem(ACTOR_ENVIRONMENT_REGISTRY, worker, "previous")
    register_actor_environments()
    project = Path(__file__).resolve().parents[2]
    for worker in (POLICY_WORKER, REWARD_WORKER):
        assert shlex.split(ACTOR_ENVIRONMENT_REGISTRY[worker]) == [
            "uv",
            "run",
            "--locked",
            "--directory",
            str(project),
        ]


def test_actors_respect_system_interpreter(monkeypatch):
    monkeypatch.setenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")
    for worker in (POLICY_WORKER, REWARD_WORKER):
        monkeypatch.setitem(ACTOR_ENVIRONMENT_REGISTRY, worker, "previous")
    register_actor_environments()
    for worker in (POLICY_WORKER, REWARD_WORKER):
        assert ACTOR_ENVIRONMENT_REGISTRY[worker] == PY_EXECUTABLES.SYSTEM
