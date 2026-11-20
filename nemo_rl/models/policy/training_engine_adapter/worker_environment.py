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

"""Worker class of the TrainingEngine policy path and the interpreter it runs under."""

import os

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.utils.venvs import DEFAULT_VENV_DIR

TRAINING_ENGINE_WORKER_FQN = "nemo_rl.models.policy.workers.training_engine_policy_worker.TrainingEnginePolicyWorker"
_MEGATRON_WORKER_FQN = (
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
)


def register_worker_environment() -> None:
    """[WAR] Workaround: run the engine worker with the Megatron worker's interpreter.

    The standard path is an entry in ACTOR_ENVIRONMENTS, which builds an environment
    for the worker from the lock file. The lock file pins a Transformer Engine release
    the engine recipes cannot use, so the worker borrows the Megatron worker's
    environment instead. That environment must already exist in the image and hold a
    Transformer Engine the recipe supports.
    TODO: once the lock file carries such a Transformer Engine, add the worker to
    ACTOR_ENVIRONMENTS and delete this function. See ENGINE_TODOS.md.
    """
    venv_dir = os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    ACTOR_ENVIRONMENT_REGISTRY.setdefault(
        TRAINING_ENGINE_WORKER_FQN,
        os.path.join(venv_dir, _MEGATRON_WORKER_FQN, "bin", "python"),
    )
