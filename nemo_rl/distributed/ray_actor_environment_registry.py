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

import os

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

USE_SYSTEM_EXECUTABLE = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"
VLLM_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.VLLM
)
SGLANG_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.SGLANG
)
MCORE_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.MCORE
)

ACTOR_ENVIRONMENT_REGISTRY: dict[str, str] = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": VLLM_EXECUTABLE,
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": VLLM_EXECUTABLE,
    "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker": SGLANG_EXECUTABLE,
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": PY_EXECUTABLES.FSDP,
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": PY_EXECUTABLES.AUTOMODEL,
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": MCORE_EXECUTABLE,
    "nemo_rl.environments.math_environment.MathEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.vlm_environment.VLMEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.code_environment.CodeEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.reward_model_environment.RewardModelEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.code_jaccard_environment.CodeJaccardEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.games.sliding_puzzle.SlidingPuzzleEnv": PY_EXECUTABLES.SYSTEM,
    # AsyncTrajectoryCollector needs vLLM environment to handle exceptions from VllmGenerationWorker
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector": PY_EXECUTABLES.VLLM,
    # ReplayBuffer needs vLLM environment to handle trajectory data from VllmGenerationWorker
    "nemo_rl.algorithms.async_utils.ReplayBuffer": PY_EXECUTABLES.VLLM,
    "nemo_rl.environments.tools.retriever.RAGEnvironment": PY_EXECUTABLES.SYSTEM,
    "nemo_rl.environments.nemo_gym.NemoGym": PY_EXECUTABLES.NEMO_GYM,
}


def get_actor_python_env(actor_class_fqn: str) -> str:
    if actor_class_fqn in ACTOR_ENVIRONMENT_REGISTRY:
        return ACTOR_ENVIRONMENT_REGISTRY[actor_class_fqn]
    else:
        raise ValueError(
            f"No actor environment registered for {actor_class_fqn}"
            f"You're attempting to create an actor ({actor_class_fqn})"
            "without specifying a python environment for it. Please either"
            "specify a python environment in the registry "
            "(nemo_rl.distributed.ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY) "
            "or pass a py_executable to the RayWorkerBuilder. If you're unsure about which "
            "environment to use, a good default is PY_EXECUTABLES.SYSTEM for ray actors that "
            "don't have special dependencies. If you do have special dependencies (say, you're "
            "adding a new generation framework or training backend), you'll need to specify the "
            "appropriate environment. See uv.md for more details."
        )
