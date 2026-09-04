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
"""Which uv extras each Ray actor needs. The single source of truth.

Two readers:

* ``nemo_rl.distributed.ray_actor_environment_registry`` imports
  ``ACTOR_ENVIRONMENTS`` and turns it into ``py_executable`` strings at runtime.
* ``docker/Dockerfile`` runs this file **as a script** to list the venvs it must
  pre-build, so the image ships one venv per actor.

DO NOT IMPORT ANYTHING FROM ``nemo_rl`` HERE, AND KEEP IT STDLIB-ONLY.
The Dockerfile runs this from the dependency layer, where only this file and
``pyproject.toml``/``uv.lock`` exist -- the rest of the source tree has not been
copied in yet. Running it as a script (rather than importing it) is also what
keeps ``nemo_rl/__init__.py`` from executing there. An import added here breaks
the image build in its most expensive layer.
``tests/unit/distributed/test_actor_environments.py`` enforces this.
"""

# Keeps the annotations below from being evaluated at runtime, so this module also
# runs under interpreters older than the one the image pins.
from __future__ import annotations

# Actor fully-qualified name -> the uv extras its virtual environment needs.
# ``None`` means the actor runs on the driver's interpreter and gets no venv.
# Every extra must exist in ``[project.optional-dependencies]`` of pyproject.toml.
ACTOR_ENVIRONMENTS: dict[str, list[str] | None] = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": ["vllm"],
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": [
        "vllm"
    ],
    "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker": ["sglang"],
    "nemo_rl.models.generation.dynamo.dynamo_worker.DynamoVllmWorker": None,
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": ["fsdp"],
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": [
        "automodel"
    ],
    "nemo_rl.models.value.workers.dtensor_value_worker_v2.DTensorValueWorkerV2": [
        "automodel"
    ],
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": [
        "mcore"
    ],
    "nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker": ["mcore"],
    "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker": [
        "trtllm"
    ],
    "nemo_rl.environments.math_environment.MathEnvironment": None,
    "nemo_rl.environments.math_environment.MathMultiRewardEnvironment": None,
    "nemo_rl.environments.vlm_environment.VLMEnvironment": None,
    "nemo_rl.environments.code_environment.CodeEnvironment": None,
    "nemo_rl.environments.reward_model_environment.RewardModelEnvironment": None,
    "nemo_rl.environments.code_jaccard_environment.CodeJaccardEnvironment": None,
    "nemo_rl.environments.games.sliding_puzzle.SlidingPuzzleEnv": None,
    # AsyncTrajectoryCollector needs the vLLM environment to handle exceptions
    # from VllmGenerationWorker.
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector": ["vllm"],
    # ReplayBuffer needs the vLLM environment to handle trajectory data from
    # VllmGenerationWorker.
    "nemo_rl.algorithms.async_utils.ReplayBuffer": ["vllm"],
    # SyncRolloutActor doesn't import vllm directly -- policy_generation is a Ray
    # actor handle. The vLLM env is needed because (1) transfer_queue is bundled
    # into the vLLM venv (and the policy training venvs), and the actor writes
    # flattened tensors to TQ via dp_client.put_samples; (2) same-node colocation
    # with VllmGenerationWorker avoids duplicate venv caches.
    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": ["vllm"],
    "nemo_rl.environments.tools.retriever.RAGEnvironment": None,
    "nemo_rl.environments.nemo_gym.NemoGym": ["nemo_gym"],
    # ModelOpt quantization-aware workers
    "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker": [
        "modelopt",
        "vllm",
    ],
    "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker": [
        "modelopt",
        "vllm",
    ],
    "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker.DTensorQuantPolicyWorker": [
        "modelopt",
        "automodel",
    ],
    "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker_v2.DTensorQuantPolicyWorkerV2": [
        "modelopt",
        "automodel",
    ],
    "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker.MegatronQuantPolicyWorker": [
        "modelopt",
        "mcore",
    ],
}


def _build_stage(extras: list[str]) -> str:
    """Which image layer can finish this venv.

    The tensorrt_llm wheel is only built in the TRT-LLM layer, so those venvs get
    their third-party packages in two steps; everything else finishes in the
    dependency layer.
    """
    return "trtllm" if "trtllm" in extras else "deps"


def main(argv: list[str]) -> int:
    r"""Print "<actor FQN>\t<stage>\t<uv sync extra flags>" for uv-managed actors.

    Usage: actor_environments.py [<stage>] [<skip extra> ...]

    ``<stage>`` is "deps" or "trtllm" (omit for all). Any extras listed after it
    are skipped, which is how the Dockerfile honors SKIP_VLLM_BUILD and friends.
    Filtering on the declared extras -- rather than on a substring of the actor
    name -- is what makes SKIP_VLLM_BUILD also skip actors like
    AsyncTrajectoryCollector, whose name contains no "vllm".
    """
    stage = argv[1] if len(argv) > 1 and argv[1] != "all" else None
    skip = set(argv[2:])
    for actor_fqn, extras in sorted(ACTOR_ENVIRONMENTS.items()):
        if extras is None or skip & set(extras):
            continue
        actor_stage = _build_stage(extras)
        if stage is not None and actor_stage != stage:
            continue
        flags = " ".join(f"--extra {extra}" for extra in extras)
        print(actor_fqn, actor_stage, flags, sep="\t")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
