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

"""Bind SpeechLM actors to the project's separately locked environments."""

import shlex
from pathlib import Path

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)

POLICY_WORKER = "speechlm_gspo.policy_worker.SpeechLMPolicyWorker"
GENERATION_WORKER = "speechlm_gspo.generation_worker.SpeechLMVllmGenerationWorker"
ASR_ENVIRONMENT = "speechlm_gspo.environment.ASREnvironment"

ENVS_DIR = Path(__file__).resolve().parents[1] / "envs"


def register_actor_environments() -> None:
    """Register project-local actor commands before constructing any workers.

    The policy environment pins Speech, Automodel, Skills and transformers
    independently of the root lock; the ASR reward also needs Skills.
    Generation adds the Speech vLLM plugin to the vLLM stack.
    """
    ACTOR_ENVIRONMENT_REGISTRY[POLICY_WORKER] = _uv_project(ENVS_DIR / "policy")
    ACTOR_ENVIRONMENT_REGISTRY[ASR_ENVIRONMENT] = _uv_project(ENVS_DIR / "policy")
    ACTOR_ENVIRONMENT_REGISTRY[GENERATION_WORKER] = _uv_project(ENVS_DIR / "generation")


def _uv_project(project_dir: Path) -> str:
    return f"uv run --locked --directory {shlex.quote(str(project_dir))}"
