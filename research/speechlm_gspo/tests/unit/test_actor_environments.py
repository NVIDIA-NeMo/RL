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

from unittest import mock

import pytest
from speechlm_gspo import actor_environments as envs

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    get_actor_python_env,
)


@pytest.mark.parametrize(
    "actor, env",
    [
        (envs.POLICY_WORKER, "policy"),
        (envs.ASR_ENVIRONMENT, "policy"),
        (envs.GENERATION_WORKER, "generation"),
    ],
)
def test_actors_use_project_locked_environments(actor, env):
    project = envs.ENVS_DIR / env
    with mock.patch.dict(ACTOR_ENVIRONMENT_REGISTRY):
        envs.register_actor_environments()
        assert get_actor_python_env(actor) == f"uv run --locked --directory {project}"
    assert (project / "pyproject.toml").is_file()
    assert (project / "uv.lock").is_file()
