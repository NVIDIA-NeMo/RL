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

import pytest

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES


def test_system_executable_override_applies_to_every_registered_actor(monkeypatch):
    actor_fqn = next(
        name
        for name, executable in ACTOR_ENVIRONMENT_REGISTRY.items()
        if executable != PY_EXECUTABLES.SYSTEM
    )
    monkeypatch.setenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")

    assert get_actor_python_env(actor_fqn) == PY_EXECUTABLES.SYSTEM


def test_registered_executable_is_used_without_override(monkeypatch):
    actor_fqn = next(
        name
        for name, executable in ACTOR_ENVIRONMENT_REGISTRY.items()
        if executable != PY_EXECUTABLES.SYSTEM
    )
    monkeypatch.delenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", raising=False)

    assert get_actor_python_env(actor_fqn) == ACTOR_ENVIRONMENT_REGISTRY[actor_fqn]


def test_system_executable_override_does_not_hide_unknown_actor(monkeypatch):
    monkeypatch.setenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")

    with pytest.raises(ValueError, match="No actor environment registered"):
        get_actor_python_env("example.UnknownActor")
