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
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.environments.utils import (
    ENV_REGISTRY,
    register_env,
    shutdown_environments,
)


def test_register_new_env_success():
    """Test successfully registering a new environment."""
    # Save original registry state
    original_registry = ENV_REGISTRY.copy()
    try:
        # Register a new environment
        env_name = "test_custom_env"
        actor_class_fqn = "my_custom_module.CustomEnvironmentActor"
        register_env(env_name, actor_class_fqn)
        # Verify the environment is registered
        assert env_name in ENV_REGISTRY
        assert ENV_REGISTRY[env_name]["actor_class_fqn"] == actor_class_fqn
    finally:
        # Restore original registry state
        ENV_REGISTRY.clear()
        ENV_REGISTRY.update(original_registry)


def test_register_env_duplicate_raises_error():
    """Test that registering a duplicate environment name raises ValueError."""
    # Save original registry state
    original_registry = ENV_REGISTRY.copy()
    try:
        # First registration should succeed
        env_name = "test_duplicate_env"
        actor_class_fqn = "my_custom_module.CustomEnvironmentActor"
        register_env(env_name, actor_class_fqn)
        # Second registration with same name should fail
        with pytest.raises(ValueError, match=f"Env name {env_name} already registered"):
            register_env(env_name, "another_module.AnotherActor")
    finally:
        # Restore original registry state
        ENV_REGISTRY.clear()
        ENV_REGISTRY.update(original_registry)


def _fake_env(name):
    env = MagicMock(name=name)
    env.shutdown.remote.return_value = f"{name}-ref"
    return env


def test_shutdown_environments_dedupes_shared_handles():
    """Runners bind the same actor to train and val; it must be stopped once.

    Shutting an already-stopped NeMo-Gym actor down a second time raises and
    would escalate a graceful teardown into a ray.kill.
    """
    env = _fake_env("gym")
    task_to_env = {"nemo_gym": env}
    val_task_to_env = task_to_env  # the aliasing real runners use

    with patch("nemo_rl.environments.utils.ray") as mock_ray:
        shutdown_environments(task_to_env, val_task_to_env)

    env.shutdown.remote.assert_called_once_with()
    mock_ray.get.assert_called_once_with("gym-ref", timeout=10.0)
    mock_ray.kill.assert_not_called()


def test_shutdown_environments_stops_each_distinct_actor():
    train_env, val_env = _fake_env("train"), _fake_env("val")

    with patch("nemo_rl.environments.utils.ray") as mock_ray:
        shutdown_environments({"a": train_env}, {"a": val_env}, None)

    train_env.shutdown.remote.assert_called_once_with()
    val_env.shutdown.remote.assert_called_once_with()
    assert mock_ray.get.call_count == 2
    mock_ray.kill.assert_not_called()


def test_shutdown_environments_kills_only_on_failure():
    healthy, wedged = _fake_env("healthy"), _fake_env("wedged")

    with patch("nemo_rl.environments.utils.ray") as mock_ray:
        mock_ray.get.side_effect = (
            lambda ref, timeout: None
            if ref == "healthy-ref"
            else (_ for _ in ()).throw(TimeoutError("no response"))
        )
        shutdown_environments({"healthy": healthy, "wedged": wedged})

    mock_ray.kill.assert_called_once_with(wedged)


def test_shutdown_environments_tolerates_empty_input():
    with patch("nemo_rl.environments.utils.ray") as mock_ray:
        shutdown_environments(None, {})

    mock_ray.get.assert_not_called()
    mock_ray.kill.assert_not_called()
