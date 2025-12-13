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
import pytest

from nemo_rl.environments.utils import ENV_REGISTRY, register_env


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
