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
from typing import Any, Dict, NotRequired, TypedDict

from hydra.utils import get_object

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.environments.interfaces import EnvironmentInterface


# Environment registry entry schema.
class EnvRegistryEntry(TypedDict, total=False):
    actor_class_fqn: str
    default_processor: NotRequired[str]


# Environment registry. Key is the env name, value is a dictionary with the actor class FQN and optional default processor.
ENV_REGISTRY: Dict[str, EnvRegistryEntry] = {
    "math_default": {
        "actor_class_fqn": "nemo_rl.environments.math_environment.MathEnvironment",
    },
    "math": {
        "actor_class_fqn": "nemo_rl.environments.math_environment.MathEnvironment",
    },
    "code": {
        "actor_class_fqn": "nemo_rl.environments.code_environment.CodeEnvironment",
    },
    "reward_model": {
        "actor_class_fqn": "nemo_rl.environments.reward_model_environment.RewardModelEnvironment",
    },
    "code_jaccard": {
        "actor_class_fqn": "nemo_rl.environments.code_jaccard_environment.CodeJaccardEnvironment",
    },
    "vlm": {
        "actor_class_fqn": "nemo_rl.environments.vlm_environment.VLMEnvironment",
    },
    "nemo_gym": {
        "actor_class_fqn": "nemo_rl.environments.nemo_gym.NemoGym",
    },
}


def chunk_list_to_workers(to_chunk: list[Any], num_workers: int) -> list[list[Any]]:
    """Chunk a list into a list of lists, where each sublist is assigned to a worker. Keeps ordering of elements.

    If the list is not divisible by the number of workers, the last worker may have fewer elements.
    If there are more workers than elements, the first len(list) workers will have a single element each,
    and the remaining workers will have empty lists.

    Args:
        list: The list to be chunked.
        num_workers: The number of workers to distribute the list to.

    Returns:
        A list of lists, where each sublist contains elements assigned to a worker.

    Examples:
    ```{doctest}
    >>> from nemo_rl.environments.utils import chunk_list_to_workers
    >>> chunk_list_to_workers([1, 2, 3, 4, 5], 3)
    [[1, 2], [3, 4], [5]]
    ```
    """
    if not to_chunk:
        return [[] for _ in range(num_workers)]

    # Handle case where we have more workers than elements
    if len(to_chunk) <= num_workers:
        result = [[item] for item in to_chunk]
        # Add empty lists for remaining workers
        result.extend([[] for _ in range(num_workers - len(to_chunk))])
        return result

    # Calculate chunk size (ceiling division to ensure all elements are covered)
    chunk_size = (len(to_chunk) + num_workers - 1) // num_workers

    # Create chunks
    chunks = []
    for i in range(0, len(to_chunk), chunk_size):
        chunks.append(to_chunk[i : i + chunk_size])

    # If we somehow ended up with more chunks than workers (shouldn't happen with ceiling division)
    # merge the last chunks
    if len(chunks) > num_workers:
        chunks[num_workers - 1 :] = [sum(chunks[num_workers - 1 :], [])]

    return chunks


def create_env(env_name: str, env_config: dict) -> EnvironmentInterface:
    assert env_name in ENV_REGISTRY, (
        f"Env name {env_name} is not registered in ENV_REGISTRY. Please call register_env() to register the environment."
    )
    actor_class_fqn = ENV_REGISTRY[env_name]["actor_class_fqn"]
    actor_class = get_object(actor_class_fqn)
    env = actor_class.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(actor_class_fqn),
            "env_vars": dict(os.environ),
        }
    ).remote(env_config)
    return env


def register_env(env_name: str, actor_class_fqn: str) -> None:
    if env_name in ENV_REGISTRY:
        raise ValueError(f"Env name {env_name} already registered")

    ENV_REGISTRY[env_name] = {"actor_class_fqn": actor_class_fqn}
