import os
import time

import pytest
import ray

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.environments.simulated_user.unique_numbers import (
    UniqueNumbersEnv,
    UniqueNumbersMetadata,
)


@pytest.fixture(scope="module")
def unique_env():
    env = UniqueNumbersEnv.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.simulated_user.unique_numbers.UniqueNumbersEnv"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(cfg={"max_turns": 5, "min_length": 3, "max_length": 3})
    yield env
    env.shutdown.remote()
    ray.kill(env)
    time.sleep(0.1)


def test_query_and_reward(unique_env):
    metadata = UniqueNumbersMetadata(numbers=[1, 2, 1], unique_count=2, turn=0, max_turns=5)
    query_log = [[{"role": "assistant", "content": "what is number 2?"}]]
    result = ray.get(unique_env.step.remote(query_log, [metadata]))

    assert result.observations[0]["content"] == "2"
    assert result.rewards[0] == 0.0
    assert result.terminateds[0] is False

    guess_meta = UniqueNumbersMetadata(numbers=[1, 2, 1], unique_count=2, turn=3, max_turns=5)
    guess_log = [[{"role": "assistant", "content": "there are 2 unique numbers"}]]
    guess_result = ray.get(unique_env.step.remote(guess_log, [guess_meta]))
    assert guess_result.terminateds[0] is True
    assert guess_result.rewards[0] == 1.0
