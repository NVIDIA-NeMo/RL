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

import time

import pytest
import ray

from nemo_rl.environments.code_jaccard_environment import CodeJaccardEnvConfig
from nemo_rl.environments.utils import create_env


@pytest.fixture(scope="module")
def code_jaccard_env_config():
    return CodeJaccardEnvConfig(num_workers=2)


@pytest.fixture(scope="module")
def code_jaccard_env(code_jaccard_env_config):
    env = create_env("code_jaccard", code_jaccard_env_config)
    yield env
    env.shutdown.remote()
    ray.kill(env)
    time.sleep(0.1)


def test_code_jaccard_basic_alignment(code_jaccard_env):
    """Identical responses should be aligned with score 1.0."""
    message_log_batch = [
        [
            {"role": "user", "content": "Return the string 'hello'"},
            {"role": "assistant", "content": "print('hello')"},
        ],
        [
            {"role": "user", "content": "Add two numbers"},
            {"role": "assistant", "content": "a + b"},
        ],
    ]
    metadata = [
        {"ground_truth": "print('hello')"},
        {"ground_truth": "a + b"},
    ]

    result = ray.get(code_jaccard_env.step.remote(message_log_batch, metadata))

    # Observations and metadata lengths match
    assert len(result.observations) == 2
    assert len(result.metadata) == 2
    # Aligned messages should be labeled aligned
    assert all(
        obs["content"].startswith("Environment: jaccard aligned")
        for obs in result.observations
    )
    # Rewards are in [0,1] and close to 1 for identical strings
    assert result.rewards.shape == (2,)
    assert float(result.rewards[0]) == pytest.approx(1.0, rel=0, abs=1e-6)
    assert float(result.rewards[1]) == pytest.approx(1.0, rel=0, abs=1e-6)
    # Terminated flags set
    assert result.terminateds.shape == (2,)
    assert all(result.terminateds == 1.0)


def test_code_jaccard_misalignment(code_jaccard_env):
    """Different responses should be misaligned with score below threshold."""
    message_log_batch = [
        [
            {"role": "user", "content": "Return the string 'hello'"},
            {"role": "assistant", "content": "goodbye world"},
        ],
    ]
    metadata = [{"ground_truth": "print('hello')"}]

    result = ray.get(code_jaccard_env.step.remote(message_log_batch, metadata))

    assert len(result.observations) == 1
    assert result.observations[0]["content"].startswith(
        "Environment: jaccard misaligned"
    )
    # Reward should be between 0 and 1, and reasonably low for disjoint tokens
    score = float(result.rewards[0])
    assert 0.0 <= score <= 0.5
    assert result.terminateds.shape == (1,)
    assert result.terminateds[0] == 1.0


def test_code_jaccard_answers_return(code_jaccard_env):
    """When return_extracted_answer=True, answers should be returned as assistant text."""
    message_log_batch = [
        [
            {"role": "user", "content": "Compute"},
            {"role": "assistant", "content": "x = a + b"},
        ],
        [
            {"role": "user", "content": "Explain"},
            {"role": "assistant", "content": "def add(a, b): return a + b"},
        ],
    ]
    metadata = [{"ground_truth": "x = a + b"}, {"ground_truth": "return a + b"}]

    result = ray.get(
        code_jaccard_env.step.remote(
            message_log_batch, metadata, return_extracted_answer=True
        )
    )

    assert result.answers is not None
    assert result.answers == ["x = a + b", "def add(a, b): return a + b"]


def test_code_jaccard_empty_input(code_jaccard_env):
    """Empty batches should return empty outputs."""
    result = ray.get(code_jaccard_env.step.remote([], []))
    assert len(result.observations) == 0
    assert len(result.metadata) == 0
    assert result.rewards.shape == (0,)
    assert result.terminateds.shape == (0,)
