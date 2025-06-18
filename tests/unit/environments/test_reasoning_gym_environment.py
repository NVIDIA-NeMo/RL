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
import time

import pytest
import ray

from nemo_rl.environments.reasoning_gym_environment import ReasoningGymEnvironment


@pytest.fixture
def rg_env(monkeypatch):
    """Patch scorer and create a ReasoningGymEnvironment actor for testing."""

    # Dummy scorer: exact-match reward
    def dummy_get_score_answer_fn(dataset_name):
        def _score_fn(pred_response: str, entry):
            return 1.0 if pred_response.strip() == entry["answer"].strip() else 0.0

        return _score_fn

    import nemo_rl.environments.reasoning_gym_environment as rg_mod

    monkeypatch.setattr(rg_mod, "get_score_answer_fn", dummy_get_score_answer_fn, False)

    env = ReasoningGymEnvironment.options(
        runtime_env={
            "py_executable": ReasoningGymEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),
        }
    ).remote({"num_workers": 1})

    yield env

    # Clean up
    env.shutdown.remote()
    ray.kill(env)
    time.sleep(0.1)


@pytest.fixture
def sample_data():
    """Return a simple correct prompt-response pair plus metadata."""
    message_log_batch = [
        [
            {
                "role": "user",
                "content": "Spell this word backward: idealess",
            },
            {
                "role": "assistant",
                "content": "sselaedi",  # correct answer
            },
        ]
    ]
    metadata = [
        {
            "source_dataset": "spell_backward",
            "ground_truth": "sselaedi",
        }
    ]
    return {"message_log_batch": message_log_batch, "metadata": metadata}


def test_reasoning_gym_correct(rg_env, sample_data):
    """Assistant gives correct answer → reward 1.0 and `Environment: correct`."""
    result = ray.get(rg_env.step.remote(**sample_data))

    assert result.rewards.shape == (1,)
    assert result.rewards[0] == 1.0
    assert result.observations[0]["content"] == "Environment: correct"


def test_reasoning_gym_incorrect(rg_env, sample_data):
    """Incorrect answer should yield zero reward and `Environment: incorrect`."""
    wrong_batch = sample_data.copy()
    wrong_batch["message_log_batch"] = [
        [
            {
                "role": "user",
                "content": "Spell this word backward: idealess",
            },
            {
                "role": "assistant",
                "content": "not_the_answer",
            },
        ]
    ]
    result = ray.get(rg_env.step.remote(**wrong_batch))

    assert result.rewards[0] == 0.0
    assert result.observations[0]["content"] == "Environment: incorrect"
