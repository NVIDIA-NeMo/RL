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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.async_utils import validation_engines
from nemo_rl.algorithms.async_utils.validation_engines import (
    EngineState,
    ValidationEnginePool,
    split_validation_urls,
)

URLS = [f"http://node:{port}/v1" for port in range(8000, 8004)]


def _make_pool(serve_training_steps):
    generation = SimpleNamespace(
        dp_openai_server_base_urls=list(URLS), set_discard_refits=MagicMock()
    )
    router = SimpleNamespace(set_serving_backends=SimpleNamespace(remote=MagicMock()))
    return ValidationEnginePool(
        generation=generation,
        router=router,
        nemo_gym=object(),
        num_engines=2,
        serve_training_steps=serve_training_steps,
    )


@pytest.fixture(autouse=True)
def _ray_get_passthrough(monkeypatch):
    monkeypatch.setattr(validation_engines.ray, "get", lambda value: value)


@pytest.fixture
def pool():
    return _make_pool(serve_training_steps=0)


def _serving_pushes(pool):
    return [c.args[0] for c in pool.router.set_serving_backends.remote.call_args_list]


def _discard_calls(pool):
    return [c.args for c in pool.generation.set_discard_refits.call_args_list]


def test_split_lends_the_last_engines():
    assert split_validation_urls(URLS, 1) == (URLS[:3], URLS[3:])
    assert split_validation_urls(URLS, 3) == (URLS[:1], URLS[1:])


@pytest.mark.parametrize("num_engines", [0, 4, 5])
def test_split_keeps_at_least_one_training_engine(num_engines):
    with pytest.raises(ValueError, match="between 1 and 3"):
        split_validation_urls(URLS, num_engines)


def test_engines_cycle_from_training_to_validation_and_back(pool):
    assert pool.shard_indices == [2, 3]
    assert pool.serving_training

    pool.lend_to_validation(10)
    assert pool.state is EngineState.VALIDATING
    assert _serving_pushes(pool) == [URLS[:2]]
    assert _discard_calls(pool) == [([2, 3], True)]

    # Refits while the validation runs pass the engines by.
    pool.before_refit(validation_running=True)
    pool.after_refit(11)
    assert pool.state is EngineState.VALIDATING
    assert _discard_calls(pool) == [([2, 3], True)]

    # Once it finishes, the next refit updates them before they rejoin training.
    pool.before_refit(validation_running=False)
    assert pool.state is EngineState.CATCHING_UP
    assert _discard_calls(pool) == [([2, 3], True), ([2, 3], False)]
    assert not pool.serving_training
    assert _serving_pushes(pool) == [URLS[:2]]

    pool.after_refit(12)
    assert pool.state is EngineState.TRAINING
    assert _serving_pushes(pool) == [URLS[:2], URLS]


def test_engines_cannot_be_lent_twice(pool):
    pool.lend_to_validation(10)
    with pytest.raises(RuntimeError, match="validating"):
        pool.lend_to_validation(11)


def test_refits_without_a_validation_change_nothing(pool):
    pool.before_refit(validation_running=False)
    pool.after_refit(1)
    assert pool.state is EngineState.TRAINING
    assert _serving_pushes(pool) == []
    assert _discard_calls(pool) == []


def test_frozen_engines_serve_training_until_too_stale():
    pool = _make_pool(serve_training_steps=2)
    pool.lend_to_validation(10)
    # Frozen, but still in the router.
    assert _discard_calls(pool) == [([2, 3], True)]
    assert _serving_pushes(pool) == []

    pool.before_refit(validation_running=True)
    pool.after_refit(11)
    assert _serving_pushes(pool) == []

    # Two steps past their weights they leave training.
    pool.before_refit(validation_running=True)
    pool.after_refit(12)
    assert _serving_pushes(pool) == [URLS[:2]]
    assert pool.state is EngineState.VALIDATING

    pool.before_refit(validation_running=False)
    pool.after_refit(13)
    assert pool.state is EngineState.TRAINING
    assert _serving_pushes(pool) == [URLS[:2], URLS]


def test_validation_done_while_still_serving_training_keeps_router():
    pool = _make_pool(serve_training_steps=3)
    pool.lend_to_validation(10)
    pool.before_refit(validation_running=False)
    assert pool.state is EngineState.CATCHING_UP
    pool.after_refit(11)
    # They never left the router, and the refit just updated them.
    assert pool.state is EngineState.TRAINING
    assert _serving_pushes(pool) == []
    assert _discard_calls(pool) == [([2, 3], True), ([2, 3], False)]
