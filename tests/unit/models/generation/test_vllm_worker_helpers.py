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

"""Tests for vLLM worker helper functions."""

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.worker_utils import (
    configure_refit_runtime,
    refit_cache_loader_routes_enabled,
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
)


@pytest.mark.parametrize("enabled", [False, True])
def test_refit_loader_cache_round_trips_through_additional_config(enabled):
    vllm_kwargs = {"additional_config": {"existing": "value"}}

    configure_refit_runtime(
        {"refit_cache_loader_routes": enabled},
        vllm_kwargs,
    )

    assert vllm_kwargs["additional_config"]["existing"] == "value"
    vllm_config = SimpleNamespace(additional_config=vllm_kwargs["additional_config"])
    assert refit_cache_loader_routes_enabled(vllm_config) is enabled


def test_refit_loader_cache_defaults_to_disabled():
    vllm_kwargs = {}

    configure_refit_runtime({}, vllm_kwargs)

    vllm_config = SimpleNamespace(additional_config=vllm_kwargs["additional_config"])
    assert refit_cache_loader_routes_enabled(vllm_config) is False


@pytest.mark.parametrize(
    ("tp", "pp", "ep", "expected"),
    [
        (2, 1, 2, "ray"),
        (1, 2, 2, "ray"),
        (1, 1, 8, "uni"),
        (1, 1, 1, None),
    ],
)
def test_resolve_distributed_executor_backend(tp, pp, ep, expected):
    assert resolve_distributed_executor_backend(tp, pp, ep) == expected


@pytest.mark.parametrize(
    ("rank", "model_parallel_size", "executor_backend", "expected"),
    [
        (7, 1, "uni", 0),
        (6, 2, "ray", 3),
    ],
)
def test_resolve_data_parallel_local_rank(
    rank, model_parallel_size, executor_backend, expected
):
    assert (
        resolve_data_parallel_local_rank(rank, model_parallel_size, executor_backend)
        == expected
    )
