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
"""Tier 2 (functional) fixtures — Ray + transfer_queue, single-node, no GPU."""

from __future__ import annotations

import os
import uuid

import pytest


@pytest.fixture
def ray_namespace() -> str:
    """Per-test Ray namespace so xdist-style parallel runs don't collide."""
    return f"dp-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def ray_session(ray_namespace):
    """Init Ray with a unique namespace; tear down after the test."""
    pytest.importorskip("ray")
    pytest.importorskip("transfer_queue")
    import ray

    if ray.is_initialized():
        ray.shutdown()
    ray.init(namespace=ray_namespace, include_dashboard=False, log_to_driver=False)
    try:
        yield ray_namespace
    finally:
        if ray.is_initialized():
            ray.shutdown()


@pytest.fixture
def tq_simple_cfg():
    """Minimal SimpleStorage config for TQ functional tests."""
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "simple",
        "storage_capacity": 1024,
        "num_storage_units": 1,
    }


def pytest_collection_modifyitems(config, items):
    """If transfer_queue isn't installed, mark all tests in this dir
    as skipped with a clear reason — no silent skip."""
    try:
        import transfer_queue  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(
            reason="transfer_queue not installed (it's a base dep — "
            "try `uv sync` to refresh)"
        )
        for item in items:
            item.add_marker(skip)
