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

"""Tests for worker shutdown, router un-registration, and recovery.

Each test creates a **fresh** worker so that shutdown / crash is
non-destructive to the rest of the session.
"""

import time

import pytest
import ray
import requests

from helpers import create_worker

pytestmark = pytest.mark.sglang


def _get_worker_count(router):
    """Get the number of workers registered with the router."""
    resp = requests.get(
        f"http://{router['ip']}:{router['port']}/workers", timeout=10
    )
    return len(resp.json().get("workers", []))


def _wait_for_worker_count(router, expected, timeout=15):
    """Poll until the router reports the expected worker count."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _get_worker_count(router) == expected:
            return True
        time.sleep(1)
    return False


# ------------------------------------------------------------------
# shutdown
# ------------------------------------------------------------------
def test_shutdown_worker(ray_cluster, router):
    """Worker shutdown completes without error."""
    worker = create_worker(router, base_gpu_id=0, tp_size=1, rank=0)
    ray.get(worker.shutdown.remote())  # should not raise


def test_shutdown_unregisters_from_router(ray_cluster, router):
    """After shutdown the worker is no longer in the router's list."""
    count_before_create = _get_worker_count(router)
    worker = create_worker(router, base_gpu_id=0, tp_size=1, rank=0)

    # Wait for the worker to appear in the router
    assert _wait_for_worker_count(router, count_before_create + 1), (
        f"Worker never appeared in router (expected {count_before_create + 1}, "
        f"got {_get_worker_count(router)})"
    )

    ray.get(worker.shutdown.remote())

    # Wait for the worker to disappear
    assert _wait_for_worker_count(router, count_before_create), (
        f"Worker still in router after shutdown (expected {count_before_create}, "
        f"got {_get_worker_count(router)})"
    )


def test_new_worker_after_shutdown(ray_cluster, router):
    """A new worker can be created on the same GPU after shutdown."""
    w1 = create_worker(router, base_gpu_id=0, tp_size=1, rank=0)
    ray.get(w1.shutdown.remote())

    w2 = create_worker(router, base_gpu_id=0, tp_size=1, rank=0)
    assert ray.get(w2.health_generate.remote()) is True
    ray.get(w2.shutdown.remote())


def test_simulate_crash(ray_cluster, router):
    """simulate_crash (which calls shutdown) does not raise."""
    worker = create_worker(router, base_gpu_id=0, tp_size=1, rank=0)
    ray.get(worker.simulate_crash.remote())
