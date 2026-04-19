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

"""Tests for the SGLangGeneration init chain — multi-worker orchestration,
router integration, and the Lock actor.

Instead of instantiating the full SGLangGeneration class (which needs
RayVirtualCluster), we test each component that __init__ wires together:
  • multiple SGLangGenerationWorker actors started in parallel
  • all workers registered with the real router
  • the Lock Ray actor used for rollout_engine_lock
"""

import pytest
import ray
import requests

from nemo_rl.models.generation.sglang.ray_utils import Lock

from helpers import create_worker

pytestmark = pytest.mark.sglang


# ------------------------------------------------------------------
# Multi-worker orchestration
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_workers(ray_cluster, router):
    """Start two TP=1 workers on GPUs 0 and 1."""
    w0 = create_worker(router, base_gpu_id=0, tp_size=1, rank=0)
    w1 = create_worker(router, base_gpu_id=1, tp_size=1, rank=1)
    yield [w0, w1]
    for w in [w0, w1]:
        try:
            ray.get(w.shutdown.remote())
        except Exception:
            pass


def test_multiple_workers_init(two_workers):
    """Two workers start successfully on separate GPUs."""
    for w in two_workers:
        assert ray.get(w.health_generate.remote()) is True


def test_workers_register_with_router(two_workers, router):
    """Both workers appear in the router's /workers list."""
    resp = requests.get(
        f"http://{router['ip']}:{router['port']}/workers", timeout=10
    )
    assert resp.status_code == 200
    workers_list = resp.json().get("workers", [])
    assert len(workers_list) >= 2


def test_workers_have_distinct_urls(two_workers):
    """Each worker reports a unique base URL."""
    urls = [ray.get(w.get_base_url.remote()) for w in two_workers]
    assert len(set(urls)) == 2
    for url in urls:
        assert url.startswith("http://")


# ------------------------------------------------------------------
# Lock actor
# ------------------------------------------------------------------
def test_lock_actor_acquire_release(ray_cluster):
    """Lock.acquire / release round-trip works."""
    lock = Lock.options(num_cpus=0.1, num_gpus=0).remote()
    try:
        assert ray.get(lock.acquire.remote()) is True
        ray.get(lock.release.remote())
    finally:
        ray.kill(lock)


def test_lock_actor_mutual_exclusion(ray_cluster):
    """A second acquire fails while the lock is held."""
    lock = Lock.options(num_cpus=0.1, num_gpus=0).remote()
    try:
        assert ray.get(lock.acquire.remote()) is True
        assert ray.get(lock.acquire.remote()) is False  # already held
        ray.get(lock.release.remote())
        assert ray.get(lock.acquire.remote()) is True  # free again
        ray.get(lock.release.remote())
    finally:
        ray.kill(lock)
