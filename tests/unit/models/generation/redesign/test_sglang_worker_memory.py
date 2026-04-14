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

"""Tests for SGLangGenerationWorker memory management:
flush_cache, release_memory_occupation, resume_memory_occupation.

Uses a real SGLang server (Qwen3-0.6B), parametrised over two
configurations so the same tests exercise both a single-rank TP=1
worker and a TP=2 worker:

  • tp1 — 1 worker × TP=1
  • tp2_2workers — 2 workers × TP=2 (the memory tests target worker 0,
    but both workers share the router)

Each test is fully self-contained — it leaves the server in the same
state it found it.
"""

import pytest
import ray

from helpers import create_worker, post_and_assert_200

pytestmark = pytest.mark.sglang


@pytest.fixture(
    scope="module",
    params=[
        pytest.param({"tp_size": 1, "num_workers": 1}, id="tp1"),
        pytest.param({"tp_size": 2, "num_workers": 2}, id="tp2_2workers"),
    ],
)
def worker(request, ray_cluster, router):
    """Worker(s) dedicated to memory tests.

    For ``tp1`` a single TP=1 worker is created.  For ``tp2_2workers``
    two TP=2 workers share the same router (mirroring the 2-servers
    configuration exercised elsewhere); memory tests run against the
    first worker but the second is kept alive so the router has the
    multi-worker topology in place.
    """
    tp_size = request.param["tp_size"]
    num_workers = request.param["num_workers"]

    workers = []
    for rank in range(num_workers):
        workers.append(
            create_worker(
                router,
                base_gpu_id=rank * tp_size,
                tp_size=tp_size,
                rank=rank,
            )
        )

    yield workers[0]

    for w in workers:
        try:
            ray.get(w.shutdown.remote())
        except Exception:
            pass


# ------------------------------------------------------------------
# flush_cache
# ------------------------------------------------------------------
def test_flush_cache_success(worker):
    """flush_cache returns without error on a healthy server."""
    ray.get(worker.flush_cache.remote())


# ------------------------------------------------------------------
# release / resume — weights (self-contained)
# ------------------------------------------------------------------
def test_release_and_resume_memory_weights(worker):
    """release_memory_weights followed by resume succeeds."""
    ray.get(worker.release_memory_weights.remote())
    ray.get(worker.resume_memory_weights.remote())


# ------------------------------------------------------------------
# release / resume — KV cache + CUDA graphs (self-contained)
# ------------------------------------------------------------------
def test_release_and_resume_memory_kv_cache_and_cuda_graph(worker):
    """release then resume KV cache + CUDA graphs succeeds."""
    ray.get(worker.release_memory_kv_cache_and_cuda_graph.remote())
    ray.get(worker.resume_memory_kv_cache_and_cuda_graph.remote())


# ------------------------------------------------------------------
# full offload / onload cycle
# ------------------------------------------------------------------
def test_full_offload_onload_cycle(worker):
    """Full offload (weights then KV) then onload (weights then KV) works."""
    ray.get(worker.release_memory_weights.remote())
    ray.get(worker.release_memory_kv_cache_and_cuda_graph.remote())
    ray.get(worker.resume_memory_weights.remote())
    ray.get(worker.resume_memory_kv_cache_and_cuda_graph.remote())


def test_health_after_memory_cycle(worker):
    """health_generate passes after a full offload / onload cycle."""
    ray.get(worker.release_memory_weights.remote())
    ray.get(worker.release_memory_kv_cache_and_cuda_graph.remote())
    ray.get(worker.resume_memory_weights.remote())
    ray.get(worker.resume_memory_kv_cache_and_cuda_graph.remote())
    assert ray.get(worker.health_generate.remote()) is True


def test_flush_cache_after_resume(worker):
    """flush_cache succeeds after a release → resume round-trip."""
    ray.get(worker.release_memory_weights.remote())
    ray.get(worker.resume_memory_weights.remote())
    ray.get(worker.flush_cache.remote())


# ------------------------------------------------------------------
# Equivalent tests using _make_request directly — verify HTTP 200
# ------------------------------------------------------------------
def test_offload_onload_via_http_200(worker):
    """Full offload/onload cycle driven by direct HTTP POST, asserting 200.

    Uses ``post_and_assert_200`` (which checks ``resp.status_code == 200``
    explicitly) rather than ``_make_request`` — ``_make_request`` throws the
    status code away inside ``raise_for_status()`` so callers cannot inspect it.
    """
    base_url = ray.get(worker.get_base_url.remote())
    assert base_url is not None

    # Release weights (flush_cache first, mirroring release_memory_occupation)
    ray.get(worker.flush_cache.remote())
    post_and_assert_200(
        base_url, "release_memory_occupation", {"tags": ["weights"]}
    )
    # Release KV cache + CUDA graphs
    ray.get(worker.flush_cache.remote())
    post_and_assert_200(
        base_url, "release_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]}
    )
    # Resume weights
    post_and_assert_200(
        base_url, "resume_memory_occupation", {"tags": ["weights"]}
    )
    # Resume KV cache + CUDA graphs
    post_and_assert_200(
        base_url, "resume_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]}
    )
    assert ray.get(worker.health_generate.remote()) is True


def test_double_offload_onload_cycle(worker):
    """Two back-to-back full offload/onload cycles via direct HTTP, asserting 200 on every call.

    Exercises the same endpoints twice to catch state leaks across cycles.
    Uses ``post_and_assert_200`` so each of the eight POSTs explicitly
    verifies ``resp.status_code == 200``.
    """
    base_url = ray.get(worker.get_base_url.remote())
    assert base_url is not None

    for _ in range(2):
        ray.get(worker.flush_cache.remote())
        post_and_assert_200(
            base_url, "release_memory_occupation", {"tags": ["weights"]}
        )
        ray.get(worker.flush_cache.remote())
        post_and_assert_200(
            base_url,
            "release_memory_occupation",
            {"tags": ["kv_cache", "cuda_graph"]},
        )
        post_and_assert_200(
            base_url, "resume_memory_occupation", {"tags": ["weights"]}
        )
        post_and_assert_200(
            base_url,
            "resume_memory_occupation",
            {"tags": ["kv_cache", "cuda_graph"]},
        )
    assert ray.get(worker.health_generate.remote()) is True
