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

"""Tests for SGLangGenerationWorker.init — server launch and router registration.

Uses a real Ray cluster, a real sglang router, and a real SGLang server
(Qwen3-0.6B, TP=1).  A module-scoped worker is shared across all tests
in this file.
"""

import pytest
import ray
import requests

from .helpers import create_worker

pytestmark = [
    pytest.mark.sglang,
    # Temporarily skipped: starts a real SGLang server, which fails during CUDA
    # graph capture ("CuTe Experimental module is only supported on Cuda toolkit
    # 13.1 and above!"). Same environment issue as PR #2881 — reproduces on main,
    # not caused by this branch.
    pytest.mark.skip(
        reason="SGLang server CUDA-graph capture fails (CuTe requires CUDA "
        "toolkit >= 13.1); same env issue as PR #2881, fails on main too."
    ),
]


@pytest.fixture(scope="module")
def ray_cluster():
    """Initialise Ray once for this module's tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def router(ray_cluster):
    """Start a real sglang router that lives for the module."""
    from nemo_rl.models.generation.sglang.sglang_router import RouterActor

    actor = RouterActor.remote()
    ip, port = ray.get(actor.init.remote({}))
    yield {"actor": actor, "ip": ip, "port": port}
    try:
        ray.get(actor.stop.remote())
    except Exception:
        pass
    ray.kill(actor)


@pytest.fixture(scope="module")
def worker(ray_cluster, router):
    """Create a single TP=1 worker for this module's tests."""
    w = create_worker(router, base_gpu_id=0, tp_size=1, rank=0)
    yield w
    try:
        ray.get(w.shutdown.remote())
    except Exception:
        pass


# ------------------------------------------------------------------
def test_init_server_healthy(worker):
    """After init, the underlying SGLang server is healthy."""
    result = ray.get(worker.health_generate.remote())
    assert result is True


def test_init_sets_base_url(worker):
    """get_base_url returns a valid http:// URL after init."""
    url = ray.get(worker.get_base_url.remote())
    assert url is not None
    assert url.startswith("http://")


def test_init_registers_with_router(worker, router):
    """The worker registers itself with the session router on init."""
    worker_url = ray.get(worker.get_base_url.remote())
    resp = requests.get(f"http://{router['ip']}:{router['port']}/workers", timeout=10)
    assert resp.status_code == 200
    workers_list = resp.json().get("workers", [])
    assert any(w["url"] == worker_url for w in workers_list)


def test_health_generate_returns_true(worker):
    """health_generate succeeds multiple times (idempotent)."""
    for _ in range(3):
        assert ray.get(worker.health_generate.remote()) is True
