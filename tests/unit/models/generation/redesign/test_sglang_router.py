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

"""Tests for RouterActor lifecycle — start, port allocation, stop.

All tests use a real Ray cluster and a real sglang_router subprocess.
Each test creates its own RouterActor to avoid cross-test interference.
"""

import pytest
import ray
import requests

from nemo_rl.models.generation.redesign.ray_utils import find_available_port
from nemo_rl.models.generation.redesign.sglang_router import RouterActor

pytestmark = pytest.mark.sglang


def _start_and_cleanup(actor, router_cfg):
    """Start a router, return (ip, port), register cleanup on failure."""
    ip, port = ray.get(actor.start.remote(router_cfg))
    return ip, port


def _stop_router(actor):
    try:
        ray.get(actor.stop.remote())
    except Exception:
        pass
    ray.kill(actor)


def test_start_returns_ip_and_port(ray_cluster):
    """RouterActor.start returns a (str, int) tuple."""
    actor = RouterActor.remote()
    try:
        ip, port = _start_and_cleanup(actor, {})
        assert isinstance(ip, str) and len(ip) > 0
        assert isinstance(port, int) and port > 0
    finally:
        _stop_router(actor)


def test_start_uses_configured_port(ray_cluster):
    """When sglang_router_port is set, the router uses that exact port."""
    configured_port = find_available_port(9000)
    actor = RouterActor.remote()
    try:
        ip, port = _start_and_cleanup(actor, {"sglang_router_port": configured_port})
        assert port == configured_port
    finally:
        _stop_router(actor)


def test_start_finds_port_when_not_configured(ray_cluster):
    """When sglang_router_port is None, the router picks one automatically."""
    actor = RouterActor.remote()
    try:
        ip, port = _start_and_cleanup(actor, {})
        assert isinstance(port, int) and port > 0
    finally:
        _stop_router(actor)


def test_stop_terminates_process(ray_cluster):
    """stop() completes without error after a successful start."""
    actor = RouterActor.remote()
    ray.get(actor.start.remote({}))
    ray.get(actor.stop.remote())  # should not raise
    ray.kill(actor)


def test_router_serves_workers_endpoint(ray_cluster):
    """A started router exposes the /workers HTTP endpoint."""
    actor = RouterActor.remote()
    try:
        ip, port = _start_and_cleanup(actor, {})
        resp = requests.get(f"http://{ip}:{port}/workers", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "workers" in data
    finally:
        _stop_router(actor)
