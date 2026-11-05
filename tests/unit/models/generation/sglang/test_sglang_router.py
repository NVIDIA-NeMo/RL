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

"""Tests for RouterActor configuration and lifecycle.

Lifecycle tests use a real Ray cluster and a real sglang_router subprocess.
Configuration tests validate plumbing without starting a router.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray
import requests

from nemo_rl.distributed.virtual_cluster import _get_free_port_local
from nemo_rl.models.generation.sglang import sglang_router
from nemo_rl.models.generation.sglang.sglang_router import RouterActor, _start_router

from . import (
    helpers,  # noqa: F401  — installs env vars + module stubs before nemo_rl imports
)

pytestmark = pytest.mark.sglang


@pytest.mark.parametrize(
    "retry_options, expected",
    [
        ({}, (5, 10)),
        ({"retry_max_retries": 7}, (7, 10)),
        ({"cb_failure_threshold": 2}, (5, 2)),
        ({"retry_max_retries": 1, "cb_failure_threshold": 1}, (1, 1)),
    ],
)
def test_router_retry_options_preserve_defaults_and_forward_overrides(
    monkeypatch, retry_options, expected
):
    args = SimpleNamespace(retry_max_retries=5, cb_failure_threshold=10)
    monkeypatch.setitem(
        sys.modules,
        "sglang_router.launch_router",
        SimpleNamespace(RouterArgs=lambda: args),
    )
    monkeypatch.setattr(sglang_router, "get_host_info", lambda: ("host", "127.0.0.1"))
    monkeypatch.setattr(sglang_router, "_get_free_port_local", lambda *_: 19001)
    router = RouterActor.__ray_metadata__.modified_class()
    start = Mock()
    monkeypatch.setattr(router, "start", start)

    assert router.init({"sglang_router_port": 19000, **retry_options}) == (
        "127.0.0.1",
        19000,
    )
    start.assert_called_once_with(args)
    assert (args.retry_max_retries, args.cb_failure_threshold) == expected


@pytest.mark.parametrize("key", ["retry_max_retries", "cb_failure_threshold"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3", None])
def test_router_retry_options_reject_invalid_values_before_launch(
    monkeypatch, key, value
):
    options = Mock()
    monkeypatch.setattr(RouterActor, "options", options)
    with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
        _start_router({key: value})
    options.assert_not_called()

    router = RouterActor.__ray_metadata__.modified_class()
    start = Mock()
    monkeypatch.setattr(router, "start", start)
    with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
        router.init({key: value})
    start.assert_not_called()


def test_external_router_does_not_apply_managed_retry_options(monkeypatch):
    options = Mock()
    monkeypatch.setattr(RouterActor, "options", options)
    assert _start_router(
        {
            "use_external_router": True,
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 19000,
            "retry_max_retries": 0,
            "cb_failure_threshold": 0,
        }
    ) == ("127.0.0.1", 19000, None)
    options.assert_not_called()


@pytest.fixture(scope="module")
def ray_cluster():
    """Initialise Ray once for this module's tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def _start_and_cleanup(actor, router_cfg):
    """Start a router, return (ip, port), register cleanup on failure."""
    ip, port = ray.get(actor.init.remote(router_cfg))
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
    configured_port = _get_free_port_local(9000, 10000)
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
    ray.get(actor.init.remote({}))
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


# ---------------------------------------------------------------------------
# use_external_router feature
# ---------------------------------------------------------------------------
def test_start_router_external_returns_configured_endpoint():
    """When use_external_router=True, return the configured endpoint and a
    ``None`` actor handle without spawning a RouterActor."""
    cfg = {
        "use_external_router": True,
        "sglang_router_ip": "10.0.0.42",
        "sglang_router_port": 12345,
    }
    ip, port, actor = _start_router(cfg)
    assert ip == "10.0.0.42"
    assert port == 12345
    # ``None`` is the contract: SGLangGeneration.shutdown skips terminate
    # when the router is externally owned.
    assert actor is None


@pytest.mark.parametrize(
    "cfg",
    [
        {"use_external_router": True},
        {"use_external_router": True, "sglang_router_ip": "10.0.0.42"},
        {"use_external_router": True, "sglang_router_port": 12345},
    ],
    ids=["missing_both", "missing_port", "missing_ip"],
)
def test_start_router_external_requires_ip_and_port(cfg):
    """Both ``sglang_router_ip`` and ``sglang_router_port`` must be set
    when ``use_external_router`` is True."""
    with pytest.raises(AssertionError):
        _start_router(cfg)


def test_start_router_external_targets_running_router(ray_cluster):
    """End-to-end: an externally-launched router is reachable via
    ``_start_router`` and the call must not return an owning actor handle."""
    external = RouterActor.remote()
    try:
        ext_ip, ext_port = ray.get(external.init.remote({}))
        cfg = {
            "use_external_router": True,
            "sglang_router_ip": ext_ip,
            "sglang_router_port": ext_port,
        }
        ip, port, actor = _start_router(cfg)
        assert (ip, port) == (ext_ip, ext_port)
        assert actor is None

        resp = requests.get(f"http://{ip}:{port}/workers", timeout=10)
        assert resp.status_code == 200
        assert "workers" in resp.json()
    finally:
        _stop_router(external)
