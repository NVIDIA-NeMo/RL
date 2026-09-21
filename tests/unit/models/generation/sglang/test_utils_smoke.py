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

"""Smoke tests for utility modules (ray_utils, misc, async_utils).

These tests verify basic functionality of helper utilities and do NOT
require a running SGLang server or GPU.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import torch
from torch.multiprocessing import reductions

from . import (
    helpers,  # noqa: F401  — installs env vars + module stubs before nemo_rl imports
)

pytestmark = pytest.mark.sglang

from nemo_rl.models.generation.sglang.config import SGLangHttpClientConfig
from nemo_rl.models.generation.sglang.utils import http_utils, train_utils
from nemo_rl.models.generation.sglang.utils.http_utils import HttpClient
from nemo_rl.models.generation.sglang.utils.ip_port_utils import _wrap_ipv6
from nemo_rl.models.generation.sglang.utils.ray_utils import get_host_info
from nemo_rl.models.generation.sglang.utils.train_utils import (
    MultiprocessingSerializer,
)


# ---------------------------------------------------------------------------
# ray_utils
# ---------------------------------------------------------------------------
def test_wrap_ipv6_noop_for_ipv4():
    """IPv4 addresses are returned unchanged by _wrap_ipv6."""
    assert _wrap_ipv6("192.168.1.1") == "192.168.1.1"
    assert _wrap_ipv6("10.0.0.1") == "10.0.0.1"
    assert _wrap_ipv6("127.0.0.1") == "127.0.0.1"


def test_wrap_ipv6_brackets_ipv6():
    """IPv6 addresses are wrapped in [] by _wrap_ipv6, idempotently."""
    # Bare IPv6 → wrapped.
    assert _wrap_ipv6("::1") == "[::1]"
    assert _wrap_ipv6("2001:db8::1") == "[2001:db8::1]"
    # Already-bracketed input stays a single pair of brackets.
    assert _wrap_ipv6("[::1]") == "[::1]"
    assert _wrap_ipv6("[2001:db8::1]") == "[2001:db8::1]"


def test_get_host_info_returns_tuple():
    """get_host_info returns (hostname, ip_address) strings."""
    hostname, ip = get_host_info()
    assert isinstance(hostname, str) and len(hostname) > 0
    assert isinstance(ip, str) and len(ip) > 0


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def test_serializer_roundtrip():
    """serialize → deserialize returns the original object."""
    obj = {"key": "value", "numbers": [1, 2, 3], "nested": {"a": True}}
    serialized = MultiprocessingSerializer.serialize(obj, output_str=True)
    assert isinstance(serialized, str) and len(serialized) > 0
    deserialized = MultiprocessingSerializer.deserialize(serialized)
    assert deserialized == obj


def test_reduce_tensor_modified_preserves_cpu_reduction(monkeypatch):
    """The CUDA UUID patch must leave the shorter CPU reducer protocol intact."""
    tensor = torch.tensor([1, 2, 3])
    monkeypatch.setattr(
        reductions,
        "_reduce_tensor_original",
        getattr(reductions, "_reduce_tensor_original", reductions.reduce_tensor),
        raising=False,
    )

    output_fn, output_args = train_utils._reduce_tensor_modified(tensor)

    assert output_fn is reductions.rebuild_tensor
    assert len(output_args) == 3
    torch.testing.assert_close(output_fn(*output_args), tensor)


def test_reduce_tensor_modified_converts_cuda_device_to_uuid(monkeypatch):
    """The dense CUDA reducer still converts its device argument to a UUID."""
    cuda_output_args = tuple(range(15))
    monkeypatch.setattr(
        reductions,
        "_reduce_tensor_original",
        lambda *_args, **_kwargs: (
            train_utils._rebuild_cuda_tensor_modified,
            cuda_output_args,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        train_utils,
        "_device_to_uuid",
        lambda device: f"cuda-uuid-{device}",
    )

    output_fn, output_args = train_utils._reduce_tensor_modified(object())

    assert output_fn is train_utils._rebuild_cuda_tensor_modified
    assert output_args[:6] == cuda_output_args[:6]
    assert output_args[6] == "cuda-uuid-6"
    assert output_args[7:] == cuda_output_args[7:]


# ---------------------------------------------------------------------------
# HTTP retry budgets (no real HTTP server or Ray cluster)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "args, expected",
    [
        (None, 3),
        ({"sglang_cfg": {}}, 3),
        ({"sglang_cfg": {"sglang_http_client_config": {}}}, 3),
        ({"sglang_cfg": {"sglang_http_client_config": {"max_retries": 5}}}, 5),
        (
            {
                "sglang_cfg": {
                    "sglang_http_client_config": SGLangHttpClientConfig(max_retries=2)
                }
            },
            2,
        ),
        (
            {
                "sglang_cfg": {
                    "sglang_router_config": {"use_external_router": True},
                    "sglang_http_client_config": {"max_retries": 4},
                }
            },
            4,
        ),
    ],
)
def test_http_client_retry_budget_without_gpu_config(monkeypatch, args, expected):
    client = HttpClient(args)
    post_local = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_post_local", post_local)

    assert asyncio.run(client.post("http://router/generate", {"input_ids": [1]})) == {
        "ok": True
    }
    post_local.assert_awaited_once_with(
        "http://router/generate", {"input_ids": [1]}, expected, action="post"
    )


def test_http_client_explicit_override_preserves_configured_budget(monkeypatch):
    client = HttpClient(
        {"sglang_cfg": {"sglang_http_client_config": {"max_retries": 4}}}
    )
    post_local = AsyncMock()
    monkeypatch.setattr(client, "_post_local", post_local)

    asyncio.run(client.post("http://router/generate", {}, max_retries=1))
    post_local.assert_awaited_once_with("http://router/generate", {}, 1, action="post")
    post_local.reset_mock()
    asyncio.run(client.post("http://router/generate", {}))
    post_local.assert_awaited_once_with("http://router/generate", {}, 4, action="post")


def test_http_client_config_applies_with_gpu_and_distributed_setup(monkeypatch):
    init_actors = Mock()
    monkeypatch.setattr(HttpClient, "_init_ray_distributed_post", init_actors)
    args = {
        "sglang_cfg": {
            "sglang_http_client_config": {"max_retries": 5},
            "sglang_server_config": {
                "num_gpus": 2,
                "num_gpus_per_engine": 1,
                "sglang_server_concurrency": 4,
            },
            "sglang_router_config": {
                "use_distributed_post": True,
                "use_external_router": True,
            },
        }
    }
    client = HttpClient(args)
    assert client._max_retries == 5
    assert client._client_concurrency == 8
    assert client._distributed_post_enabled
    init_actors.assert_called_once_with(args)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3", None])
def test_http_client_rejects_invalid_config_before_gpu_check(value):
    with pytest.raises(ValueError, match="max_retries"):
        HttpClient(
            {"sglang_cfg": {"sglang_http_client_config": {"max_retries": value}}}
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_http_client_rejects_invalid_override_before_dispatch(monkeypatch, value):
    client = HttpClient()
    post_local = AsyncMock()
    actor = SimpleNamespace(do_post=SimpleNamespace(remote=Mock()))
    client._distributed_post_enabled = True
    client._post_actors = [actor]
    monkeypatch.setattr(client, "_post_local", post_local)

    with pytest.raises(ValueError, match="max_retries"):
        asyncio.run(client.post("http://router/generate", {}, max_retries=value))
    actor.do_post.remote.assert_not_called()
    post_local.assert_not_called()


@pytest.mark.parametrize("override, expected", [(None, 4), (1, 1)])
@pytest.mark.parametrize("distributed_fails", [False, True])
def test_http_client_distributed_budget_and_local_fallback(
    monkeypatch, override, expected, distributed_fails
):
    client = HttpClient(
        {"sglang_cfg": {"sglang_http_client_config": {"max_retries": 4}}}
    )
    actor = SimpleNamespace(do_post=SimpleNamespace(remote=Mock(return_value="ref")))
    client._distributed_post_enabled = True
    client._post_actors = [actor]
    post_local = AsyncMock(return_value={"local": True})
    monkeypatch.setattr(client, "_post_local", post_local)
    get = Mock(
        return_value={"remote": True},
        side_effect=RuntimeError("remote exhausted") if distributed_fails else None,
    )
    monkeypatch.setattr(http_utils.ray, "get", get)

    result = asyncio.run(
        client.post("http://router/generate", {"input_ids": [1]}, max_retries=override)
    )
    actor.do_post.remote.assert_called_once_with(
        "http://router/generate", {"input_ids": [1]}, expected, action="post"
    )
    get.assert_called_once_with("ref")
    if distributed_fails:
        assert result == {"local": True}
        post_local.assert_awaited_once_with(
            "http://router/generate", {"input_ids": [1]}, expected, action="post"
        )
    else:
        assert result == {"remote": True}
        post_local.assert_not_called()


@pytest.mark.parametrize("execution", ["local", "actor"])
@pytest.mark.parametrize("budget", [1, 3, 5])
@pytest.mark.parametrize("succeeds", [False, True])
def test_http_retry_loops_obey_total_attempt_budget(
    monkeypatch, execution, budget, succeeds
):
    request = httpx.Request("POST", "http://router/generate")
    unavailable = httpx.Response(500, request=request)
    success = httpx.Response(200, json={"ok": True}, request=request)
    responses = [unavailable] * (budget - 1) + [success if succeeds else unavailable]
    post = AsyncMock(side_effect=responses)
    transport = SimpleNamespace(post=post)
    sleep = AsyncMock()
    monkeypatch.setattr(http_utils.asyncio, "sleep", sleep)

    if execution == "local":
        client = HttpClient(
            {"sglang_cfg": {"sglang_http_client_config": {"max_retries": budget}}}
        )
        monkeypatch.setattr(client, "_get_client", lambda: transport)
        pending = client.post(str(request.url), {})
    else:
        pending = http_utils._HttpPosterActor.__ray_metadata__.modified_class.do_post(
            SimpleNamespace(_client=transport), str(request.url), {}, budget
        )

    if succeeds:
        assert asyncio.run(pending) == {"ok": True}
    else:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(pending)
    assert post.await_count == budget
    assert sleep.await_count == budget - 1
