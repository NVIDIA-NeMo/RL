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

from multiprocessing.reduction import ForkingPickler

import pytest
import torch
from torch.multiprocessing import reductions

from . import (
    helpers,  # noqa: F401  — installs env vars + module stubs before nemo_rl imports
)

pytestmark = pytest.mark.sglang

from nemo_rl.models.generation.sglang.utils import train_utils
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


def test_reduce_tensor_modified_preserves_cpu_reduction():
    """The CUDA UUID patch must leave the shorter CPU reducer protocol intact."""
    tensor = torch.tensor([1, 2, 3])
    # No substitution needed: the module global is captured before any patching,
    # so it is the stock reducer whether or not this process has been patched.
    output_fn, output_args = train_utils._reduce_tensor_modified(tensor)

    assert output_fn is reductions.rebuild_tensor
    assert len(output_args) == 3
    torch.testing.assert_close(output_fn(*output_args), tensor)


def test_reduce_tensor_modified_converts_cuda_device_to_uuid(monkeypatch):
    """The dense CUDA reducer still converts its device argument to a UUID."""
    cuda_output_args = tuple(range(15))
    monkeypatch.setattr(
        train_utils,
        "_REDUCE_TENSOR_ORIGINAL",
        lambda *_args, **_kwargs: (
            train_utils._rebuild_cuda_tensor_modified,
            cuda_output_args,
        ),
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
# torch.multiprocessing reduction patch
# ---------------------------------------------------------------------------
class _ReducedCudaTensorStandIn:
    """Pickles the way the patched CUDA reducer emits its payload.

    Building the payload explicitly keeps this test off a real GPU. The reducer
    that produces this shape is covered by
    ``test_reduce_tensor_modified_converts_cuda_device_to_uuid``.
    """

    def __reduce__(self):
        # Index 6 is the device slot; an int keeps _device_from_maybe_uuid off CUDA.
        return (train_utils._rebuild_cuda_tensor_modified, tuple(range(15)))


def _rebuild_in_unpatched_child(payload, result_queue):
    """Runs in a fresh interpreter that never calls monkey_patch_torch_reductions().

    nvidia_resiliency_ext's spawned async-checkpoint worker does exactly this: it
    unpickles tensors reduced by the patched parent, so the rebuild wrapper has to
    work without any state that only the patch installs.
    """
    try:
        report = {
            "patched": hasattr(reductions, "_reduce_tensor_original"),
            "original_is_stock": (
                train_utils._REBUILD_CUDA_TENSOR_ORIGINAL
                is reductions.rebuild_cuda_tensor
            ),
        }
        # Stand in for the native rebuild so what the assertions exercise is the
        # wrapper's own lookup, not CUDA IPC.
        train_utils._REBUILD_CUDA_TENSOR_ORIGINAL = lambda *args: ("rebuilt", args)
        report["rebuilt"] = ForkingPickler.loads(payload)
        result_queue.put(("ok", report))
    except BaseException as exc:  # noqa: BLE001 - the failure itself is the result
        result_queue.put(("error", repr(exc)))


def test_rebuild_cuda_tensor_runs_in_unpatched_spawn():
    """A reduced CUDA tensor must rebuild in a process that never applied the patch.

    The wrapper used to delegate to ``reductions._rebuild_cuda_tensor_original``,
    an attribute only the patching process holds, so the spawned
    async-checkpoint worker died with AttributeError and the run hung until the
    scheduler timed it out.
    """
    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    payload = bytes(ForkingPickler.dumps(_ReducedCudaTensorStandIn()))
    child = ctx.Process(
        target=_rebuild_in_unpatched_child, args=(payload, result_queue)
    )
    child.start()
    try:
        status, detail = result_queue.get(timeout=180)
    finally:
        child.join(timeout=30)
        if child.is_alive():
            child.terminate()
            child.join(timeout=30)
        if child.is_alive():
            child.kill()
            child.join(timeout=30)
        result_queue.close()
        result_queue.join_thread()

    # Reporting "ok" is not the same as exiting cleanly: a child that hangs
    # afterwards gets terminated above, and a negative exitcode is the only
    # thing left that still says so.
    assert child.exitcode == 0, f"child exited with {child.exitcode}"
    assert status == "ok", f"child raised: {detail}"
    assert not detail["patched"], "child must model a process that never patched"
    assert detail["original_is_stock"]
    rebuilt_marker, rebuilt_args = detail["rebuilt"]
    assert rebuilt_marker == "rebuilt"
    assert rebuilt_args[6] == 6
