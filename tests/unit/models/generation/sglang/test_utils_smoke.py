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

import multiprocessing

from nemo_rl.models.generation.sglang.utils.ray_utils import (
    _wrap_ipv6,
    find_available_port,
    get_host_info,
    is_port_available,
)
from nemo_rl.models.generation.sglang.utils.router_utils import (
    terminate_process,
)
from nemo_rl.models.policy.torch_reductions_utils import (
    MultiprocessingSerializer,
)


# ---------------------------------------------------------------------------
# ray_utils
# ---------------------------------------------------------------------------
def test_find_available_port():
    """find_available_port returns a port that passes is_port_available."""
    port = find_available_port(20000)
    assert isinstance(port, int)
    assert port > 0
    assert is_port_available(port)


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


def test_terminate_process_already_dead():
    """terminate_process does not raise when the process is already dead."""
    p = multiprocessing.Process(target=lambda: None)
    p.start()
    p.join()
    # Process has already exited — should be a harmless no-op
    terminate_process(p)
