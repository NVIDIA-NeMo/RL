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
"""Plan §4.3 — production factory rejects disabled and unknown impls.

NoOp via factory is forbidden by design (plan §4.8 R-C10). The
NoOpDataPlaneClient is reachable only as a direct import from tests —
verified by the architecture invariants in test_architecture_invariants.
"""

from __future__ import annotations

import pytest

from nemo_rl.data_plane import build_data_plane_client


def test_factory_none_cfg_rejected():
    """T1-factory-none-cfg — None config must fail-fast, not silently
    construct anything."""
    with pytest.raises(ValueError):
        build_data_plane_client(None)


def test_factory_disabled_rejected():
    """T1-factory-disabled-rejected — production factory must not
    silently hand back a NoOp on enabled=False."""
    with pytest.raises(ValueError, match=r"disabled|enabled"):
        build_data_plane_client({"enabled": False, "impl": "transfer_queue"})


def test_factory_noop_impl_rejected():
    """T1-factory-noop-rejected-in-prod — NoOp is not selectable from
    the factory. Catches R-C10 (NoOp leaks into production)."""
    with pytest.raises(ValueError):
        build_data_plane_client({"enabled": True, "impl": "noop"})


def test_factory_unknown_impl_rejected():
    """T1-factory-unknown-impl — unknown impl name fails-fast with a
    message naming the offending value."""
    with pytest.raises(ValueError, match=r"unknown.*impl"):
        build_data_plane_client({"enabled": True, "impl": "no_such_thing"})


def test_factory_disabled_error_message_helpful():
    """When the factory rejects a disabled config, the error message
    should point users at the legacy trainer escape hatch."""
    with pytest.raises(ValueError) as excinfo:
        build_data_plane_client({"enabled": False, "impl": "transfer_queue"})
    msg = str(excinfo.value)
    # Some pointer to the legacy path so users can self-recover.
    assert "grpo" in msg.lower() or "legacy" in msg.lower(), (
        f"factory rejection should reference the legacy trainer; got: {msg}"
    )
