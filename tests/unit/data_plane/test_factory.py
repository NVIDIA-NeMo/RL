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


@pytest.fixture
def stub_tq_adapter(monkeypatch):
    """Stand in for the TQ adapter, which needs mooncake and a live cluster."""
    import sys
    from types import ModuleType

    from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient

    module = ModuleType("nemo_rl.data_plane.adapters.transfer_queue")

    class _StubClient(NoOpDataPlaneClient):
        def __init__(self, cfg, bootstrap=True):
            super().__init__()

    module.TQDataPlaneClient = _StubClient
    monkeypatch.setitem(
        sys.modules, "nemo_rl.data_plane.adapters.transfer_queue", module
    )
    return _StubClient


def _is_wrapped(client) -> bool:
    from nemo_rl.data_plane.observability import MetricsDataPlaneClient

    return isinstance(client, MetricsDataPlaneClient)


def _logs_events(client) -> bool:
    """Whether the wrapper's event sink is the logger rather than a no-op."""
    from nemo_rl.data_plane.observability import log_event

    return client._on_event is log_event


def test_telemetry_alone_installs_the_metrics_wrapper(monkeypatch, stub_tq_adapter):
    """Transfer-queue spans must not depend on data-plane event logging.

    The wrapper carries both, and requiring users to enable an unrelated
    logging feature to get spans is how the queue stayed absent from traces.
    """
    import nemo_rl.data_plane.factory as factory_mod

    monkeypatch.setattr(factory_mod, "telemetry_enabled_in_env", lambda: True)

    client = build_data_plane_client(
        {"enabled": True, "impl": "transfer_queue"}, bootstrap=False
    )
    assert _is_wrapped(client)
    # Event logging stays off: only spans were asked for, and attaching the
    # log sink would add a per-op log line nobody enabled.
    assert not _logs_events(client)
    client.close()


def test_observability_alone_still_installs_the_event_callback(
    monkeypatch, stub_tq_adapter
):
    import nemo_rl.data_plane.factory as factory_mod

    monkeypatch.setattr(factory_mod, "telemetry_enabled_in_env", lambda: False)

    client = build_data_plane_client(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "observability": {"enabled": True},
        },
        bootstrap=False,
    )
    assert _is_wrapped(client)
    assert _logs_events(client)
    client.close()


def test_no_wrapper_when_neither_telemetry_nor_observability_is_on(
    monkeypatch, stub_tq_adapter
):
    import nemo_rl.data_plane.factory as factory_mod

    monkeypatch.setattr(factory_mod, "telemetry_enabled_in_env", lambda: False)

    client = build_data_plane_client(
        {"enabled": True, "impl": "transfer_queue"}, bootstrap=False
    )
    assert not _is_wrapped(client)
    client.close()
