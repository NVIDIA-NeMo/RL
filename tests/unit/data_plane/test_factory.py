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

from unittest.mock import MagicMock

import pytest

from nemo_rl.data_plane import build_data_plane_client


@pytest.mark.parametrize("backend", ["simple", "mooncake_cpu"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_factory_passes_checkpoint_runtime_mode_to_bootstrap(
    monkeypatch, backend, checkpointing
) -> None:
    from nemo_rl.data_plane.adapters import tq_mooncake_checkpoint
    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    bootstrap = MagicMock()
    connect = MagicMock()
    monkeypatch.setattr(adapter, "_init_tq", bootstrap)
    monkeypatch.setattr(adapter, "_connect_existing", connect)
    monkeypatch.setattr(adapter, "_get_local_node_ip", lambda: "")
    monkeypatch.setattr(adapter, "_patch_mooncake_register_check", lambda: None)
    monkeypatch.setattr(
        tq_mooncake_checkpoint, "install_tq_mooncake_checkpoint_plugin", lambda: None
    )
    cfg = {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": backend,
        "claim_meta_poll_interval_s": 0.5,
        "mooncake_cpu": {"reuse_registered_buffers": False},
    }

    client = build_data_plane_client(cfg, checkpointing=checkpointing)

    bootstrap.assert_called_once_with(cfg, checkpointing=checkpointing)
    connect.assert_not_called()
    assert client._supports_checkpointing is True


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


def _has_event_sink(client) -> bool:
    """Whether the wrapper got a per-op sink at all, rather than spans only."""
    return client._on_event is not None


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
    # Event logging stays off: only spans were asked for, and attaching a sink
    # would add a per-op callback nobody enabled.
    assert not _has_event_sink(client)
    client.close()


def test_observability_alone_installs_the_wrapper_without_a_default_sink(
    monkeypatch, stub_tq_adapter
):
    """Observability alone gets the wrapper, but no per-op sink by default.

    The metrics surface is ``get_step_metrics``, which the trainer logs once a
    step; a sink here fires on every single transfer, so ``log_event`` is opt-in
    via ``observability.callback``.
    """
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
    assert not _has_event_sink(client)
    client.close()


def test_observability_callback_is_the_opt_in_path_to_per_op_events(
    monkeypatch, stub_tq_adapter
):
    """An explicitly configured callback is wired straight through."""
    import nemo_rl.data_plane.factory as factory_mod

    monkeypatch.setattr(factory_mod, "telemetry_enabled_in_env", lambda: False)

    def sink(event):
        pass

    client = build_data_plane_client(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "observability": {"enabled": True, "callback": sink},
        },
        bootstrap=False,
    )
    assert _is_wrapped(client)
    assert client._on_event is sink
    client.close()


def test_disabled_observability_block_is_not_armed_by_telemetry(
    monkeypatch, stub_tq_adapter
):
    """A telemetry-only run must not honour a disabled observability block.

    Telemetry installs the wrapper for its spans, which puts the ``callback``
    and ``verify_tensor_hash`` fields of an ``enabled: false`` block back in
    reach. Acting on them would start a per-op sink and re-read every tensor
    element on both put and get for a feature the user switched off.
    """
    import nemo_rl.data_plane.factory as factory_mod

    monkeypatch.setattr(factory_mod, "telemetry_enabled_in_env", lambda: True)

    def sink(event):
        pass

    client = build_data_plane_client(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "observability": {
                "enabled": False,
                "callback": sink,
                "verify_tensor_hash": True,
            },
        },
        bootstrap=False,
    )
    assert _is_wrapped(client)
    assert not _has_event_sink(client)
    assert not client._verify_tensor_hash
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
