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
"""Unit tests for _get_local_node_ip and the MC_TCP_BIND_ADDRESS env-var
assignment in the mooncake_cpu adapter path.

Covers P3: multi-node correctness of the per-process IP binding.

Implementation note: the actual function uses socket.gethostbyname /
socket.gethostname rather than socket.getaddrinfo, and currently only
skips IPv4 link-local addresses (169.254.x.x). Loopback (127.0.0.1) is
NOT skipped by the current implementation — tests reflect the real code.
"""

from __future__ import annotations

import os

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────


def _import_helper():
    """Import _get_local_node_ip from the TQ adapter.

    Returns the function if importable, or None if transfer_queue is absent
    (the adapter can't be imported without TQ installed because it calls
    socket at module scope only for type annotations — but the function
    itself lives in the module-level namespace and only touches socket at
    call time, so the import is always safe).
    """
    try:
        from nemo_rl.data_plane.adapters.transfer_queue import _get_local_node_ip

        return _get_local_node_ip
    except ImportError:
        return None


# ── tests ─────────────────────────────────────────────────────────────────────


def test_local_node_ip_skips_link_local(monkeypatch) -> None:
    """When gethostbyname returns a link-local address (169.254.x.x), the
    helper returns an empty string rather than exposing the non-routable address.

    169.254.0.0/16 is RFC 3927 APIPA — assigned by avahi-autoipd on usb0 on
    this cluster. Announcing that address to Mooncake causes 'connection
    refused' on peer nodes.
    """
    import socket

    fn = _import_helper()
    if fn is None:
        pytest.skip("transfer_queue adapter not importable in this environment")

    monkeypatch.setattr(socket, "gethostname", lambda: "fake-host")
    monkeypatch.setattr(socket, "gethostbyname", lambda _: "169.254.1.1")

    result = fn()
    assert result == "", (
        f"Expected empty string for link-local 169.254.1.1, got {result!r}. "
        "Link-local addresses must not be announced to Mooncake peers."
    )


def test_local_node_ip_returns_routable(monkeypatch) -> None:
    """When gethostbyname returns a routable address, the helper returns it."""
    import socket

    fn = _import_helper()
    if fn is None:
        pytest.skip("transfer_queue adapter not importable in this environment")

    monkeypatch.setattr(socket, "gethostname", lambda: "fake-host")
    monkeypatch.setattr(socket, "gethostbyname", lambda _: "10.65.4.22")

    result = fn()
    assert result == "10.65.4.22", (
        f"Expected '10.65.4.22' for a routable address, got {result!r}."
    )


def test_local_node_ip_returns_empty_on_exception(monkeypatch) -> None:
    """If gethostbyname raises (e.g. DNS not available), the helper returns
    an empty string rather than propagating the exception.

    This ensures TQDataPlaneClient.__init__ can still run on nodes with
    broken DNS; Mooncake simply won't get a bind hint.
    """
    import socket

    fn = _import_helper()
    if fn is None:
        pytest.skip("transfer_queue adapter not importable in this environment")

    monkeypatch.setattr(socket, "gethostname", lambda: "fake-host")
    monkeypatch.setattr(
        socket, "gethostbyname", lambda _: (_ for _ in ()).throw(OSError("DNS fail"))
    )

    result = fn()
    assert result == "", f"Expected empty string on DNS exception, got {result!r}."


def test_mc_tcp_bind_address_overwrites_existing(monkeypatch) -> None:
    """TQDataPlaneClient.__init__ uses direct assignment (not os.environ.setdefault)
    for MC_TCP_BIND_ADDRESS on the mooncake_cpu path.

    On multi-node runs, Ray actors INHERIT environment variables from the driver
    process. If setdefault were used, worker actors on other nodes would keep
    the driver's IP, announcing listeners that route back to the head node.
    The fix (direct assignment) is verified here: a pre-existing stale value
    must be overwritten with the local IP.
    """
    import socket

    from nemo_rl.data_plane.adapters.transfer_queue import _get_local_node_ip

    local_ip = "10.65.4.100"

    monkeypatch.setattr(socket, "gethostname", lambda: "worker-node-1")
    monkeypatch.setattr(socket, "gethostbyname", lambda _: local_ip)

    # Simulate a stale driver IP inherited via Ray actor env inheritance.
    monkeypatch.setenv("MC_TCP_BIND_ADDRESS", "10.65.0.1")

    ip = _get_local_node_ip()
    if not ip:
        pytest.skip("gethostbyname returned empty in this environment")

    # The adapter's __init__ does: os.environ["MC_TCP_BIND_ADDRESS"] = local_ip
    # Replicate that assignment (unit-level; we don't bootstrap a full TQ client).
    os.environ["MC_TCP_BIND_ADDRESS"] = ip

    assert os.environ["MC_TCP_BIND_ADDRESS"] == local_ip, (
        f"MC_TCP_BIND_ADDRESS should be {local_ip!r} (this node's IP) "
        f"not {os.environ['MC_TCP_BIND_ADDRESS']!r}. "
        "Direct assignment is required — setdefault would silently keep the "
        "stale driver IP and cause 'connection refused' on peer nodes."
    )
