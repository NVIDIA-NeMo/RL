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
"""Device selection for the mooncake transport: all IB rails, never RoCE
alongside them. A regression here still trains, just slower, so nothing
else would catch it.
"""

import os

import pytest

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter


@pytest.fixture
def fake_fabric(monkeypatch):
    """Install a synthetic device inventory.

    ``rdma_devices`` reads real sysfs, so the layout is injected: the uverbs
    glob gates on device availability and the link_layer glob enumerates.
    """

    def _install(layers: dict[str, str], *, uverbs: bool = True):
        def fake_glob(pattern: str):
            if pattern.startswith("/dev/infiniband/uverbs"):
                return ["/dev/infiniband/uverbs0"] if uverbs else []
            return [f"/sys/class/infiniband/{d}/ports/1/link_layer" for d in layers]

        monkeypatch.setattr(tq_adapter.glob, "glob", fake_glob)
        monkeypatch.setattr(
            tq_adapter,
            "_link_layer",
            lambda name: layers[name],
        )
        monkeypatch.setattr(os, "environ", dict(os.environ))
        monkeypatch.delenv("MC_MOONCAKE_DEVICE", raising=False)

    return _install


# The real pool0 layout: eight 400 Gb/s IB rails plus one 100 Gb/s RoCE port.
_MIXED = {
    "mlx5_0": "InfiniBand",
    "mlx5_1": "InfiniBand",
    "mlx5_2": "InfiniBand",
    "mlx5_3": "Ethernet",
    "mlx5_4": "InfiniBand",
    "mlx5_5": "InfiniBand",
    "mlx5_6": "InfiniBand",
    "mlx5_7": "InfiniBand",
    "mlx5_8": "InfiniBand",
}


def test_prefers_infiniband_and_excludes_roce(fake_fabric):
    """The regression this guards: mlx5_3 was chosen over eight IB rails.

    Exact equality also pins the three things the format depends on: all
    eight rails (not one), no space after the comma (mooncake splits on ","
    only), and no RoCE device mixed in.
    """
    fake_fabric(_MIXED)
    assert (
        tq_adapter.rdma_devices()
        == "mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
    )


def test_falls_back_to_roce_only_when_no_ib(fake_fabric):
    fake_fabric({"mlx5_0": "Ethernet", "mlx5_1": "Ethernet"})
    assert tq_adapter.rdma_devices() == "mlx5_0,mlx5_1"


def test_empty_without_verbs_node(fake_fabric):
    """Containers see /sys without /dev/infiniband; mooncake fails late there."""
    fake_fabric(_MIXED, uverbs=False)
    assert tq_adapter.rdma_devices() == ""


def test_env_override_wins_verbatim(fake_fabric, monkeypatch):
    fake_fabric(_MIXED)
    monkeypatch.setenv("MC_MOONCAKE_DEVICE", "mlx5_9,mlx5_10")
    assert tq_adapter.rdma_devices() == "mlx5_9,mlx5_10"


def test_transport_config_is_rdma_and_carries_all_rails(fake_fabric):
    """The device list must reach mooncake, and the transport stays RDMA."""
    fake_fabric(_MIXED)
    cfg = tq_adapter._mooncake_transport_config()
    assert cfg["protocol"] == "rdma"
    assert cfg["device_name"] == tq_adapter.rdma_devices()


def test_gid_index_left_unset_for_infiniband(fake_fabric, monkeypatch):
    """GID 3 is a RoCEv2 convention; IB numbers GIDs differently, so the
    pin must not be applied when IB is selected."""
    monkeypatch.delenv("MC_GID_INDEX", raising=False)
    fake_fabric(_MIXED)
    tq_adapter._mooncake_transport_config()
    assert "MC_GID_INDEX" not in os.environ


def test_gid_index_pinned_for_roce(fake_fabric, monkeypatch):
    monkeypatch.delenv("MC_GID_INDEX", raising=False)
    fake_fabric({"mlx5_3": "Ethernet"})
    tq_adapter._mooncake_transport_config()
    assert os.environ["MC_GID_INDEX"] == "3"


def test_raises_when_no_device_since_mooncake_is_rdma_only(fake_fabric):
    fake_fabric(_MIXED, uverbs=False)
    with pytest.raises(RuntimeError, match="requires RDMA"):
        tq_adapter._mooncake_transport_config()


# ── Peer-rail pairing ────────────────────────────────────────────────────────
#
# Mooncake picks the peer rail at random unless told otherwise. Where each rail
# is its own subnet (the RoCE-only gb200 CI runners) a cross-rail pair has no
# route, which was 100% of the failures observed there.


def _mooncake_cfg() -> dict:
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "mooncake_cpu",
        "claim_meta_poll_interval_s": 0.5,
    }


@pytest.fixture
def stub_client(monkeypatch):
    """Build a TQDataPlaneClient without touching TQ, mooncake, or the network."""

    def _build(cfg: dict):
        monkeypatch.setattr(tq_adapter, "_connect_existing", lambda: None)
        monkeypatch.setattr(tq_adapter, "_get_local_node_ip", lambda: "10.0.0.1")
        monkeypatch.setattr(tq_adapter, "_patch_mooncake_register_check", lambda: None)
        monkeypatch.setattr(tq_adapter, "_patch_mooncake_staging_buffers", lambda: None)
        monkeypatch.setattr(os, "environ", dict(os.environ))
        return tq_adapter.TQDataPlaneClient(cfg, bootstrap=False)

    return _build


def test_peer_rail_is_pinned_to_the_local_rail(stub_client):
    """Same-rail pairing keeps every rail in use instead of narrowing to one."""
    stub_client(_mooncake_cfg())
    assert os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "1"


def test_peer_rail_pairing_is_overridable(stub_client, monkeypatch):
    """A fully-routable fabric can opt back into random peer selection."""
    monkeypatch.setenv("MC_ENABLE_DEST_DEVICE_AFFINITY", "0")
    stub_client(_mooncake_cfg())
    assert os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "0"


def test_peer_rail_pairing_not_applied_to_simple_backend(stub_client):
    """The knob is mooncake-only; `simple` never touches RDMA."""
    stub_client({**_mooncake_cfg(), "backend": "simple"})
    assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in os.environ
