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
import sys

import pytest

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter
from nemo_rl.data_plane.adapters import transfer_queue_env as tq_env


@pytest.fixture
def fake_fabric(monkeypatch):
    """Install a synthetic device inventory.

    The sysfs scan lives in ``tq_env.rail_link_layers`` and the uverbs gate in
    the adapter, but both reach it through the same stdlib ``glob`` module
    object, so one patch covers both: the uverbs glob gates on device
    availability and the link_layer glob enumerates.
    """

    def _install(layers: dict[str, str], *, uverbs: bool = True):
        def fake_glob(pattern: str):
            if pattern.startswith("/dev/infiniband/uverbs"):
                return ["/dev/infiniband/uverbs0"] if uverbs else []
            return [f"/sys/class/infiniband/{d}/ports/1/link_layer" for d in layers]

        def fake_read_text(path, *args, **kwargs):
            return layers[path.parents[2].name]

        monkeypatch.setattr(tq_env.glob, "glob", fake_glob)
        monkeypatch.setattr(tq_env.Path, "read_text", fake_read_text)
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


def test_every_roce_rail_is_offered(fake_fabric):
    """The gb200 CI layout — 4 RoCE rails, two per NUMA domain — yields all 4.

    Correctness does not come from thinning this list; it comes from
    MC_ENABLE_DEST_DEVICE_AFFINITY pinning each transfer's peer rail to the
    local one. Keeping one rail per domain does NOT make this safe: the pair
    that survives such a filter (mlx5_0, mlx5_2) is itself cross-rail and was
    measured failing on that fleet.
    """
    fake_fabric(
        {
            "mlx5_0": "Ethernet",
            "mlx5_1": "Ethernet",
            "mlx5_2": "Ethernet",
            "mlx5_3": "Ethernet",
        }
    )
    assert tq_adapter.rdma_devices() == "mlx5_0,mlx5_1,mlx5_2,mlx5_3"


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
def clean_env(monkeypatch):
    """Isolate os.environ and pretend the engine has not been imported yet.

    The real ``sys.modules`` always has ``transfer_queue`` in it here — this
    test module imports the adapter — so the "not yet imported" case has to be
    injected rather than arranged.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.delenv("MC_ENABLE_DEST_DEVICE_AFFINITY", raising=False)
    monkeypatch.delenv("MC_STORE_MEMCPY", raising=False)
    monkeypatch.setattr(tq_env, "_engine_already_imported", lambda: None)


@pytest.fixture
def engine_imported(clean_env, monkeypatch):
    """Flip ``clean_env``'s verdict: pretend the engine is already loaded."""
    monkeypatch.setattr(tq_env, "_engine_already_imported", lambda: "transfer_queue")


def test_affinity_pinned_on_a_roce_only_fabric(clean_env, fake_fabric):
    """Same-rail pairing is what makes offering every rail safe."""
    fake_fabric({"mlx5_0": "Ethernet", "mlx5_1": "Ethernet"})
    tq_env.configure_engine_env(_mooncake_cfg())
    assert os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "1"


def test_affinity_left_alone_on_infiniband(clean_env, fake_fabric):
    """IB routes cross-rail, so the hint is not our call to make there.

    Scoped deliberately: the cross-rail failure was only ever measured on RoCE,
    and this cluster cannot test IB.
    """
    fake_fabric({"mlx5_0": "InfiniBand", "mlx5_1": "InfiniBand"})
    tq_env.configure_engine_env(_mooncake_cfg())
    assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in os.environ


def test_roce_gate_does_not_fail_open_when_sysfs_is_empty(clean_env, fake_fabric):
    """No rails at all must not read as "InfiniBand, skip the hint"."""
    fake_fabric({})
    assert tq_env.fabric_is_roce_only() is False


def test_existing_value_is_not_clobbered(clean_env, fake_fabric):
    """A launcher-supplied value wins; we only fill the gap."""
    fake_fabric({"mlx5_0": "Ethernet"})
    os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] = "0"
    tq_env.configure_engine_env(_mooncake_cfg())
    assert os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "0"


def test_not_applied_to_simple_backend(clean_env, fake_fabric):
    """The knob is mooncake-only; `simple` never touches RDMA."""
    fake_fabric({"mlx5_0": "Ethernet"})
    tq_env.configure_engine_env({**_mooncake_cfg(), "backend": "simple"})
    assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in os.environ


def test_raises_when_the_engine_was_already_imported(engine_imported, fake_fabric):
    """The whole point of the split: too-late must be loud, not silent.

    Mooncake snapshots MC_* as its extension loads, so a value set after that
    reads back fine from os.environ while the engine ignores it — which is how
    this cost several CI runs before being caught.
    """
    fake_fabric({"mlx5_0": "Ethernet"})
    with pytest.raises(RuntimeError, match="already imported"):
        tq_env.configure_engine_env(_mooncake_cfg())


def test_no_raise_when_already_set_even_if_engine_imported(
    engine_imported, fake_fabric
):
    """The normal worker path: Ray handed down the driver's environment."""
    fake_fabric({"mlx5_0": "Ethernet"})
    os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] = "1"
    os.environ["MC_STORE_MEMCPY"] = "0"
    tq_env.configure_engine_env(_mooncake_cfg())  # must not raise


def test_engine_already_imported_detects_a_loaded_module(monkeypatch):
    """Pin the detector itself, since every other test stubs it out."""
    monkeypatch.setitem(sys.modules, "transfer_queue", object())
    assert tq_env._engine_already_imported() == "transfer_queue"
    for name in tq_env._ENGINE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert tq_env._engine_already_imported() is None
