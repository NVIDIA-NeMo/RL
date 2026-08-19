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

    def _install(
        layers: dict[str, str],
        *,
        uverbs: bool = True,
        numa: dict[str, str] | None = None,
    ):
        numa = numa or {}

        def fake_glob(pattern: str):
            if pattern.startswith("/dev/infiniband/uverbs"):
                return ["/dev/infiniband/uverbs0"] if uverbs else []
            return [f"/sys/class/infiniband/{d}/ports/1/link_layer" for d in layers]

        def fake_read_text(path, *args, **kwargs):
            if path.name == "numa_node":
                # No NUMA info by default, matching every test that doesn't
                # pass `numa=`: rdma_devices() must then fall back to
                # unique-per-device (no dedup), not collapse every rail.
                device = path.parents[1].name
                if device not in numa:
                    raise OSError("no numa info")
                return numa[device]
            return layers[path.parents[2].name]

        monkeypatch.setattr(tq_adapter.glob, "glob", fake_glob)
        monkeypatch.setattr(tq_adapter.Path, "read_text", fake_read_text)
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


def test_dedupes_redundant_rails_on_the_same_numa_domain(fake_fabric):
    """The regression this guards: two rails on one domain leave mooncake's
    same-name peer hint something to be ambiguous about, and it was measured
    picking a cross-rail pair at random — every cross-rail pair among
    mlx5_0..3 failed on the real fleet, no same-rail pair ever did.

    Deduplicating to the first rail per domain removes the ambiguity while
    still giving every domain (not just one) a dedicated rail, so a job that
    spans both domains still gets two rails, not one.
    """
    fake_fabric(
        {
            "mlx5_0": "Ethernet",
            "mlx5_1": "Ethernet",
            "mlx5_2": "Ethernet",
            "mlx5_3": "Ethernet",
        },
        numa={"mlx5_0": "0", "mlx5_1": "0", "mlx5_2": "1", "mlx5_3": "1"},
    )
    assert tq_adapter.rdma_devices() == "mlx5_0,mlx5_2"


def test_keeps_every_rail_when_numa_info_is_unavailable(fake_fabric):
    """Missing/absent NUMA info must never be treated as "same domain" —
    that would collapse every rail on the host to one, exactly the
    single-rail regression this dedup logic must not reintroduce.
    """
    fake_fabric({"mlx5_0": "Ethernet", "mlx5_1": "Ethernet", "mlx5_2": "Ethernet"})
    assert tq_adapter.rdma_devices() == "mlx5_0,mlx5_1,mlx5_2"


def test_infiniband_is_never_deduped(fake_fabric):
    """Dedup is a RoCE-only fallback, never applied to IB.

    This PR's own intent is every IB rail in use (see the commit titled
    "use all IB rails and reuse RDMA-registered buffers"), and the
    cross-rail-ambiguity failure the dedup guards against has only been
    measured on RoCE. Two same-domain IB rails must both survive even
    though the equivalent RoCE layout in the test above collapses to one.
    """
    fake_fabric(
        {"mlx5_0": "InfiniBand", "mlx5_1": "InfiniBand"},
        numa={"mlx5_0": "0", "mlx5_1": "0"},
    )
    assert tq_adapter.rdma_devices() == "mlx5_0,mlx5_1"


def test_dedupe_can_be_disabled(fake_fabric):
    """MooncakeCpuConfig.dedupe_rails_per_numa_domain=False escape hatch."""
    fake_fabric(
        {"mlx5_0": "Ethernet", "mlx5_1": "Ethernet"},
        numa={"mlx5_0": "0", "mlx5_1": "0"},
    )
    assert tq_adapter.rdma_devices(dedupe_per_numa_domain=False) == "mlx5_0,mlx5_1"


def test_mc_mooncake_device_override_bypasses_dedupe(fake_fabric, monkeypatch):
    """An explicit override is verbatim regardless of dedupe — same guarantee
    as test_env_override_wins_verbatim, pinned separately because dedupe
    logic runs after the override check and must never second-guess it.
    """
    fake_fabric(
        {"mlx5_0": "Ethernet", "mlx5_1": "Ethernet"},
        numa={"mlx5_0": "0", "mlx5_1": "0"},
    )
    monkeypatch.setenv("MC_MOONCAKE_DEVICE", "mlx5_0,mlx5_1")
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
        monkeypatch.setattr(
            tq_adapter, "_patch_mooncake_staging_buffers", lambda max_bytes: None
        )
        monkeypatch.setattr(os, "environ", dict(os.environ))
        return tq_adapter.TQDataPlaneClient(cfg, bootstrap=False)

    return _build


def test_peer_rail_is_pinned_to_the_local_rail(stub_client, monkeypatch):
    """Same-rail pairing keeps every rail in use instead of narrowing to one."""
    monkeypatch.delenv("MC_ENABLE_DEST_DEVICE_AFFINITY", raising=False)
    stub_client(_mooncake_cfg())
    assert os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "1"


def test_setdefault_respects_an_already_set_value(stub_client, monkeypatch):
    """setdefault must not clobber a value an operator already set.

    Does NOT prove the mooncake engine treats "0" as disabled — verified
    against the pinned wheel, MC_ENABLE_DEST_DEVICE_AFFINITY parsing is
    ``if (std::getenv(...))``: presence, not value, so any set value (even
    "0") enables it at the engine level. There is currently no way to
    disable it once set; this only pins NeMo-RL's own os.environ handling.
    """
    monkeypatch.setenv("MC_ENABLE_DEST_DEVICE_AFFINITY", "0")
    stub_client(_mooncake_cfg())
    assert os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "0"


def test_peer_rail_pairing_not_applied_to_simple_backend(stub_client, monkeypatch):
    """The knob is mooncake-only; `simple` never touches RDMA.

    delenv first: the assertion is that *we* do not set it, not that the machine
    running the tests happens to have it unset.
    """
    monkeypatch.delenv("MC_ENABLE_DEST_DEVICE_AFFINITY", raising=False)
    stub_client({**_mooncake_cfg(), "backend": "simple"})
    assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in os.environ
