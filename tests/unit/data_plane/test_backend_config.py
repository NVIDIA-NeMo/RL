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
"""Resolution of the per-backend sizing block.

``data_plane`` carries one block per backend (``simple:`` / ``mooncake_cpu:``)
and only the selected one is read, falling back to that backend's defaults when
absent. Getting this wrong would silently run a job at the wrong RDMA segment
size or with the staging pool off, neither of which fails loudly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pydantic
import pytest
from omegaconf import OmegaConf
from pydantic import TypeAdapter

from nemo_rl.data_plane.interfaces import (
    DataPlaneConfig,
    MooncakeCpuConfig,
    SimpleStorageConfig,
    SsdOffloadConfig,
    backend_config,
    data_plane_supports_checkpointing,
)

_BASE = {
    "enabled": True,
    "impl": "transfer_queue",
    "claim_meta_poll_interval_s": 0.5,
}


def _cfg(backend: str, **extra) -> dict:
    return {**_BASE, "backend": backend, **extra}


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("simple", True),
        ("mooncake_cpu", True),
        ("future_backend", False),
    ],
)
def test_checkpointing_capability_defaults_to_unsupported(
    backend: str, expected: bool
) -> None:
    assert data_plane_supports_checkpointing(_cfg(backend)) is expected


def test_nested_block_is_used() -> None:
    cfg = _cfg(
        "mooncake_cpu",
        mooncake_cpu={
            "global_segment_size": 111,
            "reuse_registered_buffers": False,
            "use_gdr": True,
            "gdr_staging_buffer_mb": 512,
        },
    )
    resolved = backend_config(cfg)
    assert isinstance(resolved, MooncakeCpuConfig)
    assert resolved.global_segment_size == 111
    assert resolved.reuse_registered_buffers is False
    assert resolved.use_gdr is True
    assert resolved.gdr_staging_buffer_mb == 512


def test_absent_block_falls_back_to_model_defaults() -> None:
    """The point of the nesting: a config need not mention a backend it isn't using.

    Pins the literals rather than comparing against MooncakeCpuConfig()'s own
    attributes — that would hold for any value the class default was changed
    to and couldn't catch a regression of the sizing itself.
    """
    resolved = backend_config(_cfg("mooncake_cpu"))
    assert resolved.global_segment_size == 68719476736  # 64 GiB per client process
    assert resolved.local_buffer_size == 2147483648  # 2 GiB = 4 x 512 MiB staging slots
    # The opt-out flag defaults on, so omitting it must not disable the pool.
    assert resolved.reuse_registered_buffers is True
    assert resolved.use_gdr is False
    assert resolved.gdr_staging_buffer_mb == 1024


@pytest.mark.parametrize("value", [0, -1])
def test_gdr_staging_size_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(pydantic.ValidationError):
        backend_config(
            _cfg("mooncake_cpu", mooncake_cpu={"gdr_staging_buffer_mb": value})
        )


def test_accepts_an_already_coerced_model() -> None:
    """Configs arriving via pydantic have the block coerced to a model already."""
    cfg = _cfg("mooncake_cpu", mooncake_cpu=MooncakeCpuConfig(global_segment_size=555))
    assert backend_config(cfg).global_segment_size == 555


def test_partial_nested_block_keeps_other_defaults() -> None:
    cfg = _cfg("mooncake_cpu", mooncake_cpu={"local_buffer_size": 7})
    resolved = backend_config(cfg)
    assert resolved.local_buffer_size == 7
    assert resolved.global_segment_size == MooncakeCpuConfig().global_segment_size


def test_mooncake_has_no_backend_specific_checkpoint_knobs() -> None:
    assert {"checkpoint", "hard_pin", "offload"}.isdisjoint(
        MooncakeCpuConfig.model_fields
    )


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("use_gdr", [False, True])
def test_mooncake_checkpoint_mode_is_internal(
    monkeypatch, checkpointing, use_gdr
) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    captured = {}
    monkeypatch.setattr(adapter, "_get_local_node_ip", lambda: "10.0.0.7")
    monkeypatch.setattr(
        adapter,
        "_mooncake_transport_config",
        lambda: {"protocol": "rdma", "device_name": "mlx5_test"},
    )
    monkeypatch.setattr(adapter.os, "chmod", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapter.tq,
        "init",
        lambda *, conf: captured.setdefault("conf", conf),
    )

    cfg = _cfg("mooncake_cpu")
    cfg["mooncake_cpu"] = {"use_gdr": use_gdr, "gdr_staging_buffer_mb": 384}
    adapter._init_tq(cfg, checkpointing=checkpointing)

    conf = OmegaConf.to_container(captured["conf"], resolve=True)
    mooncake = conf["backend"]["MooncakeStore"]
    assert mooncake["hard_pin"] is (True if checkpointing else None)
    # TQ merges its own offload defaults into the resolved backend config.
    assert mooncake["offload"]["enabled"] is False
    assert mooncake["checkpoint"] == {"enabled": checkpointing}
    assert mooncake["use_gdr"] is use_gdr
    assert mooncake["gdr_staging_buffer_mb"] == 384


_MC_SSD = {"enabled": True, "path": "/local/ssd"}


def _mooncake_overlay(
    monkeypatch, mooncake_cpu, checkpointing=False, watch=lambda url, path: None
) -> dict:
    from nemo_rl.data_plane.adapters import mooncake_ssd_offload
    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    # A stub by default: no real thread polling a master that does not exist.
    monkeypatch.setattr(mooncake_ssd_offload, "watch_master_disk_evictions", watch)
    captured = {}
    monkeypatch.setattr(adapter, "_get_local_node_ip", lambda: "10.0.0.7")
    monkeypatch.setattr(adapter, "_mooncake_transport_config", lambda: {})
    monkeypatch.setattr(adapter.os, "chmod", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapter.tq, "init", lambda *, conf: captured.setdefault("conf", conf)
    )
    cfg = _cfg("mooncake_cpu", mooncake_cpu=mooncake_cpu)
    # Storage units are only started under a plan; the overlay ignores the IDs.
    plan = ["a" * 56] if mooncake_cpu.get("storage_unit_segment_size") else None
    adapter._init_tq(cfg, checkpointing=checkpointing, storage_unit_node_ids=plan)
    conf = OmegaConf.to_container(captured["conf"], resolve=True)
    return conf["backend"]["MooncakeStore"]


@pytest.mark.parametrize("unit_segment_size", [1 << 30, 0])
def test_mooncake_ssd_offload_makes_objects_evictable(
    monkeypatch, unit_segment_size
) -> None:
    from transfer_queue.storage.bootstrap import mooncake_bootstrap

    storage = _mooncake_overlay(
        monkeypatch,
        {
            "storage_unit_segment_size": unit_segment_size,
            "ssd_offload": {**_MC_SSD, "offload_threshold": 0.2},
        },
    )
    assert storage["hard_pin"] is False
    # TQ's own offload client would own none of the data; it stays off.
    assert storage["offload"]["enabled"] is False
    extra = mooncake_bootstrap.subprocess.extra_args
    assert "--offload_on_evict=true" in extra
    assert "--offload_force_evict=false" in extra
    assert "--eviction_high_watermark_ratio=0.2" in extra
    assert "--default_kv_lease_ttl=1000" in extra


@pytest.mark.parametrize("ssd_offload", [False, True])
def test_mooncake_ssd_offload_watches_the_master_for_disk_evictions(
    monkeypatch, ssd_offload
) -> None:
    watched = []
    _mooncake_overlay(
        monkeypatch,
        {"ssd_offload": _MC_SSD} if ssd_offload else {},
        watch=lambda url, path: watched.append((url, path)),
    )

    if not ssd_offload:
        assert watched == []
        return
    ((url, path),) = watched
    assert url.startswith("http://") and url.endswith("/metrics")
    assert path == "/local/ssd"


def test_mooncake_without_ssd_offload_keeps_tq_master_flags(monkeypatch) -> None:
    from transfer_queue.storage.bootstrap import mooncake_bootstrap

    storage = _mooncake_overlay(monkeypatch, {})
    assert storage["hard_pin"] is None
    assert mooncake_bootstrap.subprocess.extra_args == ()


def test_mooncake_ssd_offload_is_checkpointable() -> None:
    # Saves read evictable replicas through native GET (tq_mooncake_checkpoint).
    cfg = _cfg(
        "mooncake_cpu",
        mooncake_cpu={"storage_unit_segment_size": 1 << 30, "ssd_offload": _MC_SSD},
    )
    assert data_plane_supports_checkpointing(cfg) is True


@pytest.mark.parametrize("ssd_offload", [False, True])
def test_mooncake_checkpointing_pins_replicas_unless_offloading(
    monkeypatch, ssd_offload
) -> None:
    storage = _mooncake_overlay(
        monkeypatch,
        {"ssd_offload": _MC_SSD} if ssd_offload else {},
        checkpointing=True,
    )
    # Offload must evict; without it, saves borrow hard-pinned memory directly.
    assert storage["hard_pin"] is (not ssd_offload)
    assert storage["offload"]["enabled"] is False


@pytest.mark.parametrize(
    ("mooncake_cpu", "match"),
    [
        ({"storage_unit_segment_size": 1, "ssd_offload": {"enabled": True}}, "path"),
        (
            {
                "storage_unit_segment_size": 1,
                "ssd_offload": {"enabled": True, "path": "rel/ssd"},
            },
            "path",
        ),
        (
            {
                "storage_unit_segment_size": 1,
                "ssd_offload": {**_MC_SSD, "offload_threshold": 0},
            },
            "offload_threshold",
        ),
    ],
)
def test_mooncake_ssd_offload_config_is_checked_at_load(mooncake_cpu, match) -> None:
    with pytest.raises(pydantic.ValidationError, match=match):
        backend_config(_cfg("mooncake_cpu", mooncake_cpu=mooncake_cpu))


def test_simple_backend_nested_block_is_used() -> None:
    cfg = _cfg("simple", simple={"storage_capacity": 7, "num_storage_units": 3})
    resolved = backend_config(cfg)
    assert isinstance(resolved, SimpleStorageConfig)
    assert resolved.storage_capacity == 7
    assert resolved.num_storage_units == 3


def _simple_overlay(monkeypatch, **simple) -> dict:
    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    captured = {}
    monkeypatch.setattr(
        adapter.tq, "init", lambda *, conf: captured.setdefault("conf", conf)
    )
    cfg = _cfg("simple", simple={"num_storage_units": 2, **simple})
    adapter._init_tq(cfg)
    conf = OmegaConf.to_container(captured["conf"], resolve=True)
    return conf["backend"]["SimpleStorage"]


def test_simple_storage_units_skip_nodes_without_cpu(monkeypatch) -> None:
    """A head started with --num-cpus=0 must not get a storage unit.

    TQ places each unit (num_cpus=1) with hard node affinity round-robin over
    alive nodes, so a CPU-less node would leave its unit unschedulable. The
    overlay asks TQ for nodes that advertise CPU; check TQ honors that.
    """
    import ray
    from transfer_queue.storage.bootstrap import simple_storage_bootstrap

    storage = _simple_overlay(monkeypatch)
    assert storage["required_node_resource"] == "CPU"

    cpu_less = {"NodeID": "a" * 56, "Alive": True, "Resources": {"memory": 1.0}}
    worker = {"NodeID": "b" * 56, "Alive": True, "Resources": {"CPU": 8.0}}
    monkeypatch.setattr(ray, "nodes", lambda: [cpu_less, worker])
    strategies = simple_storage_bootstrap.get_node_round_robin_scheduling_strategies(
        3, required_node_resource=storage["required_node_resource"]
    )
    assert [s.node_id for s in strategies] == [worker["NodeID"]] * 3


def test_simple_ssd_offload_is_off_by_default(monkeypatch) -> None:
    # Whatever TQ's own base config carries, we must not turn it on.
    storage = _simple_overlay(monkeypatch)
    assert storage.get("ssd_offload", {}).get("enabled") is not True


def test_simple_ssd_offload_reaches_tq_config(monkeypatch) -> None:
    storage = _simple_overlay(
        monkeypatch,
        ssd_offload={"enabled": True, "path": "/local/ssd", "threshold_bytes": 4096},
    )
    assert storage["ssd_offload"]["enabled"] is True
    assert storage["ssd_offload"]["path"] == "/local/ssd"
    assert storage["ssd_offload"]["threshold_bytes"] == 4096


@pytest.mark.parametrize("path", [None, "", "relative/ssd"])
def test_ssd_offload_requires_absolute_path_when_enabled(path) -> None:
    """Fails at config load, not later inside driver bootstrap."""
    with pytest.raises(pydantic.ValidationError, match="ssd_offload.path"):
        TypeAdapter(DataPlaneConfig).validate_python(
            _cfg(
                "simple",
                simple={
                    "num_storage_units": 2,
                    "ssd_offload": {"enabled": True, "path": path},
                },
            )
        )


def test_ssd_offload_defaults_are_off_and_path_optional_when_disabled() -> None:
    absent = backend_config(_cfg("simple", simple={"num_storage_units": 2}))
    assert absent.ssd_offload == SsdOffloadConfig(
        enabled=False, path=None, threshold_bytes=1048576
    )
    partial = backend_config(
        _cfg(
            "simple",
            simple={"num_storage_units": 2, "ssd_offload": {"path": "/local/ssd"}},
        )
    )
    assert partial.ssd_offload.path == "/local/ssd"
    assert partial.ssd_offload.enabled is False
    assert partial.ssd_offload.threshold_bytes == 1048576


def test_simple_backend_num_storage_units_has_no_default() -> None:
    """No static default is correct across cluster sizes (see the field's
    docstring), so an absent/incomplete simple: block must raise rather than
    silently run at a node count the config never chose."""
    with pytest.raises(pydantic.ValidationError, match="num_storage_units"):
        backend_config(_cfg("simple"))
    with pytest.raises(pydantic.ValidationError, match="num_storage_units"):
        backend_config(_cfg("simple", simple={"storage_capacity": 7}))


def test_only_the_selected_backend_is_read() -> None:
    """A mooncake block must not leak into a simple run, or vice versa."""
    cfg = _cfg(
        "simple",
        simple={"storage_capacity": 5, "num_storage_units": 1},
        mooncake_cpu={"global_segment_size": 999},
    )
    resolved = backend_config(cfg)
    assert isinstance(resolved, SimpleStorageConfig)
    assert not hasattr(resolved, "global_segment_size")


def test_schema_validates_without_any_backend_block() -> None:
    """Regression guard: a required backend key is what broke SingleController CI.

    ``data_plane`` built from scratch — not inherited from the exemplar — must
    validate, otherwise MasterConfig fails before training starts.
    """
    TypeAdapter(DataPlaneConfig).validate_python(_cfg("simple"))


@pytest.fixture
def simple_units(monkeypatch) -> list[tuple[dict, tuple, dict]]:
    """Fake SimpleStorageUnit: records (options, args, kwargs) per unit built.

    Also gives TQ's provider a Ray-free default placement and restores it after
    the test, since _pin_simple_storage_units swaps that hook in place.
    """
    from transfer_queue.storage import simple_storage
    from transfer_queue.storage.bootstrap import simple_storage_bootstrap
    from transfer_queue.utils import zmq_utils

    built: list[tuple[dict, tuple, dict]] = []

    def options(**opts):
        def remote(*args, **kwargs):
            built.append((opts, args, kwargs))
            return opts["name"]

        return SimpleNamespace(remote=remote)

    monkeypatch.setattr(simple_storage.SimpleStorageUnit, "options", options)
    monkeypatch.setattr(
        simple_storage_bootstrap,
        "get_node_round_robin_scheduling_strategies",
        lambda num_actors, required_node_resource=None: [
            SimpleNamespace(node_id="unpinned") for _ in range(num_actors)
        ],
    )
    for module in (simple_storage_bootstrap, zmq_utils):
        monkeypatch.setattr(module, "process_zmq_server_info", lambda h: dict(h))
    return built


def _simple_conf(units: int, total: int):
    return OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": units,
                    "total_storage_size": total,
                },
            }
        }
    )


def test_simple_storage_units_are_pinned_to_the_resolved_nodes(
    monkeypatch, simple_units
) -> None:
    """simple.storage_unit_placement -> one hard node affinity per unit."""
    from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider

    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    monkeypatch.setattr(adapter.tq, "init", lambda *, conf: None)
    cfg = _cfg("simple", simple={"storage_capacity": 9, "num_storage_units": 3})
    n0, n1 = "a" * 56, "b" * 56  # Ray node IDs are 28-byte hex strings
    adapter._init_tq(cfg, storage_unit_node_ids=[n0, n1, n0])
    conf = _simple_conf(3, 9)
    handles = StorageBootstrapProvider.get_provider("SimpleStorage")(conf)

    assert list(handles) == [f"TransferQueueStorageUnit#{i}" for i in range(3)]
    strategies = [opts["scheduling_strategy"] for opts, _, _ in simple_units]
    assert [s.node_id for s in strategies] == [n0, n1, n0]
    assert all(s.soft is False for s in strategies)
    configs = [kw["config"] for _, _, kw in simple_units]
    assert all(set(kw) == {"config"} for _, _, kw in simple_units)
    assert all(
        c["num_data_storage_units"] == 3 and c["total_storage_size"] == 9
        for c in configs
    )
    assert len({c["_run_id"] for c in configs}) == 1  # one run ID shared by all units
    assert conf.backend.SimpleStorage.zmq_info == handles


def test_pinned_simple_provider_matches_tq_except_placement(simple_units) -> None:
    """_pin_simple_storage_units swaps only the placement of TQ's own provider.

    Run both against the same conf and compare everything but the scheduling
    options and the per-run ID, so a TQ bump that changes how units are built
    or placed fails here.
    """
    from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider

    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    placement_keys = {
        "placement_group",
        "placement_group_bundle_index",
        "scheduling_strategy",
    }

    def without_placement(calls):
        return [
            (
                {k: v for k, v in o.items() if k not in placement_keys},
                a,
                {
                    k: {ck: cv for ck, cv in v.items() if ck != "_run_id"}
                    if k == "config"
                    else v
                    for k, v in kw.items()
                },
            )
            for o, a, kw in calls
        ]

    tq_conf = _simple_conf(3, 10)
    tq_handles = StorageBootstrapProvider.get_provider("SimpleStorage")(tq_conf)
    tq_calls = without_placement(simple_units)
    simple_units.clear()

    adapter._pin_simple_storage_units(["a" * 56, "b" * 56, "a" * 56])
    ours_conf = _simple_conf(3, 10)
    ours_handles = StorageBootstrapProvider.get_provider("SimpleStorage")(ours_conf)

    assert ours_handles == tq_handles
    assert without_placement(simple_units) == tq_calls
    assert ours_conf == tq_conf


def test_pinned_simple_units_keep_ssd_offload(monkeypatch, simple_units) -> None:
    """TQ's provider still builds pinned units, so ssd_offload reaches each one."""
    from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider

    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    captured = {}
    monkeypatch.setattr(
        adapter.tq, "init", lambda *, conf: captured.setdefault("conf", conf)
    )
    cfg = _cfg(
        "simple",
        simple={
            "storage_capacity": 9,
            "num_storage_units": 2,
            "ssd_offload": {
                "enabled": True,
                "path": "/local/ssd",
                "threshold_bytes": 4096,
            },
        },
    )
    adapter._init_tq(cfg, storage_unit_node_ids=["a" * 56, "b" * 56])
    StorageBootstrapProvider.get_provider("SimpleStorage")(captured["conf"])

    assert len(simple_units) == 2
    for opts, _, kw in simple_units:
        ssd = kw["config"]["ssd_offload"]
        assert ssd["enabled"] is True
        assert ssd["path"] == "/local/ssd"
        assert ssd["threshold_bytes"] == 4096
        assert opts["runtime_env"]["env_vars"]["MALLOC_MMAP_THRESHOLD_"]


@pytest.mark.parametrize(
    "cfg",
    [
        _cfg(
            "simple",
            simple={"num_storage_units": 2, "storage_unit_placement": ["inference"]},
        ),
        _cfg("mooncake_cpu", mooncake_cpu={"storage_unit_segment_size": 1 << 30}),
    ],
    ids=["simple-placement", "mooncake-units"],
)
def test_storage_unit_settings_need_the_single_controller_plan(
    monkeypatch, cfg
) -> None:
    """Off the SingleController nothing starts or places units: fail at bootstrap.

    Otherwise mooncake runs with no memory owner (every put fails with -200) and
    simple placement is silently ignored.
    """
    from nemo_rl.data_plane.adapters import transfer_queue as adapter

    monkeypatch.setattr(
        adapter.tq, "init", lambda *, conf: pytest.fail("bootstrapped anyway")
    )
    with pytest.raises(ValueError, match="only supported by the SingleController"):
        adapter._init_tq(cfg)
