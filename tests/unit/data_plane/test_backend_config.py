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


def test_simple_backend_nested_block_is_used() -> None:
    cfg = _cfg("simple", simple={"storage_capacity": 7, "num_storage_units": 3})
    resolved = backend_config(cfg)
    assert isinstance(resolved, SimpleStorageConfig)
    assert resolved.storage_capacity == 7
    assert resolved.num_storage_units == 3


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

    Also restores TQ's provider registry after the test, since
    _pin_simple_storage_units replaces the SimpleStorage provider in it.
    """
    from transfer_queue.storage import simple_storage
    from transfer_queue.storage.bootstrap import simple_storage_bootstrap
    from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
    from transfer_queue.utils import zmq_utils

    built: list[tuple[dict, tuple, dict]] = []

    def options(**opts):
        def remote(*args, **kwargs):
            built.append((opts, args, kwargs))
            return opts["name"]

        return SimpleNamespace(remote=remote)

    monkeypatch.setattr(simple_storage.SimpleStorageUnit, "options", options)
    monkeypatch.setattr(
        simple_storage_bootstrap, "get_placement_group", lambda *a, **k: object()
    )
    for module in (simple_storage_bootstrap, zmq_utils):
        monkeypatch.setattr(module, "process_zmq_server_info", lambda h: dict(h))
    monkeypatch.setitem(
        StorageBootstrapProvider._providers,
        "simplestorage",
        StorageBootstrapProvider.get_provider("SimpleStorage"),
    )
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
    assert all(kw == {"storage_unit_size": 3} for _, _, kw in simple_units)
    assert conf.backend.SimpleStorage.zmq_info == handles


def test_pinned_simple_provider_matches_tq_except_placement(simple_units) -> None:
    """_pin_simple_storage_units copies TQ's initialize_simple_storage.

    Run both against the same conf and compare everything but the scheduling
    options, so a TQ bump that changes how units are built fails here. Total
    10 over 3 units also tells ceiling from floor division.
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
            ({k: v for k, v in o.items() if k not in placement_keys}, a, kw)
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
