# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""mooncake_master's metrics server must land on a reserved port.

The master binds its metrics socket before it consults
``enable_metric_reporting`` and exits non-zero if that bind fails, so the
``metrics_port`` gflag default (9003) is a port the job depends on whether or
not anything scrapes it — and it is inside the ephemeral range these nodes hand
out as source ports. TransferQueue passes neither ``--metrics_port`` nor a
``--config_path`` file that could carry one, so the only way the port moves is
the argv patch exercised here.

Two things can quietly undo that: TQ launching the master by some other route
(the flag stops landing), and the reservation being dropped so the port is no
longer allocated from the band ray.sub documents. Both are asserted.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DATA_PLANE_PORT_RANGE_HIGH,
    DEFAULT_DATA_PLANE_PORT_RANGE_LOW,
)

mooncake_bootstrap = pytest.importorskip(
    "transfer_queue.storage.bootstrap.mooncake_bootstrap",
    reason="transfer_queue not installed",
)
bootstrap_provider = pytest.importorskip(
    "transfer_queue.storage.bootstrap.provider",
    reason="transfer_queue not installed",
)
StorageBootstrapProvider = bootstrap_provider.StorageBootstrapProvider


@pytest.fixture
def unpatched_tq_bootstrap() -> Iterator[None]:
    """Run the test against a pristine TQ bootstrap, then restore what was found.

    "Not yet patched" cannot be assumed: the patch is process-wide, and
    anything that reaches ``_init_tq`` first installs it -- the session-scoped
    mooncake client in ``conftest.py``, or simply ``test_mooncake_gdr.py``,
    which pytest collects before this file. An already-installed patch sends
    ``_patch_mooncake_master_metrics_port`` down its idempotent early return,
    which repoints the port without re-registering the provider, so the
    installation assertions below would read the previous test's state.
    """
    original_subprocess = mooncake_bootstrap.subprocess
    # The port a session-scoped client reserved has to survive these tests.
    original_port = getattr(original_subprocess, "metrics_port", None)
    original_provider = StorageBootstrapProvider.get_provider("MooncakeStore")
    if isinstance(original_subprocess, tq_adapter._MooncakeMasterArgv):
        # ``_wrapped`` is the real subprocess module, and the module-level
        # ``initialize_mooncake_storage`` is the object register_provider put
        # in the registry, so this pair is the state TQ boots with.
        mooncake_bootstrap.subprocess = original_subprocess._wrapped
        StorageBootstrapProvider._providers["mooncakestore"] = (
            mooncake_bootstrap.initialize_mooncake_storage
        )
    yield
    mooncake_bootstrap.subprocess = original_subprocess
    if original_port is not None:
        original_subprocess.metrics_port = original_port
    # Assigned rather than re-registered: register_provider would wrap the
    # entry a second time and leave the registry holding a different object
    # than the one the next test starts from.
    StorageBootstrapProvider._providers["mooncakestore"] = original_provider


def _recording_subprocess() -> tuple[Any, list]:
    launched: list = []
    return SimpleNamespace(Popen=lambda args, **kwargs: launched.append(args)), launched


def test_master_argv_gains_the_reserved_metrics_port() -> None:
    """The flag TQ omits, appended to the argv TQ builds."""
    stub, launched = _recording_subprocess()
    argv = tq_adapter._MooncakeMasterArgv(stub, 1157)

    argv.Popen(["mooncake_master", "--rpc_port=1151"], stdout=None)

    assert launched == [["mooncake_master", "--rpc_port=1151", "--metrics_port=1157"]]
    assert argv.master_launched


def test_flag_is_appended_so_gflags_takes_it_last() -> None:
    """gflags honours the last occurrence, so appending survives TQ starting
    to pass a metrics port of its own."""
    stub, launched = _recording_subprocess()
    argv = tq_adapter._MooncakeMasterArgv(stub, 1157)

    argv.Popen(["mooncake_master", "--metrics_port=9003"])

    assert launched[0][-1] == "--metrics_port=1157"


def test_binary_is_matched_by_basename() -> None:
    """_init_tq puts the wheel's mooncake dir on PATH, but a future TQ could
    resolve the binary to an absolute path instead."""
    stub, launched = _recording_subprocess()
    argv = tq_adapter._MooncakeMasterArgv(stub, 1157)

    argv.Popen(["/opt/venv/lib/mooncake/mooncake_master", "--rpc_port=1151"])

    assert launched[0][-1] == "--metrics_port=1157"


def test_other_launches_pass_through_untouched() -> None:
    """The offload client takes no metrics_port flag, and seeing it must not
    count as the master having launched."""
    stub, launched = _recording_subprocess()
    argv = tq_adapter._MooncakeMasterArgv(stub, 1157)

    argv.Popen(["mooncake_client", "-port=42052"])

    assert launched == [["mooncake_client", "-port=42052"]]
    assert not argv.master_launched


def test_unknown_attributes_delegate_to_the_real_module() -> None:
    """The bootstrap reaches for subprocess.STDOUT on the same reference."""
    argv = tq_adapter._MooncakeMasterArgv(subprocess, 1157)

    assert argv.STDOUT is subprocess.STDOUT
    assert argv.Popen is not subprocess.Popen


def test_patch_covers_the_registry_entry_that_is_actually_called(
    unpatched_tq_bootstrap,
) -> None:
    """The registry holds a ``functools.wraps`` wrapper closed over the
    original bootstrap, so rebinding the module attribute alone would leave the
    called function unpatched — the same trap ``extract_field_schema`` has."""
    before = StorageBootstrapProvider.get_provider("MooncakeStore")

    tq_adapter._patch_mooncake_master_metrics_port(1157)

    assert StorageBootstrapProvider.get_provider("MooncakeStore") is not before
    assert isinstance(mooncake_bootstrap.subprocess, tq_adapter._MooncakeMasterArgv)


def test_reinstalling_repoints_the_port_without_stacking(
    unpatched_tq_bootstrap,
) -> None:
    """A second init in the same process must get the port it just reserved."""
    tq_adapter._patch_mooncake_master_metrics_port(1157)
    first = mooncake_bootstrap.subprocess

    tq_adapter._patch_mooncake_master_metrics_port(1158)

    assert mooncake_bootstrap.subprocess is first
    assert first.metrics_port == 1158


def test_bootstrap_that_skips_the_wrapped_popen_is_a_drift_error(
    unpatched_tq_bootstrap,
) -> None:
    """If the flag stops landing, the metrics server is silently back on 9003
    inside the ephemeral range — the failure this whole patch exists to avoid.
    """
    StorageBootstrapProvider.register_provider("MooncakeStore")(lambda conf: None)
    tq_adapter._patch_mooncake_master_metrics_port(1157)
    bootstrap = StorageBootstrapProvider.get_provider("MooncakeStore")

    with pytest.raises(RuntimeError, match="without launching mooncake_master"):
        bootstrap(None)


def test_init_tq_reserves_every_master_port_from_the_band(
    tmp_path, monkeypatch, unpatched_tq_bootstrap
) -> None:
    """End state: three distinct in-band ports, one of them carried by argv.

    The metrics port is the one that does not travel in the config TQ reads, so
    a test that only inspected the conf would pass with the flag never applied.
    """
    mooncake_dir = tmp_path / "mooncake"
    mooncake_dir.mkdir()
    mooncake_init = mooncake_dir / "__init__.py"
    mooncake_init.touch()
    (mooncake_dir / "mooncake_master").touch()

    mooncake_module = ModuleType("mooncake")
    mooncake_module.__file__ = str(mooncake_init)
    mooncake_module.__path__ = [str(mooncake_dir)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", ModuleType("mooncake.store"))

    monkeypatch.setattr(tq_adapter.os, "environ", dict(tq_adapter.os.environ))
    monkeypatch.setattr(tq_adapter, "_get_local_node_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(tq_adapter, "rdma_devices", lambda: "mlx5_0")
    captured: dict = {}
    monkeypatch.setattr(
        tq_adapter.tq, "init", lambda *, conf: captured.update(conf=conf)
    )

    tq_adapter._init_tq(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "mooncake_cpu",
            "claim_meta_poll_interval_s": 0.5,
            "mooncake_cpu": {},
        }
    )

    store_cfg = captured["conf"]["backend"]["MooncakeStore"]
    # An inert key: TQ's bootstrap never reads one, which is why the flag goes
    # through the argv instead.
    assert "metrics_port" not in store_cfg
    ports = [
        int(store_cfg["metadata_server"].rsplit(":", 1)[1]),
        int(store_cfg["master_server_address"].rsplit(":", 1)[1]),
        mooncake_bootstrap.subprocess.metrics_port,
    ]
    assert len(set(ports)) == 3, f"the master's servers must differ, got {ports}"
    for port in ports:
        assert (
            DEFAULT_DATA_PLANE_PORT_RANGE_LOW
            <= port
            < DEFAULT_DATA_PLANE_PORT_RANGE_HIGH
        ), f"port {port} escaped the band ray.sub's map documents"
