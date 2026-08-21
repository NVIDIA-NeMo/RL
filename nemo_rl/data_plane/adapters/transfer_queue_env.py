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
"""Mooncake engine environment, configured *before* the engine is imported.

Mooncake snapshots its whole ``MC_*`` configuration as its extension loads, so
these variables only take effect if they are in ``os.environ`` beforehand. That
makes import order load-bearing, and the failure is silent: a late write lands
in ``os.environ`` — where it still reads back correctly — while the engine keeps
the value it captured. On a rail-isolated RoCE fabric that silence costs a run,
with every transfer dying as "transport retry counter exceeded".

This module therefore deliberately imports **nothing** from ``transfer_queue``
or ``mooncake``, so importing it can never be the thing that loads the engine.
Keep it that way — a convenience import here, or one added to this package's
``__init__``, would defeat the whole point. :func:`configure_engine_env` turns
the ordering violation into an error rather than a silent misconfiguration.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_rl.data_plane.interfaces import DataPlaneConfig

# Importing either loads mooncake's extension (transfer_queue's storage.clients
# package eagerly imports mooncake_client, which does `from mooncake.store
# import ...` at module scope), which is the point after which MC_* can no
# longer be configured. Submodules need no entry of their own: importing
# mooncake.store registers the mooncake package first.
_ENGINE_MODULES = ("transfer_queue", "mooncake")


def _engine_already_imported() -> str | None:
    """Return the first engine module already in ``sys.modules``, else None."""
    return next((m for m in _ENGINE_MODULES if m in sys.modules), None)


def rail_link_layers() -> dict[str, str]:
    """Map each mlx5 rail to its port-1 link layer, read from sysfs."""
    layers: dict[str, str] = {}
    for path in sorted(glob.glob("/sys/class/infiniband/mlx5_*/ports/1/link_layer")):
        try:
            layers[Path(path).parents[2].name] = Path(path).read_text().strip()
        except OSError:
            continue
    return layers


def fabric_is_roce_only() -> bool:
    """True when the host has RoCE rails and no InfiniBand.

    Deliberately requires *seeing* InfiniBand to answer False, so an empty or
    unreadable sysfs cannot silently opt a RoCE host out of the pairing hint.
    """
    layers = set(rail_link_layers().values())
    return "Ethernet" in layers and "InfiniBand" not in layers


def nvidia_peermem_loaded() -> bool:
    """True when the ``nvidia_peermem`` kernel module is present.

    Mooncake's GPU-memory registration takes one of two routes and picks by env
    var, not by probing, so this is what decides whether its default can work.
    """
    try:
        with open("/proc/modules") as f:
            return any(line.startswith("nvidia_peermem") for line in f)
    except OSError:
        return False


def _wanted_engine_env(gdr: bool = False) -> dict[str, str]:
    """The engine values this backend needs, for this host's fabric.

    Args:
        gdr: whether any client in this run registers GPU memory.
    """
    # LOCAL_MEMCPY reinterpret_casts cross-process pointers and segfaults
    # MemcpyWorkerPool; upstream PR #1995 fixes it but is not in a published
    # wheel. Drop once the pinned wheel includes it.
    wanted = {"MC_STORE_MEMCPY": "0"}
    if gdr and not nvidia_peermem_loaded():
        # WITH_NVIDIA_PEERMEM (no MC_ prefix) defaults to *true* in the pinned
        # wheel (mooncake-common/src/environ.cpp:216), which registers GPU
        # memory with a plain ibv_reg_mr() and so needs the nvidia_peermem
        # module. Without that module every CUDA registration fails with
        # ERR_CONTEXT (-202) — measured on these GB200/GB300 nodes, where
        # /proc/modules has no nvidia_peermem. Zero selects the DMA-BUF route
        # (cuMemGetHandleForAddressRange), which is the supported path on this
        # generation. Only set when the module is absent, so a host that does
        # have it keeps upstream's behaviour.
        wanted["WITH_NVIDIA_PEERMEM"] = "0"
    if fabric_is_roce_only():
        # Pin each transfer's peer rail to the local one by name. Mooncake
        # otherwise picks the peer independently (Topology::selectDevice), and
        # on RoCE each rail is its own subnet, so a cross-rail pair has no
        # route. Measured on the gb200 CI runners: every cross-rail pair failed,
        # no same-rail pair ever did. InfiniBand routes cross-rail, so it is
        # left alone. Mooncake reads this presence-only (config.cpp:318), so
        # `=0` enables it too — unsetting it is the only way to disable it.
        wanted["MC_ENABLE_DEST_DEVICE_AFFINITY"] = "1"
    return wanted


def configure_engine_env(cfg: DataPlaneConfig) -> None:
    """Set the mooncake knobs that must be identical in every process.

    No-op unless the backend runs the engine (``mooncake_cpu`` or
    ``transfer_engine``); ``simple`` has none. Which knobs are wanted depends on
    the fabric and on whether the run registers GPU memory — see
    :func:`_wanted_engine_env`. Values already present in the environment are
    left alone, so a launcher can override any of them — except
    ``MC_ENABLE_DEST_DEVICE_AFFINITY``, which mooncake reads presence-only, so a
    launcher trying to override it to ``"0"`` enables it instead; unsetting it
    is the only way to disable it.

    Call this before anything imports ``transfer_queue`` or ``mooncake``.
    :func:`nemo_rl.data_plane.factory.maybe_configure_data_plane_env` does, on
    the driver before ``init_ray``, and Ray hands the result to every worker.

    Raises:
        RuntimeError: if a variable still needs setting but the engine is
            already imported, i.e. the value can no longer reach it. Fatal on
            purpose — the alternative is a run that looks configured and is not.
    """
    # Both RDMA backends run on the same engine: register mode offers every
    # rail too, so it needs the same peer-rail pinning. ``simple`` has no engine.
    if cfg["backend"] not in ("mooncake_cpu", "transfer_engine"):
        return

    # Read by hand rather than through interfaces.backend_config: this module
    # keeps its import list empty on purpose (see the module docstring), and a
    # convenience import is exactly how the engine would sneak in later.
    # A GPU-less driver evaluates this too — the flag must be identical in
    # every process, and the workers are where GPU registration happens.
    nested = cfg.get(cfg["backend"]) or {}
    if not isinstance(nested, dict):
        nested = {"use_gdr": getattr(nested, "use_gdr", False)}
    # Register mode puts GPU memory on the wire by default; mooncake_cpu opts in.
    gdr = bool(nested.get("use_gdr", cfg["backend"] == "transfer_engine"))

    missing = {k: v for k, v in _wanted_engine_env(gdr).items() if k not in os.environ}
    if not missing:
        # Already set by the launcher, or by an earlier call in this process.
        return

    imported = _engine_already_imported()
    if imported is not None:
        raise RuntimeError(
            f"mooncake's engine was already imported (via {imported!r}) before "
            f"{sorted(missing)} could be set, so the engine would never see "
            "them: it reads its MC_* configuration once, as the extension "
            "loads. Configure before anything imports transfer_queue or "
            "mooncake, or export the variables in the launcher environment."
        )

    os.environ.update(missing)
