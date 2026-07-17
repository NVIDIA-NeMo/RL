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
"""Adapter wiring :class:`DataPlaneClient` onto the ``transfer_queue`` package.

Pure plumbing — it owns the TQ controller / client handle and translates
:class:`KVBatchMeta` ↔ TQ's own ``BatchMeta`` / ``KVBatchMeta``. No
business logic. Backend init is lifted from
``rl-arena/arena/backends.py``; the call shapes are lifted from
``rl-arena/arena/dataplane_client.py``.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import time
from importlib import resources
from typing import Any

import torch
import transfer_queue as tq
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import (
    DataPlaneClient,
    DataPlaneConfig,
    KVBatchMeta,
)

# ──────────────────────────────────────────────────────────────────────────
# Backend init — lifted from rl-arena/arena/backends.py.
# ──────────────────────────────────────────────────────────────────────────


def _get_local_node_ip() -> str:
    """Return THIS process's host IP, not the cluster head's.

    Each Ray actor process must use its own node's IP so Mooncake's
    announce address (``MC_TCP_BIND_ADDRESS`` → ``desc.ip_or_host_name``
    in ``transfer_engine_impl.cpp``) is routable cross-node.
    Non-routable addresses are rejected:

    * Link-local (169.254/16, fe80::/10) — ``gethostbyname`` can
      resolve to APIPA on hosts where ``avahi-autoipd`` is active.
    * Loopback (127.0.0.0/8, ::1) — hosts whose ``/etc/hosts`` maps
      the hostname to 127.0.0.1 would otherwise announce an
      unroutable address to Mooncake peers, causing cross-node
      ``connection refused``.
    """
    try:
        ip = socket.gethostbyname(socket.gethostname())
        addr = ipaddress.ip_address(ip)
        if addr.is_link_local or addr.is_loopback:
            return ""
        return ip
    except Exception:
        return ""


def _mooncake_transport_config() -> dict:
    protocol = os.environ.get("MC_MOONCAKE_PROTOCOL", "tcp")
    if protocol != "rdma":
        return {"protocol": "tcp"}
    device = os.environ.get("MC_MOONCAKE_DEVICE", "")
    if not device:
        try:
            out = subprocess.run(
                [
                    "sh",
                    "-c",
                    "for d in /sys/class/infiniband/mlx5_*/ports/1/link_layer; do "
                    "  test -f $d && grep -q Ethernet $d && basename $(dirname $(dirname $d)); "
                    "done | head -1",
                ],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            device = out or ""
        except Exception:
            device = ""
    if device:
        os.environ.setdefault("MC_GID_INDEX", os.environ.get("MC_GID_INDEX", "3"))
    return {"protocol": "rdma", "device_name": device}


def _connect_existing() -> None:
    """Worker-process path: connect this process's client to the Ray cluster.

    Connects to the already-running named controller actor. Mirrors
    rl-arena/arena/dataplane_client.py's `tq.init()` (no args) call.
    """
    tq.init()


def _positive_int_or_none(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


_MOONCAKE_CLIENT_DEFAULTS: tuple[int, int] | None = None
_MOONCAKE_INDEX_REUSE_PATCHED = False
_MOONCAKE_CLEAR_SEMANTICS_PATCHED = False
_MOONCAKE_TQ_ACTOR_SETUP_HOOK = (
    "nemo_rl.data_plane.adapters.transfer_queue._setup_mooncake_tq_actor_hooks"
)


def _mooncake_client_defaults(mooncake_client: Any) -> tuple[int, int]:
    global _MOONCAKE_CLIENT_DEFAULTS
    if _MOONCAKE_CLIENT_DEFAULTS is None:
        _MOONCAKE_CLIENT_DEFAULTS = (
            mooncake_client.BATCH_SIZE_LIMIT,
            mooncake_client.MAX_WORKER_THREADS,
        )
    return _MOONCAKE_CLIENT_DEFAULTS


def _configure_mooncake_client_limits(cfg: DataPlaneConfig) -> None:
    """Optionally throttle Mooncake tensor write bursts in TQ's Python client.

    TransferQueue's Mooncake client defaults to batches of 200 tensor keys and
    four writer threads. That is fine for small per-key tensors, but top-k
    distillation writes can make each key large. These knobs let large-topk
    jobs reduce burst size without forking TransferQueue.
    """
    if cfg["backend"] != "mooncake_cpu":
        return

    from transfer_queue.storage.clients import mooncake_client

    default_batch_limit, default_max_threads = _mooncake_client_defaults(
        mooncake_client
    )
    batch_limit_raw = cfg.get("mooncake_batch_size_limit")
    if batch_limit_raw is None or batch_limit_raw == "":
        batch_limit_raw = os.environ.get("NRL_TQ_MOONCAKE_BATCH_SIZE_LIMIT")
    batch_limit = _positive_int_or_none(
        batch_limit_raw,
        "mooncake_batch_size_limit",
    )
    max_threads_raw = cfg.get("mooncake_max_worker_threads")
    if max_threads_raw is None or max_threads_raw == "":
        max_threads_raw = os.environ.get("NRL_TQ_MOONCAKE_MAX_WORKER_THREADS")
    max_threads = _positive_int_or_none(
        max_threads_raw,
        "mooncake_max_worker_threads",
    )

    mooncake_client.BATCH_SIZE_LIMIT = (
        batch_limit if batch_limit is not None else default_batch_limit
    )
    mooncake_client.MAX_WORKER_THREADS = (
        max_threads if max_threads is not None else default_max_threads
    )


def _disable_mooncake_index_reuse() -> None:
    """Avoid recycling TQ global indexes for Mooncake KV keys.

    TransferQueue's KV storage keys are ``"{global_index}@{field}"``. Its
    controller normally recycles released global indexes, which is fine for
    mutable KV stores. Mooncake logs a warning that upsert/update is not
    supported, and large top-k writeback has failed when a later step reused
    keys such as ``205@teacher_topk_indices``. Keeping indexes monotonic avoids
    writing over recently removed Mooncake keys while preserving sample cleanup.
    """
    global _MOONCAKE_INDEX_REUSE_PATCHED
    if _MOONCAKE_INDEX_REUSE_PATCHED:
        return

    from transfer_queue.controller import PartitionIndexManager

    if getattr(PartitionIndexManager, "_nemo_mooncake_no_reuse", False):
        _MOONCAKE_INDEX_REUSE_PATCHED = True
        return

    def allocate_indexes(self, partition_id, count=1) -> list[int]:
        if count <= 0:
            raise ValueError(
                f"Number of indexes needed must be larger than 0, but got {count}"
            )

        start_index = self.global_index_counter
        end_index = start_index + count
        indexes = list(range(start_index, end_index))

        self.allocated_indexes.update(indexes)
        self.global_index_counter = end_index
        self.partition_to_indexes[partition_id].update(indexes)
        self.reusable_indexes.clear()

        return indexes

    def release_partition(self, partition_id) -> list[int]:
        indexes = self.partition_to_indexes.pop(partition_id, set())
        for idx in indexes:
            self.allocated_indexes.discard(idx)
        self.reusable_indexes.clear()
        return list(indexes)

    def release_indexes(self, partition_id: str, indexes_to_release: list[int]):
        if partition_id not in self.partition_to_indexes:
            return []

        partition_indexes = self.partition_to_indexes[partition_id]
        if not set(indexes_to_release).issubset(partition_indexes):
            raise ValueError(
                "Some indexes to release do not belong to the specified partition."
            )

        partition_indexes.difference_update(indexes_to_release)
        self.allocated_indexes.difference_update(indexes_to_release)
        self.reusable_indexes.clear()

        if not partition_indexes:
            self.partition_to_indexes.pop(partition_id, None)

    PartitionIndexManager.allocate_indexes = allocate_indexes
    PartitionIndexManager.release_partition = release_partition
    PartitionIndexManager.release_indexes = release_indexes
    PartitionIndexManager._nemo_mooncake_no_reuse = True
    _MOONCAKE_INDEX_REUSE_PATCHED = True
    print(
        "Mooncake TQ index reuse disabled: using monotonic global indexes",
        flush=True,
    )


_TQ_RUNTIME_ENV_PATCH_LEVEL = 0


def _resolve_tq_pin() -> str:
    """Return the ``TransferQueue`` requirement string from nemo-rl metadata.

    Single source of truth is ``pyproject.toml`` — we read it back via
    ``importlib.metadata.requires`` so the runtime_env injection cannot
    drift from the dependency declaration.
    """
    from importlib.metadata import requires

    for req in requires("nemo-rl") or []:
        spec = req.split(";")[0].strip()
        if spec.lower().startswith("transferqueue"):
            return spec
    raise RuntimeError(
        "Could not resolve TransferQueue dependency from nemo-rl metadata. "
        "Check pyproject.toml under [project.dependencies]."
    )


def _patch_mooncake_clear_semantics() -> None:
    """Clear Mooncake storage before releasing TQ controller indexes.

    Upstream TQ clears controller metadata first, which immediately returns
    dense global indexes to the reusable pool. For Mooncake, that can make the
    next step write the same ``global_index@field`` key while the previous
    key's storage removal is still in progress. Storage-first clear keeps the
    controller from recycling indexes until Mooncake deletion has completed,
    and raising on failed removes prevents silent top-k payload leaks.
    """
    global _MOONCAKE_CLEAR_SEMANTICS_PATCHED
    if _MOONCAKE_CLEAR_SEMANTICS_PATCHED:
        return

    from transfer_queue.client import AsyncTransferQueueClient, logger
    from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient

    if not getattr(MooncakeStoreClient, "_nemo_raise_on_clear_fail", False):

        def clear(self, keys: list[str], custom_backend_meta=None) -> None:
            ret_codes = self._store.batch_remove(keys, force=True)
            if len(ret_codes) != len(keys):
                raise RuntimeError(
                    f"batch_remove returned {len(ret_codes)} results, "
                    f"expected {len(keys)}"
                )
            failed = [
                (keys[i], ret)
                for i, ret in enumerate(ret_codes)
                if not (ret == 0 or ret == -704)
            ]
            if failed:
                detail = ", ".join(f"{key}:{ret}" for key, ret in failed[:8])
                suffix = "" if len(failed) <= 8 else f", ... +{len(failed) - 8} more"
                raise RuntimeError(f"batch_remove failed for {detail}{suffix}")

        MooncakeStoreClient.clear = clear
        MooncakeStoreClient._nemo_raise_on_clear_fail = True

    if not getattr(AsyncTransferQueueClient, "_nemo_storage_first_clear", False):

        async def async_clear_partition(self, partition_id: str):
            try:
                if not hasattr(self, "storage_manager") or self.storage_manager is None:
                    raise RuntimeError(
                        f"[{self.client_id}]: Storage manager not initialized. "
                        "Call initialize_storage_manager() before performing storage operations."
                    )

                if not self._controller:
                    raise RuntimeError("No controller registered")

                metadata = await self._get_partition_meta(partition_id)

                if not metadata:
                    logger.warning(
                        f"Try to clear an non-exist partition {partition_id}. "
                        "No action will be taken."
                    )
                    return

                await self.storage_manager.clear_data(metadata)
                await self._clear_partition_in_controller(partition_id)

                logger.debug(
                    f"[{self.client_id}]: Clear operation for partition_id "
                    f"{partition_id} completed."
                )
            except Exception as e:
                raise RuntimeError(f"Error in clear operation: {str(e)}") from e

        async def async_clear_samples(self, metadata):
            try:
                if not hasattr(self, "storage_manager") or self.storage_manager is None:
                    raise RuntimeError(
                        f"[{self.client_id}]: Storage manager not initialized. "
                        "Call initialize_storage_manager() before performing storage operations."
                    )

                if metadata.size == 0:
                    logger.warning(
                        f"[{self.client_id}]: Empty BatchMeta provided to clear_samples. "
                        "No action taken."
                    )
                    return

                if not self._controller:
                    raise RuntimeError("No controller registered")

                await self.storage_manager.clear_data(metadata)
                await self._clear_meta_in_controller(metadata)

                logger.debug(
                    f"[{self.client_id}]: Clear operation for batch {metadata} completed."
                )
            except Exception as e:
                raise RuntimeError(f"Error in clear_samples operation: {str(e)}") from e

        AsyncTransferQueueClient.async_clear_partition = async_clear_partition
        AsyncTransferQueueClient.async_clear_samples = async_clear_samples
        AsyncTransferQueueClient._nemo_storage_first_clear = True

    _MOONCAKE_CLEAR_SEMANTICS_PATCHED = True
    print(
        "Mooncake TQ clear semantics patched: storage clear before controller release",
        flush=True,
    )


def _setup_mooncake_tq_actor_hooks() -> None:
    _disable_mooncake_index_reuse()
    _patch_mooncake_clear_semantics()


def _pythonpath_with_repo_root() -> str:
    repo_root = "/opt/nemo-rl"
    current = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if repo_root not in entries:
        entries.insert(0, repo_root)
    return os.pathsep.join(entries)


def _patch_tq_actor_runtime_env(*, patch_mooncake_semantics: bool = False) -> None:
    """Inject a per-actor ``runtime_env`` pin into TQ's actor ``.options()``.

    TQ spawns ``SimpleStorageUnit`` and ``TransferQueueController`` via
    ``Cls.options(...).remote(...)`` without a runtime_env, so they
    inherit the job-level env. In a multi-node container deployment
    where each node has its own ``/opt/nemo_rl_venv``, the driver's
    ``uv sync`` only updates ray-head's venv and a worker-node actor
    fails with ``ModuleNotFoundError``. This monkey-patch makes Ray
    pip-install TQ into a per-actor runtime_env on first spawn (cached
    per-node by Ray afterwards). Idempotent. Couples us to TQ's internal
    class layout — if TQ restructures, this becomes a no-op with a
    logged warning and we fall back to per-node ``uv sync``.

    The pin is sourced from nemo-rl's installed metadata via
    :func:`_resolve_tq_pin` so it cannot drift from ``pyproject.toml``.

    TODO(zhiyul): remove this patch once the nightly container image
    is published with ``TransferQueue`` baked in via ``pyproject.toml``.
    When every node starts from that image, the base env already has TQ
    and Ray actors inherit it — this injection then becomes pure
    overhead (Ray builds a redundant per-actor pip env on top of the
    container's existing TQ install). Drop the call from
    ``TQDataPlaneClient.__init__`` and delete this function.
    """
    global _TQ_RUNTIME_ENV_PATCH_LEVEL
    desired_patch_level = 2 if patch_mooncake_semantics else 1
    if _TQ_RUNTIME_ENV_PATCH_LEVEL >= desired_patch_level:
        return

    runtime_env: dict[str, Any] = {"pip": [_resolve_tq_pin()]}
    if patch_mooncake_semantics:
        runtime_env["env_vars"] = {"PYTHONPATH": _pythonpath_with_repo_root()}
        runtime_env["worker_process_setup_hook"] = _MOONCAKE_TQ_ACTOR_SETUP_HOOK

    def _install(cls) -> bool:
        if not hasattr(cls, "options"):
            return False
        original = cls.options

        def patched(*args, **kwargs):
            kwargs.setdefault("runtime_env", runtime_env)
            return original(*args, **kwargs)

        cls.options = patched  # type: ignore[method-assign]
        return True

    patched_any = False
    try:
        from transfer_queue.storage.simple_backend import SimpleStorageUnit

        patched_any |= _install(SimpleStorageUnit)
    except ImportError:
        pass
    try:
        from transfer_queue.controller import TransferQueueController

        patched_any |= _install(TransferQueueController)
    except ImportError:
        pass

    if not patched_any:
        # Soft-fail: TQ may have moved its actor classes. The driver will
        # still work; multi-node TQ may need the per-node `uv sync` workaround.
        import warnings

        warnings.warn(
            "Could not patch TQ actor classes for runtime_env injection. "
            "Multi-node TQ may fail with ModuleNotFoundError: 'transfer_queue' "
            "on worker nodes. Workaround: run `uv sync` inside each node's "
            "container before the driver runs.",
            RuntimeWarning,
            stacklevel=2,
        )
    if patched_any:
        _TQ_RUNTIME_ENV_PATCH_LEVEL = desired_patch_level


def _init_tq(cfg: DataPlaneConfig) -> None:
    """Driver-process path: bootstrap the TQ controller for the chosen backend."""
    from omegaconf import OmegaConf

    base = OmegaConf.load(str(resources.files("transfer_queue") / "config.yaml"))

    backend = cfg["backend"]
    storage_capacity = cfg["storage_capacity"]
    num_storage_units = cfg["num_storage_units"]

    # polling_mode=True: controller returns empty BatchMeta instead of raising
    # TimeoutError when no samples are ready yet. The client-side blocking
    # loop in `claim_meta` drives the retry cadence.
    controller_overlay = {"controller": {"polling_mode": True}}

    if backend == "simple":
        overlay = {
            **controller_overlay,
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": storage_capacity,
                    "num_data_storage_units": num_storage_units,
                },
            },
        }
    elif backend == "mooncake_cpu":
        # The mooncake-transfer-engine wheel ships `mooncake_master` at
        # <site-packages>/mooncake/, NOT on $PATH. TQ's
        # subprocess.Popen(["mooncake_master", ...]) fails with
        # FileNotFoundError unless we put the package dir on PATH first.
        import mooncake  # type: ignore[import-not-found]

        # TQ's mooncake_client masks any underlying ImportError as
        # "Please install via pip install mooncake-transfer-engine".
        # Force the real cause (e.g. ``libcudart.so.X: cannot open
        # shared object file``) to surface by importing here.
        import mooncake.store  # type: ignore[import-not-found]  # noqa: F401

        _moon_pkg = os.path.dirname(mooncake.__file__)
        _master = os.path.join(_moon_pkg, "mooncake_master")
        try:
            os.chmod(_master, 0o755)
        except OSError as e:
            if not os.access(_master, os.X_OK):
                raise RuntimeError(
                    f"Failed to make {_master} executable: {e}. "
                    f"Mooncake bootstrap requires this binary."
                ) from e
        _existing_path = os.environ.get("PATH", "")
        if _moon_pkg not in _existing_path.split(os.pathsep):
            os.environ["PATH"] = _moon_pkg + os.pathsep + _existing_path
        # Per-process MC_TCP_BIND_ADDRESS / KV-path promotion already
        # set by TQDataPlaneClient.__init__ (runs on every process,
        # including this driver). _init_tq only needs local_ip below
        # for the metadata/master server URLs (driver-bound).
        local_ip = _get_local_node_ip()
        if not local_ip:
            raise RuntimeError(
                "Mooncake backend requires a local node IP; "
                "_get_local_node_ip() returned empty."
            )
        # Mooncake virtual segment / local buffer sizing. Defaults sized
        # for production-scale rollouts (multi-iter DAPO, large
        # message_log object payloads); under-sized values cause
        # ``batch_get_tensor returned None`` once mooncake exhausts its
        # internal allocator headroom. Lazy-mmap'd, so RSS is bounded
        # by actual traffic. Override per-recipe via
        # ``data_plane.global_segment_size`` /
        # ``data_plane.local_buffer_size`` (bytes).
        overlay = {
            **controller_overlay,
            "backend": {
                "storage_backend": "MooncakeStore",
                "MooncakeStore": {
                    "global_segment_size": int(cfg["global_segment_size"]),
                    "local_buffer_size": int(cfg["local_buffer_size"]),
                    # _init_tq runs on the driver only — driver IS the
                    # head, so local_ip here is also the head's IP that
                    # mooncake_master + the metadata server bind to.
                    "metadata_server": f"{local_ip}:50050",
                    "master_server_address": f"{local_ip}:50051",
                    **_mooncake_transport_config(),
                },
            },
        }
    else:
        raise ValueError(f"unknown TQ backend: {backend!r}")

    conf = OmegaConf.merge(base, overlay)

    # Inject runtime_env into TQ's actor spawn so SimpleStorageUnit /
    # TransferQueueController land on workers with transfer_queue available
    # — see _patch_tq_actor_runtime_env() docstring for the why.
    _patch_tq_actor_runtime_env(patch_mooncake_semantics=backend == "mooncake_cpu")

    # pyrefly: ignore  # bad-argument-type
    tq.init(conf=conf)


# ──────────────────────────────────────────────────────────────────────────
# Adapter-level enforcement that nothing but tensors crosses the bus.
# ──────────────────────────────────────────────────────────────────────────


def _assert_no_key_loss(src_dict: dict, new_td: TensorDict, fn: str) -> None:
    """Guard against silent leaf drops through TensorDict constructor rebuild.

    tensordict's constructor has historically dropped NonTensorStack /
    NonTensorData leaves when built from a plain dict. Compare the
    source dict's keys against the rebuilt TD's top-level keys.
    """
    new_keys = set(new_td.keys())
    if set(src_dict.keys()) != new_keys:
        dropped = sorted(set(src_dict.keys()) - new_keys)
        raise RuntimeError(
            f"{fn} lost leaves through TensorDict rebuild: dropped={dropped}."
        )


def _promote_1d_leaves(td: TensorDict) -> TensorDict:
    """Unsqueeze 1D tensor leaves to ``(N, 1)`` — mooncake_cpu KV-path workaround.

    Works around TQ's ``KVStorageManager`` 1D schema/data mismatch;
    :func:`_from_wire` squeezes the trailing 1 back on read. Symmetric
    with `_from_wire` — callers gate on ``self._promote_1d``.
    ``NonTensorStack`` / ``NonTensorData`` leaves pass through.

    Args:
        td: ``TensorDict`` whose 1D tensor leaves should be promoted.

    Returns:
        ``TensorDict`` with 1D tensor leaves unsqueezed to ``(N, 1)``;
        all other leaves pass through unchanged.
    """
    # td.keys() (top-level) includes NonTensorData / NonTensorStack leaves.
    # keys(include_nested=True, leaves_only=True) enumerates tensor leaves
    # only — non-tensor leaves would silently fall out of the rebuilt dict.
    new_dict: dict[str, Any] = {}
    changed = False
    for k in td.keys():
        v = td.get(k)
        if isinstance(v, torch.Tensor) and not v.is_nested and v.dim() == 1:
            new_dict[str(k)] = v.unsqueeze(-1).contiguous()
            changed = True
        else:
            new_dict[str(k)] = v
    if not changed:
        return td
    new_td = TensorDict(new_dict, batch_size=td.batch_size)
    _assert_no_key_loss(new_dict, new_td, "_promote_1d_leaves")
    return new_td


def _from_wire(td: TensorDict) -> TensorDict:
    """Inverse of `_promote_1d_leaves`: squeeze trailing 1 back to (N,)."""
    # Same top-level iteration as `_promote_1d_leaves`: NonTensorData /
    # NonTensorStack leaves are only visible via td.keys(), not leaves_only.
    new_dict: dict[str, Any] = {}
    changed = False
    for k in td.keys():
        v = td.get(k)
        if (
            isinstance(v, torch.Tensor)
            and not v.is_nested
            and v.dim() >= 2
            and v.shape[-1] == 1
        ):
            new_dict[str(k)] = v.squeeze(-1).contiguous()
            changed = True
        else:
            new_dict[str(k)] = v
    if not changed:
        return td
    new_td = TensorDict(new_dict, batch_size=td.batch_size)
    _assert_no_key_loss(new_dict, new_td, "_from_wire")
    return new_td


class TQDataPlaneClient(DataPlaneClient):
    """Adapter façade — maps NeMo-RL calls onto TransferQueue's public API."""

    def __init__(self, cfg: DataPlaneConfig, *, bootstrap: bool = True) -> None:
        """Construct a TQ-backed client.

        Args:
            cfg: data-plane config (backend selection, poll cadence, …).
            bootstrap: True (driver) bootstraps the TQ controller using
                ``cfg``. False (worker) connects this process to an
                already-running named controller actor in the Ray
                cluster — ``cfg`` is then only consulted for client-side
                knobs (poll interval).
        """
        # mooncake_cpu setup must run BEFORE _init_tq / _connect_existing
        # — once tq.init/connect runs, Mooncake's engine.so reads the
        # env vars and they can't be changed. Three per-process knobs
        # needed in EVERY process that builds a TQ client (driver,
        # SyncRolloutActor, every MegatronPolicyWorker rank):
        #   1. MC_TCP_BIND_ADDRESS — Mooncake engine.so writes this into
        #      desc.ip_or_host_name, the address peers receive from the
        #      metadata service. Without it, getifaddrs()[0] picks usb0
        #      (169.254.x APIPA) and peers fail to connect.
        #   2. MC_STORE_MEMCPY=0 — Mooncake LOCAL_MEMCPY fast-path
        #      reinterpret_casts cross-process pointers, segfaulting
        #      MemcpyWorkerPool. PR #1995 (merged 2026-04-30) fixes the
        #      root cause but isn't in any published wheel yet
        #      (mooncake-transfer-engine 0.3.10.post2 was bumped before
        #      that merge). Drop this once the wheel includes the fix.
        #   3. KV-path 1D promotion — works around TQ's
        #      extract_field_schema schema/data mismatch for 1D fields.
        if cfg["backend"] == "mooncake_cpu":
            local_ip = _get_local_node_ip()
            if local_ip:
                # Force-assign per-process: Ray actors inherit env vars
                # from the driver, so a setdefault on the worker would
                # be a no-op and the actor would announce the driver's
                # IP — peers fail with "connection refused".
                os.environ["MC_TCP_BIND_ADDRESS"] = local_ip
            os.environ.setdefault("MC_STORE_MEMCPY", "0")
            _disable_mooncake_index_reuse()
            _patch_mooncake_clear_semantics()

        _configure_mooncake_client_limits(cfg)

        # Workaround for TQ KVStorageManager's 1D-field schema/data
        # mismatch (only `mooncake_cpu` goes through that path; `simple`
        # is unaffected). Writer unsqueezes 1D → (N, 1) on put; reader
        # squeezes the trailing 1 back on get. Drop when upstream TQ
        # unifies the schema/data shapes for 1D fields.
        self._promote_1d = cfg["backend"] == "mooncake_cpu"

        if bootstrap:
            _init_tq(cfg)
        else:
            _connect_existing()
        self._poll_interval_s = cfg["claim_meta_poll_interval_s"]
        self._closed = False

    # ── (A) task-mediated ───────────────────────────────────────────────

    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
        enums: dict[str, list[str]] | None = None,
    ) -> None:
        # Pre-populate ``Partition.field_name_mapping`` with the full
        # field schema by doing a single synchronous placeholder put on
        # the driver before any worker producer/consumer is live for
        # this partition.
        #
        # Why: TQ's controller registers new field names lazily inside
        # ``update_production_status`` (controller.py:538) without a lock,
        # while ``kv_retrieve_meta`` (controller.py:1645) iterates the
        # same dict — interleaved threads raise ``RuntimeError: dictionary
        # changed size during iteration`` and kill the controller's
        # ProcessRequestThread (no try/except around the while-loop).
        # Registering everything from a single driver thread before any
        # client request races with a put removes the trigger entirely.
        if not fields:
            return
        client = tq.get_client()
        dummy_td = TensorDict(
            {f: torch.zeros(1) for f in fields},
            batch_size=[1],
        )
        meta = client.put(data=dummy_td, partition_id=partition_id)
        client.clear_samples(metadata=meta)

    def claim_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta:
        client = tq.get_client()
        deadline = time.time() + max(0.0, timeout_s)
        sampling_config: dict[str, Any] = {}
        if dp_rank is not None:
            sampling_config["dp_rank"] = dp_rank

        while True:
            tq_meta = client.get_meta(
                data_fields=list(required_fields),
                batch_size=int(batch_size),
                partition_id=partition_id,
                task_name=task_name,
                mode="fetch",
                sampling_config=sampling_config,
            )
            if getattr(tq_meta, "size", 0) > 0:
                break
            if not blocking:
                return KVBatchMeta(
                    partition_id=partition_id,
                    task_name=task_name,
                    sample_ids=[],
                    fields=list(required_fields),
                )
            if time.time() >= deadline:
                raise TimeoutError(
                    f"claim_meta(partition={partition_id}, task={task_name}) "
                    f"timed out after {timeout_s}s"
                )
            time.sleep(self._poll_interval_s)

        keys: list[str] = client.kv_retrieve_keys(
            global_indexes=list(tq_meta.global_indexes),
            partition_id=partition_id,
        )

        # Propagate per-key tags. ``sequence_lengths`` is lifted out of
        # the ``input_lengths`` tag if present (kept as a typed list
        # because shard_meta_for_dp reads it directly), but the rest
        # of the tag dict travels through unchanged so consumers can
        # filter on it without fetching data.
        tags = list(tq_meta.custom_meta) if tq_meta.custom_meta else [{} for _ in keys]
        seqlens: list[int] | None = None
        if tags and any("input_lengths" in t for t in tags):
            seqlens = [int(t.get("input_lengths", 0)) for t in tags]

        return KVBatchMeta(
            partition_id=partition_id,
            task_name=task_name,
            sample_ids=keys,
            fields=list(required_fields),
            sequence_lengths=seqlens,
            tags=tags if tags else None,
        )

    def get_data(
        self,
        meta: KVBatchMeta,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        fields = select_fields if select_fields is not None else meta.fields
        if fields is None:
            raise ValueError(
                "get_data requires either select_fields or meta.fields; "
                "silently fetching all fields is forbidden."
            )
        return self.get_samples(meta.sample_ids, meta.partition_id, list(fields))

    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool:
        client = tq.get_client()
        for t in task_names:
            if not client.check_consumption_status(
                task_name=t, partition_id=partition_id
            ):
                return False
        return True

    # ── (B) direct-by-key ──────────────────────────────────────────────

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        if not sample_ids:
            return KVBatchMeta(
                partition_id=partition_id, task_name=None, sample_ids=[], fields=None
            )
        if tags is None:
            tags = [{} for _ in sample_ids]

        wire_fields: TensorDict | None = None
        field_names: list[str] | None = None
        if fields is not None:
            # No ``.contiguous()``: under tensordict==0.12.2 it strips
            # non-tensor leaves (NonTensorStack stored as LinkedList) to empty
            # TDs. TQ's encoder forces ``.contiguous()`` per tensor leaf
            # itself, so the call here was redundant for tensors and
            # destructive for non-tensors.
            wire_fields = fields.detach()  # type: ignore[bad-assignment,missing-argument]
            if self._promote_1d:
                wire_fields = _promote_1d_leaves(wire_fields)  # type: ignore[bad-argument-type]
            field_names = list(wire_fields.keys())

        # TQ's wire vocabulary is `keys=` — translation point.
        tq.kv_batch_put(
            keys=list(sample_ids),
            partition_id=partition_id,
            fields=wire_fields,
            tags=tags,
        )

        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            sample_ids=list(sample_ids),
            fields=field_names,
            tags=[dict(t) for t in tags] if tags else None,
        )

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        if not sample_ids:
            return TensorDict({}, batch_size=(0,))
        # TQ's wire vocabulary is `keys=` — translation point.
        td = tq.kv_batch_get(
            keys=list(sample_ids),
            partition_id=partition_id,
            select_fields=select_fields,
        )
        if self._promote_1d:
            td = _from_wire(td)
        return td

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        cleared_via_none = sample_ids is None
        if sample_ids is None:
            # No local state — ask TQ's controller for the current key
            # set in this partition. ``kv_list`` errors propagate; we
            # don't want a network blip to silently turn into "cleared
            # nothing".
            listing = tq.kv_list(partition_id=partition_id)
            sample_ids = list(listing.get(partition_id, {}).keys())
        if not sample_ids:
            if cleared_via_none:
                import warnings

                warnings.warn(
                    f"clear_samples(sample_ids=None, partition_id={partition_id!r}) "
                    "found nothing to clear — TQ's kv_list returned no keys for "
                    "this partition. The partition may already be empty, never "
                    "have been written to, or be unknown to the controller. "
                    "Callers that hold a ``KVBatchMeta`` should pass its "
                    "``sample_ids`` explicitly for a deterministic clear.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return
        # TQ's wire vocabulary is `keys=` — translation point.
        tq.kv_clear(keys=list(sample_ids), partition_id=partition_id)

    # ── (C) lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            tq.close()
        except Exception:
            pass
