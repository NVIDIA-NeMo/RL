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

import contextlib
import glob
import importlib
import ipaddress
import json
import os
import resource
import socket
import threading
import time
import warnings
import weakref
from importlib import resources
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, cast

import torch

# Loading this loads mooncake, which snapshots MC_* on the way in. Configure the
# engine before this import — see nemo_rl.data_plane.adapters.transfer_queue_env.
import transfer_queue as tq
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.transfer_queue_env import rail_link_layers
from nemo_rl.data_plane.interfaces import (
    DataPlaneClient,
    DataPlaneConfig,
    KVBatchMeta,
    backend_config,
    data_plane_supports_checkpointing,
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


def rdma_devices() -> str:
    """Return this host's RDMA devices as mooncake's comma-separated list.

    ``MC_MOONCAKE_DEVICE`` wins and is passed through verbatim (one device or
    a list). Otherwise every rail is offered: the NICs are split across NUMA
    domains, so naming only one device makes the other domain's ranks cross
    the socket on every transfer.

    Offering every rail is only safe because
    ``MC_ENABLE_DEST_DEVICE_AFFINITY`` pins each transfer's peer rail to the
    local one by name, so a cross-rail pair is never formed — see
    :mod:`nemo_rl.data_plane.adapters.transfer_queue_env`. Without it, on a
    fabric where each rail is its own subnet, a cross-rail draw has no route and
    dies with "transport retry counter exceeded".

    IB and RoCE are never mixed; InfiniBand is preferred when present.

    Also the skip predicate for the mooncake tests — ``mooncake_cpu`` is
    RDMA-only, so they cannot run without a device.
    """
    override = os.environ.get("MC_MOONCAKE_DEVICE", "")
    if override:
        return override
    # sysfs lists devices the kernel knows about; libibverbs can only open the
    # ones exposed as /dev/infiniband/uverbs*. Containers routinely have the
    # former without the latter, where mooncake fails with "No available RNIC"
    # well after setup has begun — so treat a missing verbs node as no device.
    if not glob.glob("/dev/infiniband/uverbs*"):
        return ""
    layers = rail_link_layers()
    ib = [n for n, layer in layers.items() if layer == "InfiniBand"]
    roce = [n for n, layer in layers.items() if layer == "Ethernet"]
    # No space after the comma: mooncake splits on "," only.
    return ",".join(ib or roce)


def _mooncake_transport_config() -> dict:
    # mooncake_cpu exists for the zero-copy RDMA MooncakeStore path (TQ v0.1.8),
    # so RDMA is the only transport it runs: there is no TCP fallback, and a
    # host without an RDMA device fails here rather than quietly degrading.
    # Runs on the driver only, so it assumes homogeneous nodes — the device it
    # finds is broadcast to every client.
    devices = rdma_devices()
    if not devices:
        raise RuntimeError(
            "data_plane.backend='mooncake_cpu' requires RDMA, but no usable "
            "mlx5 device was found. Check that /dev/infiniband/uverbs* exists "
            "(a container does not inherit it from the host even though it "
            "does see /sys/class/infiniband) — name a device with "
            "MC_MOONCAKE_DEVICE=<dev>, or use data_plane.backend='simple'."
        )
    return {"protocol": "rdma", "device_name": devices}


# A slot is held for exactly one transfer, so waiting minutes for one means
# this process runs more concurrent transfers than the pool has slots — not
# that a transfer is slow. Fail with that diagnosis rather than block forever.
_STAGING_SLOT_TIMEOUT_S = 600.0


def _memlock_limit() -> str:
    """Return this process's RLIMIT_MEMLOCK soft limit, for error messages."""
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return "unlimited" if soft == resource.RLIM_INFINITY else f"{soft} bytes"


def _register_checked(store: Any, ptr: int, nbytes: int) -> None:
    """``store.register_buffer`` with its status actually checked.

    Mooncake returns a status int here, and TQ drops it at every call site
    (``mooncake_client.py``'s ``_register_all_buffers``). A registration that
    fails is then invisible: the transfer into that unmapped region comes
    back as the generic ``TRANSFER_FAIL`` (-800), which carries no root
    cause, and burns its three retries against the same unmapped memory.
    Registration pins pages with ``ibv_reg_mr`` once per RDMA rail, so it is
    exactly the call that a memlock rlimit or a missing ``IPC_LOCK`` breaks.

    ``None`` counts as success — the binding's return type has varied across
    mooncake wheels, so only an explicit non-zero status is a failure.
    """
    status = store.register_buffer(ptr, nbytes)
    if status is not None and status != 0:
        raise RuntimeError(
            f"mooncake register_buffer(0x{ptr:x}, {nbytes} bytes) failed with "
            f"status {status}. Registration pins the pages with ibv_reg_mr "
            f"once per rail (devices={rdma_devices() or 'none'}), so it needs "
            f"IPC_LOCK and a high memlock rlimit — RLIMIT_MEMLOCK is "
            f"{_memlock_limit()} here. Lower data_plane.mooncake_cpu.global_segment_size / local_buffer_size if the limit is the bound."
        )


class _StagingPool:
    """RDMA-registered host buffers, owned by one mooncake client.

    Not thread-local: the ``ThreadPoolExecutor`` is rebuilt inside each
    get/put, so thread-local buffers would be discarded every call. Sized to
    the executor width so no worker normally waits for a slot.

    A slot's buffer is registered for as long as the pool holds it. The
    invariant that matters is the converse: **no buffer is ever freed while
    still registered**, because mooncake would keep a mapping over an address
    the allocator immediately hands to the next caller.
    """

    def __init__(self, store: Any, n_slots: int, max_bytes: int) -> None:
        self._store = store
        self._free: SimpleQueue = SimpleQueue()
        for _ in range(n_slots):
            self._free.put(None)  # allocated on first use
        self._n_slots = n_slots
        self._max_bytes = max_bytes

    @contextlib.contextmanager
    def buffer(self, nbytes: int):
        # Outliers bypass the pool: slots only ever grow, so admitting one
        # long-sequence sample would pin that size in every slot for the
        # rest of the run. Registering it transiently is the cheaper trade.
        if nbytes > self._max_bytes:
            tmp = torch.empty(nbytes, dtype=torch.uint8)
            _register_checked(self._store, tmp.data_ptr(), tmp.nbytes)
            try:
                yield tmp
            finally:
                self._store.unregister_buffer(tmp.data_ptr())
            return
        try:
            buf = self._free.get(timeout=_STAGING_SLOT_TIMEOUT_S)
        except Empty:
            raise RuntimeError(
                f"No mooncake staging slot free after {_STAGING_SLOT_TIMEOUT_S}s. "
                f"The pool has {self._n_slots} slots, sized to one TQ worker "
                "pool, so this means overlapping put/get calls in this process. "
                "Set data_plane.mooncake_cpu.reuse_registered_buffers=false to "
                "fall back to upstream's per-call registration."
            ) from None
        try:
            if buf is None or buf.nbytes < nbytes:
                if buf is not None:
                    status = self._store.unregister_buffer(buf.data_ptr())
                    if status is not None and status != 0:
                        # Dropping it now would hand memory the NIC may still
                        # map back to the allocator — see _register_checked.
                        raise RuntimeError(
                            f"mooncake unregister_buffer(0x{buf.data_ptr():x}) "
                            f"failed with status {status}; refusing to free a "
                            "buffer that may still be registered."
                        )
                    # Empty the slot before allocating: if the registration
                    # below fails, the slot must come back empty rather than
                    # holding a buffer the NIC no longer maps.
                    buf = None
                grown = torch.empty(nbytes, dtype=torch.uint8)
                _register_checked(self._store, grown.data_ptr(), grown.nbytes)
                buf = grown
            yield buf
        finally:
            self._free.put(buf)


class _StagingPoolRegistry:
    """Owns each client's staging pool, keyed weakly so it dies with the client.

    Weak keys because the registry is reachable from the patched class for the
    process lifetime; a strong table would pin every client's registered
    buffers for that long.
    """

    def __init__(self, n_slots: int, max_bytes: int) -> None:
        self._n_slots = n_slots
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._pools: weakref.WeakKeyDictionary[Any, _StagingPool] = (
            weakref.WeakKeyDictionary()
        )

    def pool_for(self, client: Any) -> _StagingPool:
        """Return ``client``'s pool, building it at most once across threads.

        Locked because ``put``/``get`` drive the thread workers from a
        ``ThreadPoolExecutor``, so two of them reach a cold client at once
        whenever a call splits into more than one ``BATCH_SIZE_LIMIT`` batch.
        Unsynchronized, the loser's pool is dropped on the floor and its buffers
        are freed while still registered — see :func:`_register_checked` for why
        that surfaces as a bare ``TRANSFER_FAIL``. The lock is taken on every
        lookup rather than double-checked: it is uncontended after the first
        transfer, and nanoseconds against a millisecond RDMA transfer is not
        worth reasoning about visibility.
        """
        with self._lock:
            pool = self._pools.get(client)
            if pool is None:
                pool = self._pools[client] = _StagingPool(
                    client._store, self._n_slots, self._max_bytes
                )
            return pool


def _tq_shape_drift_error(
    missing: str, consequence: str, target: str, *, opt_out: bool = False
) -> RuntimeError:
    """Build the error shared by the monkey-patch shape guards below.

    All three guards fire for the same reason — the pinned ``transfer_queue``
    revision no longer has the internals a patch depends on — so they share
    this message shape rather than each hand-rolling it.
    """
    remedy = f"re-point the patch at the new {target}"
    if opt_out:
        remedy += (
            ", or set data_plane.mooncake_cpu.reuse_registered_buffers=false "
            "to run on upstream's per-call registration deliberately"
        )
    return RuntimeError(
        f"transfer_queue's {missing}, so {consequence}. The TQ pin in "
        f"pyproject.toml has moved: {remedy}."
    )


def _patch_mooncake_register_check() -> None:
    """Make a failed RDMA registration fail at the registration.

    Upstream's ``_register_all_buffers`` ignores ``register_buffer``'s
    status, so every worker that uses it — including the two bytes workers
    :func:`_patch_mooncake_staging_buffers` leaves alone — transfers into
    memory the NIC may never have mapped and reports only ``TRANSFER_FAIL``
    (-800). Applied for every ``mooncake_cpu`` client, independent of
    ``reuse_registered_buffers``, so the check survives disabling the pool.

    Raises if ``_register_all_buffers`` is missing, rather than returning
    early: this check has no ``reuse_registered_buffers``-style opt-out, so
    silently skipping it would put a failed registration back to surfacing
    only as a bare ``TRANSFER_FAIL``, which is exactly the diagnosability
    this patch exists to add.
    """
    try:
        from transfer_queue.storage.clients import mooncake_client as _mc
    except ImportError:
        return

    cls = getattr(_mc, "MooncakeStoreClient", None)
    if cls is None or getattr(cls, "_nrl_register_checked", False):
        return
    if not hasattr(cls, "_register_all_buffers"):
        raise _tq_shape_drift_error(
            "MooncakeStoreClient no longer has _register_all_buffers",
            "a failed RDMA registration would go back to surfacing only as "
            "a bare TRANSFER_FAIL (-800) with no root cause",
            "call site",
        )

    def _register_all_buffers(self, ptrs, sizes):  # type: ignore[no-untyped-def]
        for ptr, size in zip(ptrs, sizes, strict=True):
            _register_checked(self._store, ptr, size)

    cls._register_all_buffers = _register_all_buffers
    cls._nrl_register_checked = True


def _assert_tq_stores_scalar_rows_0d() -> None:
    """Confirm a dense 1-D field really is stored as 0-d rows.

    :func:`_patch_scalar_field_schema` rewrites the reported sample shape to
    ``()`` on that premise, and nothing reshapes the payload to compensate
    any more. If a TQ revision started storing 1-D fields as ``(1,)`` rows
    instead — fixing the same bug from the other side — the rewrite would
    turn a correct schema into a wrong one, and the symptom would be
    corrupt reads rather than an import error.

    So ask TQ directly rather than trusting the pin.

    Raises rather than skipping when the storage module is gone: the caller
    reached here only after importing ``transfer_queue.metadata``, so "TQ isn't
    installed" is no longer a live explanation — a missing module means the
    layout moved, which is exactly what this guard exists to catch.
    """
    try:
        from transfer_queue.storage.managers.base import KVStorageManager
    except ImportError as e:
        raise _tq_shape_drift_error(
            "storage.managers.base is no longer importable",
            "the dense-1-D storage layout the scalar schema patch assumes "
            "cannot be verified, and a wrong assumption corrupts reads",
            "probe",
        ) from e

    generate = getattr(KVStorageManager, "_generate_values", None)
    if generate is None:
        raise _tq_shape_drift_error(
            "KVStorageManager no longer has _generate_values",
            "the dense-1-D storage layout the scalar schema patch assumes "
            "cannot be verified, and a wrong assumption corrupts reads",
            "probe",
        )

    probe = TensorDict({"_nrl_probe": torch.zeros(2)}, batch_size=[2])
    rows = generate(probe)
    if len(rows) != 2 or any(getattr(r, "ndim", None) != 0 for r in rows):
        shapes = [tuple(getattr(r, "shape", ())) for r in rows]
        raise _tq_shape_drift_error(
            "a dense 1-D field no longer stores as 0-d rows "
            f"(probe yielded {len(rows)} rows with shapes {shapes})",
            "rewriting the reported sample shape to () would now disagree "
            "with the stored rows and corrupt scalar columns",
            "patch (it may simply be unnecessary — check whether upstream "
            "fixed extract_field_schema)",
        )


def _patch_scalar_field_schema() -> None:
    """Report the true ``()`` sample shape for dense 1-D fields.

    Upstream ``transfer_queue.metadata.extract_field_schema`` rebinds a
    *local* for 1-D inputs::

        if len(value.shape) == 1:
            value = value.unsqueeze(-1)     # local only
        first_item = value[0]               # -> shape (1,)

    but the value that reaches storage is the original ``(N,)`` tensor,
    which ``KVStorageManager._generate_values`` iterates into ``N`` **0-d**
    rows. So the schema claims a per-sample shape of ``(1,)`` while the
    stored rows are ``()``.

    Only the KV path notices. ``BatchMeta.get_shapes`` repeats the uniform
    ``shape`` per sample for non-nested fields, and ``KVStorageManager``
    hands that list to the client, which reshapes raw bytes with it — so
    a scalar column reconstructs as ``(1,)`` rows and
    ``_merge_tensors_to_tensordict`` then re-nests it instead of taking
    its ``all(dim() == 0) -> torch.stack`` branch. ``SimpleStorage``
    fetches stored objects by ``(index, field)`` and never consults the
    schema, which is why the symptom is ``mooncake_cpu``-only.

    Byte counts are unaffected either way (``prod(()) == prod((1,)) == 1``);
    this is a reshape/dtype-of-container bug, not a sizing one.

    Applied on every backend so one partition's schema cannot disagree with
    itself across processes. There is no payload-side fallback, so the
    premise is verified against TQ itself before the patch is installed —
    see :func:`_assert_tq_stores_scalar_rows_0d`.
    """
    try:
        from transfer_queue import metadata as _md
    except ImportError:
        return
    if getattr(_md, "_nrl_scalar_schema_patched", False):
        return

    orig = getattr(_md, "extract_field_schema", None)
    if orig is None:
        raise _tq_shape_drift_error(
            "metadata module no longer exposes extract_field_schema",
            "dense 1-D fields would keep reporting a (1,) sample shape and "
            "reconstruct as nested (1,) rows on the KV path",
            "function",
        )

    _assert_tq_stores_scalar_rows_0d()

    # Bound to a fresh name after the ``None`` check: a type checker does not
    # carry narrowing of ``orig`` into the closure below, since a closure can
    # run after its captured names change.
    upstream = orig

    def extract_field_schema(data):  # type: ignore[no-untyped-def]
        schema = upstream(data)
        for name in data.keys():
            value = data.get(name)
            if (
                isinstance(value, torch.Tensor)
                and not value.is_nested
                and value.dim() == 1
                and str(name) in schema
            ):
                # ``_generate_values`` iterates this into 0-d rows; say so.
                schema[str(name)]["shape"] = torch.Size([])
        return schema

    # Both storage managers bound the name at import time
    # (``from transfer_queue.metadata import extract_field_schema``), so
    # rebinding only the defining module would leave them on the original.
    _md.extract_field_schema = extract_field_schema
    for mod_path in (
        "transfer_queue.storage.managers.base",
        "transfer_queue.storage.managers.simple_storage_manager",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        if hasattr(mod, "extract_field_schema"):
            mod.extract_field_schema = extract_field_schema
    _md._nrl_scalar_schema_patched = True


def _patch_mooncake_staging_buffers(max_bytes: int) -> None:
    """Reuse RDMA-registered host buffers for mooncake tensor GETs and PUTs.

    Upstream's thread workers allocate a fresh destination per call and
    register/unregister it on the critical path. Pinning pages for DMA costs
    several times the wire time for the same bytes, and because the buffers
    are freed each call the pointers are always new, so nothing can be
    cached. This keeps a small pool of registered buffers alive instead.

    Monkey-patched because TransferQueue is pinned by git SHA in
    ``pyproject.toml``. Raises if the internals it drives are not shaped as
    expected, rather than returning early: a silent return would leave
    ``reuse_registered_buffers: true`` reading as on while the pool is
    never built, with no symptom besides lost throughput.
    """
    try:
        from transfer_queue.storage.clients import mooncake_client as _mc
        from transfer_queue.utils.mooncake_utils import _aligned_offsets, split_by_bytes
        from transfer_queue.utils.tensor_utils import get_nbytes
    except ImportError:
        return

    cls = getattr(_mc, "MooncakeStoreClient", None)
    if cls is None or getattr(cls, "_nrl_staging_patched", False):
        return
    if not all(
        hasattr(cls, a)
        for a in (
            "_get_tensors_thread_worker",
            "_batch_get_into_with_retry",
            "_put_tensors_thread_worker",
            "_batch_upsert_with_retry",
        )
    ):
        raise _tq_shape_drift_error(
            "MooncakeStoreClient no longer has the methods the staging pool patches",
            "reuse_registered_buffers cannot be honoured and every transfer "
            "would silently re-register its buffers",
            "call sites",
            opt_out=True,
        )

    _n_slots_raw = getattr(_mc, "MAX_BATCH_WORKER_THREADS", None)
    if not isinstance(_n_slots_raw, int):
        raise _tq_shape_drift_error(
            "mooncake_client module no longer exposes MAX_BATCH_WORKER_THREADS "
            "as an int",
            "the staging pool cannot be sized and reuse_registered_buffers "
            "cannot be honoured",
            "constant",
            opt_out=True,
        )
    n_slots: int = _n_slots_raw
    registry = _StagingPoolRegistry(n_slots, max_bytes)

    def _get_tensors_thread_worker(
        self, batch_keys, batch_shapes, batch_dtypes, indexes
    ):  # type: ignore[no-untyped-def]
        pool = registry.pool_for(self)
        batch_nbytes = get_nbytes(batch_dtypes, batch_shapes)
        tensors: list[Any] = [None] * len(batch_keys)
        # Split the payload to fit a bounded buffer rather than sizing the
        # buffer to the payload — this is what keeps the pool footprint fixed.
        for idxs in split_by_bytes(batch_nbytes, max_bytes):
            g_keys = [batch_keys[i] for i in idxs]
            g_nbytes = [batch_nbytes[i] for i in idxs]
            offsets, total = _aligned_offsets(g_nbytes)
            with pool.buffer(total) as buf:
                base = buf.data_ptr()
                self._batch_get_into_with_retry(
                    g_keys, [base + off for off in offsets], g_nbytes
                )
                # Clone: the buffer is reused by the next group and next call.
                for pos, off, nb in zip(idxs, offsets, g_nbytes, strict=True):
                    tensors[pos] = (
                        buf[off : off + nb]
                        .view(batch_dtypes[pos])
                        .reshape(tuple(batch_shapes[pos]))
                        .clone()
                    )
        return tensors, indexes

    def _put_tensors_thread_worker(self, batch_keys, batch_tensors):  # type: ignore[no-untyped-def]
        """PUT direction of the GET patch: stage into the pooled buffer, then transfer."""
        pool = registry.pool_for(self)
        contiguous = [t.contiguous() for t in batch_tensors]
        nbytes = [t.nbytes for t in contiguous]
        for idxs in split_by_bytes(nbytes, max_bytes):
            g_nbytes = [nbytes[i] for i in idxs]
            offsets, total = _aligned_offsets(g_nbytes)
            with pool.buffer(total) as buf:
                base = buf.data_ptr()
                for i, off, nb in zip(idxs, offsets, g_nbytes, strict=True):
                    # reshape(-1) before view(uint8): view() resizes the last
                    # dim and raises on the 0-d scalars real payloads carry.
                    buf[off : off + nb].copy_(
                        contiguous[i].reshape(-1).view(torch.uint8)
                    )
                self._batch_upsert_with_retry(
                    [batch_keys[i] for i in idxs],
                    [base + off for off in offsets],
                    g_nbytes,
                )

    cls._get_tensors_thread_worker = _get_tensors_thread_worker
    cls._put_tensors_thread_worker = _put_tensors_thread_worker
    cls._nrl_staging_patched = True


def _connect_existing() -> None:
    """Worker-process path: connect this process's client to the Ray cluster.

    Connects to the already-running named controller actor. Mirrors
    rl-arena/arena/dataplane_client.py's `tq.init()` (no args) call.
    """
    tq.init()


def _init_tq(cfg: DataPlaneConfig) -> None:
    """Driver-process path: bootstrap the TQ controller for the chosen backend."""
    from omegaconf import OmegaConf

    base = OmegaConf.load(str(resources.files("transfer_queue") / "config.yaml"))

    backend = cfg["backend"]

    # polling_mode=True: controller returns empty BatchMeta instead of raising
    # TimeoutError when no samples are ready yet. The client-side blocking
    # loop in `claim_meta` drives the retry cadence.
    controller_overlay = {"controller": {"polling_mode": True}}

    if backend == "simple":
        # Resolved here, not above: MooncakeStore has no unit count and no
        # sample cap — it sizes from global_segment_size/local_buffer_size —
        # so both keys are SimpleStorage-only and reading them at the top
        # implied otherwise.
        simple_cfg = backend_config(cfg)
        overlay = {
            **controller_overlay,
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": simple_cfg.storage_capacity,
                    "num_data_storage_units": simple_cfg.num_storage_units,
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
        # Sizes are per client process and RDMA-pinned — see MooncakeCpuConfig
        # in nemo_rl/data_plane/interfaces.py for the per-node arithmetic.
        mooncake_cfg = backend_config(cfg)
        overlay = {
            **controller_overlay,
            "backend": {
                "storage_backend": "MooncakeStore",
                "MooncakeStore": {
                    "global_segment_size": int(mooncake_cfg.global_segment_size),
                    "local_buffer_size": int(mooncake_cfg.local_buffer_size),
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


def _from_wire(td: TensorDict) -> TensorDict:
    """Densify uniform nested tensors coming back from TQ.

    Both storage managers reconstruct every non-scalar field as a nested
    tensor, including fields whose rows all share a shape. Densify those so
    regular batched inputs retain their dense representation; truly ragged
    fields stay nested.

    Per-sample scalar columns need no handling here: with
    :func:`_patch_scalar_field_schema` applied they are stored and reported
    as 0-d rows, which ``_merge_tensors_to_tensordict`` stacks into a dense
    ``(N,)`` column before it ever reaches this function.

    Packed multimodal fields are excluded: their rows are per-sample media,
    not a padded sequence, and "all rows share a shape" is a data-dependent
    accident (every sample happening to carry one image). Stacking them
    discards the row boundaries that ``PackedTensor.from_wire`` needs, and
    the dense value then fails the ``is_nested`` check in
    ``codec.materialize`` and reaches ``get_multimodal_dict`` unreassembled.
    ``codec.materialize`` applies the same exclusion.
    """
    # NonTensorData / NonTensorStack leaves are only visible via td.keys(),
    # not keys(leaves_only=True) — iterating the latter would silently drop
    # them from the rebuilt dict.
    # Deferred: ``multimodal_utils`` pulls PIL, requests and a few hundred
    # transformers submodules, and this adapter is imported by every process
    # that constructs a TQ client. ``codec.materialize`` defers the same import
    # for the same reason.
    from nemo_rl.data.multimodal_utils import PACKED_MULTIMODAL_FIELDS

    new_dict: dict[str, Any] = {}
    changed = False
    for k in td.keys():
        v = td.get(k)
        field_name = str(k)
        if (
            isinstance(v, torch.Tensor)
            and v.is_nested
            and field_name not in PACKED_MULTIMODAL_FIELDS
        ):
            rows = list(v.unbind())
            if rows and all(row.shape == rows[0].shape for row in rows[1:]):
                v = torch.stack(rows)
                changed = True
        new_dict[field_name] = v
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
        # env vars and they can't be changed. Two per-process knobs are
        # needed in EVERY process that builds a TQ client (driver,
        # SyncRolloutActor, every MegatronPolicyWorker rank):
        #   1. MC_TCP_BIND_ADDRESS — Mooncake engine.so writes this into
        #      desc.ip_or_host_name, the address peers receive from the
        #      metadata service. Without it, getifaddrs()[0] picks usb0
        #      (169.254.x APIPA) and peers fail to connect.
        #   2. KV-path 1D promotion — works around TQ's
        #      extract_field_schema schema/data mismatch for 1D fields.
        # The cluster-wide MC_* knobs are NOT among them; they are set
        # once on the driver, before this module is importable — see
        # nemo_rl.data_plane.adapters.transfer_queue_env.
        if cfg["backend"] == "mooncake_cpu":
            local_ip = _get_local_node_ip()
            if local_ip:
                # Force-assign per-process: Ray actors inherit env vars
                # from the driver, so a setdefault on the worker would
                # be a no-op and the actor would announce the driver's
                # IP — peers fail with "connection refused".
                os.environ["MC_TCP_BIND_ADDRESS"] = local_ip
            # Do not add MC_* setup here — mooncake snapshotted its config when
            # this module imported, so a write now is silently ignored.
            # Both must run before the first get, in every process with a TQ
            # client. The registration check is unconditional: it also covers
            # the two bytes workers the staging patch leaves untouched, which
            # is where an unchecked registration surfaces as TRANSFER_FAIL.
            _patch_mooncake_register_check()
            # Opt-out flag, defaulted on MooncakeCpuConfig rather than here:
            # an absent mooncake_cpu block means "this backend's defaults",
            # so the pool is on unless a config deliberately turns it off.
            mooncake_cfg = backend_config(cfg)
            if mooncake_cfg.reuse_registered_buffers:
                _patch_mooncake_staging_buffers(mooncake_cfg.staging_buffer_size)

        self._backend = cfg["backend"]
        self._supports_checkpointing = data_plane_supports_checkpointing(cfg)
        # Fix TQ's 1-D field schema at the source rather than reshaping the
        # payload around it: the schema now reports the ``()`` sample shape
        # the stored rows actually have. Applied on every backend and in
        # every process that builds a client, before ``_init_tq`` /
        # ``_connect_existing``, so no put can land under the old schema.
        # Self-verifying — see :func:`_assert_tq_stores_scalar_rows_0d`.
        _patch_scalar_field_schema()

        if bootstrap:
            _init_tq(cfg)
        else:
            _connect_existing()
        self._poll_interval_s = cfg["claim_meta_poll_interval_s"]
        self._closed = False
        # TQ restore is non-transactional and requires a globally clean system.
        # This process-local guard catches incorrect ordering through this
        # adapter; setup must still ensure no other client has touched TQ.
        self._data_operations_started = False
        # Fields whose schema this process has already warmed, per partition.
        # The controller's field map is append-only, so each field only needs
        # warming once for the lifetime of this client.
        self._warmed_fields: dict[str, set[str]] = {}

    def _require_checkpointing_support(self) -> None:
        """Reject backends that cannot round-trip all data-plane state."""
        if not self._supports_checkpointing:
            raise NotImplementedError(
                "TQ checkpointing is not supported for "
                f"data_plane.backend={self._backend!r}: the backend cannot "
                "persist and restore all storage rows."
            )

    def _mark_data_operation_started(self) -> None:
        """Make a later checkpoint load fail instead of mixing TQ states."""
        self._data_operations_started = True

    def _require_clean_for_load(self) -> None:
        """Reject restore after this client has performed a data operation."""
        if self._data_operations_started:
            raise RuntimeError(
                "load_checkpoint requires a clean TQ client before any "
                "register, claim, get, list, put, clear, or consumption operation"
            )

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
        #
        # Only new field names need warming: the controller's
        # ``field_name_mapping`` is append-only (never deleted, and our
        # ``clear_samples`` zeroes rows without popping the partition).
        already = self._warmed_fields.setdefault(partition_id, set())
        fields = [f for f in fields if f not in already]
        if not fields:
            return
        # Use a unique KV key instead of ``client.put``'s default row id
        # (``0@field`` at the Mooncake storage layer). Mooncake does not
        # support upsert, so repeated schema warmups can collide with
        # stale metadata from a previous registration.
        self._mark_data_operation_started()
        schema_key = (
            f"__schema__:{partition_id}:{os.getpid()}:{id(self)}:{time.time_ns()}"
        )
        dummy_td = TensorDict(
            {f: torch.zeros(1) for f in fields},
            batch_size=[1],
        )
        tq.kv_batch_put(
            keys=[schema_key],
            partition_id=partition_id,
            fields=dummy_td,
            tags=[{}],
        )
        tq.kv_clear(keys=[schema_key], partition_id=partition_id)
        # Only mark warmed once the write actually landed — otherwise a
        # failed put (mooncake's own retries already exhausted) poisons the
        # cache and a future retry of this call would wrongly skip warmup.
        already.update(fields)

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
        self._mark_data_operation_started()
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
        self._mark_data_operation_started()
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
        user_tags = (
            [{} for _ in sample_ids] if tags is None else [dict(tag) for tag in tags]
        )
        wire_fields: TensorDict | None = None
        field_names: list[str] | None = None
        if fields is not None:
            # No ``.contiguous()``: under tensordict==0.12.2 it strips
            # non-tensor leaves (NonTensorStack stored as LinkedList) to empty
            # TDs. TQ's encoder forces ``.contiguous()`` per tensor leaf
            # itself, so the call here was redundant for tensors and
            # destructive for non-tensors.
            detached_fields = cast(
                TensorDict,
                fields.detach(),  # type: ignore[missing-argument]
            )
            wire_fields = detached_fields
            field_names = [str(key) for key in detached_fields.keys()]

        self._mark_data_operation_started()
        # TQ's wire vocabulary is `keys=` — translation point.
        tq.kv_batch_put(
            keys=list(sample_ids),
            partition_id=partition_id,
            fields=wire_fields,
            tags=user_tags,
        )

        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            sample_ids=list(sample_ids),
            fields=field_names,
            tags=user_tags if user_tags else None,
        )

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        if not sample_ids:
            return TensorDict({}, batch_size=(0,))
        self._mark_data_operation_started()
        td = tq.kv_batch_get(
            keys=list(sample_ids),
            partition_id=partition_id,
            select_fields=select_fields,
        )
        return _from_wire(td)

    def list_sample_ids(self, partition_id: str) -> list[str]:
        """List TQ keys in ``partition_id`` without fetching tensor payloads."""
        self._mark_data_operation_started()
        listing = tq.kv_list(partition_id=partition_id)
        return sorted(listing.get(partition_id, {}).keys())

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        cleared_via_none = sample_ids is None
        if sample_ids is None:
            self._mark_data_operation_started()
            # No local state — ask TQ's controller for the current key
            # set in this partition. ``kv_list`` errors propagate; we
            # don't want a network blip to silently turn into "cleared
            # nothing".
            listing = tq.kv_list(partition_id=partition_id)
            sample_ids = list(listing.get(partition_id, {}).keys())
        if not sample_ids:
            if cleared_via_none:
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
        self._mark_data_operation_started()
        # TQ's wire vocabulary is `keys=` — translation point.
        tq.kv_clear(keys=list(sample_ids), partition_id=partition_id)

    # ── (C) lifecycle ──────────────────────────────────────────────────

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save TQ controller metadata and storage data."""
        self._require_checkpointing_support()
        _connect_existing()
        tq.save_checkpoint(checkpoint_dir, metadata=metadata)

    def load_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        """Restore TQ state after initialization and before data operations.

        The local lifecycle guard cannot observe operations issued by another
        TQ client, so the recovery coordinator must also guarantee globally
        clean setup ordering.
        """
        self._require_checkpointing_support()
        self._require_clean_for_load()
        # Validate the adapter-owned metadata before starting TQ's
        # non-transactional storage/controller restore.
        metadata_path = Path(checkpoint_dir) / "metadata.json"
        with metadata_path.open() as metadata_file:
            checkpoint_metadata = json.load(metadata_file)
        user_metadata = checkpoint_metadata.get("user_metadata", {})
        if not isinstance(user_metadata, dict):
            raise ValueError("TQ checkpoint user_metadata must be a dictionary")
        _connect_existing()
        # A failed TQ load may have partially modified distributed storage, so
        # this client is no longer safe for a retry even when an error escapes.
        self._mark_data_operation_started()
        tq.load_checkpoint(checkpoint_dir)
        return dict(user_metadata)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            tq.close()
        except Exception:
            pass
