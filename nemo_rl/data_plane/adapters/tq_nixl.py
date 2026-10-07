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
"""TransferQueue glue for the NIXL data plane (``data_plane.backend: nixl``).

Importing this module registers the ``NixlStore`` backend with TransferQueue:

- ``StorageClientFactory["NixlStoreClient"]`` -> :class:`NixlKVClient`
- ``StorageManagerFactory["NixlStore"]`` -> :class:`NixlStorageManager`
- ``StorageBootstrapProvider["NixlStore"]`` -> :func:`initialize_nixl_storage`

**Client.** ``put`` lays all values of one call out as raw bytes in one blob and
hands it to the store (:mod:`nemo_rl.data_plane.nixl`); the returned per-key
*placement-free* location meta ``{"b": blob_id, "o": off, "n": len, "k": kind,
"d": dtype, "s": shape, ...store extras}`` is what TQ stores as
``custom_backend_meta``. ``get`` asks the store to fill byte ranges and
reconstructs from that meta, not from TQ's field schema, so a TQ shape bug
cannot corrupt a read. ``clear`` releases entry refcounts. All NIXL access
happens under one lock: the TQ KV manager calls ``put`` from executor threads
and ``get`` from its event loop.

**Bootstrap.** Called once by ``transfer_queue.init`` in the process that
creates the TQ controller: starts the BlobDirectory and the storage units
(:mod:`nemo_rl.data_plane.nixl_storage_unit`). Placement is SimpleStorage style
(``num_storage_units`` round-robin over eligible nodes) or per node
(``storage_units_per_node`` on each node of ``node_ids``, which the caller
resolves from ``storage_unit_placement``). ``transfer_queue.close`` does not stop
third-party backends, so the owner calls :func:`shutdown` after it.

**Manager.** Everything on the data path is inherited from ``KVStorageManager``;
checkpointing goes through :mod:`nemo_rl.data_plane.adapters.tq_nixl_checkpoint`.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import uuid
from typing import Any, Iterable

import numpy as np
import ray
import torch
from ray.util.placement_group import placement_group_table
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.storage.clients.base import StorageClientFactory, StorageKVClient
from transfer_queue.storage.managers.base import KVStorageManager, StorageManagerFactory

from nemo_rl.data_plane.adapters import tq_nixl_checkpoint
from nemo_rl.data_plane.nixl import blob_format
from nemo_rl.data_plane.nixl.blobstore import LostBlobs, UnitSlabStore, make_store
from nemo_rl.data_plane.nixl.bufpool import BufferPool
from nemo_rl.data_plane.nixl.directory import BlobDirectory
from nemo_rl.data_plane.nixl.errors import DataPlaneLostKeys, StaleRead
from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint
from nemo_rl.data_plane.nixl.placement import make_placement
from nemo_rl.data_plane.nixl_storage_unit import NixlStorageUnit
from nemo_rl.distributed import numa_utils

BACKEND_NAME = "NixlStore"
DEFAULT_NAMESPACE = "nemo_rl_nixl"


def _as_dict(cfg: Any) -> dict[str, Any]:
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    return dict(cfg or {})


def _check_finite(tag: str, keys: list[str], values: list[Any]) -> None:
    """Debug aid (``debug_check_finite`` / ``NVDP_CHECK_FINITE=1``).

    Report rows whose float tensors hold NaN/inf or whose bool masks are all False, at both ends of
    the wire. A row reported on ``get`` but never on ``put`` is corruption in the
    data plane; a row reported on both was already like that upstream.
    """
    for k, v in zip(keys, values):
        if not isinstance(v, torch.Tensor) or v.is_nested or v.numel() == 0:
            continue
        if v.is_floating_point():
            bad = ~torch.isfinite(v)
            if bad.any():
                n_nan = int(torch.isnan(v).sum())
                n_inf = int(bad.sum()) - n_nan
                print(
                    f"[nvdp-check] {tag} key={k} dtype={v.dtype} shape={tuple(v.shape)} nan={n_nan} inf={n_inf} "
                    f"first={v.reshape(-1)[:4].tolist()}",
                    file=sys.stderr,
                    flush=True,
                )
        elif v.dtype == torch.bool and not v.any():
            print(
                f"[nvdp-check] {tag} key={k} bool all-False shape={tuple(v.shape)}",
                file=sys.stderr,
                flush=True,
            )


def _copy_into(dst: np.ndarray, src: memoryview, parallel_min: int) -> None:
    """Copy raw bytes into a registered buffer; torch's parallel copy_ when large.

    Single-threaded memcpy tops out near 14-28 GB/s on GB300; torch ``copy_``
    reaches ~107 GB/s from ~16 MB up, but costs ~0.4 ms of thread fan-out.
    """
    n = dst.nbytes
    if n >= parallel_min and not src.readonly:
        torch.from_numpy(dst).copy_(torch.frombuffer(src, dtype=torch.uint8, count=n))
    else:
        memoryview(dst)[:] = src


@StorageClientFactory.register("NixlStoreClient")
class NixlKVClient(StorageKVClient):
    def __init__(self, config: dict[str, Any]):
        cfg = _as_dict(config)
        super().__init__(cfg)
        self.namespace = cfg.get("namespace", DEFAULT_NAMESPACE)
        directory = ray.get_actor(
            cfg.get("directory_name", "BlobDirectory"), namespace=self.namespace
        )
        nixl_cfg = _as_dict(cfg.get("nixl") or {})
        store_cfg = _as_dict(cfg.get("store") or {})
        kind = store_cfg.get("kind", "unit")
        # Socket binding (nemo_rl/distributed/numa_utils.py): a client already pinned
        # inside one socket writes to its socket's unit first.
        init_params = dict(nixl_cfg.get("backend_init_params") or {})
        self.numa: int | None = None
        sockets = numa_utils.cpu_nodes() if cfg.get("numa", "auto") == "auto" else {}
        if len(sockets) >= 2:
            # Socket-first placement only; NICs are not restricted (see nixl_storage_unit).
            self.numa = numa_utils.socket_of_affinity(sockets)
        self.ep = NixlEndpoint(
            f"client-{os.getpid()}-{uuid.uuid4().hex[:6]}",
            backend=nixl_cfg.get("backend_name", "UCX"),
            init_params=init_params,
            prog_thread=False,
            extra_backends=(
                [store_cfg.get("file_backend", "POSIX")] if kind == "file" else ()
            ),
            require_rdma=bool(nixl_cfg.get("require_rdma", True)),
        )
        # Serialises every NIXL agent call (the agent is not thread-safe). Copies
        # into and out of pool buffers happen outside it, so concurrent TQ
        # threads overlap their memory traffic with each other's transfers.
        self._lock = threading.RLock()
        self.pool = BufferPool(
            self.ep,
            self._lock,
            base_bytes=int(cfg.get("client_staging_bytes", 256 << 20)),
            base_count=int(cfg.get("client_staging_buffers", 2)),
            max_oversize_bytes=int(cfg.get("client_pool_max_bytes", 8 << 30)),
        )
        # Zero-copy put: register the caller's own value memory and RDMA from it.
        # Pays off only where registration is cheap (UCX caches reused heap
        # ranges): ~0.02 ms at 16 MB, but ~70 ms per fresh 1 GiB mapping, where a
        # parallel copy into a pooled buffer takes ~10 ms. Hence a size window.
        self.zero_copy_min = int(cfg.get("zero_copy_min_bytes", 1 << 20))
        self.zero_copy_max = int(cfg.get("zero_copy_max_bytes", 32 << 20))
        self.parallel_copy_min = int(cfg.get("parallel_copy_min_bytes", 16 << 20))
        self.timeout_s = float(cfg.get("transfer_timeout_s", 120.0))
        # Must match the units' pin TTL: a READ that completes after its pin
        # expired may have observed a reused region and is discarded (§8).
        self.read_pin_s = float(cfg.get("read_pin_s", 60.0))
        self._check = (
            bool(cfg.get("debug_check_finite", False))
            or os.environ.get("NVDP_CHECK_FINITE") == "1"
        )
        if kind == "unit":
            pl = _as_dict(cfg.get("placement") or {})
            self.store = make_store(
                "unit",
                self.ep,
                directory,
                namespace=self.namespace,
                placement=make_placement(
                    pl.get("policy", "local_first"), start=os.getpid(), numa=self.numa
                ),
                node_id=ray.get_runtime_context().get_node_id(),
                timeout_s=self.timeout_s,
            )
        else:
            self.store = make_store(
                kind,
                self.ep,
                directory,
                root=store_cfg["root"],
                timeout_s=self.timeout_s,
                backend=store_cfg.get("file_backend", "POSIX"),
                max_open=int(store_cfg.get("max_open_files", 256)),
            )

    # ------------------------------------------------------------------ put
    def put(self, keys: list[str], values: list[Any]) -> list[Any] | None:
        if not keys:
            return []
        if self._check:
            _check_finite("put", keys, list(values))
        if isinstance(self.store, UnitSlabStore):
            blob_id = uuid.uuid4().hex
            # The blob id rides in the footer tag so RPC-free readers can verify it.
            p = blob_format.plan(keys, values, tag=bytes.fromhex(blob_id))
            blob_id, extra = self._put_segments(p, blob_id)
        else:  # file store: one contiguous blob
            p = blob_format.plan(keys, values)
            with self.pool.acquire(p.nbytes) as (base, buf):
                blob_format.write(p, memoryview(buf)[: p.nbytes])
                with self._lock:
                    blob_id, extra = self.store.put(base, p)
        return [{"b": blob_id, **e.meta(), **extra} for e in p.entries]

    @staticmethod
    def _backing_bytes(owner: Any) -> int:
        """Size of the allocation a value lives in (not the value itself)."""
        if isinstance(owner, torch.Tensor):
            return int(owner.untyped_storage().nbytes())
        base = owner
        while isinstance(base, np.ndarray) and base.base is not None:
            base = base.base
        return int(getattr(base, "nbytes", 0) or 0)

    def _zero_copy_ok(self, e: blob_format.Entry, mv: memoryview, owner: Any) -> bool:
        # Gated on the *backing allocation*, not the value: registration is only
        # cheap for ranges UCX has cached, i.e. reused heap allocations. Rows that
        # slice one big freshly mmap'd tensor (a 256 MB batch -> 16 x 16 MB rows)
        # cost ~1-2 ms each to register (bench: 256 MB put 33 -> 51 ms).
        return (
            e.kind in (blob_format.KIND_TENSOR, blob_format.KIND_NUMPY)
            and self.zero_copy_min <= e.len <= self.zero_copy_max
            and self._backing_bytes(owner) <= self.zero_copy_max
            and not mv.readonly
            and mv.c_contiguous
        )

    def _put_segments(
        self, p: blob_format.BlobPlan, blob_id: str
    ) -> tuple[str, dict[str, Any]]:
        """Lay out a blob without building it.

        Big values go zero-copy, the rest (small values, the index and the footer) is packed into one pool buffer.
        """
        zero = [
            (e, mv)
            for e, mv, owner in zip(p.entries, p.buffers, p.keepalive)
            if e.len and self._zero_copy_ok(e, mv, owner)
        ]
        zero_ids = {id(e) for e, _ in zero}
        packed = [
            (e, mv)
            for e, mv in zip(p.entries, p.buffers)
            if e.len and id(e) not in zero_ids
        ]
        tail = len(p.index_bytes) + blob_format.FOOTER_SIZE
        layout, cur = [], 0
        for e, mv in packed:
            layout.append((e, mv, cur))
            cur = blob_format._align_up(cur + e.len)
        tail_off = cur
        packed_bytes = tail_off + tail

        regs: list[int] = []
        try:
            with self.pool.acquire(packed_bytes) as (base, buf):
                segments: list[tuple[int, int, int]] = []
                for e, mv, at in layout:
                    _copy_into(
                        buf[at : at + e.len], mv.cast("B"), self.parallel_copy_min
                    )
                    segments.append((base + at, e.off, e.len))
                blob_format.write_tail(p, memoryview(buf)[tail_off : tail_off + tail])
                segments.append((base + tail_off, p.index_off, tail))
                with self._lock:
                    for e, mv in zero:
                        arr = np.frombuffer(mv.cast("B"), dtype=np.uint8)
                        addr = self.ep.register(arr)
                        regs.append(addr)
                        segments.append((addr, e.off, e.len))
                    return self.store.put_segments(
                        segments, p.nbytes, len(p.entries), blob_id=blob_id
                    )
        finally:
            if regs:
                with self._lock:
                    for addr in regs:
                        self.ep.deregister(addr)

    # ------------------------------------------------------------------ get
    def get(
        self, keys: list[str], shapes=None, dtypes=None, custom_backend_meta=None
    ) -> list[Any]:
        if not keys:
            return []
        metas = list(custom_backend_meta or [])
        if len(metas) != len(keys) or any(m is None for m in metas):
            bad = [k for k, m in zip(keys, metas or [None] * len(keys)) if m is None]
            raise DataPlaneLostKeys(
                bad or list(keys), reason="no location meta for keys"
            )
        for _attempt in range(2):
            t0 = time.monotonic()
            out = self._get_once(keys, metas)
            if time.monotonic() - t0 <= self.read_pin_s:
                if self._check:
                    _check_finite("get", keys, out)
                return out
        raise StaleRead(
            f"read of {len(keys)} keys outlived its {self.read_pin_s}s pin twice"
        )

    def _get_once(self, keys: list[str], metas: list[dict[str, Any]]) -> list[Any]:
        by_blob: dict[str, list[int]] = {}
        for i, m in enumerate(metas):
            by_blob.setdefault(m["b"], []).append(i)
        total = sum(int(m["n"]) for m in metas)
        # Room after the payload for one footer per blob: the RPC-free read path
        # fetches each blob's footer with the data and checks its tag.
        foot_at = blob_format._align_up(total)
        foot_len = len(by_blob) * blob_format.FOOTER_SIZE
        with self.pool.acquire(foot_at + foot_len) as (lbase, lbuf):
            lview = memoryview(lbuf)
            scratch = (lbase + foot_at, lview[foot_at : foot_at + foot_len])
            cur = 0
            local_off = [0] * len(keys)
            requests: dict[str, list[tuple[int, int, int]]] = {}
            for b, idxs in by_blob.items():
                rs = requests.setdefault(b, [])
                for i in idxs:
                    n = int(metas[i]["n"])
                    local_off[i] = cur
                    rs.append((int(metas[i]["o"]), n, lbase + cur))
                    cur += n
            try:
                with self._lock:
                    self.store.read(
                        requests,
                        {b: metas[idxs[0]] for b, idxs in by_blob.items()},
                        scratch,
                    )
            except LostBlobs as e:
                keys_lost = [keys[i] for b in e.blob_ids for i in by_blob.get(b, [])]
                raise DataPlaneLostKeys(keys_lost or list(keys), reason=e.reason) from e
            return [
                blob_format.materialize(
                    blob_format.decode_entry(
                        lview[local_off[i] : local_off[i] + int(m["n"])],
                        blob_format.Entry.from_meta(k, m),
                    )
                )
                for i, (k, m) in enumerate(zip(keys, metas))
            ]

    # ------------------------------------------------------------------ clear
    def clear(self, keys: list[str], custom_backend_meta=None) -> None:
        counts: dict[str, int] = {}
        hints: dict[str, dict[str, Any]] = {}
        for m in custom_backend_meta or []:
            if m:
                counts[m["b"]] = counts.get(m["b"], 0) + 1
                hints.setdefault(m["b"], m)
        if not counts:
            return
        with self._lock:
            self.store.release(counts, hints)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.pool.close()
        with self._lock:
            self.store.close()
            self.ep.close()


# ============================================================================ bootstrap
def _unit_name(i: int) -> str:
    return f"NixlStorageUnit#{i}"


# ----------------------------------------------------------------------------- node selection
def alive_node_ids(required_node_resource: str | None = None) -> list[str]:
    ids = sorted(
        n["NodeID"]
        for n in ray.nodes()
        if n.get("Alive", False)
        and (
            required_node_resource is None
            or n.get("Resources", {}).get(required_node_resource, 0) > 0
        )
    )
    if not ids:
        raise RuntimeError("No alive Ray nodes found. Is Ray initialized?")
    return ids


def placement_group_node_ids(placement_groups: Iterable[Any]) -> list[str]:
    """Ray node ids occupied by the given placement groups (one GCS read).

    Pass the inference and/or train clusters' placement groups to implement
    ``storage_unit_placement: all | inference | train``.
    """
    table = placement_group_table()
    return sorted(
        {
            node_id
            for pg in placement_groups
            for node_id in table[pg.id.hex()]["bundles_to_node_id"].values()
        }
    )


def unit_node_ids(cfg: dict[str, Any]) -> list[str]:
    """One node id per unit to create, from the backend config block."""
    nodes = list(cfg.get("node_ids") or []) or alive_node_ids(
        cfg.get("required_node_resource")
    )
    per_node = cfg.get("storage_units_per_node")
    if per_node:
        return [n for n in nodes for _ in range(int(per_node))]
    total = int(cfg.get("num_storage_units", 2))
    return [nodes[i % len(nodes)] for i in range(total)]


# ----------------------------------------------------------------------------- lifecycle
def _kill_named(name: str, namespace: str) -> bool:
    try:
        h = ray.get_actor(name, namespace=namespace)
    except ValueError:
        return False
    ray.kill(h, no_restart=True)
    return True


def shutdown(conf_or_backend_cfg: Any) -> None:
    """Kill the BlobDirectory and every storage unit of a NixlStore system.

    Units are named ``NixlStorageUnit#<i>``; kill consecutively until a name
    is missing, so a larger earlier system in the same namespace is swept too.
    """
    cfg = conf_or_backend_cfg
    if hasattr(cfg, "backend"):
        cfg = cfg.backend.NixlStore
    cfg = _as_dict(cfg)
    namespace = cfg.get("namespace", DEFAULT_NAMESPACE)
    i = 0
    while _kill_named(_unit_name(i), namespace):
        i += 1
    _kill_named(cfg.get("directory_name", "BlobDirectory"), namespace)


@StorageBootstrapProvider.register_provider("NixlStore")
def initialize_nixl_storage(conf: Any) -> dict[str, Any]:
    cfg = _as_dict(conf.backend.NixlStore)
    namespace = cfg.get("namespace", DEFAULT_NAMESPACE)
    dir_name = cfg.get("directory_name", "BlobDirectory")
    nixl_cfg = _as_dict(cfg.get("nixl") or {})
    store_cfg = _as_dict(cfg.get("store") or {})
    kind = store_cfg.get("kind", "unit")

    # A fresh TQ controller means a fresh data plane: stale named actors from a
    # previous system in the same namespace would otherwise block creation.
    shutdown(cfg)

    directory = BlobDirectory.options(name=dir_name, namespace=namespace).remote()
    handles: dict[str, Any] = {"BlobDirectory": directory}

    if kind == "file":
        # Actor-less store: one file per blob under store.root (Lustre / NVMe).
        root = store_cfg.get("root")
        if not root:
            raise ValueError(
                "backend.NixlStore.store.root is required for store.kind='file'"
            )
        os.makedirs(root, exist_ok=True)
    else:
        units = []
        per_node: dict[str, int] = {}
        for i, node_id in enumerate(unit_node_ids(cfg)):
            numa_index = per_node.get(
                node_id, 0
            )  # k-th unit on this node -> socket k % sockets
            per_node[node_id] = numa_index + 1
            unit = NixlStorageUnit.options(
                name=_unit_name(i),
                namespace=namespace,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
            ).remote(
                i,
                int(cfg.get("unit_slab_bytes", 1 << 30)),
                read_pin_s=float(cfg.get("read_pin_s", 60.0)),
                nixl_backend=nixl_cfg.get("backend_name", "UCX"),
                nixl_init_params=nixl_cfg.get("backend_init_params") or {},
                require_rdma=bool(nixl_cfg.get("require_rdma", True)),
                numa_mode=str(cfg.get("numa", "auto")),
                numa_index=numa_index,
                directory=directory,
            )
            units.append(unit)
            handles[_unit_name(i)] = unit
        infos = ray.get([u.info.remote() for u in units])
        ray.get([directory.register_unit.remote(info) for info in infos])

    conf.backend.NixlStore.directory_name = dir_name
    conf.backend.NixlStore.namespace = namespace
    return handles


# ============================================================================ manager
@StorageManagerFactory.register("NixlStore")
class NixlStorageManager(KVStorageManager):
    def __init__(self, controller_info: Any, config: Any, zmq_context: Any = None):
        cfg = _as_dict(config)
        cfg["client_name"] = "NixlStoreClient"
        # Newer TQ offers to share the client's ZMQ context; the pinned SHA does not
        # take that argument. KV managers only use ZMQ for the controller notify path,
        # so owning a private context is correct on both.
        super().__init__(controller_info, cfg)

    # TQ calls these from its client event loop (sync wrappers block on them);
    # the work is Ray RPCs plus file I/O, so run it off the loop.
    async def save_checkpoint(self, checkpoint_dir: str) -> None:
        await asyncio.to_thread(
            tq_nixl_checkpoint.save_storage_checkpoint,
            self.storage_client,
            checkpoint_dir,
        )

    async def load_checkpoint(self, checkpoint_dir: str) -> None:
        await asyncio.to_thread(
            tq_nixl_checkpoint.load_storage_checkpoint,
            self.storage_client,
            checkpoint_dir,
        )

    def close(self) -> None:
        try:
            client = getattr(self, "storage_client", None)
            if client is not None and hasattr(client, "close"):
                client.close()
        finally:
            super().close()
