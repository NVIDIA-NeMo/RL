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
"""Register mode: publish addresses on put, move bytes once on get.

``mooncake_cpu`` copies every payload into the store before ``put`` returns —
through host staging, and out of GPU memory first if that is where the tensor
lived. Register mode does neither. ``put`` registers the producer's own
allocation with the local NIC and publishes ``{endpoint, base, offset, size}``
through TransferQueue's ``custom_backend_meta``; the bytes move exactly once,
when a consumer's ``get`` pulls them one-sided out of that memory. With
``use_gdr`` the source stays in HBM and the destination is HBM, so a tensor
crosses the fabric GPU-to-GPU with no host bounce and no staging copy on either
end.

This is Mooncake P2P-Store's "register is seeding" model reached through the
Transfer Engine's Python bindings. Mooncake classifies a registered range by
probing the pointer (``getMemoryLocation`` → ``cudaPointerGetAttributes``), so a
CUDA allocation registers as ``cuda:N`` and picks up that GPU's affine rail from
the same topology every other transfer uses.

Everything here plugs into TransferQueue through its decorator registries, so
importing this module is what makes ``backend: transfer_engine`` resolvable.

Two properties a caller must respect:
  - the producer must outlive the keys it published and must not mutate a
    registered buffer until ``clear`` releases it;
  - every read of a key is served by the producing process's NIC, so read
    fan-out concentrates instead of spreading across the cluster.

See ``docs/design-docs/tq-register-mode.md``.
"""

from __future__ import annotations

import json
import logging
import operator
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import reduce
from time import perf_counter
from typing import Any, Callable, cast

import torch
import zmq
from omegaconf import DictConfig
from torch import Tensor
from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.storage.clients.base import StorageClientFactory, StorageKVClient
from transfer_queue.storage.managers.base import KVStorageManager, StorageManagerFactory
from transfer_queue.utils import serial_utils
from transfer_queue.utils.tensor_utils import (
    allocate_empty_tensors,
    compute_stride,
    get_nbytes,
)
from transfer_queue.utils.zmq_utils import (
    ZMQServerInfo,
    create_zmq_socket,
    format_zmq_address,
    get_free_port,
    get_node_ip_address,
)

logger = logging.getLogger(__name__)

TRANSFER_ENGINE_IMPORTED: bool = True
try:
    from mooncake.engine import TransferEngine
except ImportError:
    TRANSFER_ENGINE_IMPORTED = False

# Mirrors MooncakeStoreClient: one batch of descriptors per engine call.
BATCH_SIZE_LIMIT: int = 400
MAX_BATCH_WORKER_THREADS: int = 4
MAX_SERIAL_WORKER_THREADS: int = 4


@dataclass
class _Pin:
    """One registered allocation: the reference keeping it alive and its use count."""

    storage: Any
    refcount: int = 0


class _PinTable:
    """Source regions this process has registered, keyed by allocation base.

    Two invariants force this shape. The Transfer Engine rejects overlapping
    registrations (``ERR_ADDRESS_OVERLAPPED``) while TQ hands ``put`` one value
    per sample — all views into one batched allocation — so the base allocation
    is registered once and each value records its offset. That also collapses
    registration cost to one ``ibv_reg_mr`` per batch, which matters most for
    CUDA memory.

    The strong reference is mandatory, not defensive: holding only the address
    would let PyTorch's caching allocator reissue the block after the tensor is
    collected, and consumers would silently read another tensor's data.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._pins: dict[int, _Pin] = {}

    def pin(self, tensor: Tensor) -> tuple[int, int]:
        """Register ``tensor``'s allocation once; return ``(base, offset)``."""
        storage = tensor.untyped_storage()
        base = storage.data_ptr()
        with self._lock:
            pin = self._pins.get(base)
            if pin is None:
                status = self._engine.register_memory(base, storage.nbytes())
                if status is not None and status != 0:
                    raise RuntimeError(
                        f"register_memory(0x{base:x}, {storage.nbytes()} bytes) "
                        f"failed with status {status}. Registration pins the "
                        f"pages with ibv_reg_mr, so it needs IPC_LOCK and a high "
                        f"memlock rlimit; a CUDA range additionally needs the "
                        f"peer-memory/dmabuf path the GDR transport uses."
                    )
                pin = self._pins[base] = _Pin(storage=storage)
            pin.refcount += 1
        return base, tensor.data_ptr() - base

    def unpin(self, base: int) -> None:
        """Drop one reference to ``base``; unregister and release it at zero."""
        with self._lock:
            pin = self._pins.get(base)
            if pin is None:
                return
            pin.refcount -= 1
            if pin.refcount == 0:
                del self._pins[base]
                self._engine.unregister_memory(base)

    def unpin_all(self) -> None:
        """Unregister every region regardless of refcount; used on shutdown."""
        with self._lock:
            for base in self._pins:
                self._engine.unregister_memory(base)
            self._pins.clear()

    def storage_for(self, base: int) -> Any:
        """The registered allocation at ``base``, or None if this process has none."""
        with self._lock:
            pin = self._pins.get(base)
            return pin.storage if pin is not None else None


class _ReleaseChannel:
    """Delivers "you may unpin these keys" to the process that published them.

    Only the owner can ``unregister_memory`` its own address space, but the
    moment a key becomes releasable is known centrally — it is exactly TQ's
    ``clear``, which runs in whichever process calls it (the driver, in both
    trainers). Without a way to reach the owner, a producer's registrations
    accumulate for the life of the process.

    PUSH/PULL rather than REQ/REP: a release is fire-and-forget, and a producer
    that has already exited must not be able to block the caller of ``clear``.
    ``IMMEDIATE`` plus a send timeout turns "nobody is listening" into a logged
    ``zmq.Again`` instead of a queued message that never lands.
    """

    _SEND_TIMEOUT_MS = 1000

    def __init__(
        self, ip: str, on_release: Callable[[list[tuple[str, int]]], None]
    ) -> None:
        self._ip = ip
        self._on_release = on_release
        self._ctx = zmq.Context(io_threads=1)
        self._inbox = create_zmq_socket(self._ctx, zmq.PULL, ip)
        self.port = get_free_port(ip)
        self._inbox.bind(format_zmq_address(ip, self.port))
        self._senders: dict[str, zmq.Socket] = {}
        self._senders_lock = threading.Lock()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="tq-register-release", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._inbox, zmq.POLLIN)
        while not self._closed.is_set():
            try:
                if not poller.poll(timeout=200):
                    continue
                payload = [
                    (key, seq) for key, seq in json.loads(self._inbox.recv().decode())
                ]
            except (zmq.ZMQError, ValueError, UnicodeDecodeError) as e:
                if self._closed.is_set():
                    return
                logger.warning("register-mode release listener: %s", e)
                continue
            try:
                self._on_release(payload)
            except Exception:
                # A failed release leaks one region; killing the listener would
                # leak every future one.
                logger.exception(
                    "register-mode release of %d keys failed", len(payload)
                )

    def send(self, ip: str, port: int, keys: list[tuple[str, int]]) -> None:
        """Ask the client at ``ip:port`` to release ``keys``. Never blocks."""
        address = format_zmq_address(ip, port)
        with self._senders_lock:
            sender = self._senders.get(address)
            if sender is None:
                sender = create_zmq_socket(self._ctx, zmq.PUSH, ip)
                sender.setsockopt(zmq.IMMEDIATE, 1)
                sender.setsockopt(zmq.SNDTIMEO, self._SEND_TIMEOUT_MS)
                sender.setsockopt(zmq.LINGER, 0)
                sender.connect(address)
                self._senders[address] = sender
        try:
            sender.send(json.dumps(keys).encode())
        except zmq.ZMQError as e:
            # The producer is gone, so its registrations died with it.
            logger.debug("release to %s dropped (%s); owner is gone", address, e)

    def close(self) -> None:
        """Stop serving and drop every socket."""
        self._closed.set()
        self._thread.join(timeout=2)
        with self._senders_lock:
            for sender in self._senders.values():
                sender.close(linger=0)
            self._senders.clear()
        self._inbox.close(linger=0)
        self._ctx.term()


def _allocate_receive_buffers(
    dtypes: list[torch.dtype], shapes: list[tuple], device: torch.device
) -> tuple[list[Tensor], list[int], list[int], list[int]]:
    """Allocate receive tensors as views into one contiguous region per dtype.

    Grouping by dtype is what keeps registration cheap: the region is what gets
    registered, not each tensor. On CPU this is exactly TQ's
    ``allocate_empty_tensors``; the CUDA case repeats its layout because that
    helper has no device parameter and ``.cuda()`` on its result would be the
    copy this backend exists to avoid.

    Returns:
        ``(tensors, tensor_ptrs, region_ptrs, region_sizes)``.
    """
    if device.type == "cpu":
        return allocate_empty_tensors(dtypes, shapes)

    groups: dict[torch.dtype, list[int]] = defaultdict(list)
    for i, dtype in enumerate(dtypes):
        groups[dtype].append(i)

    tensors: list[Tensor] = [torch.empty(())] * len(dtypes)
    ptrs: list[int] = [0] * len(dtypes)
    region_ptrs: list[int] = []
    region_sizes: list[int] = []

    for dtype, indices in groups.items():
        counts = [reduce(operator.mul, tuple(shapes[i]), 1) for i in indices]
        region = torch.empty(sum(counts), dtype=dtype, device=device)
        region_ptrs.append(region.data_ptr())
        region_sizes.append(region.nbytes)
        offset = 0
        for i, count in zip(indices, counts, strict=True):
            shape = tuple(shapes[i])
            view = region.as_strided(
                size=shape, stride=compute_stride(shape), storage_offset=offset
            )
            tensors[i] = view
            ptrs[i] = view.data_ptr()
            offset += count

    return tensors, ptrs, region_ptrs, region_sizes


@StorageClientFactory.register("TransferEngineClient")
class TransferEngineClient(StorageKVClient):
    """Storage client that publishes buffer addresses instead of copying payloads.

    ``put`` returns per-key ``{"endpoint", "base", "offset", "size"}``. TQ keeps
    it as ``custom_backend_meta`` and hands it back on ``get`` and ``clear``.
    That address book is the only state this backend has — no master, no storage
    pool, no eviction.
    """

    def __init__(self, config: dict[str, Any]):
        """Bring this process's Transfer Engine up and publish its endpoint.

        Args:
            config: TQ backend config block. ``local_hostname`` is resolved
                per-process when empty, so the driver-built config stays
                node-agnostic.
        """
        super().__init__(config)
        if not TRANSFER_ENGINE_IMPORTED:
            raise ImportError(
                "Mooncake Transfer Engine not installed. Install via: "
                "pip install mooncake-transfer-engine"
            )

        # Empty means "resolve here": the driver ships one config to every node.
        self.local_hostname = config["local_hostname"] or get_node_ip_address()
        # 0 lets the engine pick a free port, which co-located clients need.
        self.rpc_port = int(config["rpc_port"])
        self.protocol = config["protocol"]
        self.device_name = config["device_name"]
        self.metadata_server = config["metadata_server"]

        # GDR is per process, not per cluster: a CPU-only client (the
        # SingleController driver) shares a config with GPU workers and simply
        # keeps host destinations. Registration itself never cares — mooncake
        # classifies a source range by probing the pointer.
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if bool(config["use_gdr"]) and torch.cuda.is_initialized()
            else torch.device("cpu")
        )
        # Independent of the receive side: a producer can hand its bytes to host
        # memory and still have consumers pull them straight into HBM.
        self._offload_source = bool(config["offload_source_to_host"])

        self._engine: Any = TransferEngine()
        status = self._engine.initialize(
            f"{self.local_hostname}:{self.rpc_port}",
            self.metadata_server,
            self.protocol,
            self.device_name,
        )
        if status != 0:
            raise RuntimeError(
                f"TransferEngine initialization failed with status {status} "
                f"(protocol={self.protocol}, device_name={self.device_name!r})."
            )

        # Segment name peers read this process's buffers from; it rides in
        # every meta entry this client publishes.
        self._endpoint = f"{self.local_hostname}:{self._engine.get_rpc_port()}"
        self._pins = _PinTable(self._engine)
        # key -> (base, seq) for every key this process still has registered.
        # The sequence number is what makes a replayed release safe: TQ recycles
        # `<global_idx>@<field>` keys across steps, so a stale clear naming a key
        # that has since been republished must not unpin the new registration.
        # Matching on (key, seq) rejects it; matching on the key alone would not,
        # and matching on the address alone would not either — the caching
        # allocator hands back the same block all the time.
        self._published: dict[str, tuple[int, int]] = {}
        self._published_seq = 0
        self._published_lock = threading.Lock()
        self._release = _ReleaseChannel(self.local_hostname, self._release_keys)
        # The one line that says which path this process actually took: without
        # it a GDR client that silently fell back to host receive buffers looks
        # identical in the logs to one that did not.
        logger.info(
            "TransferQueue register mode ready: endpoint=%s protocol=%s "
            "device_name=%s receive_device=%s source=%s release_port=%d",
            self._endpoint,
            self.protocol,
            self.device_name,
            self._device,
            "host (offloaded)" if self._offload_source else "in place",
            self._release.port,
        )

    @property
    def endpoint(self) -> str:
        """Segment name (``host:rpc_port``) peers pull this process's buffers from."""
        return self._endpoint

    @property
    def receive_device(self) -> torch.device:
        """Where this client's ``get`` lands data — CUDA under GDR, else host."""
        return self._device

    def put(self, keys: list[str], values: list[Any]) -> list[dict]:
        """Register the caller's buffers and publish their addresses.

        No payload byte is copied and nothing changes device: a CUDA tensor is
        registered in HBM exactly where the producer built it.

        Args:
            keys: Unique string identifiers.
            values: Values to publish (tensors, scalars, dicts, ...).

        Returns:
            Per-key ``{"endpoint", "base", "offset", "size"}`` aligned with
            ``keys``. Each buffer stays registered until ``clear`` releases it
            and must not be modified in the meantime.
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        started = perf_counter()

        tensor_indices: list[int] = []
        non_tensor_indices: list[int] = []
        for i, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)

        custom_backend_meta: list[dict] = [{} for _ in keys]

        tensors = {
            i: self._prepare_source(cast(Tensor, values[i])) for i in tensor_indices
        }
        # Publishing an address makes the buffer readable by a peer NIC, which
        # is not ordered against the stream that filled it. One sync per put
        # covers every CUDA source in the batch.
        if any(t.is_cuda for t in tensors.values()):
            torch.cuda.synchronize()
        for i, tensor in tensors.items():
            custom_backend_meta[i] = self._pin_and_describe(
                keys[i], tensor, tensor.nbytes
            )

        if non_tensor_indices:
            # Python objects have no stable buffer, so they are serialized into
            # a freshly allocated host region. A fresh region cannot overlap an
            # existing registration, so it publishes exactly like a tensor.
            def alloc(sizes: list[int]) -> list[Tensor]:
                buffers, _, _, _ = allocate_empty_tensors(
                    [torch.uint8] * len(sizes), [(s,) for s in sizes]
                )
                return buffers

            buffers, packed_sizes = serial_utils.batch_encode_into(
                [values[i] for i in non_tensor_indices],
                alloc,
                num_workers=MAX_SERIAL_WORKER_THREADS,
            )
            for i, buffer, packed_size in zip(
                non_tensor_indices, buffers, packed_sizes, strict=True
            ):
                custom_backend_meta[i] = self._pin_and_describe(
                    keys[i], cast(Tensor, buffer), packed_size
                )

        # The counterpart of the get line: what register mode substitutes for a
        # store write is this pin, so its cost is the other half of the ledger.
        logger.info(
            "register_mode put: total=%dus keys=%d (tensor=%d object=%d) "
            "bytes=%d pinned_keys=%d",
            (perf_counter() - started) * 1e6,
            len(keys),
            len(tensor_indices),
            len(non_tensor_indices),
            sum(entry["size"] for entry in custom_backend_meta),
            self.pinned_keys,
        )
        return custom_backend_meta

    def get(
        self,
        keys: list[str],
        shapes: list[Any] | None = None,
        dtypes: list[Any] | None = None,
        custom_backend_meta: list[dict | None] | None = None,
    ) -> list[Any]:
        """Pull values one-sided from the processes that published them.

        Tensors land on :attr:`receive_device` — under GDR that is this
        process's GPU, so the payload goes producer HBM to consumer HBM in one
        hop. Non-tensor payloads are always decoded from host memory.

        Args:
            keys: Keys to fetch.
            shapes: Expected tensor shapes (``[]`` for scalars).
            dtypes: Expected dtypes; ``None`` marks non-tensor data.
            custom_backend_meta: Per-key address entries returned by ``put``.

        Returns:
            Retrieved values in the same order as ``keys``.
        """
        if shapes is None or dtypes is None or custom_backend_meta is None:
            raise ValueError(
                "TransferEngineClient needs shapes, dtypes and "
                "custom_backend_meta to locate the published buffers."
            )
        if not (len(keys) == len(shapes) == len(dtypes) == len(custom_backend_meta)):
            raise ValueError(
                "Lengths of keys, shapes, dtypes and custom_backend_meta must match"
            )

        metas: list[dict] = []
        for key, meta in zip(keys, custom_backend_meta, strict=True):
            if not meta:
                raise ValueError(
                    f"Missing custom_backend_meta for key `{key}`; "
                    f"it was never published."
                )
            metas.append(meta)

        # Receive buffers: tensors keep their declared dtype/shape, non-tensor
        # payloads land in uint8 buffers sized by the producer's packed length.
        dst_dtypes = [dtype if dtype is not None else torch.uint8 for dtype in dtypes]
        dst_shapes = [
            tuple(shape) if dtype is not None else (meta["size"],)
            for shape, dtype, meta in zip(shapes, dtypes, metas, strict=True)
        ]
        sizes = [meta["size"] for meta in metas]
        for key, dtype, expected, size in zip(
            keys, dtypes, get_nbytes(dst_dtypes, dst_shapes), sizes, strict=True
        ):
            if dtype is not None and expected != size:
                raise ValueError(
                    f"Key `{key}` was published as {size} bytes but its "
                    f"metadata describes {expected}."
                )

        tensor_indices = [i for i, dtype in enumerate(dtypes) if dtype is not None]
        non_tensor_indices = [i for i, dtype in enumerate(dtypes) if dtype is None]
        results: list[Any] = [None] * len(keys)

        # Non-tensor payloads are msgpack buffers that get decoded on the host,
        # so they land in host memory even when tensors are going to HBM.
        for indices, device in (
            (tensor_indices, self._device),
            (non_tensor_indices, torch.device("cpu")),
        ):
            if not indices:
                continue
            buffers = self._fetch_group(
                [dst_dtypes[i] for i in indices],
                [dst_shapes[i] for i in indices],
                [sizes[i] for i in indices],
                [metas[i] for i in indices],
                device,
            )
            for i, buffer in zip(indices, buffers, strict=True):
                results[i] = buffer

        if non_tensor_indices:
            decoded = serial_utils.batch_decode_from(
                [results[i] for i in non_tensor_indices]
            )
            for i, value in zip(non_tensor_indices, decoded, strict=True):
                results[i] = value
        return results

    def clear(
        self, keys: list[str], custom_backend_meta: list[dict | None] | None = None
    ) -> None:
        """Release the registrations backing ``keys``, wherever they live.

        Only the owner can unregister its own memory, and ``clear`` runs in
        whichever process called it — the driver, in both trainers. Keys this
        process published are unpinned here; the rest are forwarded to the
        client that published them, which is why a producer's registrations do
        not outlive the data (see :class:`_ReleaseChannel`).
        """
        if custom_backend_meta is None:
            raise ValueError(
                "TransferEngineClient needs custom_backend_meta to locate the "
                "registrations to release."
            )

        mine: list[tuple[str, int]] = []
        remote: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
        for key, meta in zip(keys, custom_backend_meta, strict=True):
            if not meta:
                continue
            if meta["endpoint"] == self._endpoint:
                mine.append((key, meta["seq"]))
            else:
                host = meta["endpoint"].rsplit(":", 1)[0]
                remote[(host, meta["release_port"])].append((key, meta["seq"]))

        self._release_keys(mine)
        for (host, port), owned in remote.items():
            self._release.send(host, port, owned)

    def close(self) -> None:
        """Stop serving releases, unregister every published region, drop the engine."""
        if self._engine is not None:
            self._release.close()
            self._pins.unpin_all()
            with self._published_lock:
                self._published.clear()
            self._engine = None

    @property
    def pinned_keys(self) -> int:
        """Keys this process still has registered — the leak metric, if it regrows."""
        with self._published_lock:
            return len(self._published)

    def _prepare_source(self, tensor: Tensor) -> Tensor:
        """Return the tensor whose allocation ``put`` will register.

        ``offload_source_to_host`` trades one D2H per put for HBM the producer
        would otherwise keep registered until ``clear``; consumers still pull
        one-sided into their own HBM, since the receive side is chosen
        independently.
        """
        if self._offload_source and tensor.is_cuda:
            return tensor.to("cpu", copy=False).contiguous()
        return tensor.contiguous()

    def _pin_and_describe(self, key: str, tensor: Tensor, size: int) -> dict:
        base, offset = self._pins.pin(tensor)
        with self._published_lock:
            self._published_seq += 1
            seq = self._published_seq
            superseded = self._published.get(key)
            self._published[key] = (base, seq)
        if superseded is not None:
            # Re-publishing a key TQ never cleared: the old value is dead the
            # moment the new one lands, and dropping its reference here is what
            # keeps an upsert from leaking a registration.
            self._pins.unpin(superseded[0])
        return {
            "endpoint": self._endpoint,
            "base": base,
            "offset": offset,
            "size": size,
            # Where to send this key's release, and which publication it refers
            # to. The host comes from ``endpoint``, so only the port is carried.
            "release_port": self._release.port,
            "seq": seq,
        }

    def _release_keys(self, versioned_keys: list[tuple[str, int]]) -> None:
        """Unpin the named publications. A replay or a duplicate is a no-op."""
        bases: list[int] = []
        with self._published_lock:
            for key, seq in versioned_keys:
                published = self._published.get(key)
                if published is not None and published[1] == seq:
                    bases.append(published[0])
                    del self._published[key]
        for base in bases:
            self._pins.unpin(base)

    def _fetch_group(
        self,
        dtypes: list[torch.dtype],
        shapes: list[tuple],
        sizes: list[int],
        metas: list[dict],
        device: torch.device,
    ) -> list[Tensor]:
        """Allocate receive buffers on ``device`` and fill them from their producers.

        Emits one timing line per call, deliberately mirroring mooncake's own
        ``batch_get_into: <n>us ... key count: <n>`` so a register-mode run and a
        mooncake_cpu run can be compared on per-call latency rather than on step
        time, where a data-plane difference of tens of milliseconds is invisible.
        The registration leg is timed separately because registering receive
        regions per get — rather than pooling them, which would reintroduce a
        copy — is the cost most likely to make this backend slower.
        """
        started = perf_counter()
        buffers, dst_ptrs, region_ptrs, region_sizes = _allocate_receive_buffers(
            dtypes, shapes, device
        )
        allocated_at = perf_counter()
        register_s = 0.0

        groups: dict[str, list[int]] = defaultdict(list)
        for i, meta in enumerate(metas):
            groups[meta["endpoint"]].append(i)

        local = groups.pop(self._endpoint, [])
        remote = bool(groups)

        if remote:
            # The engine can only write into registered memory, so the receive
            # regions are registered for the duration of the read. Registering
            # per get rather than pooling keeps the pull copy-free: a pooled
            # destination would need a second copy out of the pool.
            register_started = perf_counter()
            status = self._engine.batch_register_memory(region_ptrs, region_sizes)
            register_s = perf_counter() - register_started
            if status is not None and status != 0:
                raise RuntimeError(
                    f"batch_register_memory of {len(region_ptrs)} receive "
                    f"regions on {device} failed with status {status}."
                )
        try:
            # Keys this process published are already in its address space, so
            # a local copy beats a loopback transfer through the NIC.
            for i in local:
                self._copy_local(buffers[i], metas[i])

            if remote:
                with ThreadPoolExecutor(
                    max_workers=MAX_BATCH_WORKER_THREADS
                ) as executor:
                    futures = [
                        executor.submit(
                            self._read_from_endpoint,
                            endpoint,
                            indices,
                            dst_ptrs,
                            sizes,
                            metas,
                        )
                        for endpoint, indices in groups.items()
                    ]
                    for future in futures:
                        future.result()
        finally:
            if remote:
                self._engine.batch_unregister_memory(region_ptrs)

        finished = perf_counter()
        logger.info(
            "register_mode get: total=%dus alloc=%dus register=%dus move=%dus "
            "keys=%d (local=%d remote=%d) bytes=%d peers=%d device=%s",
            (finished - started) * 1e6,
            (allocated_at - started) * 1e6,
            register_s * 1e6,
            (finished - allocated_at - register_s) * 1e6,
            len(metas),
            len(local),
            len(metas) - len(local),
            sum(sizes),
            len(groups),
            device,
        )
        return buffers

    def _copy_local(self, dst: Tensor, meta: dict) -> None:
        """Fill ``dst`` from a key this process published, without the fabric.

        The pin table still holds the source allocation, so the copy is a
        ``Tensor.copy_`` between two views of live storages — which also gets
        the device pairing right when producer and consumer differ (a CUDA
        source read by a CPU-only client, say).
        """
        storage = self._pins.storage_for(meta["base"])
        if storage is None:
            raise RuntimeError(
                f"Key published by this endpoint ({self._endpoint}) at "
                f"0x{meta['base']:x} is no longer registered here; it was "
                f"cleared or the client was closed before this get."
            )
        element_size = dst.element_size()
        if meta["offset"] % element_size:
            raise RuntimeError(
                f"Published offset {meta['offset']} is not a multiple of the "
                f"{element_size}-byte element size of {dst.dtype}."
            )
        source = torch.empty(0, dtype=dst.dtype, device=storage.device).set_(
            storage, meta["offset"] // element_size, tuple(dst.shape)
        )
        dst.copy_(source)

    def _read_from_endpoint(
        self,
        endpoint: str,
        indices: list[int],
        dst_ptrs: list[int],
        sizes: list[int],
        metas: list[dict],
    ) -> None:
        for start in range(0, len(indices), BATCH_SIZE_LIMIT):
            batch = indices[start : start + BATCH_SIZE_LIMIT]
            status = self._engine.batch_transfer_sync_read(
                endpoint,
                [dst_ptrs[i] for i in batch],
                [metas[i]["base"] + metas[i]["offset"] for i in batch],
                [sizes[i] for i in batch],
            )
            if status < 0:
                raise RuntimeError(
                    f"batch_transfer_sync_read of {len(batch)} keys from "
                    f"{endpoint} failed with status {status}."
                )


@StorageManagerFactory.register("TransferEngine")
class TransferEngineStorageManager(KVStorageManager):
    """Storage manager binding the register-mode client into TQ's KV path."""

    def __init__(self, controller_info: ZMQServerInfo, config: dict[str, Any]):
        """Select ``TransferEngineClient`` and defer to the KV manager.

        Signature mirrors ``MooncakeStorageManager`` at the pinned TQ commit,
        where ``StorageManagerFactory.create`` passes exactly these two
        positionally.
        """
        config["client_name"] = "TransferEngineClient"
        super().__init__(controller_info, config)


@StorageBootstrapProvider.register_provider("TransferEngine")
def initialize_transfer_engine_storage(conf: DictConfig) -> dict:
    """Bootstrap the register-mode backend — there is nothing central to start.

    Register mode has no master and no storage pool: each client registers its
    own buffers and serves reads from them. With ``P2PHANDSHAKE`` there is not
    even a metadata server, so this exists only to keep TQ from logging an
    unregistered-backend error.

    Args:
        conf: Full TQ config; ``conf.backend.TransferEngine`` is read.

    Returns:
        An empty dict — this backend owns no external resource. TQ's ``close()``
        has no branch for it and logs "not supported for now"; there is
        genuinely nothing to tear down.

    Raises:
        ValueError: If a metadata server other than ``P2PHANDSHAKE`` is
            configured, which NeMo-RL does not wire up.
    """
    metadata_server = str(conf.backend.TransferEngine.metadata_server).strip()
    if metadata_server.upper() != "P2PHANDSHAKE":
        raise ValueError(
            f"TransferEngine backend supports metadata_server='P2PHANDSHAKE' "
            f"only, got {metadata_server!r}."
        )
    return {}
