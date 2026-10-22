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
"""NIXL endpoint for the data plane: registration, metadata, blocking transfers.

The agent is created by :func:`nemo_rl.utils.checkpoint_engines.nixl._create_nixl_agent`,
the same path the checkpoint-engine refit uses.

One ``NixlEndpoint`` per process. The agent is not thread-safe; callers
serialise access (the KV client holds a lock, the unit is a Ray actor).

Memory type: ``DRAM`` host buffers over UCX.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Mapping, Sequence

import numpy as np

from nemo_rl.data_plane.nixl.errors import (
    DataPlaneFatal,
    TransferError,
    TransportPolicyError,
)
from nemo_rl.utils.checkpoint_engines.nixl import (
    NIXL_DEFAULT_BACKEND_NAME,
    _create_nixl_agent,
)


log = logging.getLogger(__name__)

# UCX transport names that carry bytes off the host over RDMA verbs.
_RDMA_TLS_PREFIXES = ("rc", "dc", "ud", "ib")


def check_rdma_only(env: Mapping[str, str] | None = None) -> str:
    """Raise :class:`TransportPolicyError` unless UCX can only use RDMA.

    Allowed: an explicit ``UCX_TLS`` list with at least one RDMA transport
    (``rc``, ``rc_x``, ``rc_mlx5``, ``dc``, ``ud``, ``ib`` ...) and no ``tcp``
    or ``all``, plus same-host helpers (``self``, ``sm``, ``shm``, ``cma`` ...);
    or an exclusion list (``^...``) that names ``tcp``. ``UCX_NET_DEVICES``,
    if set, must name RDMA devices (``<dev>:<port>``), never a netdev such as
    ``eth0``. Unset ``UCX_TLS`` is rejected: UCX would then fall back to TCP
    whenever RDMA is unavailable. Returns a one-line description.
    """
    env = os.environ if env is None else env
    tls = (env.get("UCX_TLS") or "").strip()
    if not tls:
        raise TransportPolicyError(
            "UCX_TLS is unset, so UCX may fall back to tcp. Set it to an RDMA list, "
            "e.g. UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm (maybe_configure_data_plane_env "
            "sets this when unset)."
        )
    items = [t.strip() for t in tls.split(",") if t.strip()]
    if items[0].startswith("^"):
        excluded = {items[0][1:], *items[1:]}
        if "tcp" not in excluded:
            raise TransportPolicyError(
                f"UCX_TLS={tls} excludes some transports but not tcp"
            )
    else:
        bad = [t for t in items if t == "all" or t.startswith("tcp")]
        if bad:
            raise TransportPolicyError(
                f"UCX_TLS={tls} allows {bad}; the NIXL data plane never runs over tcp"
            )
        if not any(t.split("_")[0] in _RDMA_TLS_PREFIXES for t in items):
            raise TransportPolicyError(
                f"UCX_TLS={tls} has no RDMA transport (rc/dc/ud/ib)"
            )
    devs = (env.get("UCX_NET_DEVICES") or "all").strip()
    if devs != "all":
        netdevs = [d for d in devs.split(",") if d and ":" not in d]
        if netdevs:
            raise TransportPolicyError(
                f"UCX_NET_DEVICES={devs} names network interfaces {netdevs}; use RDMA "
                "devices with a port, e.g. rdma_vf_rail0:1"
            )
    return f"UCX_TLS={tls} UCX_NET_DEVICES={devs}"


class Graveyard:
    """Every NIXL request in this process that failed short of DONE.

    One per process (:data:`GRAVEYARD`), so dropping an endpoint or adapter
    can't free an entry through garbage collection. An entry holds the
    request's handle (never released), its descriptor lists, the agent that
    owns it, and its local ranges; callers keep the memory behind those
    ranges alive (the pool retires it, the adapter keeps zero-copy sources).

    Past ``max_bytes`` of retained memory (local ranges, plus every buffer the
    pool retires whole) the process is POISONED: every later transfer raises
    :class:`DataPlaneFatal` before touching NIXL.
    """

    def __init__(self, max_bytes: int = 4 << 30) -> None:
        self.max_bytes = int(max_bytes)
        self.entries: list[tuple[Any, ...]] = []
        self.bytes = 0
        self.poisoned: str | None = None  # why, once POISONED

    def retain(self, agent: Any, h: Any, ldesc: Any, rdesc: Any, local) -> None:
        self.entries.append((agent, h, ldesc, rdesc, list(local)))
        self._charge(sum(int(n) for _, n in local))

    def retain_buffer(self, nbytes: int) -> None:
        """Charge a pool buffer retired whole because a failed transfer touches it.

        Its transferred ranges were charged by :meth:`retain`, so they count
        twice: the limit errs on the side of poisoning.
        """
        self._charge(int(nbytes))

    def _charge(self, nbytes: int) -> None:
        self.bytes += nbytes
        if self.bytes > self.max_bytes and self.poisoned is None:
            self.poisoned = (
                f"{len(self.entries)} unfinished NIXL transfer(s) retain "
                f"{self.bytes} B, over graveyard_max_bytes={self.max_bytes}"
            )
            log.error("data plane POISONED: %s", self.poisoned)

    def check(self) -> None:
        if self.poisoned is not None:
            raise DataPlaneFatal(f"data plane is POISONED: {self.poisoned}")


GRAVEYARD = Graveyard()


def nixl_available() -> bool:
    try:
        import nixl._api  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def addr_of(arr: np.ndarray) -> int:
    return int(arr.ctypes.data)


class NixlEndpoint:
    def __init__(
        self,
        name: str,
        *,
        backend: str = NIXL_DEFAULT_BACKEND_NAME,
        init_params: dict[str, Any] | None = None,
        prog_thread: bool = False,
        require_rdma: bool = True,
    ) -> None:
        if backend == "UCX" and require_rdma:
            # Checked before the agent exists: NIXL's UCX backend takes UCX_*
            # env vars over backend_init_params, so the env decides the transport.
            log.info(
                "nixl endpoint %s pid=%d transport: %s",
                name,
                os.getpid(),
                check_rdma_only(),
            )
        self.name = name
        self.backend = backend
        self.agent = _create_nixl_agent(
            name, backend, dict(init_params or {}), enable_prog_thread=prog_thread
        )
        self.backends = [backend]
        self._regs: dict[Any, tuple[Any, list[str]]] = {}
        self._remotes: set[str] = set()
        # Local (addr, nbytes) ranges of transfers that failed without reaching
        # DONE: a READ may still write them, a WRITE may still read them. Never
        # cleared: no NIXL call proves such a transfer has stopped.
        self.unfinished: list[tuple[int, int]] = []
        # Their handles and descriptor lists, kept alive and never released;
        # also held by the process-wide GRAVEYARD, which outlives this object.
        self.unfinished_handles: list[tuple[Any, Any, Any]] = []

    def touches_unfinished(self, addr: int, nbytes: int) -> bool:
        """True if ``[addr, addr+nbytes)`` overlaps a transfer that may still run."""
        return any(a < addr + nbytes and addr < a + n for a, n in self.unfinished)

    # ------------------------------------------------------------------ registration
    def register(self, arr: np.ndarray) -> int:
        """Register a contiguous uint8 host array; returns its base address."""
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("array must be C-contiguous")
        addr = addr_of(arr)
        descs = self.agent.get_reg_descs([(addr, arr.nbytes, 0, "")], "DRAM")
        reg = self.agent.register_memory(descs, backends=list(self.backends))
        if reg is None:
            raise RuntimeError("register_memory(DRAM) failed")
        self._regs[addr] = (reg, list(self.backends))
        return addr

    def deregister(self, addr: int) -> None:
        item = self._regs.pop(addr, None)
        if item is not None:
            reg, backends = item
            self.agent.deregister_memory(reg, backends=backends)

    # ------------------------------------------------------------------ metadata
    def metadata(self) -> bytes:
        return self.agent.get_agent_metadata()

    def add_remote(self, md: bytes) -> str:
        name = self.agent.add_remote_agent(md)
        if isinstance(name, bytes):
            name = name.decode()
        self._remotes.add(name)
        return name

    def has_remote(self, name: str) -> bool:
        return name in self._remotes or name == self.name

    def remove_remote(self, name: str) -> None:
        if name in self._remotes:
            self._remotes.discard(name)
            try:
                self.agent.remove_remote_agent(name)
            except Exception:
                pass

    # ------------------------------------------------------------------ transfer
    def transfer(
        self,
        op: str,
        local: Sequence[tuple[int, int]],
        remote: Sequence[tuple[int, int]],
        remote_agent: str,
        *,
        timeout_s: float = 120.0,
    ) -> None:
        """Blocking one-sided ``WRITE`` (local→remote) or ``READ`` (remote→local).

        ``local`` and ``remote`` are parallel lists of ``(addr, nbytes)``.
        """
        err = self.transfer_many(
            [(op, local, remote, remote_agent)], timeout_s=timeout_s
        )[0]
        if err is not None:
            raise err

    def transfer_many(
        self,
        ops: Sequence[
            tuple[str, Sequence[tuple[int, int]], Sequence[tuple[int, int]], str]
        ],
        *,
        timeout_s: float = 120.0,
    ) -> list[TransferError | None]:
        """Post every ``(op, local, remote, remote_agent)``, then wait for all of them.

        The transfers run at the same time, so N units cost about the slowest
        one, not the sum. One ``timeout_s`` bounds the wait for the whole batch.
        Returns one entry per op: ``None`` once it is DONE, else its
        :class:`TransferError`. Ops are independent: a failure of one does not
        stop the others. Invalid arguments raise ``ValueError`` before anything
        is posted.

        Every posted transfer ends DONE (handle released) or retained (see
        :meth:`_retain`), on every exit, including an exception.
        """
        GRAVEYARD.check()
        for op, local, remote, _ in ops:
            if op not in ("WRITE", "READ"):
                raise ValueError(op)
            if len(local) != len(remote) or not local:
                raise ValueError("descriptor lists must be non-empty and equal length")
            for (_, a), (_, b) in zip(local, remote):
                if a != b:
                    raise ValueError("descriptor lengths differ")
        errs: list[TransferError | None] = [None] * len(ops)
        live: dict[int, tuple[Any, Any, Any]] = {}  # op index -> (h, ldesc, rdesc)
        try:
            for i, (op, local, remote, remote_agent) in enumerate(ops):
                errs[i] = self._post(i, op, local, remote, remote_agent, live)
            deadline = time.monotonic() + timeout_s
            while live:
                for i in list(live):
                    try:
                        state = self.agent.check_xfer_state(live[i][0])
                    except Exception as e:  # noqa: BLE001
                        errs[i] = self._fail(
                            i, ops[i], live, f"{type(e).__name__}: {e}", e
                        )
                        continue
                    if state == "DONE":
                        self._done(live.pop(i))
                    elif state != "PROC":
                        errs[i] = self._fail(i, ops[i], live, f"ended in state {state}")
                if live and time.monotonic() > deadline:
                    for i in list(live):
                        errs[i] = self._fail(
                            i, ops[i], live, f"timed out after {timeout_s}s"
                        )
        except BaseException:
            for i in list(live):
                self._retain(live.pop(i), ops[i][1])
            raise
        return errs

    def _post(self, i, op, local, remote, remote_agent, live) -> TransferError | None:
        """Post one transfer; ``live[i]`` is set while it is still running."""
        ldesc = self.agent.get_xfer_descs([(a, n, 0) for a, n in local], "DRAM")
        rdesc = self.agent.get_xfer_descs([(a, n, 0) for a, n in remote], "DRAM")
        # NIXL reports a broken peer by raising (NIXL_ERR_REMOTE_DISCONNECT,
        # NIXL_ERR_BACKEND, ...) from any of these calls. Surface all of them as
        # TransferError. The stores report it as retryable (TransferFailed ->
        # DataPlaneTransferFailed) unless the unit is confirmed dead.
        try:
            h = self.agent.initialize_xfer(op, ldesc, rdesc, remote_agent)
        except Exception as e:  # noqa: BLE001 - NIXL binding errors have no common base
            err = TransferError(f"{op} to {remote_agent}: {type(e).__name__}: {e}")
            err.__cause__ = e
            return err
        # From here on the transfer may be posted. Anything short of DONE leaves
        # it possibly still running, so its local ranges go on ``unfinished``
        # for good: a READ's are its destination, a WRITE's its source, whose
        # memory and registration must outlive it too (BufferPool and the
        # zero-copy put check them). A failed WRITE's destination is remote;
        # the caller retires it on the unit.
        live[i] = (h, ldesc, rdesc)
        try:
            state = self.agent.transfer(h)
        except Exception as e:  # noqa: BLE001
            return self._fail(
                i,
                (op, local, remote, remote_agent),
                live,
                f"{type(e).__name__}: {e}",
                e,
            )
        if state == "DONE":
            self._done(live.pop(i))
        elif state != "PROC":
            return self._fail(
                i, (op, local, remote, remote_agent), live, f"ended in state {state}"
            )
        return None

    def _fail(self, i, op_args, live, why, cause=None) -> TransferError:
        """Retain transfer ``i`` (it may still run) and return its error."""
        op, local, _, remote_agent = op_args
        self._retain(live.pop(i), local)
        err = TransferError(f"{op} to {remote_agent}: {why}", in_flight=True)
        err.__cause__ = cause
        return err

    def _retain(self, xfer: tuple[Any, Any, Any], local) -> None:
        """Keep a transfer that may still run, for good.

        The handle is never released: POSIX hands the request object to its I/O
        callbacks and deleting it under a pending I/O is a use-after-free.
        """
        h, ldesc, rdesc = xfer
        self.unfinished_handles.append((h, ldesc, rdesc))
        self.unfinished.extend(local)
        GRAVEYARD.retain(self.agent, h, ldesc, rdesc, local)

    def _done(self, xfer: tuple[Any, Any, Any]) -> None:
        try:
            self.agent.release_xfer_handle(xfer[0])
        except Exception:  # noqa: BLE001 - nothing is pending on a DONE handle
            pass

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self.unfinished_handles:
            # A transfer may still be running: keep the agent, remotes and
            # registrations alive until the process exits.
            log.warning(
                "nixl endpoint %s: %d unfinished transfer(s); leaving NIXL state "
                "registered until exit",
                self.name,
                len(self.unfinished_handles),
            )
            return
        for name in list(self._remotes):
            self.remove_remote(name)
        for key in list(self._regs):
            try:
                self.deregister(key)  # type: ignore[arg-type]
            except Exception:
                pass
        self.agent = None
