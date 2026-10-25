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
"""ZMQ control plane between clients and storage units (no Ray on the op path).

Same shape as TransferQueue's SimpleStorage: each storage unit binds a ZMQ
ROUTER on ``node_ip:<free port>`` and serves small control messages from one
thread; clients talk to it over a DEALER socket per unit. Ray is used only to
start the units and to publish their addresses once (bootstrap).

Messages are msgpack ``[op, request_id, args]``; replies ``[request_id, ok,
result_or_error]``. ``notify`` sends ops that need no reply (``unpin``,
``abort``); the unit sends nothing back for those.

An error reply is ``{"kind", "msg", "reason", "blobs"}`` so the client can
rebuild the exception the unit raised instead of reporting every failure the
same way:

======== =================================== ==========================
kind     client exception                    meaning
======== =================================== ==========================
UnitFull ``UnitFull``                        try the next unit
Busy     ``DataPlaneBusy(reason)``           retry this unit later
Missing  ``LostBlobs(blobs)``                the unit no longer has them
Protocol ``TransferError``                   any other server exception
======== =================================== ==========================
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import uuid
from typing import Any, Callable

import msgspec
import zmq

from nemo_rl.data_plane.nixl.errors import (
    DataPlaneBusy,
    LostBlobs,
    TransferError,
    UnitFull,
)

log = logging.getLogger(__name__)

_enc = msgspec.msgpack.Encoder()
_dec = msgspec.msgpack.Decoder()

NO_REPLY = {"unpin", "abort"}


def error_reply(e: BaseException) -> dict[str, Any]:
    """The wire form of a server-side exception (see the module docstring)."""
    out: dict[str, Any] = {"kind": "Protocol", "msg": f"{type(e).__name__}: {e}"}
    if isinstance(e, UnitFull):
        out = {"kind": "UnitFull", "msg": str(e)}
    elif isinstance(e, DataPlaneBusy):
        out = {"kind": "Busy", "msg": str(e), "reason": e.reason}
    elif isinstance(e, LostBlobs):
        out = {"kind": "Missing", "msg": e.reason, "blobs": e.blob_ids}
    out.setdefault("reason", None)
    out.setdefault("blobs", None)
    return out


def raise_reply(address: str, op: str, err: Any) -> None:
    """Rebuild and raise the exception behind an error reply."""
    if not isinstance(err, dict):  # pre-structured peer: [type_name, msg]
        err = {"kind": "Protocol", "msg": ": ".join(map(str, err))}
    kind, msg = err.get("kind"), err.get("msg", "")
    if kind == "UnitFull":
        raise UnitFull(msg)
    if kind == "Busy":
        raise DataPlaneBusy(f"unit {address} {op}: {msg}", reason=err.get("reason"))
    if kind == "Missing":
        raise LostBlobs(list(err.get("blobs") or []), msg or "blob not on unit")
    raise TransferError(f"unit {address} {op}: {msg}")


def node_ip() -> str:
    """This node's routable IP (what Ray advertises), falling back to a UDP probe."""
    try:
        import ray

        return ray.util.get_node_ip_address()
    except Exception:  # noqa: BLE001
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()


class UnitServer:
    """ROUTER server thread for one storage unit.

    ``handlers`` maps op name -> callable(**args). They run on this thread under
    ``lock``, which the unit also takes for its (rare) Ray-called methods.
    """

    def __init__(
        self, handlers: dict[str, Callable[..., Any]], lock: threading.Lock
    ) -> None:
        self.handlers = handlers
        self.lock = lock
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.setsockopt(zmq.LINGER, 0)
        ip = node_ip()
        port = self.sock.bind_to_random_port(
            f"tcp://{ip}", min_port=20000, max_port=60000
        )
        self.address = f"tcp://{ip}:{port}"
        self._stop = threading.Event()
        self.thread = threading.Thread(
            target=self._serve, name="nvdp-unit-zmq", daemon=True
        )
        self.thread.start()

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                if not dict(poller.poll(500)):
                    continue
                ident, payload = self.sock.recv_multipart()
            except zmq.ContextTerminated:
                return
            except Exception:  # noqa: BLE001 - malformed peer; keep serving
                continue
            try:
                op, rid, args = _dec.decode(payload)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(op, str):  # unhashable op would kill this thread
                continue
            fn = self.handlers.get(op)
            try:
                if fn is None:
                    raise ValueError(f"unknown op {op!r}")
                with self.lock:
                    result, ok = fn(**args), True
            except Exception as e:  # noqa: BLE001 - reported to the caller
                result, ok = error_reply(e), False
            if op not in NO_REPLY:
                try:
                    self.sock.send_multipart([ident, _enc.encode([rid, ok, result])])
                except Exception:  # noqa: BLE001 - unencodable result; keep serving
                    log.exception("could not reply to %s", op)

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)
        self.sock.close(0)
        self.ctx.term()


class UnitClient:
    """DEALER sockets to every unit this client talks to.

    Not thread-safe: callers serialise use (the KV client holds its lock around store calls).
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        self.ctx = zmq.Context()
        self.timeout_ms = int(timeout_s * 1000)
        self._socks: dict[str, zmq.Socket] = {}

    def _sock(self, address: str) -> zmq.Socket:
        s = self._socks.get(address)
        if s is None:
            s = self.ctx.socket(zmq.DEALER)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(address)
            self._socks[address] = s
        return s

    def _drop(self, address: str) -> None:
        s = self._socks.pop(address, None)
        if s is not None:
            s.close(0)

    def call(
        self, address: str, op: str, *, timeout_s: float | None = None, **args: Any
    ) -> Any:
        """Request/reply.

        Raises what the unit raised (see the module docstring), and
        ``TransferError`` when no reply comes within ``timeout_s`` (default: the
        client's timeout; callers pass their remaining operation budget).
        """
        timeout_ms = self._timeout_ms(timeout_s)
        rid = self._send(address, op, args, timeout_ms)
        return self._recv(address, op, rid, time.monotonic() + timeout_ms / 1000)

    def call_many(
        self,
        op: str,
        args_by_address: dict[str, dict[str, Any]],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """``call`` to every address at once: all requests go out before any reply is read.

        Returns each address's result, or the exception its call raised, so the
        caller decides what one failure means for the others. ``timeout_s``
        bounds the whole batch, not each call.
        """
        timeout_ms = self._timeout_ms(timeout_s)
        end = time.monotonic() + timeout_ms / 1000
        out: dict[str, Any] = {}
        rids: dict[str, str] = {}
        for address, args in args_by_address.items():
            try:
                rids[address] = self._send(address, op, args, timeout_ms)
            except Exception as e:  # noqa: BLE001 - reported per address
                out[address] = e
        for address, rid in rids.items():
            try:
                out[address] = self._recv(address, op, rid, end)
            except Exception as e:  # noqa: BLE001 - reported per address
                out[address] = e
        return out

    def _timeout_ms(self, timeout_s: float | None) -> int:
        if timeout_s is None:
            return self.timeout_ms
        return max(1, min(self.timeout_ms, int(timeout_s * 1000)))

    def _send(
        self, address: str, op: str, args: dict[str, Any], timeout_ms: int
    ) -> str:
        rid = uuid.uuid4().hex[:12]
        s = self._sock(address)
        # A unit that stopped reading fills the send queue: wait no longer
        # than the reply would be waited for.
        s.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            s.send(_enc.encode([op, rid, args]))
        except zmq.Again:
            raise TransferError(
                f"unit {address} send queue stayed full for {op} "
                f"({timeout_ms / 1000:.1f}s)"
            ) from None
        return rid

    def _recv(self, address: str, op: str, rid: str, end: float) -> Any:
        s = self._sock(address)
        while True:
            wait_ms = max(0, int((end - time.monotonic()) * 1000))
            if not s.poll(wait_ms, zmq.POLLIN):
                self._drop(address)  # a late reply must not reach the next caller
                raise TransferError(f"unit {address} did not answer {op} in time")
            got_rid, ok, result = _dec.decode(s.recv())
            if got_rid != rid:
                continue  # stale reply from an earlier timed-out call
            if ok:
                return result
            raise_reply(address, op, result)

    def notify(self, address: str, op: str, **args: Any) -> None:
        """Fire-and-forget (no reply is sent for ``NO_REPLY`` ops)."""
        assert op in NO_REPLY, op
        try:
            self._sock(address).send(_enc.encode([op, "", args]), flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

    def close(self) -> None:
        for a in list(self._socks):
            self._drop(a)
        self.ctx.term()
