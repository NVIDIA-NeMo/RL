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
thread; clients talk to it over DEALER sockets lent by TQ's socket pool. Ray is used only to
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
UnitFull ``UnitFull``                        put fails (SSD on: spills here)
Busy     ``DataPlaneBusy(reason)``           retry this unit later
Missing  ``LostBlobs(blobs)``                the unit no longer has them
Protocol ``TransferError``                   any other server exception
======== =================================== ==========================
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
import uuid
from typing import Any, Callable

import msgspec
import zmq
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQServerInfo, ZMQSocketPool

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


def reply_error(address: str, op: str, err: Any) -> Exception:
    """Rebuild the exception behind an error reply."""
    if not isinstance(err, dict):  # pre-structured peer: [type_name, msg]
        err = {"kind": "Protocol", "msg": ": ".join(map(str, err))}
    kind, msg = err.get("kind"), err.get("msg", "")
    if kind == "UnitFull":
        return UnitFull(msg)
    if kind == "Busy":
        return DataPlaneBusy(f"unit {address} {op}: {msg}", reason=err.get("reason"))
    if kind == "Missing":
        return LostBlobs(list(err.get("blobs") or []), msg or "blob not on unit")
    return TransferError(f"unit {address} {op}: {msg}")


def node_ip() -> str:
    """This node's routable IP (what Ray advertises), falling back to a UDP probe."""
    try:
        import ray

        return ray.util.get_node_ip_address()
    except Exception:  # noqa: BLE001
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip


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
    """Requests to storage units over DEALER sockets lent by TransferQueue's pool.

    A socket goes back to the pool only after a complete reply with its
    request's id; after anything else a late reply may still arrive, so it is closed.
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        self.ctx = zmq.Context()
        self.timeout_ms = int(timeout_s * 1000)
        self._pool = ZMQSocketPool(self.ctx, "nvdp_unit_client", "ctl")

    def call(
        self, address: str, op: str, *, timeout_s: float | None = None, **args: Any
    ) -> Any:
        """Request/reply.

        Raises what the unit raised (see the module docstring), and
        ``TransferError`` when no reply comes within ``timeout_s`` (default: the
        client's timeout; callers pass their remaining operation budget).
        """
        end = time.monotonic() + self._timeout_ms(timeout_s) / 1000
        with self._pool.lease(_peer(address)) as s:
            ok, result = self._recv(s, address, op, self._send(s, op, args), end)
        if not ok:  # a complete reply: raised outside the lease, which keeps the socket
            raise reply_error(address, op, result)
        return result

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
        end = time.monotonic() + self._timeout_ms(timeout_s) / 1000
        out: dict[str, Any] = {}
        sent: dict[str, tuple[zmq.Socket, str]] = {}
        with contextlib.ExitStack() as leases:
            for address, args in args_by_address.items():
                s = None
                try:
                    s = leases.enter_context(self._pool.lease(_peer(address)))
                    sent[address] = (s, self._send(s, op, args))
                except Exception as e:  # noqa: BLE001 - reported per address
                    if s is not None:  # caught, so the lease would keep it: close
                        s.close(linger=0)
                    out[address] = e
            for address, (s, rid) in sent.items():
                try:
                    ok, result = self._recv(s, address, op, rid, end)
                except Exception as e:  # noqa: BLE001 - reported per address
                    s.close(linger=0)
                    out[address] = e
                    continue
                out[address] = result if ok else reply_error(address, op, result)
        return out

    def _timeout_ms(self, timeout_s: float | None) -> int:
        if timeout_s is None:
            return self.timeout_ms
        return max(1, min(self.timeout_ms, int(timeout_s * 1000)))

    @staticmethod
    def _send(s: zmq.Socket, op: str, args: dict[str, Any]) -> str:
        rid = uuid.uuid4().hex[:12]
        try:  # TQ's DEALER has no send high-water mark, so this does not block
            s.send(_enc.encode([op, rid, args]), flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            raise TransferError(f"could not send {op}: {e}") from None
        return rid

    @staticmethod
    def _recv(s: zmq.Socket, address: str, op: str, rid: str, end: float) -> Any:
        if not s.poll(max(0, int((end - time.monotonic()) * 1000)), zmq.POLLIN):
            raise TransferError(f"unit {address} did not answer {op} in time")
        got_rid, ok, result = _dec.decode(s.recv())
        if got_rid != rid:
            raise TransferError(
                f"unit {address} answered {op} with another request's id"
            )
        return ok, result

    def notify(self, address: str, op: str, **args: Any) -> None:
        """Fire-and-forget (no reply is sent for ``NO_REPLY`` ops)."""
        assert op in NO_REPLY, op
        with self._pool.lease(_peer(address)) as s:
            try:
                s.send(_enc.encode([op, "", args]), flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

    def close(self) -> None:
        self._pool.close()
        self.ctx.term()


def _peer(address: str) -> ZMQServerInfo:
    """A unit's ``tcp://ip:port`` as the ``ZMQServerInfo`` TQ's pool dials."""
    ip, port = address.removeprefix("tcp://").rsplit(":", 1)
    return ZMQServerInfo(Role.STORAGE, address, ip, {"ctl": int(port)})
