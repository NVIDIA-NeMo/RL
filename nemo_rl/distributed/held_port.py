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
import socket
import threading

import ray

from nemo_rl.distributed.virtual_cluster import _get_node_ip_local

_HANDOFF_RELEASED = b"\x01"


def _held_port_uds_name(port: int) -> str:
    """Abstract-namespace Unix socket where a HeldPortReservation serves its fd."""
    return f"\0nemo_rl_held_port_{port}"


def receive_held_socket(port: int) -> socket.socket:
    """Adopt the listening socket held by this node's HeldPortReservation.

    Args:
        port: The reserved port; names the same-node handoff endpoint.

    Returns:
        The live listening socket, duplicated into this process.
    """
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    fds: list[int] = []
    received_socket: socket.socket
    try:
        client.connect(_held_port_uds_name(port))
        _, fds, _, _ = socket.recv_fds(client, 1024, 1)
        if not fds:
            raise RuntimeError(f"Port holder for port {port} sent no file descriptor.")
        # The receiving process now owns a duplicate of the reservation socket,
        # but callers such as MCore close it before binding their own
        # SO_REUSEPORT listeners. Wait until the holder has closed its original
        # descriptor so those listeners cannot race the old, non-reusable
        # reservation socket and fail with EADDRINUSE.
        released = client.recv(1)
        if released != _HANDOFF_RELEASED:
            socket.close(fds.pop())
            raise RuntimeError(
                f"Port holder for port {port} did not confirm releasing its socket."
            )
        received_socket = socket.socket(fileno=fds.pop())
    except OSError as e:
        for fd in fds:
            socket.close(fd)
        raise RuntimeError(
            f"Could not receive the reserved server socket for port {port}: "
            "the port holder on this node is gone, so the pre-published URL would be unreachable."
        ) from e
    finally:
        client.close()
    return received_socket


class HeldPortReservation:
    """Bind-and-hold port reservation with fd handoff to a same-node process.

    The listening socket stays open from reservation until the eventual server adopts it,
    so there is zero gap in which the kernel could hand the port to anyone else.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # MCore starts multiple frontend replicas that each bind this port with
        # SO_REUSEPORT. Make the reservation socket part of the same reuse group
        # so their binds remain valid while the handed-off fd is being closed.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._sock.bind(("", 0))
        self._sock.listen(128)
        self._port = self._sock.getsockname()[1]
        self._node_ip = _get_node_ip_local()
        self._uds = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._uds.bind(_held_port_uds_name(self._port))
        self._uds.listen(1)
        threading.Thread(target=self._serve_fd_once, daemon=True).start()

    def address(self) -> tuple[str, int]:
        """Return (node_ip, held_port)."""
        return self._node_ip, self._port

    def _serve_fd_once(self) -> None:
        conn, _ = self._uds.accept()
        try:
            socket.send_fds(conn, [b"s"], [self._sock.fileno()])
            # The receiver holds a duplicate fd, so the port remains reserved.
            # Close this copy before acknowledging the handoff; the receiver may
            # immediately close its copy and rebind the port with SO_REUSEPORT.
            self._sock.close()
            conn.sendall(_HANDOFF_RELEASED)
        finally:
            conn.close()
            self._uds.close()
            self._sock.close()


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
# The caller pins this to the bundle rank 0 will occupy.
@ray.remote(num_cpus=0)  # pragma: no cover
class RemoteHeldPortReservation(HeldPortReservation):
    pass
