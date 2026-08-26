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
    try:
        client.connect(_held_port_uds_name(port))
        _, fds, _, _ = socket.recv_fds(client, 1024, 1)
    except OSError as e:
        raise RuntimeError(
            f"Could not receive the reserved server socket for port {port}: "
            "the port holder on this node is gone, so the pre-published URL would be unreachable."
        ) from e
    finally:
        client.close()
    if not fds:
        raise RuntimeError(f"Port holder for port {port} sent no file descriptor.")
    return socket.socket(fileno=fds[0])


class HeldPortReservation:
    """Bind-and-hold port reservation with fd handoff to a same-node process.

    The listening socket stays open from reservation until the eventual server adopts it,
    so there is zero gap in which the kernel could hand the port to anyone else.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
        finally:
            conn.close()
            self._uds.close()
            # The receiver holds a duplicate fd; the local one is done.
            self._sock.close()


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
# The caller pins this to the bundle rank 0 will occupy.
@ray.remote(num_cpus=0)  # pragma: no cover
class RemoteHeldPortReservation(HeldPortReservation):
    pass
