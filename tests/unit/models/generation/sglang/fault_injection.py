"""Node-local process-tree fault injection for real SGLang tests."""

import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import psutil
import ray
import requests


@dataclass
class KilledServer:
    url: str
    pid: int
    child_pids: list[int]
    running_requests: int


@dataclass
class ServerProcessTree:
    url: str
    # PID and creation time distinguish the captured process from PID reuse.
    processes: list[tuple[int, float]]


def _find_servers(server_urls: list[str]) -> dict[str, psutil.Process]:
    connections = psutil.net_connections(kind="tcp")
    servers = {}
    for url in server_urls:
        port = urlsplit(url).port
        assert port is not None, f"Missing server port: {url}"
        pids = {
            connection.pid
            for connection in connections
            if connection.status == psutil.CONN_LISTEN
            and connection.laddr.port == port
            and connection.pid is not None
        }
        assert len(pids) == 1, f"Expected one listener for {url}, found {pids}"
        servers[url] = psutil.Process(pids.pop())
    return servers


@ray.remote(num_cpus=0, num_gpus=0)
def snapshot_server_process_tree(server_url: str) -> ServerProcessTree:
    """Capture the exact HTTP server and GPU descendants before fault injection."""
    server = _find_servers([server_url])[server_url]
    return ServerProcessTree(
        url=server_url,
        processes=[
            (process.pid, process.create_time())
            for process in [server, *server.children(recursive=True)]
        ],
    )


@ray.remote(num_cpus=0, num_gpus=0)
def wait_for_server_process_tree_exit(
    tree: ServerProcessTree, *, timeout: float
) -> None:
    """Require captured processes to exit and the HTTP port to stop listening."""
    port = urlsplit(tree.url).port
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live_pids = []
        for pid, created_at in tree.processes:
            try:
                process = psutil.Process(pid)
                if process.create_time() == created_at and process.status() not in (
                    psutil.STATUS_ZOMBIE,
                    psutil.STATUS_DEAD,
                ):
                    live_pids.append(pid)
            except psutil.NoSuchProcess:
                pass
        listening = any(
            connection.status == psutil.CONN_LISTEN and connection.laddr.port == port
            for connection in psutil.net_connections(kind="tcp")
        )
        if not live_pids and not listening:
            return
        time.sleep(0.1)
    raise TimeoutError(
        f"Server teardown left processes {live_pids}; "
        f"{tree.url} still listening: {listening}"
    )


@ray.remote(num_cpus=0, num_gpus=0)
def kill_server_process_tree(
    server_urls: list[str], *, wait_for_inflight: bool, timeout: float
) -> KilledServer:
    """SIGKILL one identified server and its descendants without deregistering it.

    The caller must place this task on the engines' node and pause synthetic
    serving probes while observing load, so an in-flight request belongs to
    the generation batch under test.
    """
    servers = _find_servers(server_urls)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for url, server in servers.items():
            running_requests = 0
            if wait_for_inflight:
                response = requests.get(f"{url}/get_load", timeout=5)
                response.raise_for_status()
                # Pinned SGLang 3003d70f, http_server.py:772-796: /get_load
                # returns one object per DP rank, including running + waiting.
                running_requests = sum(
                    load["num_reqs"] - load["num_waiting_reqs"]
                    for load in response.json()
                )
                if running_requests == 0:
                    continue

            children = server.children(recursive=True)
            processes = [server, *children]
            # Kill the HTTP parent first so the request loses its connection;
            # use the captured descendants to reap GPU workers afterwards.
            for process in processes:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass

            reap_deadline = time.monotonic() + 30
            while time.monotonic() < reap_deadline:
                live_pids = []
                for process in processes:
                    try:
                        if process.is_running() and process.status() not in (
                            psutil.STATUS_ZOMBIE,
                            psutil.STATUS_DEAD,
                        ):
                            live_pids.append(process.pid)
                    except psutil.NoSuchProcess:
                        pass
                if not live_pids:
                    return KilledServer(
                        url=url,
                        pid=server.pid,
                        child_pids=[child.pid for child in children],
                        running_requests=running_requests,
                    )
                time.sleep(0.1)
            raise TimeoutError(f"SIGKILL left live server processes: {live_pids}")
        time.sleep(0.02)

    raise TimeoutError(
        f"No running generation request observed on {server_urls} within {timeout}s"
    )
