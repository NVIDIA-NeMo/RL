"""Exercise SGLang test helpers across real Ray workers without GPUs or SGLang."""

import json
import multiprocessing
import subprocess
import sys
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Connection

import psutil
import pytest
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tests.unit.models.generation.sglang.fault_injection import (
    KilledServer,
    ServerProcessTree,
    kill_server_process_tree,
    snapshot_server_process_tree,
    wait_for_server_process_tree_exit,
)


class _LoadHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/get_load":
            self.send_error(404)
            return
        payload = json.dumps([{"num_reqs": 2, "num_waiting_reqs": 1}]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


def _serve_load(ready: Connection) -> None:
    """Publish an owned port and descendant PID before accepting test requests."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with ThreadingHTTPServer(("127.0.0.1", 0), _LoadHandler) as server:
            ready.send((f"http://127.0.0.1:{server.server_port}", child.pid))
            ready.close()
            server.serve_forever()
    finally:
        child.kill()
        child.wait(timeout=10)


@pytest.fixture(scope="module")
def helper_ray_cluster() -> Iterator[None]:
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=2, num_gpus=0, include_dashboard=False)
    yield
    if started_here:
        ray.shutdown()


@pytest.fixture
def dummy_server() -> Iterator[tuple[str, int, int]]:
    context = multiprocessing.get_context("spawn")
    ready, sender = context.Pipe(duplex=False)
    server = context.Process(target=_serve_load, args=(sender,))
    server.start()
    sender.close()
    child = None
    try:
        assert ready.poll(30), "Dummy server did not publish its port"
        server_url, child_pid = ready.recv()
        child = psutil.Process(child_pid)
        assert server.pid is not None
        yield server_url, server.pid, child_pid
    finally:
        ready.close()
        if server.is_alive():
            server.kill()
        if child is not None:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        server.join(timeout=10)
        assert not server.is_alive(), "Dummy server survived cleanup"


@pytest.mark.parametrize("wait_for_inflight", [False, True])
def test_fault_injection_helpers_cross_real_ray_processes(
    helper_ray_cluster,
    dummy_server: tuple[str, int, int],
    wait_for_inflight: bool,
) -> None:
    server_url, server_pid, child_pid = dummy_server
    strategy = NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(), soft=False
    )
    tree = ray.get(
        snapshot_server_process_tree.options(scheduling_strategy=strategy).remote(
            server_url
        ),
        timeout=30,
    )
    assert isinstance(tree, ServerProcessTree)
    assert (
        type(tree).__module__ == "tests.unit.models.generation.sglang.fault_injection"
    )
    assert {server_pid, child_pid} <= {pid for pid, _ in tree.processes}

    killed = ray.get(
        kill_server_process_tree.options(scheduling_strategy=strategy).remote(
            [server_url], wait_for_inflight=wait_for_inflight, timeout=5
        ),
        timeout=45,
    )
    assert isinstance(killed, KilledServer)
    assert killed.url == server_url
    assert killed.pid == server_pid
    assert child_pid in killed.child_pids
    assert killed.running_requests == (1 if wait_for_inflight else 0)

    # Passing the snapshot back into another worker also exercises inbound
    # deserialization of its canonical module, not only the function import.
    ray.get(
        wait_for_server_process_tree_exit.options(scheduling_strategy=strategy).remote(
            tree, timeout=10
        ),
        timeout=20,
    )
