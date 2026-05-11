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
"""Submit a Ray Job to a named RayCluster.

Two dashboard access modes:

* **In-cluster**: when the CLI runs inside a pod that can reach the cluster
  DNS (``<raycluster>-head-svc.<ns>.svc.cluster.local:8265``), we hit the
  dashboard directly. Detected via ``KUBERNETES_SERVICE_HOST``.

* **Laptop**: spawn a ``kubectl port-forward`` subprocess to forward
  ``<raycluster>-head-svc:8265`` to a local port and submit via
  ``http://127.0.0.1:<port>``. The subprocess is killed on context exit.

No other cluster-specific assumptions: namespace, cluster name, port are
all passed in.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

DASHBOARD_PORT = 8265  # KubeRay convention (headGroupSpec containerPort).


# =============================================================================
# kubectl preflight (cached)
# =============================================================================

# Cache the probe result for the lifetime of the CLI invocation — kubectl
# auth doesn't flip between one command's validate and apply.
_KUBECTL_OK: bool | None = None


def kubectl_preflight(namespace: str) -> None:
    """Fail early with an actionable error if kubectl is missing or unauthenticated.

    Running two cheap checks up-front is much friendlier than watching a
    port-forward die with "The connection to the server … was refused" 30s
    later. Result is cached for the process so we don't re-probe on every
    ``dashboard_url`` call inside one CLI invocation.
    """
    global _KUBECTL_OK
    if _KUBECTL_OK:
        return
    if shutil.which("kubectl") is None:
        raise RuntimeError(
            "kubectl not found on PATH — install it or run nrl-k8s from an "
            "in-cluster pod."
        )
    try:
        subprocess.run(
            ["kubectl", "version", "--client"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "`kubectl version --client` failed — is kubectl installed correctly? "
            f"({exc})"
        ) from exc
    try:
        res = subprocess.run(
            ["kubectl", "auth", "can-i", "get", "pods", "-n", namespace],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"`kubectl auth can-i` timed out contacting the API server for "
            f"namespace {namespace}; is the cluster reachable? (try `aws sso login`)"
        ) from exc
    verdict = (res.stdout or "").strip().lower()
    if verdict != "yes":
        raise RuntimeError(
            f"kubectl has no permission to get pods in namespace {namespace} — "
            f"authenticate (e.g. `aws sso login`) or request the edit role."
        )
    _KUBECTL_OK = True


# =============================================================================
# Public API
# =============================================================================


def is_in_cluster() -> bool:
    """Heuristic: are we running inside a Kubernetes pod?"""
    return "KUBERNETES_SERVICE_HOST" in os.environ


class _PortForward:
    """Manage a long-running ``kubectl port-forward`` with stdout drainage.

    The subprocess pipes are drained on a daemon thread so a long
    ``tail_job_logs`` session doesn't block on a full pipe buffer.
    ``restart()`` is available so HTTP clients can recover from a forward
    that died mid-call without tearing down the whole context.
    """

    def __init__(self, cluster_name: str, namespace: str, port: int) -> None:
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.port = port
        self._proc: subprocess.Popen | None = None
        self._drain: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        cmd = [
            "kubectl",
            "-n",
            self.namespace,
            "port-forward",
            f"svc/{self.cluster_name}-head-svc",
            f"{self.port}:{DASHBOARD_PORT}",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._stop.clear()
        self._drain = threading.Thread(
            target=self._drain_pipe, name="nrl-k8s-pf-drain", daemon=True
        )
        self._drain.start()
        _wait_for_tcp("127.0.0.1", self.port, timeout_s=30, proc=self._proc)

    def _drain_pipe(self) -> None:
        """Consume stdout so a long session doesn't stall on a full pipe."""
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            for _ in iter(self._proc.stdout.readline, ""):
                if self._stop.is_set():
                    return
        except (ValueError, OSError):
            # Pipe was closed under us during shutdown — not an error.
            return

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def restart(self) -> None:
        """Kill the old process and start a new one on the same local port."""
        self.stop()
        self.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is None:
            return
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        if proc.poll() is None:
            proc.kill()
        self._proc = None


@contextlib.contextmanager
def dashboard_url(
    cluster_name: str,
    namespace: str,
    *,
    local_port: int | None = None,
) -> Iterator[str]:
    """Yield a URL to the dashboard, managing a port-forward if needed."""
    if is_in_cluster():
        yield f"http://{cluster_name}-head-svc.{namespace}.svc.cluster.local:{DASHBOARD_PORT}"
        return

    if shutil.which("kubectl") is None:
        raise RuntimeError(
            "kubectl not found on PATH — install it or run nrl-k8s from an "
            "in-cluster pod."
        )

    port = local_port or _free_port()
    pf = _PortForward(cluster_name, namespace, port)
    pf.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        pf.stop()


def submit_ray_job(
    dashboard: str,
    *,
    entrypoint: str,
    working_dir: Path,
    env_vars: dict[str, str] | None = None,
    submission_id: str | None = None,
    pip: list[str] | None = None,
) -> str:
    """Submit a Ray Job. Returns the job submission ID."""
    from ray.job_submission import JobSubmissionClient

    runtime_env: dict[str, Any] = {"working_dir": str(working_dir)}
    if env_vars:
        runtime_env["env_vars"] = dict(env_vars)
    if pip:
        runtime_env["pip"] = list(pip)

    client = JobSubmissionClient(dashboard)
    submitted = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    return submitted


def tail_job_logs(dashboard: str, job_id: str) -> Iterator[str]:
    """Yield stdout/stderr lines from a running job.

    Bridges Ray's async iterator into a sync generator via a daemon thread +
    queue. Daemon-thread status means the reader stops when the caller exits
    (including on KeyboardInterrupt) without us having to coordinate event
    loops across threads.
    """
    import asyncio
    import queue
    import threading

    from ray.job_submission import JobSubmissionClient

    q: queue.Queue = queue.Queue()
    sentinel: object = object()

    def _worker() -> None:
        async def _run() -> None:
            client = JobSubmissionClient(dashboard)
            async for line in client.tail_job_logs(job_id):
                q.put(line)

        try:
            asyncio.run(_run())
        except Exception as exc:  # noqa: BLE001
            q.put(f"\n[tail error: {exc}]\n")
        finally:
            q.put(sentinel)

    threading.Thread(target=_worker, name="nrl-k8s-tail", daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item


def wait_for_job(
    dashboard: str,
    job_id: str,
    *,
    timeout_s: int | None = None,
    poll_s: int = 10,
) -> str:
    """Block until a job reaches a terminal state; return that state."""
    from ray.job_submission import JobStatus, JobSubmissionClient

    client = JobSubmissionClient(dashboard)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while deadline is None or time.monotonic() < deadline:
        status = client.get_job_status(job_id)
        if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
            return status.value
        time.sleep(poll_s)
    raise TimeoutError(f"job {job_id} did not finish in {timeout_s}s")


# =============================================================================
# Internals
# =============================================================================


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_tcp(
    host: str, port: int, *, timeout_s: int, proc: subprocess.Popen
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = ""
            if proc.stderr:
                try:
                    stderr = proc.stderr.read().decode(errors="replace").strip()
                except Exception:
                    pass
            msg = f"kubectl port-forward exited early (rc={proc.returncode})"
            if stderr:
                msg += f": {stderr}"
            raise RuntimeError(msg)
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(
        f"kubectl port-forward didn't open {host}:{port} in {timeout_s}s"
    )


__all__ = [
    "DASHBOARD_PORT",
    "dashboard_url",
    "is_in_cluster",
    "kubectl_preflight",
    "submit_ray_job",
    "tail_job_logs",
    "wait_for_job",
]
