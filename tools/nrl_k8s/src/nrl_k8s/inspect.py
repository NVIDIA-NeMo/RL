"""Read-only introspection of the clusters a recipe owns.

Used by ``nrl-k8s status`` and ``nrl-k8s logs`` to summarise what's running
and stream logs from it. Everything here is idempotent — no apply, no
delete, no job submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from kubernetes import client

from . import k8s, submit
from .config import LoadedConfig
from .orchestrate import ALL_ROLES
from .schema import ClusterSpec, InfraConfig


@dataclass
class RayClusterPods:
    head_name: str | None = None
    head_phase: str | None = None
    worker_names: list[str] = field(default_factory=list)
    worker_phases: list[str] = field(default_factory=list)


@dataclass
class ClusterStatus:
    role: str
    name: str
    state: str                       # "ready" | "—" | other KubeRay states
    head_pod: str | None
    head_phase: str | None            # "Running" | "Pending" | ...
    worker_phases: list[str]          # one per worker pod
    daemon_submission_id: str | None
    daemon_status: str | None         # Ray JobStatus string, if reachable


# =============================================================================
# Status
# =============================================================================

def collect_status(loaded: LoadedConfig) -> list[ClusterStatus]:
    """Build a :class:`ClusterStatus` for every role declared in the recipe."""
    out: list[ClusterStatus] = []
    infra = loaded.infra
    for role in ALL_ROLES:
        cluster: ClusterSpec | None = getattr(infra.clusters, role)
        if cluster is None:
            continue
        out.append(_status_for(role, cluster, infra))
    return out


def _status_for(role: str, cluster: ClusterSpec, infra: InfraConfig) -> ClusterStatus:
    obj = k8s.get_raycluster(cluster.name, infra.namespace)
    state = (obj or {}).get("status", {}).get("state", "—") if obj else "(not found)"

    pods = list_cluster_pods(cluster.name, infra.namespace)

    daemon_id: str | None = None
    daemon_status: str | None = None
    if obj is not None and state == "ready" and cluster.daemon is not None:
        daemon_id, daemon_status = _latest_daemon_job(
            cluster.name, infra.namespace, cluster.daemon.submissionId
        )

    return ClusterStatus(
        role=role,
        name=cluster.name,
        state=state,
        head_pod=pods.head_name,
        head_phase=pods.head_phase,
        worker_phases=pods.worker_phases,
        daemon_submission_id=daemon_id,
        daemon_status=daemon_status,
    )


def list_cluster_pods(cluster_name: str, namespace: str) -> RayClusterPods:
    """Return the head and worker pods for a RayCluster."""
    k8s.load_kubeconfig()
    core = client.CoreV1Api()
    out = RayClusterPods()
    for p in core.list_namespaced_pod(
        namespace=namespace, label_selector=f"ray.io/cluster={cluster_name}"
    ).items:
        kind = (p.metadata.labels or {}).get("ray.io/node-type", "")
        if kind == "head":
            out.head_name = p.metadata.name
            out.head_phase = p.status.phase
        else:
            out.worker_names.append(p.metadata.name)
            out.worker_phases.append(p.status.phase)
    return out


def _latest_daemon_job(
    cluster_name: str, namespace: str, base_submission_id: str
) -> tuple[str | None, str | None]:
    """Find the most recent Ray Job whose submission_id matches the base id
    or a ``--replace``-suffixed variant (``<base>-<timestamp>``), and return
    its (submission_id, status) pair. Returns (None, None) on any error."""
    try:
        from ray.job_submission import JobSubmissionClient

        with submit.dashboard_url(cluster_name, namespace) as dash:
            clnt = JobSubmissionClient(dash)
            matches = [
                j for j in clnt.list_jobs()
                if j.submission_id == base_submission_id
                or j.submission_id.startswith(f"{base_submission_id}-")
            ]
            if not matches:
                return (base_submission_id, None)
            latest = max(matches, key=lambda j: j.start_time or 0)
            return (latest.submission_id, latest.status.value)
    except Exception:
        return (base_submission_id, None)


# =============================================================================
# Logs
# =============================================================================

def stream_pod_logs(
    pod_name: str,
    namespace: str,
    *,
    container: str | None = None,
    follow: bool = False,
    tail_lines: int | None = 200,
) -> Iterator[str]:
    """Stream stdout/stderr from a specific pod."""
    k8s.load_kubeconfig()
    core = client.CoreV1Api()
    stream = core.read_namespaced_pod_log(
        name=pod_name,
        namespace=namespace,
        container=container,
        follow=follow,
        tail_lines=tail_lines,
        _preload_content=False,
    )
    try:
        for raw in stream.stream():
            yield raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
    finally:
        stream.release_conn()


def head_pod_name(cluster_name: str, namespace: str) -> str:
    """Return the head pod's name for a RayCluster, raising if missing."""
    pods = list_cluster_pods(cluster_name, namespace)
    if pods.head_name is None:
        raise RuntimeError(
            f"no head pod found for RayCluster {cluster_name} in {namespace}"
        )
    return pods.head_name


__all__ = [
    "ClusterStatus",
    "RayClusterPods",
    "collect_status",
    "head_pod_name",
    "list_cluster_pods",
    "stream_pod_logs",
]
