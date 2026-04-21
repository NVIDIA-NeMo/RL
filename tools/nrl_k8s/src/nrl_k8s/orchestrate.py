"""One-shot orchestration for a disaggregated run.

``nrl-k8s run <recipe>`` delegates here. The flow:

  1. For each role in order (generation, gym, training):
       - Apply the RayCluster manifest.
       - Wait for state=ready.
       - If the role has a daemon entrypoint, stage a fresh working_dir,
         submit the daemon as a Ray Job, and (if configured) wait on a
         health-check URL.
  2. Stage a working_dir for training and submit ``infra.launch.entrypoint``
     as a Ray Job against the training cluster.
  3. Return the training job's submission ID. Callers tail logs separately.

Every cluster-specific value — namespace, cluster names, image, ports,
entrypoints, node selectors — is read from the recipe. This module has no
hardcoded cluster assumptions.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from ray.job_submission import JobStatus, JobSubmissionClient

from . import k8s, submit, workdir
from .config import LoadedConfig
from .manifest import build_raycluster_manifest
from .schema import ClusterSpec, CodeSource, InfraConfig, SubmitterMode
from .submitters import SubmissionHandle, build_submitter, save_handle

Role = Literal["generation", "gym", "training"]
ALL_ROLES: tuple[Role, ...] = ("generation", "gym", "training")


@dataclass
class RunResult:
    """Outcome of a training submission.

    ``handle`` carries the transport-specific identifiers the observability
    commands need. ``training_dashboard`` / ``training_job_id`` remain for
    back-compat with earlier callers that printed Ray submission ids
    directly; they are set to ``None``/``""`` for exec submissions.
    """

    handle: SubmissionHandle
    training_dashboard: str | None = None
    training_job_id: str = ""


# =============================================================================
# Public API
# =============================================================================

def _fresh_submission_id(base: str) -> str:
    return f"{base}-{int(time.time())}"


def bring_up_cluster(
    role: Role,
    loaded: LoadedConfig,
    *,
    log: callable,
    wait_ready: bool = True,
    ready_timeout_s: int = 900,
) -> str:
    """Apply the RayCluster for ``role`` and wait for it to be ready."""
    cluster = _require_cluster(loaded.infra, role)
    manifest = build_raycluster_manifest(cluster, loaded.infra)
    name = cluster.name
    namespace = loaded.infra.namespace

    log(f"[{role}] applying RayCluster {name} in namespace {namespace}")
    k8s.apply_raycluster(manifest, namespace)

    if wait_ready:
        log(f"[{role}] waiting for RayCluster {name} to reach state=ready ...")
        k8s.wait_for_raycluster_ready(name, namespace, timeout_s=ready_timeout_s)
        log(f"[{role}] RayCluster {name} is ready.")

    return name


def submit_daemon(
    role: Role,
    loaded: LoadedConfig,
    cluster_name: str,
    *,
    log: callable,
    repo_root: Path,
    replace: bool = False,
) -> str | None:
    """If the role has a daemon spec, stage+submit it. Returns submission_id."""
    cluster = _require_cluster(loaded.infra, role)
    daemon = cluster.daemon
    if daemon is None:
        return None

    namespace = loaded.infra.namespace

    # One port-forward for both the status check and the submit avoids a
    # startup race where a separate check returns None while the forward
    # boots.
    with submit.dashboard_url(cluster_name, namespace) as dash:
        client = JobSubmissionClient(dash)

        existing = None
        if daemon.submissionId:
            try:
                existing = client.get_job_status(daemon.submissionId)
            except Exception:
                existing = None

        if existing in (JobStatus.RUNNING, JobStatus.SUCCEEDED) and not replace:
            log(f"[{role}] daemon {daemon.submissionId} already {existing.value} — skipping submit")
            return daemon.submissionId

        if existing in (JobStatus.FAILED, JobStatus.STOPPED) and not replace:
            raise RuntimeError(
                f"daemon {daemon.submissionId} is {existing.value} — "
                f"re-run with --replace (or bump infra.clusters.{role}.daemon.submissionId)"
            )

        # Ray refuses to re-use a submissionId even after terminal state, so
        # --replace picks a fresh suffix and stops the live one if any.
        submission_id = daemon.submissionId
        if replace and existing is not None:
            if existing is JobStatus.RUNNING:
                log(f"[{role}] --replace: stopping {daemon.submissionId}")
                try:
                    client.stop_job(daemon.submissionId)
                    _wait_job_stopped(client, daemon.submissionId, log=log, role=role)
                except Exception as exc:  # noqa: BLE001
                    log(f"[{role}] warning: stop failed: {exc}")
            if daemon.submissionId:
                submission_id = _fresh_submission_id(daemon.submissionId)
                log(f"[{role}] --replace: using fresh submissionId {submission_id}")

        upload_paths = daemon.rayUploadPaths or _upload_paths(loaded.infra)
        log(f"[{role}] staging working_dir for daemon ({len(upload_paths)} paths)")
        wd = workdir.stage_workdir(repo_root, include_paths=upload_paths)

        log(f"[{role}] submitting daemon via {dash}")
        job_id = submit.submit_ray_job(
            dash,
            entrypoint=daemon.entrypoint,
            working_dir=wd,
            env_vars=daemon.env,
            submission_id=submission_id,
        )
        log(f"[{role}] daemon submitted as job {job_id}")
        if daemon.healthCheckUrl:
            _wait_for_http(daemon.healthCheckUrl, daemon.healthCheckTimeoutS, log, role)
    return job_id


def submit_training(
    loaded: LoadedConfig,
    *,
    log: callable,
    repo_root: Path,
    replace: bool = False,
    run_id: str | None = None,
) -> RunResult:
    """Submit the training job against the training cluster.

    Dispatches on ``infra.submit.submitter`` + ``infra.launch.codeSource``:

    * ``submitter=portForward`` + ``codeSource=upload`` — today's path.
      Stages a working_dir, opens port-forward, submits via Ray SDK.
    * ``submitter=portForward`` + ``codeSource in (image, lustre)`` — no
      staging; Ray job inherits the head pod's cwd. The entrypoint is
      responsible for ``cd`` + ``source``.
    * ``submitter=exec`` — ``kubectl exec`` into the head, run the user
      entrypoint under ``nohup`` + ``disown``. No staging, no
      port-forward. ``codeSource`` must not be ``upload`` on this path.

    ``run_id`` is used as the submission id / pidfile tag. Ray
    port-forward submissions generate one if ``run_id`` is None (Ray's
    default behaviour); exec submissions require a non-empty value and
    caller is expected to synthesize one via :func:`default_run_id`.
    """
    infra = loaded.infra
    launch = infra.launch
    if not launch.entrypoint:
        raise ValueError("infra.launch.entrypoint must be set for `nrl-k8s launch`")

    if replace:
        _reset_endpoint_registry(loaded, log=log)

    cluster = _require_cluster(infra, "training")
    name = cluster.name

    submitter = build_submitter(infra)
    is_exec = infra.submit.submitter is SubmitterMode.EXEC
    upload = launch.codeSource is CodeSource.UPLOAD

    if is_exec and upload:
        raise ValueError(
            "infra.submit.submitter=exec is incompatible with "
            "infra.launch.codeSource=upload — pick image or lustre, "
            "or switch submitter to portForward."
        )

    # Stage only if we're actually uploading. Exec + image/lustre modes
    # rely on the code being on the pod's filesystem already.
    wd: Path | None = None
    if upload:
        log("[training] staging working_dir ...")
        recipe_yaml = OmegaConf.to_yaml(loaded.recipe)
        wd = workdir.stage_workdir(
            repo_root,
            include_paths=_upload_paths(infra),
            extra_files={"nrl_k8s_run.yaml": recipe_yaml},
        )

    # `--replace` semantics: stop any running job on the training cluster
    # so the new one can claim GPUs. Only applies to the Ray path; exec
    # runs are keyed by run_id (unique per submission) so replace is a
    # no-op there.
    if replace and not is_exec:
        with submit.dashboard_url(name, infra.namespace) as dash:
            client = JobSubmissionClient(dash)
            for job in client.list_jobs():
                if job.status is JobStatus.RUNNING:
                    log(f"[training] --replace: stopping running job {job.submission_id}")
                    try:
                        client.stop_job(job.submission_id)
                        _wait_job_stopped(client, job.submission_id, log=log, role="training")
                    except Exception as exc:  # noqa: BLE001
                        log(f"[training] warning: stop failed: {exc}")

    if is_exec:
        run_id = run_id or default_run_id("training")
        log(f"[training] exec submitter: launching as run_id={run_id} on head pod")
    else:
        log("[training] port-forward submitter: submitting Ray Job")

    # Always expose the run id to the training entrypoint. Both transports
    # see the same variable so recipe authors can reference
    # ``$NRL_K8S_RUN_ID`` (in ``logger.wandb.name`` etc.) without caring
    # which submitter the CLI picked.
    env_vars = {**dict(launch.env)}
    if run_id:
        env_vars.setdefault("NRL_K8S_RUN_ID", run_id)

    handle = submitter.submit(
        name,
        infra.namespace,
        entrypoint=launch.entrypoint,
        run_id=run_id or "",
        env_vars=env_vars,
        working_dir=wd,
    )
    save_handle(handle)
    log(f"[training] training run handle: kind={handle.kind} id={handle.run_id}")
    return RunResult(
        handle=handle,
        training_dashboard=None,  # per-call port-forward, not persistent
        training_job_id=handle.run_id,
    )


def default_run_id(role: str) -> str:
    """Human-readable default id when the user didn't supply ``--run-id``."""
    return f"{role}-{int(time.time())}"


def run(
    loaded: LoadedConfig,
    *,
    log: callable,
    repo_root: Path,
    replace: bool = False,
    run_id: str | None = None,
) -> RunResult:
    """Do the full sequence: bring up all 3 clusters + daemons, submit training."""
    if replace:
        _reset_endpoint_registry(loaded, log=log)

    for role in ALL_ROLES:
        if _get_cluster(loaded.infra, role) is None:
            log(f"[{role}] not defined in recipe — skipping")
            continue
        name = bring_up_cluster(role, loaded, log=log)
        submit_daemon(role, loaded, name, log=log, repo_root=repo_root, replace=replace)

    return submit_training(
        loaded, log=log, repo_root=repo_root, replace=replace, run_id=run_id
    )


_JOB_ID_RE = re.compile(r"--job-id[= ]+(\S+)")


def _infer_disagg_job_id(infra: InfraConfig) -> str | None:
    """Best-effort extraction of the gym's ``--job-id`` from its entrypoint.

    The endpoint-registry ConfigMap is named ``nemo-rl-endpoints-<job_id>``;
    gym publishes ``gym_head_server`` there and training publishes
    ``vllm_base_urls``. We parse the id from the gym daemon entrypoint so
    ``--replace`` can delete the ConfigMap without a dedicated config key.
    """
    gym = infra.clusters.gym
    if gym is None or gym.daemon is None:
        return None
    m = _JOB_ID_RE.search(gym.daemon.entrypoint)
    return m.group(1) if m else None


def _reset_endpoint_registry(loaded: LoadedConfig, *, log: callable) -> None:
    """Delete the endpoint-registry ConfigMap so gym + training rendezvous
    on fresh URLs instead of caching stragglers from a prior failed run."""
    job_id = _infer_disagg_job_id(loaded.infra)
    if not job_id:
        return
    cm_name = f"nemo-rl-endpoints-{job_id}"
    if k8s.delete_configmap(cm_name, loaded.infra.namespace):
        log(f"[replace] deleted endpoint registry ConfigMap {cm_name}")


# =============================================================================
# Internals
# =============================================================================

def _get_cluster(infra: InfraConfig, role: Role) -> ClusterSpec | None:
    return getattr(infra.clusters, role)


def _require_cluster(infra: InfraConfig, role: Role) -> ClusterSpec:
    cluster = _get_cluster(infra, role)
    if cluster is None:
        raise ValueError(f"infra.clusters.{role} is not defined")
    return cluster


def _upload_paths(infra: InfraConfig) -> list[str]:
    """Resolve the list of repo-relative paths to stage for Ray uploads."""
    if infra.launch.rayUploadPaths is not None:
        return list(infra.launch.rayUploadPaths)
    return list(workdir.DEFAULT_RAY_UPLOAD_PATHS)


_TERMINAL = (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.SUCCEEDED)


def _wait_job_stopped(
    client: JobSubmissionClient,
    submission_id: str,
    *,
    log: callable,
    role: Role,
    timeout_s: int = 60,
) -> None:
    """Block until a Ray Job reaches a terminal state after a stop_job call."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status = client.get_job_status(submission_id)
        except Exception:
            return
        if status in _TERMINAL:
            log(f"[{role}] previous job {submission_id} → {status.value}")
            return
        time.sleep(2)
    log(f"[{role}] previous job {submission_id} did not stop within {timeout_s}s; continuing")


def _wait_for_http(url: str, timeout_s: int, log: callable, role: Role) -> None:
    log(f"[{role}] waiting for health-check {url} (timeout {timeout_s}s)")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if 200 <= r.status < 500:
                    log(f"[{role}] health-check {url} responded {r.status}")
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    raise TimeoutError(f"health-check {url} did not respond within {timeout_s}s")


__all__ = [
    "ALL_ROLES",
    "RunResult",
    "bring_up_cluster",
    "default_run_id",
    "run",
    "submit_daemon",
    "submit_training",
]
