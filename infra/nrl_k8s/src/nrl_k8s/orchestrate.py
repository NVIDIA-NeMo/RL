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
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from ray.job_submission import JobStatus, JobSubmissionClient

from . import k8s, submit, workdir
from .config import LoadedConfig, get_username
from .manifest import (
    build_compute_domain_manifest,
    build_raycluster_manifest,
    build_roce_template_manifest,
    dra_resources_for_cluster,
)
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
    manifest = build_raycluster_manifest(cluster, loaded.infra, role=role)
    name = cluster.name
    namespace = loaded.infra.namespace

    ensure_dra_resources(role, loaded, log=log)
    log(f"[{role}] applying RayCluster {name} in namespace {namespace}")
    k8s.apply_raycluster(manifest, namespace)

    if wait_ready:
        log(f"[{role}] waiting for RayCluster {name} to reach state=ready ...")
        k8s.wait_for_raycluster_ready(name, namespace, timeout_s=ready_timeout_s)
        log(f"[{role}] RayCluster {name} is ready.")

    return name


def ensure_cluster(
    role: Role,
    loaded: LoadedConfig,
    *,
    log: callable,
    recreate: bool = False,
    wait_ready: bool = True,
    ready_timeout_s: int = 900,
) -> str:
    """Idempotent cluster up: reuse when live matches rendered, warn on drift.

    Unlike :func:`bring_up_cluster`, we never silently patch a live cluster.
    If the RayCluster already exists and its spec matches the rendered one
    we just wait for readiness. If it exists but drifted we log a warning
    and reuse anyway — pass ``recreate=True`` to delete + re-apply instead.
    """
    cluster = _require_cluster(loaded.infra, role)
    manifest = build_raycluster_manifest(cluster, loaded.infra, role=role)
    name = cluster.name
    namespace = loaded.infra.namespace

    live = k8s.get_raycluster(name, namespace)
    if live is not None:
        live_owner = (live.get("metadata", {}).get("labels") or {}).get(
            "nrl-k8s/owner"
        )
        me = get_username()
        if live_owner and live_owner != me:
            raise RuntimeError(
                f"RayCluster {name} in namespace {namespace} is owned by "
                f"'{live_owner}' (you are '{me}'). Use a different cluster "
                f"name or ask {live_owner} to tear it down."
            )
    ensure_dra_resources(role, loaded, log=log)
    if live is None:
        log(f"[{role}] applying RayCluster {name} in namespace {namespace}")
        k8s.apply_raycluster(manifest, namespace)
    elif _spec_drifted(live.get("spec") or {}, manifest["spec"]):
        if recreate:
            log(
                f"[{role}] --recreate: RayCluster {name} has drifted from the "
                f"rendered manifest; deleting and re-applying"
            )
            k8s.delete_raycluster(name, namespace)
            k8s.wait_for_raycluster_gone(name, namespace)
            k8s.apply_raycluster(manifest, namespace)
        else:
            log(
                f"[{role}] warning: live RayCluster {name} has drifted from the "
                f"rendered manifest; reusing as-is (pass --recreate to replace)"
            )
    else:
        log(f"[{role}] RayCluster {name} already exists and matches — reusing")

    if wait_ready:
        log(f"[{role}] waiting for RayCluster {name} to reach state=ready ...")
        k8s.wait_for_raycluster_ready(name, namespace, timeout_s=ready_timeout_s)
        log(f"[{role}] RayCluster {name} is ready.")

    return name


# Server-managed fields that never appear in a rendered manifest and should
# be ignored when diffing for drift.
_DRIFT_IGNORE_TOP = ("status",)
_DRIFT_IGNORE_METADATA = (
    "creationTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "selfLink",
    "uid",
)


def _spec_drifted(live_spec: dict, rendered_spec: dict) -> bool:
    """Return True if the live RayCluster ``.spec`` diverges from rendered."""
    return _strip_server_fields(live_spec) != _strip_server_fields(rendered_spec)


def _strip_server_fields(obj):
    """Recursively drop keys the API server injects so comparisons are stable."""
    if isinstance(obj, dict):
        return {
            k: _strip_server_fields(v)
            for k, v in obj.items()
            if k not in _DRIFT_IGNORE_TOP and k not in _DRIFT_IGNORE_METADATA
        }
    if isinstance(obj, list):
        return [_strip_server_fields(v) for v in obj]
    return obj


def ensure_dra_resources(
    role: Role,
    loaded: LoadedConfig,
    *,
    log: callable,
) -> None:
    """Create ComputeDomain / RoCE ResourceClaimTemplate if the spec needs them."""
    cluster = _get_cluster(loaded.infra, role)
    if cluster is None:
        return
    namespace = loaded.infra.namespace
    resources = dra_resources_for_cluster(cluster.name, role, cluster.spec)
    for kind, name in resources:
        if kind == "compute-domain":
            log(f"[{role}] ensuring ComputeDomain {name}")
            k8s.apply_compute_domain(
                build_compute_domain_manifest(name, namespace), namespace
            )
        elif kind == "roce":
            log(f"[{role}] ensuring RoCE ResourceClaimTemplate {name}")
            k8s.apply_resource_claim_template(
                build_roce_template_manifest(name, namespace), namespace
            )


def delete_dra_resources(
    role: Role,
    loaded: LoadedConfig,
    *,
    log: callable,
) -> None:
    """Delete DRA resources for a role."""
    cluster = _get_cluster(loaded.infra, role)
    if cluster is None:
        return
    namespace = loaded.infra.namespace
    resources = dra_resources_for_cluster(cluster.name, role, cluster.spec)
    for kind, name in reversed(resources):
        if kind == "roce":
            log(f"[{role}] deleting RoCE ResourceClaimTemplate {name}")
            k8s.delete_resource_claim_template(name, namespace)
        elif kind == "compute-domain":
            log(f"[{role}] deleting ComputeDomain {name}")
            k8s.delete_compute_domain(name, namespace)


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
            log(
                f"[{role}] daemon {daemon.submissionId} already {existing.value} — skipping submit"
            )
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
        try:
            job_id = submit.submit_ray_job(
                dash,
                entrypoint=daemon.entrypoint,
                working_dir=wd,
                env_vars=daemon.env,
                submission_id=submission_id,
            )
        finally:
            shutil.rmtree(wd, ignore_errors=True)
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
        raise ValueError("infra.launch.entrypoint must be set for `nrl-k8s run` / `rayjob`")

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
    # so the new one can claim GPUs.
    if replace and is_exec:
        from .submitters.exec_ import ExecSubmitter
        ExecSubmitter(exec_tmp_dir=infra.submit.execTmpDir).stop_all_running(
            name, infra.namespace, log=log,
        )
    elif replace:
        with submit.dashboard_url(name, infra.namespace) as dash:
            client = JobSubmissionClient(dash)
            for job in client.list_jobs():
                if job.status is JobStatus.RUNNING:
                    log(
                        f"[training] --replace: stopping running job {job.submission_id}"
                    )
                    try:
                        client.stop_job(job.submission_id)
                        _wait_job_stopped(
                            client, job.submission_id, log=log, role="training"
                        )
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

    try:
        handle = submitter.submit(
            name,
            infra.namespace,
            entrypoint=launch.entrypoint,
            run_id=run_id or "",
            env_vars=env_vars,
            working_dir=wd,
        )
    finally:
        if wd is not None:
            shutil.rmtree(wd, ignore_errors=True)
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
    skip_daemons: bool = False,
    recreate: bool = False,
) -> RunResult:
    """Idempotent bring-up + daemon + training submit.

    For each declared role, reuse the live RayCluster when its spec matches
    the rendered manifest, apply when it is absent, warn + reuse on drift
    (or delete + re-apply when ``recreate=True``). Then submit daemons and
    the training entrypoint.

    ``skip_daemons=True`` bypasses gym/generation daemon submission — use
    when those roles are already healthy and you only want to re-submit
    training.
    """
    if replace:
        _reset_endpoint_registry(loaded, log=log)

    for role in ALL_ROLES:
        if _get_cluster(loaded.infra, role) is None:
            log(f"[{role}] not defined in recipe — skipping")
            continue
        name = ensure_cluster(role, loaded, log=log, recreate=recreate)
        if skip_daemons and role != "training":
            log(f"[{role}] --skip-daemons: not submitting daemon")
            continue
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
    on fresh URLs instead of caching stragglers from a prior failed run.
    """
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
    log(
        f"[{role}] previous job {submission_id} did not stop within {timeout_s}s; continuing"
    )


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
    "ensure_cluster",
    "run",
    "submit_daemon",
    "submit_training",
]
