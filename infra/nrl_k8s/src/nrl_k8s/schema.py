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
"""Pydantic schema for the ``infra:`` section of a NeMo-RL recipe.

A recipe YAML is a standard NeMo-RL recipe plus an optional top-level ``infra:``
mapping. The CLI merges recipe-level ``infra:`` with a user-level defaults file
(``~/.config/nrl-k8s/defaults.yaml``) and a shipped defaults file
(``defaults/defaults.example.yaml``) before validating the result through
:class:`InfraConfig`.

Strict validation (``extra='forbid'``) surfaces typos early. Every field that
isn't strictly cluster-identifying has a sensible default so short ``infra:``
blocks work on well-configured clusters.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    """Base pydantic model with strict extra-field rejection."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# =============================================================================
# Scheduling
# =============================================================================


class SchedulerKind(str, Enum):
    KAI = "kai"
    KUEUE = "kueue"
    DEFAULT = "default"


class SchedulerSpec(_StrictModel):
    kind: SchedulerKind = SchedulerKind.DEFAULT
    queue: str | None = None

    @model_validator(mode="after")
    def _queue_required_for_kai_kueue(self) -> "SchedulerSpec":
        if self.kind in (SchedulerKind.KAI, SchedulerKind.KUEUE) and not self.queue:
            raise ValueError(
                f"infra.scheduler.queue is required when scheduler.kind={self.kind.value}"
            )
        return self


# =============================================================================
# Placement (node selectors, tolerations)
# =============================================================================


class Toleration(_StrictModel):
    key: str
    operator: Literal["Equal", "Exists"] = "Equal"
    value: str | None = None
    effect: Literal["NoSchedule", "PreferNoSchedule", "NoExecute"] = "NoSchedule"
    tolerationSeconds: int | None = None


class PlacementSpec(_StrictModel):
    nodeSelector: dict[str, str] = Field(default_factory=dict)
    tolerations: list[Toleration] = Field(default_factory=list)
    # Optional raw affinity passthrough (rarely needed; falls back to nodeSelector).
    affinity: dict[str, Any] | None = None


# =============================================================================
# Networking
# =============================================================================


class NetworkingSpec(_StrictModel):
    hostNetwork: bool = False
    gloo_socket_ifname: str | None = None
    nccl_socket_ifname: str | None = None
    nccl_ib_disable: bool = False
    nccl_net: Literal["Socket", "IB", "OFI"] | None = None
    # Raw extra NCCL env vars if the cluster needs more; user-managed.
    extra_env: dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Storage (workspace, HF cache, checkpoints)
# =============================================================================


class WorkspaceKind(str, Enum):
    LUSTRE = "lustre"  # FSx for Lustre / managed-parallel-FS PVC
    PVC = "pvc"  # any other RWX PVC
    HOST_PATH = "hostPath"  # dev / kind only
    RAY_UPLOAD = "rayUpload"  # Ray Job SDK working_dir upload (fallback, 100 MiB cap)
    AUTO = "auto"  # prefer lustre if the pvc exists, else rayUpload


class WorkspaceSpec(_StrictModel):
    kind: WorkspaceKind = WorkspaceKind.RAY_UPLOAD
    # PVC-backed kinds (lustre, pvc)
    pvcName: str | None = None
    mountPath: str = "/mnt/nemo-rl"
    repoSubdir: str = (
        "workdirs"  # ${mountPath}/${repoSubdir}/<hash>/ holds the synced repo
    )
    size: str | None = None  # only consulted if the PVC needs to be created (lustre)
    # hostPath-backed
    hostPath: str | None = None

    @model_validator(mode="after")
    def _required_fields_by_kind(self) -> "WorkspaceSpec":
        if self.kind in (WorkspaceKind.LUSTRE, WorkspaceKind.PVC) and not self.pvcName:
            raise ValueError(
                f"infra.workspace.pvcName is required when kind={self.kind.value}"
            )
        if self.kind is WorkspaceKind.HOST_PATH and not self.hostPath:
            raise ValueError("infra.workspace.hostPath is required when kind=hostPath")
        return self


class HFCacheKind(str, Enum):
    LUSTRE = "lustre"
    PVC = "pvc"
    EMPTY_DIR = "emptyDir"
    NONE = "none"


class HFCacheSpec(_StrictModel):
    kind: HFCacheKind = HFCacheKind.NONE
    pvcName: str | None = None
    mountPath: str = "/root/.cache/huggingface"

    @model_validator(mode="after")
    def _pvc_required(self) -> "HFCacheSpec":
        if self.kind in (HFCacheKind.LUSTRE, HFCacheKind.PVC) and not self.pvcName:
            raise ValueError(
                f"infra.hf_cache.pvcName is required when kind={self.kind.value}"
            )
        return self


class CheckpointsKind(str, Enum):
    LUSTRE = "lustre"
    PVC = "pvc"
    NONE = "none"  # checkpoints land on pod-local storage (smoke tests only)


class CheckpointsSpec(_StrictModel):
    kind: CheckpointsKind = CheckpointsKind.NONE
    pvcName: str | None = None
    mountPath: str = "/mnt/nemo-rl/checkpoints"

    @model_validator(mode="after")
    def _pvc_required(self) -> "CheckpointsSpec":
        if (
            self.kind in (CheckpointsKind.LUSTRE, CheckpointsKind.PVC)
            and not self.pvcName
        ):
            raise ValueError(
                f"infra.checkpoints.pvcName is required when kind={self.kind.value}"
            )
        return self


# =============================================================================
# Submission (how the CLI gets a job onto the cluster)
# =============================================================================


class SubmitKind(str, Enum):
    SDK = "sdk"  # Ray Job SDK (default)
    RAYJOB = "rayjob"  # RayJob CRD


class PortForwardMode(str, Enum):
    KUBECTL_RAY_PLUGIN = "kubectl-ray-plugin"
    KUBECTL_PORT_FORWARD = "kubectl-port-forward"
    AUTO = "auto"


class DevPodMode(str, Enum):
    AUTO = "auto"
    REQUIRED = "required"
    SKIP = "skip"


class SubmitterMode(str, Enum):
    """How the CLI ships the training entrypoint onto the cluster.

    ``portForward`` (default) goes through Ray's Job SDK via a brief
    ``kubectl port-forward`` to the head dashboard. The Ray job carries a
    submission id the dashboard tracks.

    ``exec`` shells into the training head pod with ``kubectl exec`` and
    launches the entrypoint as a backgrounded ``nohup`` process — the
    same shape as a Slurm driver running on a login node. No Ray Job
    abstraction; the process calls ``ray.init(address="auto")`` from
    inside the head pod and joins the already-running cluster.
    """

    PORT_FORWARD = "portForward"
    EXEC = "exec"


class SubmitSpec(_StrictModel):
    kind: SubmitKind = SubmitKind.SDK
    portForward: PortForwardMode = PortForwardMode.AUTO
    devPod: DevPodMode = DevPodMode.AUTO
    # Local port when port-forwarding; default avoids collision with `kubectl-ray session`.
    localDashboardPort: int = 18265
    # Transport used to deliver the training entrypoint to the cluster.
    submitter: SubmitterMode = SubmitterMode.PORT_FORWARD
    # Pod directory ExecSubmitter uses for the launcher script, stdout log,
    # and pidfile. Override if PodSecurityPolicy makes /tmp read-only.
    execTmpDir: str = "/tmp"


# =============================================================================
# Launch (single vs disaggregated, attach mode)
# =============================================================================


class LaunchMode(str, Enum):
    SINGLE = "single"  # colocated training in one RayCluster (default)
    RAYJOB = "rayjob"  # ephemeral cluster per run (auto teardown)
    ATTACH = "attach"  # submit training onto existing RayClusters
    BRINGUP = "bringup"  # create a long-lived RayCluster, no job


class RunMode(str, Enum):
    """Interactive vs batch defaults for ``nrl-k8s run`` / ``launch``.

    ``interactive`` (default) preserves the dev-iteration UX: port-forward
    submitter, ``codeSource=upload``, foreground log tailing, exits on
    terminal state.

    ``batch`` flips production defaults: exec submitter, ``codeSource=image``,
    no wait — the CLI returns a run id and exits as soon as ``nohup`` fires
    so laptops are off the critical path for long-running training.

    Both defaults are macros — each individual dimension (submitter,
    codeSource, no_wait) can still be overridden via explicit flag.
    """

    INTERACTIVE = "interactive"
    BATCH = "batch"


class CodeSource(str, Enum):
    """Where the training code lives at submission time.

    ``upload`` (default) stages a working_dir from the laptop and uploads it
    via Ray's Job SDK. Subject to Ray's 100 MiB working_dir cap.

    ``image`` assumes the code is baked into the container image at
    ``LaunchSpec.codePath`` (typically ``/opt/nemo-rl``). No CLI-side
    staging, no upload. The entrypoint is responsible for ``cd``-ing there.

    ``lustre`` is structurally identical to ``image`` but implies the
    ``codePath`` points at a Lustre / shared-FS mount populated by an
    out-of-band snapshot step. Kept distinct so ``nrl-k8s check`` output
    and future snapshot tooling can reason about it.
    """

    UPLOAD = "upload"
    IMAGE = "image"
    LUSTRE = "lustre"


class AttachSpec(_StrictModel):
    generation: str | None = None
    gym: str | None = None
    training: str | None = None  # null = create ephemeral training cluster


class LaunchSpec(_StrictModel):
    mode: LaunchMode = LaunchMode.SINGLE
    # Interactive vs batch defaults — see :class:`RunMode`. Overridden by
    # ``--mode`` on the CLI; unset on the infra layer means "interactive".
    runMode: RunMode = RunMode.INTERACTIVE
    attach: AttachSpec = Field(default_factory=AttachSpec)
    peerWatcher: bool = True
    # Shell command the training job runs inside the Ray cluster. Required
    # for `nrl-k8s run` / `nrl-k8s rayjob`. Typically a line like
    # ``python -u examples/.../entry.py --config nrl_k8s_run.yaml ...``.
    # The CLI stages the resolved recipe as ``nrl_k8s_run.yaml`` at the
    # working_dir root so this command can reference it by name (only when
    # ``codeSource=upload``; in image/lustre mode the entrypoint must use
    # the in-container recipe path directly).
    entrypoint: str | None = None
    # Env vars injected into the training job's runtime_env.
    env: dict[str, str] = Field(default_factory=dict)
    # Repo-relative paths to stage into every Ray Job's working_dir.
    # None means "use the built-in default" (see nrl_k8s.workdir). Keeping
    # this narrow matters — Ray caps working_dir uploads at 100 MiB, so
    # recipes should exclude datasets they don't need for the run.
    # Only consulted when ``codeSource=upload``.
    rayUploadPaths: list[str] | None = None
    # Where the training code lives — see :class:`CodeSource`. When
    # ``image`` or ``lustre``, ``codePath`` must be set.
    codeSource: CodeSource = CodeSource.UPLOAD
    # Absolute path inside the container where the code lives when
    # codeSource is image or lustre. The entrypoint is expected to ``cd``
    # there and (if applicable) ``source`` any env setup.
    codePath: str | None = None

    @model_validator(mode="after")
    def _attach_fields(self) -> "LaunchSpec":
        if self.mode is LaunchMode.ATTACH:
            if not (self.attach.generation or self.attach.gym or self.attach.training):
                raise ValueError(
                    "infra.launch.mode=attach requires at least one of "
                    "infra.launch.attach.{generation,gym,training}"
                )
        return self

    @model_validator(mode="after")
    def _code_path_required_for_non_upload(self) -> "LaunchSpec":
        if (
            self.codeSource in (CodeSource.IMAGE, CodeSource.LUSTRE)
            and not self.codePath
        ):
            raise ValueError(
                f"infra.launch.codePath is required when codeSource="
                f"{self.codeSource.value}"
            )
        return self


# =============================================================================
# Resource profiles per role (CLI derives sensible defaults from cluster.*)
# =============================================================================


class PodResources(_StrictModel):
    cpu: str | None = None  # e.g. "8" or "500m"
    memory: str | None = None  # e.g. "32Gi"
    # nvidia.com/gpu is derived from cluster.gpus_per_node by default, but can be overridden.
    gpu: int | None = None


class ResourceProfile(_StrictModel):
    head: PodResources = Field(default_factory=PodResources)
    worker: PodResources = Field(default_factory=PodResources)


class ResourcesSpec(_StrictModel):
    training: ResourceProfile = Field(default_factory=ResourceProfile)
    generation: ResourceProfile = Field(default_factory=ResourceProfile)
    gym: ResourceProfile = Field(default_factory=ResourceProfile)


# =============================================================================
# Per-role cluster spec — a pointer to a raw RayCluster manifest on disk.
#
# We deliberately don't model the RayCluster topology in pydantic. Researchers
# already maintain these YAMLs (`infra/examples/disagg_*_raycluster.yaml`);
# the CLI just reads, optionally patches a handful of fields, and applies
# them via the official kubernetes Python client. The manifest is the
# authoritative source for head/worker shape, ports, node placement, etc.
# =============================================================================


class DaemonSpec(_StrictModel):
    """A long-running Ray Job to submit once the RayCluster is ready.

    The entrypoint runs via ``ray.job_submission.JobSubmissionClient`` against
    the cluster's dashboard. ``submissionId`` is the human-readable name we
    tag the job with so subsequent ``nrl-k8s job logs`` can find it.
    """

    entrypoint: str
    submissionId: str | None = None
    # Environment variables to inject into the Ray job runtime.
    env: dict[str, str] = Field(default_factory=dict)
    # Health-check URL (cluster-internal DNS). If set, the CLI polls it after
    # submission so `cluster up` returns only once the daemon is serving.
    healthCheckUrl: str | None = None
    # Seconds to wait on the health-check before giving up.
    healthCheckTimeoutS: int = 300
    # Per-daemon override for the working_dir upload set. None = use
    # ``infra.launch.rayUploadPaths``. Use this to keep each daemon's
    # upload small (e.g. the gen server doesn't need training data).
    rayUploadPaths: list[str] | None = None


class ClusterSpec(_StrictModel):
    """One long-lived RayCluster in the disaggregated setup.

    ``spec`` is the inline RayCluster ``.spec`` body — the CLI wraps it in
    ``apiVersion: ray.io/v1``, ``kind: RayCluster``, ``metadata.name/namespace``
    and patches cross-cutting fields (image, imagePullSecrets, serviceAccount)
    from the top-level ``infra`` keys before applying.

    We deliberately do *not* model the RayCluster topology in pydantic —
    it's a free-form dict so every upstream RayCluster field works without
    schema changes. ``extra='forbid'`` still catches typos in the
    surrounding keys (``name``, ``daemon``).
    """

    name: str
    spec: dict[str, Any]
    # Labels/annotations to attach to metadata (useful for KAI/Kyverno).
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    # Daemon to start on the cluster once Ready (e.g. the gen/gym server).
    daemon: DaemonSpec | None = None


class ClustersSpec(_StrictModel):
    """The 3 named RayClusters in a disaggregated run."""

    training: ClusterSpec | None = None
    generation: ClusterSpec | None = None
    gym: ClusterSpec | None = None


# =============================================================================
# Top-level InfraConfig
# =============================================================================


class InfraConfig(_StrictModel):
    # Cluster-identifying (required)
    namespace: str
    image: str
    imagePullSecrets: list[str] = Field(default_factory=list)
    # Ray version pinned on the cluster; optional — the CLI uses the image default.
    rayVersion: str | None = None
    # Pod ServiceAccount; None means "don't patch — keep whatever the manifest has".
    serviceAccount: str | None = None

    # Behaviour
    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)
    placement: PlacementSpec = Field(default_factory=PlacementSpec)
    networking: NetworkingSpec = Field(default_factory=NetworkingSpec)
    workspace: WorkspaceSpec = Field(default_factory=WorkspaceSpec)
    hf_cache: HFCacheSpec = Field(default_factory=HFCacheSpec)
    checkpoints: CheckpointsSpec = Field(default_factory=CheckpointsSpec)
    submit: SubmitSpec = Field(default_factory=SubmitSpec)
    launch: LaunchSpec = Field(default_factory=LaunchSpec)
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    clusters: ClustersSpec = Field(default_factory=ClustersSpec)

    # Opaque extra labels / annotations the platform may require (Kyverno-enforced, etc.)
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)

    @field_validator("namespace", "image")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


__all__ = [
    "AttachSpec",
    "CheckpointsKind",
    "CheckpointsSpec",
    "ClusterSpec",
    "ClustersSpec",
    "CodeSource",
    "DaemonSpec",
    "DevPodMode",
    "HFCacheKind",
    "HFCacheSpec",
    "InfraConfig",
    "LaunchMode",
    "LaunchSpec",
    "NetworkingSpec",
    "PlacementSpec",
    "PodResources",
    "PortForwardMode",
    "ResourceProfile",
    "ResourcesSpec",
    "RunMode",
    "SchedulerKind",
    "SchedulerSpec",
    "SubmitKind",
    "SubmitSpec",
    "SubmitterMode",
    "Toleration",
    "WorkspaceKind",
    "WorkspaceSpec",
]
