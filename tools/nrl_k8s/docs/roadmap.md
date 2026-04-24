# `nrl-k8s` roadmap

Features that were scaffolded and then removed from the CLI surface pending
real implementations. Re-add the commands as they land so `--help` only
advertises things that work.

## `doctor` — cluster-baseline health check

Currently removed. Intended behavior:

- confirm KubeRay operator ≥ v1.5 is installed and the `RayCluster` CRD is
  reachable
- if `infra.scheduler.kind=kai`, confirm the KAI scheduler deployment is
  healthy and the `Queue` CRD exists
- if `infra.scheduler.kind=kueue`, confirm the `ClusterQueue` CRD exists
- confirm the configured `serviceAccount` exists in `infra.namespace`
- confirm at least one node advertises `nvidia.com/gpu` in allocatable
- on AWS: confirm the EFA device plugin exposes `vpc.amazonaws.com/efa`

Exits non-zero on any failed check with a remediation hint per failure.
Good fit for CI preflight and for the `--hint` path of `_explain_and_exit`.

## `dashboard <cluster-name>` — open the Ray dashboard locally

Port-forward a RayCluster's head service to a local port and open the URL
(via `webbrowser.open`) in the default browser. Uses the existing
`submit.dashboard_url` context manager. Removed because the happy path is
already covered by `nrl-k8s logs --role training --source daemon` and
`nrl-k8s job list --role …`, but worth re-adding for the "eyeball metrics"
use case.

## `dev up` / `dev down` — in-cluster dev pod

Long-running `nrl-dev-<user>` Pod per researcher (one-time create). Image
matches the RayCluster container image so `ray.job_submission.JobSubmission
Client` upload+client Ray versions are identical. Mounts the shared PVC for
code edits and the HF cache PVC. ServiceAccount limited to the minimum
needed to `create` RayClusters + get/list/watch pods + configmaps in
`infra.namespace`.

Workflow from a laptop becomes:

```bash
nrl-k8s dev up                                 # one-time
kubectl exec -it pod/nrl-dev-$USER -- bash
# inside:
nrl-k8s run tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml
```

Main benefits: no port-forward (the pod is in-cluster so it uses CNI DNS
to reach the Ray dashboard), no laptop-speed upload of the 100 MiB working
dir, and nothing breaks when SSO tokens expire in the middle of a run.

Removed because the feature requires a shared PVC provisioned per
researcher + RBAC templates; that's a separate platform task.

## `job list` auto-lookup for training submissions

Today `job list --role training` returns zero rows because training jobs
have auto-generated submission IDs and there's no stable identifier on
the cluster to correlate a run. A future enhancement tags each submission
with a `wandbRunId` annotation (via `runtime_env.metadata`) so `job list`
can render one line per run with the wandb URL inline.

## Multi-context support

`load_kubeconfig` is cached per process; to support running against two
clusters from one shell session (e.g. staging + prod), surface
`--context <name>` on the root group and thread it down into every
`custom_objects_api()` / `CoreV1Api()` call site.

## RayJob CRD submission path

`infra.launch.mode=rayjob` is declared in the schema but `orchestrate.py`
always follows the SDK path. Adding the CRD path means rendering a
`RayJob` manifest with `clusterSelector: {ray.io/cluster: <name>}` and
`submissionMode: HTTPMode`, then applying it via the same manifest builder
used for `RayCluster`. Useful for GitOps pipelines where durable,
k8s-native job state is required. Out of scope for v1.0 unless a caller
shows up for it.

## Pluggable backends, schedulers, orchestration

The `backends/`, `schedulers/`, and `orchestration/` subpackages were
sketched early and deleted after a few iterations — none had callers and
the orchestration logic consolidated into a single `orchestrate.py`. Keep
this list so the scaffolding doesn't come back reflexively:

- **Backends** (`backends/base.py:LaunchBackend`, `kuberay.py`,
  `jobset.py`) — abstract layer for non-Ray workloads (JobSet-based pure
  PyTorch). Current code assumes KubeRay + `JobSubmissionClient`. If a
  JobSet path is needed, add one backend module that implements the same
  shape (`apply_cluster`, `submit_job`, `wait_ready`) and flip based on an
  `infra.launch.backend: kuberay|jobset` field.
- **Schedulers** (`schedulers/kai.py`, `kueue.py`, `default.py`) — per-
  scheduler manifest mutators. `infra.scheduler.kind=kai|kueue` is parsed
  into `SchedulerSpec` today but nothing patches the resulting manifest
  (no `kai.scheduler/queue` label, no `kueue.x-k8s.io/queue-name`, no
  `spec.suspend`). When re-added, wire into `manifest.build_raycluster
  _manifest` as a post-patch step keyed on `infra.scheduler.kind`.
- **Orchestration modes** (`orchestration/single.py`, `disagg.py`) —
  `LaunchMode` declares `single`, `rayjob`, `attach`, `bringup` but the
  code branches only on `.clusters.<role>` presence. If we grow a CRD-
  based or bring-up-only path, split the driver in `orchestrate.py` by
  mode rather than resurrecting the subpackage.

The empty `templates/` dir was also removed — the CLI renders RayCluster
objects by patching a Python dict (see `manifest.build_raycluster_
manifest`). Jinja stays off the table unless the generated YAML stops
fitting into that pattern.
