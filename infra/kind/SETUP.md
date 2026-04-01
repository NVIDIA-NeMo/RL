# Local K8s GPU Development Environment

nvkind-based Kubernetes cluster with NVIDIA GPU support, KAI scheduler (gang scheduling), and KubeRay.

## Prerequisites

- Docker with systemd cgroup driver and **cgroup v2**
- NVIDIA driver installed on host (`nvidia-smi` works)
- `nvidia-container-toolkit` installed on host
- `go` (for nvkind installation)
- `helmfile` (install: `curl -sSL https://github.com/helmfile/helmfile/releases/latest/download/helmfile_$(uname -s | tr '[:upper:]' '[:lower:]')_amd64.tar.gz | tar xz -C ~/bin helmfile`)

### One-time host setup (requires sudo)

```sh
# Set nvidia as Docker's default runtime and enable CDI
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
sudo systemctl restart docker

# Verify
docker info | grep "Default Runtime"   # should show "nvidia"
stat -fc %T /sys/fs/cgroup/            # should show "cgroup2fs"
```

## Quick Start (local kind cluster)

```sh
# 1. Install tools
cd infra/kind
bash install-nvkind.sh
bash get-kubectl.sh
bash get-helm.sh

# 2. Create cluster (all host GPUs exposed to a single worker node)
bash create-cluster.sh

# 3. Deploy infrastructure
cd ../helm
helmfile -e kind sync

# 4. Create KAI scheduler queues
kubectl apply -f ../examples/kai-queue.yaml

# 5. Test: gang-schedule two GPU pods
kubectl apply -f ../examples/kai_scheduled_pods.yaml
kubectl get pods -w                     # both go Running at the same time
kubectl logs gpu-test-0                 # nvidia-smi output
kubectl delete -f ../examples/kai_scheduled_pods.yaml

# 6. Test: gang-schedule two RayClusters (each with 1 GPU worker)
kubectl apply -f ../examples/kai_scheduled_rayclusters.yaml
kubectl get rayclusters -w              # both become "ready"
kubectl delete -f ../examples/kai_scheduled_rayclusters.yaml
```

## Deploy on a real cluster

```sh
cd infra/helm
helmfile -e prod sync
```

This installs the full **GPU Operator** (instead of just the device plugin) along with KAI scheduler and KubeRay. The GPU Operator manages the NVIDIA driver, container toolkit, device plugin, NFD, and DCGM exporter.

Set `driver.enabled=false` in `values/gpu-operator.yaml` if the cluster nodes already have the NVIDIA driver installed.

## Helmfile environments

| Environment | GPU component | Use case |
|-------------|---------------|----------|
| `kind` | nvidia-device-plugin | Local dev — nvkind handles toolkit/runtime |
| `prod` | gpu-operator (full) | Real clusters — operator manages everything |

Both environments include KAI scheduler and KubeRay operator.

## Tear down (kind only)

```sh
kind delete cluster --name nemo-rl
```

## Architecture

```
infra/
├── kind/                              # Cluster setup (local dev only)
│   ├── create-cluster.sh             # Creates nvkind cluster
│   ├── install-nvkind.sh             # Installs kind + nvkind
│   ├── get-kubectl.sh / get-helm.sh  # Tool installers
│   ├── nvkind-config-values.yaml     # Default: workers with all GPUs
│   ├── nvkind-config-values-dev.yaml # Dev: + local code mount
│   └── nvkind-config-template.yaml   # Custom template with extraMounts
├── helm/                              # Infrastructure (helmfile)
│   ├── helmfile.yaml                 # environments: kind, prod
│   └── values/
│       ├── nvidia-device-plugin.yaml # kind only
│       ├── gpu-operator.yaml         # prod only
│       ├── kai-scheduler.yaml
│       ├── kuberay-operator.yaml
│       ├── kyverno.yaml              # Kyverno policy engine
│       └── kube-prometheus-stack.yaml # Prometheus + Grafana
└── examples/
    ├── kai-queue.yaml                # KAI queue hierarchy
    ├── kai_scheduled_pods.yaml       # Gang-scheduled GPU test pods
    ├── kai_scheduled_rayclusters.yaml # Gang-scheduled RayClusters
    ├── kai_scheduled_sft.yaml        # Two gang-scheduled SFT RayJobs
    ├── sft_rayjob.yaml               # 2-GPU SFT RayJob
    ├── raycluster-blocker.yaml       # GPU blocker for testing KAI
    ├── gym_standalone_config.yaml    # Gym standalone server config
    ├── disagg_rl_raycluster.yaml     # Disagg RL cluster + peer-watcher
    ├── disagg_gym_raycluster.yaml    # Disagg Gym cluster + peer-watcher
    ├── endpoint-registry-rbac.yaml   # RBAC for ConfigMap endpoint registry
    ├── peer-watcher.py               # Sidecar for failure cascading
    ├── kai-queue-prod.yaml           # 256-GPU prod queue config
    ├── kyverno-kai-policies.yaml     # Queue enforcement policies
    ├── kai-service-monitors.yaml     # Prometheus ServiceMonitors for KAI
    └── kai-grafana-dashboard.yaml    # Grafana dashboard for fairshare
```

## Notes

- **nvkind vs vanilla kind**: nvkind automates GPU device injection, nvidia-container-toolkit installation inside nodes, containerd nvidia runtime configuration, and RuntimeClass registration.
- **nvidia-device-plugin** (kind only): The full GPU Operator fails in kind because its driver validation doesn't work inside kind nodes. The lightweight device plugin with CDI discovery is sufficient since nvkind handles the runtime setup. On a real cluster, use the full GPU Operator (`helmfile -e prod sync`).
- **Device plugin kind overrides**: Affinity is overridden because NFD isn't installed in kind. `runtimeClassName: nvidia` is set so the plugin pod gets NVIDIA libraries injected for NVML discovery. Neither override is needed on a real cluster.
- **KAI scheduler** creates PodGroups automatically for recognized workload types (RayCluster, Job, PyTorchJob, etc.). For bare pods, create a PodGroup manually and annotate pods with `pod-group-name`.
- **RayJob** (not RayCluster) is preferred for batch workloads — it auto-tears down the cluster after the job finishes, avoiding stale Ray state. Requires `submissionMode: HTTPMode` for KAI compatibility.

## Failure cascading for disaggregated Gym

The disaggregated RL/Gym manifests include a **peer-watcher sidecar** on each head pod that monitors the peer cluster via the K8s API. If the peer is deleted, fails, or signals an error via the ConfigMap, the watcher tears down both clusters to release resources.

- `peer-watcher.py` — Python sidecar script (deployed as a ConfigMap)
- Monitors: peer RayCluster status + ConfigMap `error` key
- `MAX_PEER_FAILURES` (default 3) consecutive failures before teardown
- Applications can signal errors via `K8sEndpointRegistry.signal_error("message")`

Setup:
```sh
kubectl create configmap peer-watcher-script --from-file=peer-watcher.py=infra/examples/peer-watcher.py
kubectl apply -f infra/examples/endpoint-registry-rbac.yaml
```

## Fairshare scheduling

KAI distributes GPU resources using hierarchical fair-share with two phases:
1. **Guaranteed quota**: Each queue gets its `quota` first, unconditionally.
2. **Over-quota surplus**: Remaining GPUs distributed by `priority` (higher served first), then `overQuotaWeight` within the same priority level.

### Queue fields

| Field | Description |
|-------|-------------|
| `quota` | Guaranteed GPUs. `-1` = unlimited, `0` = no guarantee |
| `limit` | Hard cap on total GPUs. `-1` = no limit |
| `overQuotaWeight` | Weight for surplus distribution (higher = bigger share) |
| `priority` | Over-quota allocation order (higher = served first, reclaimed last) |
| `preemptMinRuntime` | Min runtime before a higher-priority queue can preempt (default: `"4h"`) |
| `reclaimMinRuntime` | Min runtime before over-quota resources can be reclaimed (default: `"15m"`) |

### Preempt vs reclaim

- **Preempt**: A higher-priority queue takes from a lower-priority queue. (VIP takes your table.)
- **Reclaim**: A queue takes back what it's entitled to from an over-allocated queue. (Fairness — give back what you owe.)

`reclaimMinRuntime` is shorter than `preemptMinRuntime` because reclaim is about fairness (returning over-quota resources quickly), while preempt protects long-running jobs from priority-based interruption.

### Example configs

- `kai-queue.yaml` — 2-GPU kind cluster (team-a, team-b, equal quotas)
- `kai-queue-prod.yaml` — 256-GPU production cluster (3 departments, 6 teams)

### Monitoring (Grafana)

```sh
kubectl port-forward svc/kube-prometheus-stack-grafana -n monitoring 3000:80
# Open http://localhost:3000
# Login: admin / prom-operator
# Dashboard: search "KAI Scheduler Fairshare" or go to http://localhost:3000/d/kai-fairshare
```

Key metrics: `kai_queue_allocated_gpus`, `kai_queue_deserved_gpus`, `kai_e2e_scheduling_latency_milliseconds`.

### Kyverno queue enforcement

RayCluster and RayJob resources must have a `kai.scheduler/queue` label or they're rejected by Kyverno. To enable user→queue access control, uncomment Policy 2 in `kyverno-kai-policies.yaml` and configure the `kai-queue-permissions` ConfigMap.

## TODO: NVL72 topology-aware scheduling

KAI v0.14.0 added Ray topology-aware subgroup scheduling ([PR #1125](https://github.com/kai-scheduler/KAI-Scheduler/pull/1125)). Need to test on an actual NVL72 cluster:

- **Confirm `--segment=N` equivalent works**: KAI's `subGroups` with per-subgroup `topologyConstraint.requiredTopologyLevel: "rack"` should be the equivalent of Slurm's `--segment=N`. Each subgroup of N nodes is constrained to one rack. Unclear if this works correctly for cross-rack scheduling (e.g., `--segment=16` with 32 total nodes = 2 racks).
- **Auto-segmentation not yet implemented**: The design doc at [`docs/developer/designs/segmented-subgroups/`](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/developer/designs/segmented-subgroups/README.md) proposes `kai.scheduler/segment-size` annotation for automatic subgroup creation, but it depends on "Replica-Type SubGrouping" which isn't shipped yet. See [Issue #1189](https://github.com/kai-scheduler/KAI-Scheduler/issues/1189) and [PR #1127](https://github.com/kai-scheduler/KAI-Scheduler/pull/1127) (minSubGroup field, still open).
- **Test with our k8s CLI**: The `nrl-k8s submit` command (see `extensions/k8s_cli/`) should support `--segment-size` that auto-generates the PodGroup subgroups until KAI ships native support.

## TODO: Log persistence

Currently, logs are lost when RayJob pods are cleaned up (`ttlSecondsAfterFinished`). Two levels of log persistence are needed:

1. **Container stdout/stderr** (`kubectl logs`): Captured by containerd at `/var/log/pods/` on the node, but not queryable after pod deletion.
2. **Ray file logs** (`/tmp/ray/session_*/logs/`): Worker/driver logs, system logs — not sent to stdout at all.

**Planned approach**: Deploy a Loki stack (Loki + Promtail + Grafana) via helmfile for both `kind` and `prod` environments:
- Promtail DaemonSet captures container stdout/stderr from each node, auto-labels with K8s metadata (namespace, pod, job name)
- Fluent Bit sidecar in each RayJob pod tails `/tmp/ray` logs and ships to Loki
- Grafana UI for querying logs by job name, time range, and content (LogQL)
- kind: Loki stores on local PVC; prod: Loki stores on S3/GCS
