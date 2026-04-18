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

# 4. Create KAI scheduler queues + RBAC
kubectl apply -f ../examples/kai-queue.yaml
kubectl apply -f ../examples/endpoint-registry-rbac.yaml
kubectl apply -f ../examples/kyverno-kai-policies.yaml

# 5. Deploy disaggregated RL + Gym
kubectl apply -f ../examples/disagg-rayclusters.yaml
kubectl get rayclusters -w    # both should become "ready"
```

## Deploy on a real cluster

```sh
cd infra/helm
helmfile -e prod sync
kubectl apply -f infra/examples/kai-queue-prod.yaml
```

This installs the full **GPU Operator** (instead of just the device plugin) along with KAI scheduler and KubeRay. The GPU Operator manages the NVIDIA driver, container toolkit, device plugin, NFD, and DCGM exporter.

Set `driver.enabled=false` in `values/gpu-operator.yaml` if the cluster nodes already have the NVIDIA driver installed.

## Helmfile environments

| Environment | GPU component | Use case |
|-------------|---------------|----------|
| `kind` | nvidia-device-plugin | Local dev — nvkind handles toolkit/runtime |
| `prod` | gpu-operator (full) | Real clusters — operator manages everything |

Both environments include KAI scheduler, KubeRay operator, Kyverno, and Prometheus+Grafana.

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
│       ├── kyverno.yaml
│       └── kube-prometheus-stack.yaml
├── examples/
│   ├── disagg-rayclusters.yaml       # Disaggregated RL + Gym (main exemplar)
│   ├── endpoint-registry-rbac.yaml   # RBAC for ConfigMap service discovery
│   ├── gym_standalone_config.yaml    # Gym standalone server config
│   ├── kai-queue.yaml                # 2-GPU kind cluster queues
│   ├── kai-queue-prod.yaml           # 288-GPU NVL72 prod queues
│   ├── kyverno-kai-policies.yaml     # Queue enforcement policies
│   ├── kai-service-monitors.yaml     # Prometheus ServiceMonitors for KAI
│   └── kai-grafana-dashboard.yaml    # Grafana fairshare dashboard
```

## Disaggregated RL + Gym

The main workload exemplar is `disagg-rayclusters.yaml`: two RayClusters deployed together.

- **RL cluster** (`raycluster-rl`): Ray head + GPU workers for training (vLLM + Megatron)
- **Gym cluster** (`raycluster-gym`): CPU-only, runs NeMo Gym servers as HTTP service
- **Service discovery**: K8s ConfigMap endpoint registry — RL publishes vLLM URLs, Gym publishes its head server address. Both poll until the peer registers.
- **Failure cascading**: Peer-watcher sidecar (inlined Python script) on each head pod. If either cluster fails or is deleted, both are torn down.

## Notes

- **nvkind vs vanilla kind**: nvkind automates GPU device injection, nvidia-container-toolkit installation inside nodes, containerd nvidia runtime configuration, and RuntimeClass registration.
- **nvidia-device-plugin** (kind only): The full GPU Operator fails in kind because its driver validation doesn't work inside kind nodes. The lightweight device plugin with CDI discovery is sufficient since nvkind handles the runtime setup.
- **KAI scheduler** creates PodGroups automatically for recognized workload types (RayCluster, Job, PyTorchJob, JobSet, etc.). For bare pods, create a PodGroup manually and annotate with `pod-group-name`.

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

- `kai-queue.yaml` — 2-GPU kind cluster (priority-team + community, imbalanced)
- `kai-queue-prod.yaml` — 288-GPU NVL72 production cluster (priority + community departments)

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
