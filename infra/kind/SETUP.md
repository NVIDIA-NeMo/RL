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
├── kind/           # Cluster setup (local dev only)
│   ├── create-cluster.sh
│   ├── install-nvkind.sh
│   ├── get-kubectl.sh / get-helm.sh
│   └── nvkind-config-values.yaml
├── helm/           # Infrastructure (helmfile)
│   ├── helmfile.yaml               # environments: kind, prod
│   └── values/
│       ├── nvidia-device-plugin.yaml   # kind only
│       ├── gpu-operator.yaml           # prod only
│       ├── kai-scheduler.yaml
│       └── kuberay-operator.yaml
└── examples/       # Test manifests
    ├── kai-queue.yaml
    ├── kai_scheduled_pods.yaml
    └── kai_scheduled_rayclusters.yaml
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

## TODO: Log persistence

Currently, logs are lost when RayJob pods are cleaned up (`ttlSecondsAfterFinished`). Two levels of log persistence are needed:

1. **Container stdout/stderr** (`kubectl logs`): Captured by containerd at `/var/log/pods/` on the node, but not queryable after pod deletion.
2. **Ray file logs** (`/tmp/ray/session_*/logs/`): Worker/driver logs, system logs — not sent to stdout at all.

**Planned approach**: Deploy a Loki stack (Loki + Promtail + Grafana) via helmfile for both `kind` and `prod` environments:
- Promtail DaemonSet captures container stdout/stderr from each node, auto-labels with K8s metadata (namespace, pod, job name)
- Fluent Bit sidecar in each RayJob pod tails `/tmp/ray` logs and ships to Loki
- Grafana UI for querying logs by job name, time range, and content (LogQL)
- kind: Loki stores on local PVC; prod: Loki stores on S3/GCS
