# K8s Infrastructure for nemo-rl

> [!WARNING]
> These instructions are in active development and should not be relied on as stable yet.
> APIs, manifests, and tooling may change without notice.

## Overview

This gives you a local GPU-enabled K8s playground for testing RL workloads — all you need is Docker and an NVIDIA GPU. The same example manifests can be brought to a production K8s cluster, but you may need to adapt them (work with your cluster operator).

The helmfile here is for convenience. A production system should manage these components in Terraform or another infrastructure-as-code solution.

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
cd kind
bash install-nvkind.sh
bash get-kubectl.sh
bash get-helm.sh

# 2. Create cluster (all host GPUs exposed to a single worker node)
bash create-cluster.sh

# 3. Deploy infrastructure (KAI scheduler, KubeRay, JobSet controller)
cd ../helm
helmfile -e kind sync

# 4. Create KAI scheduler queues + RBAC
kubectl apply -f ../examples/kai-queue.yaml
kubectl apply -f ../examples/endpoint-registry-rbac.yaml

# 5. Deploy a workload (pick one)
kubectl apply -f ../examples/rayjob-monolithic.yaml      # single-cluster RayJob
kubectl apply -f ../examples/disagg-rayclusters.yaml      # disagg via KubeRay
kubectl apply -f ../examples/disagg-jobset.yaml           # disagg via JobSet
```

### Testing locally with kind

Once the cluster is up, you can:

```sh
kubectl get rayclusters -w           # watch cluster status
kubectl get jobsets.jobset.x-k8s.io  # watch JobSet status
kubectl get pods -o wide             # see pod placement and IPs
kubectl logs <pod-name>              # check logs

# Exec into RL head to run training manually:
kubectl exec -it <rl-head-pod> -c ray-head -- bash
cd /workspace/nemo-rl
python examples/nemo_gym/run_grpo_nemo_gym.py +env.disagg_job_id=my-job logger.wandb_enabled=false
```

## Deploy on a real cluster

```sh
cd helm
helmfile -e prod sync
kubectl apply -f examples/kai-queue-prod.yaml
```

This installs KAI scheduler, KubeRay, and JobSet. The cluster is expected to already have the GPU Operator (or equivalent GPU provisioning) installed.

## Architecture

### Colocated (single Ray cluster)

All components run on a single RayCluster — vLLM generation, Megatron training, and Gym environment servers are colocated as Ray actors on the same cluster.

```
┌───────────────────────────────────────────────┐
│              RayCluster / RayJob              │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ vLLM     │  │ Megatron │  │ Gym      │     │
│  │ (GPU)    │  │ (GPU)    │  │ (CPU)    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│            All on same Ray cluster            │
└───────────────────────────────────────────────┘
```

Example: `rayjob-monolithic.yaml` — a single RayJob with head + GPU workers. KubeRay manages the lifecycle.

### Disaggregated (separate RL + Gym clusters)

The RL cluster (vLLM + Megatron) and Gym cluster (environment servers) run independently and communicate over HTTP. A K8s ConfigMap acts as an endpoint registry for dynamic URL exchange — RL publishes vLLM URLs, Gym publishes its head server address.

```
┌───────────────────────────┐    ┌───────────────────────────┐
│    RL Ray Cluster         │    │    Gym Ray Cluster        │
│                           │    │                           │
│  ┌──────┐  ┌──────────┐   │    │  ┌──────────┐             │
│  │ vLLM │  │ Megatron │   │    │  │ Gym      │             │
│  │ (GPU)│  │ (GPU)    │   │    │  │ servers  │             │
│  └──┬───┘  └──────────┘   │    │  └────┬─────┘             │
│     │                     │    │       │                   │
└─────┼─────────────────────┘    └───────┼───────────────────┘
      │                                  │
      │    ┌──────────────────┐          │
      └───►│ ConfigMap        │◄─────────┘
           └──────────────────┘
```

There are two ways to deploy the disagg architecture:

#### Option A: Two KubeRay RayClusters (`disagg-rayclusters.yaml`)

- KubeRay operator manages each RayCluster independently
- **Failure cascading** requires a peer-watcher sidecar on each head pod (inlined Python script that monitors the peer cluster via K8s API and tears down both on failure)
- **Startup ordering** is implicit — both clusters start simultaneously, the endpoint registry handles coordination
- **Gang-of-gang scheduling** is not natively supported by KAI — each RayCluster gets its own PodGroup, and KAI cannot gang two PodGroups together. See [KAI issue #1420](https://github.com/kai-scheduler/KAI-Scheduler/issues/1420). The peer-watcher is a workaround
- **ConfigMap** is required for both vLLM URL exchange and Gym head address discovery

#### Option B: Single JobSet (`disagg-jobset.yaml`)

- JobSet controller manages all pods as a single unit (no KubeRay needed for lifecycle)
- **Failure cascading** is native via `failurePolicy: FailJobSet` — any job failure tears down everything
- **Startup ordering** uses init containers (not `dependsOn` — see note below). Workers wait for their head via `ray health-check` polling, same pattern as KubeRay
- **Gang scheduling** works naturally — KAI creates one PodGroup for the entire JobSet, so all pods are gang-scheduled together
- **DNS names** are predictable (`disagg-job-rl-head-0-0.disagg-job`), so the Gym head address can be hardcoded instead of discovered via ConfigMap. However, **ConfigMap is still needed for vLLM URL exchange** — vLLM binds to a dynamic IP:port inside the RL worker, which isn't known until runtime

> [!NOTE]
> **Why not `dependsOn`?** KAI gang-schedules all pods in a JobSet together (one PodGroup with `minMember` = total pods). `dependsOn` prevents dependent pods from being created until the head is Ready. This deadlocks: KAI waits for all pods to exist, JobSet waits for the head to be scheduled. The fix is to create all pods simultaneously and use init containers for ordering.

### Comparison

| Feature | Colocated | Disagg (KubeRay) | Disagg (JobSet) |
|---------|-----------|-------------------|-----------------|
| Resources to manage | 1 RayJob | 2 RayClusters + RBAC | 1 JobSet + RBAC |
| Failure cascading | KubeRay built-in | Peer-watcher sidecar | Native failurePolicy |
| Gang scheduling | Single PodGroup | Two PodGroups (no cross-gang) | Single PodGroup |
| Head discovery | KubeRay Service | ConfigMap registry | DNS (predictable) |
| vLLM URL exchange | N/A (colocated) | ConfigMap registry | ConfigMap registry |
| Startup ordering | KubeRay built-in | Implicit (both start) | Init containers |
| KubeRay required | Yes | Yes | No |

## File layout

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
│       ├── kai-scheduler.yaml
│       └── kuberay-operator.yaml
├── examples/                          # Workload examples
│   ├── rayjob-monolithic.yaml        # Single-cluster RayJob (1 GPU)
│   ├── disagg-rayclusters.yaml       # Disagg RL + Gym via KubeRay RayClusters
│   ├── disagg-jobset.yaml            # Disagg RL + Gym via JobSet (no KubeRay)
│   ├── endpoint-registry-rbac.yaml   # RBAC for ConfigMap service discovery
│   ├── gym_standalone_config.yaml    # Gym standalone server config
│   ├── kai-queue.yaml                # 2-GPU kind cluster queues
│   └── kai-queue-prod.yaml           # 288-GPU NVL72 prod queues
```

## Helmfile environments

| Environment | GPU component | Use case |
|-------------|---------------|----------|
| `kind` | nvidia-device-plugin | Local dev — nvkind handles toolkit/runtime |
| `prod` | (none — cluster provides GPU Operator) | Real clusters |

Both environments include KAI scheduler, KubeRay operator, and JobSet controller.

## Tear down (kind only)

```sh
kind delete cluster --name nemo-rl
```

## Notes

- **nvkind vs vanilla kind**: nvkind automates GPU device injection, nvidia-container-toolkit installation inside nodes, containerd nvidia runtime configuration, and RuntimeClass registration.
- **nvidia-device-plugin** (kind only): The GPU Operator doesn't work in kind because its driver validation fails inside kind nodes. The lightweight device plugin with CDI discovery is sufficient since nvkind handles the runtime setup.
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

- `kai-queue.yaml` — 2-GPU kind cluster (high-prio + low-prio)
- `kai-queue-prod.yaml` — 288-GPU NVL72 production cluster (priority + community departments)
