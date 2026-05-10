# Fault-Tolerant Disaggregated Generation (RL-412)

## Overview

Fault-tolerant disaggregated generation splits an RL training run into two
independent RayClusters:

- **Training RayCluster** -- runs the GRPO training driver, Megatron policy
  workers, and reward/value models. Managed by the standard `nrl-k8s` CLI.
- **Generation RayCluster** -- runs vLLM DP shards behind a
  `GenerationRouter` that provides health monitoring, shard lifecycle, and
  HTTP-based data plane proxying. Uses KubeRay autoscaler v2 to provision
  and reclaim pods dynamically.

The training driver talks to the generation cluster exclusively over HTTP
via the `RemoteGeneration` wrapper. Weight updates flow over a cross-cluster
NCCL broadcast managed by the `RefitWorker` split-actor, which isolates the
CUDA context used for the broadcast from Megatron's own CUDA contexts.

A `FaultInjector` actor can be optionally launched on the generation cluster
to simulate pod kills, actor kills, or HTTP errors for stress testing the
fault-tolerance machinery.


## Architecture

### GenerationRouter (`nemo_rl/models/generation/generation_router.py`)

FastAPI application serving both the data plane and control plane on a single
port (default 8089):

- **Data plane**: `/v1/completions`, `/v1/chat/completions` -- proxied to
  healthy vLLM shards via round-robin.
- **Control plane**: `/add_shard`, `/remove_shard`, `/update_weights_from_collective`,
  `/shards`, `/health` -- shard lifecycle management.
- **Health monitoring**: periodic `/openapi.json` probes per shard with
  configurable `failure_threshold` and `health_timeout_s`. Unhealthy shards
  are cordoned and eventually evicted.

### RefitWorker (`nemo_rl/models/policy/workers/refit_worker.py`)

Split-actor that runs on the same node and GPU as training rank 0 but in a
separate Ray actor with its own primary CUDA context. Owns the cross-cluster
`model_update_group` NCCL communicator. When a generation peer dies
mid-broadcast, only the RefitWorker's CUDA context is poisoned -- Megatron's
EP/TP/PP/DP groups remain clean. The RefitWorker is then `ray.kill()`-ed and
respawned with a fresh CUDA context for the next refit cycle.

Enabled by `NRL_USE_REFIT_WORKER=1`.

### FaultInjector (`nemo_rl/models/generation/fault_inject.py`)

Configurable fault injection driver that runs as a Ray actor on the generation
cluster. Supports three modes:

- `pod-kill` -- deletes the Kubernetes pod backing a target shard
- `actor-kill` -- `ray.kill()`s the vLLM generation worker actor
- `http-error` -- injects HTTP 500 errors on the router's proxy path

Configurable parameters include `trigger_after_s`, `recover_after_s`,
`repeat_every_s`, `rotate_target`, `burst_size_random_max`, and
`new_shard_grace_period_s`.

### RemoteGeneration (`nemo_rl/models/generation/remote_generation.py`)

HTTP-only disaggregated wrapper used by the training driver. Replaces the
in-process vLLM generation path with HTTP calls to the GenerationRouter's
`/v1/completions` endpoint. Configured via
`policy.generation.remote_generation_url` in the recipe YAML.


## How to Run

Three config files are needed:

1. **Recipe YAML** (training parameters):
   `infra/nrl_k8s/examples/qwen3_30b_if_fault_tolerant_full.yaml`
2. **Infra YAML** (KubeRay cluster specs for both train and gen):
   `infra/nrl_k8s/examples/qwen3_30b_if_fault_tolerant_full.gb300.infra.yaml`
3. **Generation standalone YAML** (vLLM config + fault injection):
   `infra/examples/generation_standalone_qwen3_30b_ft_full.yaml`

Launch with `nrl-k8s`:

```bash
nrl-k8s launch \
  --config infra/nrl_k8s/examples/qwen3_30b_if_fault_tolerant_full.yaml \
  --infra  infra/nrl_k8s/examples/qwen3_30b_if_fault_tolerant_full.gb300.infra.yaml
```

The generation cluster is brought up first (via the `launch.attach.generation`
cluster name in the infra YAML), followed by the training cluster. The gen
daemon (`run_standalone_generation_server.py`) is submitted as a Ray Job to the
gen cluster's head pod.


## Key Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `NRL_USE_REFIT_WORKER` | `1` | Enable RefitWorker split-actor for CUDA context isolation |
| `NRL_REFIT_NUM_BUFFERS` | `1` | Single-buffer refit (avoids double-buffer broadcast deadlock) |
| `NCCL_NVLS_ENABLE` | `0` | Disable NVLink SHARP on GB300 (avoids IMEX channel errors) |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `60` | Bound hung-collective wait for fast failure detection |
| `TORCH_NCCL_RETHROW_CUDA_ERRORS` | `0` | Surface CUDA errors as exceptions instead of SIGABRT |
| `NRL_REFIT_PACKED_DEBUG` | `1` | Per-chunk SHA1 fingerprints for refit corruption diagnosis |


## Known Issues

- **TP=2 on vLLM gen side**: not validated. Causes NCCL broadcast hang on the
  cross-cluster `model_update_group`. Use TP=1 (the canonical layout).
- **Double-buffering (`NRL_REFIT_NUM_BUFFERS=2`)**: causes RefitWorker
  broadcast deadlock where buffer-1's NCCL broadcast starts before buffer-0
  has been consumed by all gen shards. Use `NRL_REFIT_NUM_BUFFERS=1`.


## Hero Run

- **Hemil's hero run**: [wandb nvidia/nemorl-rl-412-fault-tolerant/runs/4kjwd1sk](https://wandb.ai/nvidia/nemorl-rl-412-fault-tolerant/runs/4kjwd1sk)
- **Terry's replication run**: [wandb nvidia/nemorl-rl-412-fault-tolerant/runs/vcf4wuyq](https://wandb.ai/nvidia/nemorl-rl-412-fault-tolerant/runs/vcf4wuyq)
- Reward parity: 0.5693 (replication) vs 0.5757 (canonical)


## Linear Issues

- [RL-412](https://linear.app/nvidia-nemo/issue/RL-412) -- fault-tolerant disaggregated generation
- [RL-741](https://linear.app/nvidia-nemo/issue/RL-741) -- related follow-up
