# Dynamo Integration

This document describes how [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
is integrated into NeMo-RL as a generation backend, and the overall structure of
that integration as it currently stands.

> **Status:** Active development on the `dynamo-k8s-integration` branch.
> Kubernetes-only. The rollout path (nemo-gym HTTP) and ModelExpress v2
> mid-training weight refit are both wired end-to-end; see
> [Current State and Limitations](#current-state-and-limitations) for what is
> and isn't supported.

## What the integration does

A `DynamoGraphDeployment` (DGD) becomes the generation backend for RL training.
This decouples generation from the trainer: inference is an independent,
Kubernetes-managed deployment, and the trainer dispatches rollouts to its
frontend over HTTP. It is the natural fit for **non-colocated** RL and for
[nemo-gym](nemo-gym-integration.md)-style agentic rollouts, which already speak
OpenAI-compatible HTTP.

The key thing to understand is the **division of ownership**, which is
deliberately lopsided:

- **The DGD owns everything inference.** The dynamo operator brings up and owns
  the entire stack — frontend, workers, etcd, NATS — and owns all engine
  arguments (model, TP, parsers, `max_model_len`, …). NeMo-RL never sees them.
- **NeMo-RL owns almost nothing on the serving side.** It resolves the frontend
  URL and dispatches to it. The generation backend is a thin client; the only
  trainer→DGD coupling beyond the rollout URL is the weight-refit path.

So the two integration surfaces worth your attention are: (1) **how the trainer
finds and talks to the DGD frontend** (config + URL resolution), and (2) **how
freshly-trained weights get into the DGD workers each step** (ModelExpress v2 over
NIXL). Everything else is standard Dynamo.

## Architecture

```
┌──────────────────────────── Kubernetes namespace ────────────────────────────┐
│                                                                               │
│   ┌─────────────── Training RayCluster ───────────────┐                       │
│   │  run_grpo_nemo_gym.py (driver)                     │                       │
│   │    └─ GRPO loop                                    │                       │
│   │         ├─ DTensorPolicyWorker (trainer GPUs)      │                       │
│   │         └─ DynamoGeneration  ── HTTP rollouts ─────┼──┐                    │
│   └────────────────────────────────────────────────────┘  │                  │
│            │ stream_weights_via_mx (NIXL RDMA publish)      │ /v1 (OpenAI API) │
│            ▼                                                ▼                  │
│   ┌──────────────────┐        ┌──────────── DynamoGraphDeployment ──────────┐ │
│   │ modelexpress-     │◀──────▶│  Frontend (HTTP :8000, router)              │ │
│   │ server (gRPC      │  pull  │     │ routes /v1 + /health                  │ │
│   │ :8001)            │        │     ▼                                       │ │
│   │  - source         │        │  VllmDecodeWorker × N                       │ │
│   │    discovery      │◀──RDMA─│     - vLLM engine                           │ │
│   │  - shape registry │ (NIXL) │     - MxRefitWorkerExtension                │ │
│   └──────────────────┘        │     - /engine/<route> admin (DYN_SYSTEM      │ │
│                               │       _PORT :9090)                           │ │
│                               │  etcd + NATS (service discovery / routing)   │ │
│                               └─────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

Two data planes connect the trainer and the DGD:

1. **Rollout path (HTTP).** nemo-gym dispatches generation requests to the DGD
   frontend's OpenAI-compatible `/v1` endpoint. The frontend load-balances across
   workers. This is the only generation path — `DynamoGeneration.generate()` is
   intentionally not implemented.
2. **Weight-refit path (NIXL RDMA, optional).** When `cluster.weight_sync.method
   = "mx"`, the trainer publishes new weights to the ModelExpress (MX) server each
   step and each DGD worker RDMA-pulls them. See
   [Weight Refit via ModelExpress v2](#weight-refit-via-modelexpress-v2).

## NeMo-RL Side

### `DynamoGeneration` backend

`nemo_rl/models/generation/dynamo/dynamo_generation.py` defines
`DynamoGeneration`, which implements `GenerationInterface` directly (it does *not*
extend `VllmGeneration`). It is selected via `policy.generation.backend = "dynamo"`
through the factory in `nemo_rl/models/generation/__init__.py`.

It is essentially a URL forwarder plus a refit dispatcher:

- **Construction** resolves the cluster-internal frontend URL. Either:
  - `dynamo_cfg.dgd_name` — derives
    `http://<dgd_name>-frontend.<namespace>.svc.cluster.local:<port>/v1` from the
    dynamo operator's stable Service naming. Requires running inside a Kubernetes
    pod (checked via `KUBERNETES_SERVICE_HOST`). nrl-k8s stamps this field
    automatically.
  - `dynamo_cfg.frontend_url` — an explicit reachable URL (escape hatch for
    non-default Service names, NodePort/Ingress, or running outside Kubernetes).
    This disables the in-pod check.

  The resolved URL is exposed as `dp_openai_server_base_urls`, which nemo-gym
  reads to dispatch rollouts.

- **Lifecycle methods** (`prepare_for_generation`, `finish_generation`,
  `shutdown`) are no-ops — the DGD lifecycle is owned by Kubernetes.
- **`generate()`** raises `NotImplementedError`; rollouts flow through the
  nemo-gym HTTP path.
- **Collective / IPC weight-sync methods** (`init_collective`,
  `update_weights_from_collective`, `update_weights_via_ipc_zmq`) raise
  `NotImplementedError` — non-colocated refit goes through MX.
- **`__getstate__` / `__setstate__`** make the object picklable so async rollouts
  can ship the `GenerationInterface` across Ray actors.

### Config

`nemo_rl/models/generation/dynamo/config.py`:

- `DynamoCfg` (the `policy.generation.dynamo_cfg` block):
  | Field | Purpose |
  |---|---|
  | `dgd_name` | `metadata.name` of the DGD; used to derive the frontend URL. |
  | `frontend_url` | Explicit URL (mutually exclusive with `dgd_name`; wins if both set). |
  | `namespace` | K8s namespace (auto-resolved from the pod if omitted). |
  | `frontend_port` | Frontend HTTP port (default `8000`). |
  | `tool_call_parser`, `reasoning_parser` | Compatibility hints; the actual parsers are configured on the DGD worker via `--dyn-*` flags. |

- `DynamoConfig(GenerationConfig)` adds `dynamo_cfg` plus `vllm_cfg` /
  `vllm_kwargs` *compatibility shims* — the DGD owns the real inference-engine
  args, but nemo-gym today reads `cfg["vllm_cfg"]["max_model_len"]` directly, so
  those fields are retained (not authoritative).

## nrl-k8s Orchestration

`infra/nrl_k8s` is the orchestration layer that brings up the DGD and wires the
recipe to it before the training entrypoint runs.

- **DGD ingestion** (`src/nrl_k8s/dgd.py`): loads a standalone DGD manifest
  (`load_dgd_manifest`), deep-merges `overrides`, patches cross-cutting `infra`
  fields (image, imagePullSecrets, serviceAccount, labels) into each service's
  `extraPodSpec` (`build_dgd_manifest`), and provides create-or-replace CRUD plus
  `wait_for_dgd_ready` (waits for `status.state == "successful"` and all pods
  Ready). CRD identifiers: `nvidia.com/v1alpha1`, kind `DynamoGraphDeployment`.
- **Schema** (`src/nrl_k8s/schema.py`): `DynamoGraphSpec` (`manifest`, `name`,
  `overrides`, `labels`, `readyTimeoutS`, …); `InfraConfig.dynamo` is a
  `dict[str, DynamoGraphSpec]` of named DGDs declared by the run.
- **Orchestration** (`src/nrl_k8s/orchestrate.py`): `ensure_dgd` applies the
  manifest idempotently (warns on drift, recreates on `--recreate`), optionally
  attaches an `ownerReference` so the DGD is GC'd with the training RayCluster,
  and waits for readiness. When exactly one DGD is declared, nrl-k8s stamps
  `policy.generation.dynamo_cfg.dgd_name` into the recipe automatically (the infra
  YAML's entrypoint passes `+policy.generation.dynamo_cfg.dgd_name=<name>`).

### Config files

A run is described by three files that reference each other:

| File | Role |
|---|---|
| `examples/nemo_gym/grpo_workplace_assistant_dynamo_*.yaml` | **Recipe** — sets `policy.generation.backend: dynamo`, GRPO hyperparameters, and (for MX) `cluster.weight_sync`. |
| `infra/nrl_k8s/examples/*.gb300.infra.yaml` | **Infra** — training RayCluster spec, image, the `dynamo:` block pointing at a DGD manifest, and the launch entrypoint. |
| `infra/nrl_k8s/examples_dgd/*.yaml` | **DGD manifest** — the `DynamoGraphDeployment` itself (Frontend + worker services, model, parsers, env). |

Current examples: a non-MX `..._smoke` variant, an MX variant
(`..._dynamo_mx`), a 16×TP2 MX scale variant (`..._dynamo_mx_16tp2`), and a
small `qwen3_0.6b_kind` variant for local Kind clusters.

## Rollout Path

nemo-gym performs agentic rollouts by issuing OpenAI-compatible HTTP requests to
the URL(s) in `DynamoGeneration.dp_openai_server_base_urls`. Those requests hit
the DGD frontend, which routes them across `VllmDecodeWorker` replicas. NeMo-RL
is not in this loop beyond having resolved the URL — there is no per-token or
per-request involvement on the trainer side.

## Weight Refit via ModelExpress v2

The hard problem in non-colocated RL is getting freshly-trained weights from the
trainer into the inference workers each step, cross-node, without a shared NCCL
process group. The integration solves this with **ModelExpress v2 (MX)** over
**NIXL RDMA**. This is the cross-node analog of vLLM's
`update_weights_from_collective`. It is selected with `cluster.weight_sync.method
= "mx"` and is **pull-based**.

### Components

- **MX server** (`infra/nrl_k8s/dynamo_mx/modelexpress-server.yaml`) — a gRPC
  service (`:8001`) that holds source discovery state and a per-version tensor
  **shape registry**. Both the trainer and the workers must point at the same
  `mx_server_url`.
- **Trainer publisher** — `DTensorPolicyWorker.stream_weights_via_mx`
  (`nemo_rl/models/policy/workers/dtensor_policy_worker.py`), built via
  `nemo_rl/distributed/mx_helpers.py::build_v2_publisher` (an
  `MxV2TrainingPublisher`).
- **Worker receiver** — each `VllmDecodeWorker` runs `MxRefitWorkerExtension`
  (registered on the dynamo side as `parallel_config.worker_extension_cls`,
  gated by `DYN_MX_REFIT_ENABLED=1`), which builds an `MxV2RefitReceiver`.
- **Refit dispatcher** — `_dispatch_update_weights_via_mx_remote` (a Ray remote
  in `dynamo_generation.py`) drives the per-worker refit over the workers'
  `/engine/<route>` admin HTTP server.

### MxConfig

`nemo_rl/distributed/mx_helpers.py::MxConfig` mirrors the
`cluster.weight_sync.mx_config` block. The trainer and workers must agree on these
because they encode the transfer contract:

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `False` | Master switch; when off, refit falls back to NCCL collective. |
| `mx_server_url` | `modelexpress-server:8001` | gRPC URL of the MX server. |
| `timeout_seconds` | `300.0` | Max wait for source discovery / RDMA receive (also the dispatcher's retry deadline). |
| `same_rank_only` | `True` | Restrict transfers to trainer-rank-N → inference-rank-N pairs. **Required** on multi-subnet RDMA fabrics (GB200/GB300/EFA). |
| `tree_scale_out` | `True` | Receivers republish themselves as sources after refit, so later receivers pull from peers instead of the trainer's NIC. |
| `moe_expert_filter` | `True` | Receivers pull only the expert shards their EP rank owns (no-op for dense models). |
| `nic_pin` | `auto` | NUMA-local NIC pinning before NIXL init (`auto` / `off` / concrete `mlx5_<i>`). |
| `retain_latest_k` | `1` | TensorHub-style retention (forward-compat; not yet enforced server-side). |

### Refit flow

The orchestration lives in `refit_policy_generation`
(`nemo_rl/algorithms/grpo.py`). The MX path is **serialized publish-then-pull**
(unlike the NCCL collective path, which must run concurrently or it deadlocks):

1. **Publish (trainer).** `policy.stream_weights_via_mx(version, mx_config)` →
   each `DTensorPolicyWorker`:
   - Lazily builds an `MxV2TrainingPublisher` once per worker (NIXL registration
     happens on the first publish; local addresses are stable across steps).
   - For each tensor in the state dict: DTensors are materialized with
     `full_tensor()` (allgather across the FSDP/TP mesh) so the published bytes
     match the global shape recorded in the registry, then registered with NIXL.
     The receiver reshapes to the global shape and vLLM applies its own TP
     sharding. *(This trades away the v2 "no-allgather" optimization; per-rank
     expert publishing for MoE/EP is a follow-up.)*
   - Calls `publish(version)` then `mark_ready()`.
   - The driver `ray.get`s the publish futures so the publish fully completes
     before any receiver is triggered.
2. **Pull (workers).** `policy_generation.update_weights_via_mx(version,
   mx_config)` dispatches `_dispatch_update_weights_via_mx_remote`, which:
   - `GET <dgd-frontend>:8000/health` to enumerate live worker instances (keyed
     by stable `instance_id`), pairing each pod IP with `DYN_SYSTEM_PORT` (9090)
     to form the per-pod `system_url`. Retries with backoff on the first pass
     (workers may be container-Ready but not yet registered in discovery).
   - For each *new* worker: `POST /engine/pause_generation` → `POST
     /engine/update_weights_via_mx` (the real NIXL receive, blocks) → `POST
     /engine/flush_cache` (drop stale prefix cache) → `POST
     /engine/resume_generation` (in a `finally`, so generation is always
     re-enabled).
   - **Re-discovers** after each pass: if new `instance_id`s appeared (a worker
     scaled in or restarted mid-cycle), it refits those too, converging over up
     to 5 passes. Workers that disappeared mid-cycle are dropped (they can't be
     serving stale weights).
   - **Retries transient refit failures** until the shared cycle deadline. At
     scale (e.g. 16 workers), early receivers can fire inside the ~1s window
     between the trainer's `mark_ready()` and that READY status propagating into
     the server's `list_sources(status_filter=READY)` index, getting "no v2
     source available". As long as the dispatcher keeps running, the trainer
     stays blocked in `ray.get` and alive, so its heartbeat holds the sources
     READY and a re-issued refit finds them. Raising on the first failure instead
     would crash the trainer, STALE the sources, and doom every remaining worker.
   - Raises (failing the run loudly) on per-worker step errors or non-convergence
     — replacing the silent stale-weight failure mode of an earlier poll-only
     design.

### RDMA fabric setup

NIXL RDMA over RoCE requires non-trivial pod plumbing, captured in the example
infra/DGD YAMLs (GB300):

- **RoCE DRA `resourceClaim`** on both the trainer worker pod and each DGD worker
  pod (nrl-k8s auto-creates the matching `ResourceClaimTemplate`). NIXL also needs
  pinned host memory, hence the `IPC_LOCK` capability on the trainer.
- **UCX env knobs**: `UCX_TLS=rc,cuda_copy` (restricted set — the broader set
  picks intra-host transports first and `prep_xfer_dlist` fails for cross-pod
  descriptors), `UCX_IB_GPU_DIRECT_RDMA=yes`, `UCX_CUDA_COPY_DMABUF=yes`,
  `MX_RDMA_NIC_PIN=auto`.
- **`NIXL_PLUGIN_DIR`** pointed at the venv-installed nixl plugins — the dynamo
  operator hardcodes a path that only has the GDS plugin, while the UCX plugin
  ships inside the bundled-UCX nixl wheel.
- **Images** (`infra/nrl_k8s/dynamo_mx/`): the trainer overlay
  (`Dockerfile.nemorl`) bakes `nixl` + `modelexpress` + `protobuf>=6` into every
  venv; the worker image is built from `ai-dynamo/dynamo` with
  `ENABLE_MODELEXPRESS_P2P=true` and a UCX-bundled nixl wheel. Trainer and worker
  nixl/modelexpress versions must align. See the directory's `README.md` for the
  exact build recipe.

## Current State and Limitations

- **Kubernetes-only.** The `dgd_name` path requires running inside a pod;
  `frontend_url` is the only way to point at a DGD from outside.
- **No `generate()`.** Direct generation is not supported; rollouts go through
  the nemo-gym HTTP path.
- **MX refit is DTensor-only.** `stream_weights_via_mx` is implemented on
  `DTensorPolicyWorker`. Megatron-backend MX support is a follow-up — MX recipes
  flip `megatron_cfg.enabled: false` / `dtensor_cfg.enabled: true`.
- **DTensors are allgathered** (`full_tensor()`) on publish rather than published
  as local shards. The per-rank/no-allgather optimization (and MoE expert
  filtering) is a known follow-up.
- **FP8 KV-cache scales** are not yet supported on the MX path.
- **No collective / IPC weight sync** for the Dynamo backend — MX is the only
  non-colocated refit mechanism.
- **Autoscaling / planner** integration (VirtualConnector-driven scaling) is not
  part of the current DGD-centric architecture; the refit dispatcher tolerates
  worker pool churn (re-discovery + convergence passes) but does not itself drive
  scaling.

## File Map

| Area | Path |
|---|---|
| Generation backend | `nemo_rl/models/generation/dynamo/dynamo_generation.py` |
| Backend config | `nemo_rl/models/generation/dynamo/config.py` |
| Backend selection | `nemo_rl/models/generation/__init__.py` |
| MX helpers (config, publisher/receiver, NIC pin) | `nemo_rl/distributed/mx_helpers.py` |
| Trainer-side publish | `nemo_rl/models/policy/workers/dtensor_policy_worker.py` (`stream_weights_via_mx`) |
| Refit orchestration | `nemo_rl/algorithms/grpo.py` (`refit_policy_generation`) |
| DGD ingestion / CRUD | `infra/nrl_k8s/src/nrl_k8s/dgd.py` |
| DGD schema | `infra/nrl_k8s/src/nrl_k8s/schema.py` (`DynamoGraphSpec`, `InfraConfig.dynamo`) |
| DGD orchestration | `infra/nrl_k8s/src/nrl_k8s/orchestrate.py` (`ensure_dgd`) |
| MX infra (server, images, README) | `infra/nrl_k8s/dynamo_mx/` |
| Example recipes | `examples/nemo_gym/grpo_workplace_assistant_dynamo_*.yaml` |
| Example infra | `infra/nrl_k8s/examples/*dynamo*.infra.yaml` |
| Example DGD manifests | `infra/nrl_k8s/examples_dgd/*.yaml` |
| Unit tests | `tests/unit/models/generation/test_dynamo_generation.py` |
