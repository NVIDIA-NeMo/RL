<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Quickstart — full NeMo-RL × Megatron × ModelExpress deployment

Stand up the whole refit stack on Kubernetes: the ModelExpress (MX) server, a
Megatron-Core trainer, and Dynamo/vLLM rollout workers, all sharing one MX
server for GPU-to-GPU weight refit. Cluster-agnostic — substitute the
`<PLACEHOLDER>` values for your environment.

## Prerequisites

- A Kubernetes cluster with GPU nodes, RDMA NICs, and the Dynamo operator + a
  Ray/KubeRay operator installed.
- A worker image built from a Dynamo + vLLM base with the **ModelExpress client
  installed** (it auto-registers its vLLM integrations via its
  `vllm.general_plugins` entry point). See `../../dynamo_mx/Dockerfile` for the
  layer that pins the client on top of a Dynamo base image.
- A trainer image with the NeMo-RL Megatron stack + MX/NIXL deps
  (`../../dynamo_mx/Dockerfile.nemorl`).
- A shared workspace/HF-cache PVC and a registry pull secret.

## The three tiers

| Tier | Manifest / config | Role |
|---|---|---|
| MX server | `modelexpress-server.yaml` | control-plane catalog (gRPC `:8001`), one per namespace |
| Trainer | `../examples/k8s_exemplars/V2/grpo_llama3_1_8b_instruct_megatron_dynamo_mx.yaml` (run config) + its `.gb300.infra.yaml` (infra) | Megatron GRPO trainer, publishes per-rank shards |
| Rollout | `examples/rollout-dgd.template.yaml` (generalized) | Dynamo/vLLM workers, pull + reshape + load |

The trainer publishes each rank's native shard to the MX server; each rollout
worker discovers the current version and pulls only the slices it needs over
NIXL RDMA. Bytes flow GPU-to-GPU; the server carries references only.

## Steps

**1. MX server** (once per namespace):

```bash
export K8S_NAMESPACE=<NAMESPACE>
export MX_SERVER_IMAGE=<MX_SERVER_IMAGE>       # ai-dynamo/modelexpress server build
export MX_IMAGE_PULL_SECRET=<IMAGE_PULL_SECRET>
export MX_SERVICE_ACCOUNT=default
envsubst < modelexpress-server.yaml | kubectl -n $K8S_NAMESPACE apply -f -
kubectl -n $K8S_NAMESPACE rollout status deploy/modelexpress-server
```

All clients reach it at `modelexpress-server.<NAMESPACE>.svc.cluster.local:8001`.

**2. Rollout workers** (DGD). Fill in the placeholders in
`examples/rollout-dgd.template.yaml` (worker image, model id, namespace, pull
secret, HF cache path, PVC, GPU product, RoCE claim), then:

```bash
kubectl -n <NAMESPACE> apply -f examples/rollout-dgd.template.yaml
kubectl -n <NAMESPACE> get pods -l nvidia.com/dynamo-component-type=worker
```

For MoE models, uncomment `--moe-backend triton`. To exercise fan-out or
elasticity, raise `VllmDecodeWorker.replicas`.

**3. Trainer.** Launch the NeMo-RL Megatron GRPO run pointed at the same MX
server. Start from the V2 exemplar run config and set the MX server URL in the
weight-sync config:

```yaml
# in the run config
cluster:
  weight_sync:
    mx_server_url: modelexpress-server.<NAMESPACE>.svc.cluster.local:8001
policy:
  megatron_cfg: {enabled: true, tensor_model_parallel_size: 1, pipeline_model_parallel_size: 1}
  generation: {backend: "dynamo"}
```

The V2 exemplar (`grpo_llama3_1_8b_instruct_megatron_dynamo_mx`) is a 16-GPU
async Megatron GRPO run; the `.gb300.infra.yaml` sibling is the K8s infra
manifest (adjust the node group / GPU product for your hardware).

## Verify the refit path (without a full GRPO run)

The smoke harnesses under `smoke/` run a trainer(publisher)/rollout(receiver)
pair against the MX server and check byte-identity:

```bash
export MX_SERVER_URL=modelexpress-server.<NAMESPACE>.svc.cluster.local:8001
export MODEL=<MODEL_ID>
# on a trainer pod:
python smoke/smoke_real_megatron_publisher.py
# on a rollout pod:
python smoke/smoke_real_megatron_receiver.py
```

Other harnesses: `smoke_mixed_tp_*` (heterogeneous TP), `smoke_qwen3_moe_*`
(MoE + `--moe-backend triton`), `smoke_fanout_receiver.py` (fan-out).

## Weight-transfer enablement: two options

- **Going-forward (native):** `--weight-transfer-config '{"backend":"mx"}'` on
  the worker. The MX client registers the `"mx"` backend via its plugin entry
  point, so this is a launch flag once the Dynamo build forwards it.
- **Current (validated):** `--load-format mx` + `DYN_MX_REFIT_ENABLED=1`, as in
  the template. Both are shown (commented) in `examples/rollout-dgd.template.yaml`.

## Where to look next

- Transport toggles, perf levers, and preliminary numbers: the NeMo-RL × MX run
  guide.
- MX client, planner, translator, native backend, MDL loader:
  `ai-dynamo/modelexpress` PR #482.
- Trainer publish path + these infra artifacts: `NVIDIA-NeMo/RL` PR #3068.
