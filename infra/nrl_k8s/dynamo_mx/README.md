# Dynamo + ModelExpress v2 Infra

This directory contains the shared Kubernetes pieces for running Dynamo as the
NeMo-RL generation backend with ModelExpress v2 weight refit.

The current integration does not use a Dynamo worker source overlay. The DGD
worker image is expected to be built from the Dynamo integration branch
`jthomson04/tokenize-endpoint-merge-main-06-09` with modelexpress, NIXL/UCX, the
tokenize/completion-token extensions, and the MX refit worker extension baked
in. DGD manifests enable the receiver with `DYN_MX_REFIT_ENABLED=1`.

## Files

| File | Purpose |
|---|---|
| `modelexpress-server.yaml` | Deployment and Service for the MX server (`:8001`). |
| `Dockerfile.nemorl` | NeMo-RL trainer image helper for MX/NIXL dependencies and SWE sandbox tooling. |
| `prometheus.yaml` | Prometheus instance used by the GlobalPlanner/GlobalRouter GP example. |

The removed Dynamo worker overlay used to live here. Build the Dynamo image from
the Dynamo repository instead of copying Python files from a mounted checkout at
pod startup.

## Refit Flow

1. The trainer publishes weights through `stream_weights_via_mx` to the MX
   server.
2. `DynamoGeneration` discovers live workers from the DGD frontend `/health`
   response and calls each worker admin server on `DYN_SYSTEM_PORT`.
3. Each worker receives `POST /engine/update_weights_via_mx`, pulls weights via
   NIXL RDMA, then receives `POST /engine/flush_cache`.
4. The dispatcher re-runs discovery so restarted or newly scaled workers are
   brought to the same version before the training step continues.

Parser selection is DGD-owned. Set tool/reasoning parsers on the
`VllmDecodeWorker` args with `--dyn-tool-call-parser` and
`--dyn-reasoning-parser`; do not put parser fields in
`policy.generation.dynamo_cfg`.

## Retained Examples

| Path | Purpose |
|---|---|
| `infra/nrl_k8s/examples/grpo_workplace_assistant_dynamo_mx_gp.gb300.infra.yaml` | Workplace-assistant GP/GlobalRouter topology. |
| `examples/nemo_gym/grpo_workplace_assistant_dynamo_mx_gp.yaml` | Trainer recipe for the GP topology. |
| `infra/nrl_k8s/examples_dgd/qwen3_4b_thinking_gb300_mx_gp_{pool0,pool1,ctrl}.yaml` | Three DGD manifests for the GP topology. |
| `infra/nrl_k8s/examples/grpo_swe2_qwen3_30b_dynamo_mx.gb300.infra.yaml` | SWE2 GB300 trainer infra. |
| `examples/nemo_gym/grpo_swe2_qwen3_30b_dynamo_mx.yaml` | SWE2 trainer recipe. |
| `infra/nrl_k8s/examples_dgd/qwen3_30b_thinking_gb300_mx_4gpu.yaml` | SWE2 DGD manifest. |
| `infra/nrl_k8s/examples/k8s_exemplars/V1` through `V7` | Versioned Dynamo + MX exemplars. `V6` is FP8 KV cache; `V7` is EAGLE. |

Before launching, apply `modelexpress-server.yaml` in the target namespace and
ensure the trainer and DGD images carry compatible modelexpress and NIXL builds.
