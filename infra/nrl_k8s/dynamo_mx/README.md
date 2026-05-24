# Dynamo + ModelExpress v2 (mid-training weight refit) — infra

This directory holds the pieces needed to run Dynamo as nemo-rl's generation
backend **with mid-training weight refit via ModelExpress v2 NIXL RDMA**. It's
the cross-node analog of vLLM's `update_weights_from_collective` — same
trainer-side `stream_weights_via_mx` (already on `dynamo-k8s-integration` via
the cherry-pick of `d58dca07`), different receiver: each Dynamo
`VllmDecodeWorker` pod runs `MxRefitWorkerExtension.update_weights_via_mx`,
registered as `parallel_config.worker_extension_cls` on the dynamo side.

## Layout

```
infra/nrl_k8s/dynamo_mx/
├── README.md                 (this file)
├── modelexpress-server.yaml  Kubernetes Deployment + Service for the MX server
├── Dockerfile.nemorl         trainer image overlay: nixl + modelexpress + protobuf-6 + wandb 0.26+
├── Dockerfile                (placeholder) dynamo worker image overlay
└── DEBUGGING_POSTMORTEM.md   chronology of the bring-up findings (cluster bugs,
                              version-mismatch failure modes, etc.)
```

## Architecture

The trainer drives each refit cycle synchronously over HTTP:

1. `GET <dgd-frontend>:8000/health` → enumerate live `VllmDecodeWorker`
   instances from the `instances[*]` array; pair each pod IP with
   `DYN_SYSTEM_PORT` (9090) to form `system_url`.
2. For each new instance:
     - `POST {system_url}/engine/pause_generation`
     - `POST {system_url}/engine/update_weights_via_mx` (blocks on NIXL receive)
     - `POST {system_url}/engine/flush_cache`
     - `POST {system_url}/engine/resume_generation` (try/finally so always re-enabled)
3. Re-discover via `/health`; if new `instance_id`s appeared (scale-up
   mid-cycle), refit those too. Bounded at 5 convergence passes.

Code lives at:

  * `nemo_rl/models/generation/dynamo/dynamo_generation.py` —
    `_dispatch_update_weights_via_mx_remote` (the dispatcher above)
  * `components/src/dynamo/vllm/handlers.py` on
    `jthomson04/tokenize-endpoint-merge-main-05-07` — adds
    `pause_generation` / `resume_generation` / `flush_cache` handlers
  * `components/src/dynamo/vllm/worker_factory.py` (same branch) — registers
    `/engine/<route>` via `runtime.register_engine_route(...)`

## Image expectations

The dynamo worker image must be built from
`jthomson04/tokenize-endpoint-merge-main-05-07` (commit `8590c2694e` or later)
with build args:

```bash
docker buildx build --platform linux/arm64 \
  --build-arg ENABLE_MODELEXPRESS_P2P=true \
  --build-arg MODELEXPRESS_REF=8594fd6 \
  -t <registry>/dynamo-arm-mx:<tag> \
  -f container/rendered.Dockerfile .
```

The `8590c2694e` commit bundles UCX into the nixl_cu12 wheel — required because
the dynamo operator hardcodes `NIXL_PLUGIN_DIR` to a path that only has the GDS
plugin. The DGD spec sets `NIXL_PLUGIN_DIR` explicitly to point at the
venv-installed nixl plugins.

The trainer image (`Dockerfile.nemorl`) similarly needs
`--build-arg NIXL_VERSION=0.10.1 --build-arg MX_REF=8594fd6` to align with the
worker; otherwise the infra YAML has runtime nixl-downgrade + PYTHONPATH
modelexpress-shadow workarounds.

## Components

| File | Purpose |
|---|---|
| `modelexpress-server.yaml` | K8s Deployment + Service for MX server (gRPC :8001) |
| `Dockerfile.nemorl` | Trainer image overlay (nemo-rl-mx) — pip-installs nixl, modelexpress, protobuf-6, wandb 0.26+ into every venv |
| `Dockerfile` | Worker image overlay placeholder; superseded by direct rebuild of `ai-dynamo/dynamo` from `jthomson04/tokenize-endpoint-merge-main-05-07` |
| `DEBUGGING_POSTMORTEM.md` | Bring-up postmortem |
| `../examples/grpo_workplace_assistant_dynamo_mx.gb300.infra.yaml` | infra YAML pointing at the MX-enabled DGD |
| `../examples_dgd/qwen3_4b_thinking_gb300_mx.yaml` | Example DGD manifest with `DYN_MX_REFIT_ENABLED=1` + RoCE claim |
| `../../../examples/nemo_gym/grpo_workplace_assistant_dynamo_mx.yaml` | Recipe with `cluster.weight_sync.method: mx` |
