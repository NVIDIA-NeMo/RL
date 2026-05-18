# Dynamo + ModelExpress v2 (mid-training weight refit) — infra + dev workflow

This directory holds the pieces needed to run Dynamo as nemo-rl's generation
backend **with mid-training weight refit via ModelExpress v2 NIXL RDMA**. It's
the cross-node analog of vLLM's `update_weights_from_collective` — same
trainer-side `stream_weights_via_mx` (already on `dynamo-k8s-integration` via
the cherry-pick of `d58dca07`), different receiver: each Dynamo `VllmDecodeWorker`
pod's vLLM process runs `MxRefitWorkerExtension.update_weights_via_mx`,
registered as `parallel_config.worker_extension_cls` on the dynamo side.

## Layout

```
infra/nrl_k8s/dynamo_mx/
├── README.md                 (this file)
├── modelexpress-server.yaml  Kubernetes Deployment + Service for the MX server
├── dev_sync.sh               rsync local dynamo checkout to Lustre for hot-reload
└── Dockerfile                (TODO) overlay image adding modelexpress + NIXL on
                              top of the base dynamo worker image
```

## How the no-rebuild dev loop works

`dynamo` is a [PEP 420 namespace
package](https://peps.python.org/pep-0420/) — there is no `dynamo/__init__.py`
at the top level. Two distributions contribute to the namespace:

  * `ai-dynamo-runtime` (the Rust pyo3 binding) installs `dynamo._core`,
    `dynamo.runtime`, `dynamo.llm`, etc. into site-packages.
  * `ai-dynamo` (the Python components) installs `dynamo.vllm`,
    `dynamo.frontend`, `dynamo.planner`, etc. into the same `dynamo/` tree.

Because the top-level `dynamo` is a namespace package, **a directory earlier on
`sys.path` wins for subpackages it provides** (e.g. `dynamo.vllm.mx_refit`)
*without* affecting subpackages it doesn't provide (e.g. `dynamo._core` still
loads from the Rust binding in site-packages). This is the trick that lets us
develop pure-Python dynamo changes without rebuilding the worker image.

### Setup (once per user)

```bash
# From the nemo-rl repo root, on the dev pod or any machine with Lustre access:
infra/nrl_k8s/dynamo_mx/dev_sync.sh
```

The script rsyncs `/mnt/rl-workspace/$USER/dynamo/components/src/dynamo/` →
`/mnt/rl-workspace/$USER/dynamo-dev/dynamo/` so it's on the Lustre PVC every
worker pod mounts at `/mnt/rl-workspace`.

### Per-iteration loop

```bash
# 1. Edit dynamo Python files at /mnt/rl-workspace/$USER/dynamo/components/src/dynamo/...
# 2. Re-sync to the dev path:
infra/nrl_k8s/dynamo_mx/dev_sync.sh
# 3. Restart the affected worker pods:
kubectl delete pod -n default -l <dgd-worker-selector>
# (the dynamo operator reconciles and recreates them; ~30s vs ~10min for a rebuild)
```

The DGD manifests in this directory set
`PYTHONPATH=/mnt/rl-workspace/$USER/dynamo-dev:$PYTHONPATH` on the `VllmDecodeWorker`
container, so the recreated pods pick up our local code without an image
rebuild.

### Limits

  * Only Python-only changes hot-reload. Rust changes (the binding, the
    frontend) still need an image rebuild.
  * Adding *new* dependencies that aren't in the base image needs an image
    rebuild. The dev loop only swaps existing module contents.
  * `modelexpress` itself isn't in the existing dynamo worker image. The
    Dockerfile in this directory bakes it in. Once that image is built and
    pushed, subsequent iteration on `dynamo.vllm.mx_refit.*` is hot-reload.

## Components

| File | Purpose |
|---|---|
| `modelexpress-server.yaml` | K8s Deployment + Service for MX server (gRPC :8001) |
| `dev_sync.sh` | rsync helper for the hot-reload workflow |
| `Dockerfile` | (TODO) image overlay: base dynamo worker + `modelexpress` Python + NIXL userspace |
| `qwen3_4b_thinking_mx.dgd.yaml` | example DGD manifest with `DYN_MX_REFIT_ENABLED=1`, RoCE claim, PYTHONPATH dev override |
| `../examples/grpo_workplace_assistant_dynamo_mx.gb300.infra.yaml` | infra YAML pointing at the MX-enabled DGD |
| `../../../examples/nemo_gym/grpo_workplace_assistant_dynamo_mx.yaml` | recipe with `cluster.weight_sync.method: mx` |
