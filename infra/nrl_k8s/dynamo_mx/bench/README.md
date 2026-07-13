<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# MX vs NCCL differentiator benchmarks

Harnesses for measuring the MX/NIXL refit path against NCCL — focused on the
scenarios where MX is expected to win (elasticity, stragglers, partial/EP
refit), plus apples-to-apples transport and per-phase refit timing.

| File | What it does |
|---|---|
| `run_differentiator_bench.sh` | Canonical analyzer orchestrator. It accepts only checkpoint-identified live artifacts for Qwen3-30B-A3B and skips scenarios whose artifacts are absent. |
| `mx_vs_nccl_refit_bench.py` | Deployment-agnostic driver: runs one excluded warmup plus N measured cycles per backend, aggregates the canonical ten stages, emits JSON/CSV, and `--compare`s old or new JSON. Optional versioned trigger/ack flags coordinate an MX publisher. |
| `mx_hf_publisher_bench.py` | One-GPU MX source for receiver-matched comparisons. Preloads checkpoint `hf_weights`, publishes each triggered version through `MxTrainingPublisher`, marks it READY, and acknowledges the matching cycle. |
| `ep_gt1_byte_pruning.py` | Planner-only developer check. It is not accepted as differentiator evidence by the canonical runner. |
| `mdl_partial_smoke.py` | Runtime smoke for MDL incremental/partial-update (cold/warm/subset/incremental, byte-identical). No transport. |
| `configs/nixl_ep4_tp1_gb200_rc_debug.yaml` | Sanitized EP4→TP1 GB200 topology and UCX/NIXL config, including the TCP-fallback baseline and validated `^tcp` fix (12/12 pulls on RDMA, ~7× steady-state refit speedup). |
| `preflight_ep8_tp2.sh` | Fail-fast Kubernetes gate for the two 4-GPU EP8 trainer pods and 2-GPU TP2 rollout: GPU count, `rdma0..3`, `^tcp`, matching package versions, and native `mx` backend registration. |
| `native_nccl_refit_bench.py` | Native-vLLM PyNccl sender/controller baseline. Coordinates one excluded warmup plus N measured packed updates with versioned trigger/ack files and emits raw plus aggregate canonical-stage records. |
| `configs/native_nccl_sender.gb200.yaml` | One-GPU GB200 sender pod for `native_nccl_refit_bench.py`, with four RDMA interfaces and the shared checkpoint PVC. |
| `configs/native_nccl_receiver_30b.gb200.yaml` | Standalone Qwen3-30B-A3B TP2 Dynamo rollout for the NCCL arm. It uses `--load-format auto`, the default vLLM Worker, and `DYN_WEIGHT_TRANSFER_BACKEND=nccl`. |
| `differentiator_suite.py` | Standard JSON analyzers and assertions for all seven differentiators: EP filtering, TP slicing, partial refit, elastic join, straggler isolation, fan-out, and trainer egress balance. |
| `fanout_bench.py` | Real-model NIXL data producer for direct-vs-tree fan-out. Trainer, seed, and receiver roles publish timeline JSON consumed by `differentiator_suite.py fanout`. |

## Run

```bash
# Analyze live differentiator artifacts. Numeric-only and synthetic inputs are
# rejected. See "Canonical artifact contract" below for required metadata.
EP_ARTIFACT=/results/ep.json FULL_EXPERT_BYTES=<measured-expert-bytes> \
TP_ARTIFACT=/results/tp.json PARTIAL_MANIFEST=/results/manifest.json \
PARTIAL_SELECTOR=model.layers.0 MX_ARTIFACT=/results/publisher.json \
MX_LOG=/results/rollout.log MX_STEPS=1 \
ELASTIC_RESULTS=/results/elastic STRAGGLER_RESULTS=/results/straggler \
FANOUT_DIRECT=/results/fanout-direct FANOUT_TREE=/results/fanout-tree \
FANOUT_N=13 OUT=./differentiator_results \
bash run_differentiator_bench.sh

# Native NCCL Real-30B baseline. First replace all <...> placeholders in the
# receiver manifest and the sender image/PVC values for the target namespace.
kubectl -n <namespace> apply -f configs/native_nccl_receiver_30b.gb200.yaml
kubectl -n <namespace> apply -f configs/native_nccl_sender.gb200.yaml
kubectl -n <namespace> wait --for=condition=Ready pod/native-nccl-refit-sender --timeout=30m
kubectl -n <namespace> wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-component-type=worker --timeout=30m

# Put native_nccl_refit_bench.py at this shared-PVC path before running:
BENCH=/mnt/rl-workspace/bench/native_nccl_refit_bench.py
RUN=/mnt/rl-workspace/bench/native-nccl-30b
SENDER_IP=$(kubectl -n <namespace> get pod native-nccl-refit-sender \
  -o jsonpath='{.status.podIP}')
WORKER=$(kubectl -n <namespace> get pod \
  -l nvidia.com/dynamo-component-type=worker \
  -o jsonpath='{.items[0].metadata.name}')

# Terminal 1: trainer rank 0. The checkpoint must contain hf_weights or a flat
# HF-compatible state dict whose names match Qwen/Qwen3-30B-A3B.
kubectl -n <namespace> exec native-nccl-refit-sender -- \
  python3 "$BENCH" sender --master-address "$SENDER_IP" --master-port 29600 \
  --checkpoint /mnt/rl-workspace/<checkpoint>.pt \
  --manifest "$RUN.manifest.json" --trigger "$RUN.trigger" \
  --result "$RUN.sender.json" --warmup-cycles 1 --cycles 5 --preload-gpu

# Terminal 2: controller in the TP2 worker (localhost reaches DYN_SYSTEM_PORT).
kubectl -n <namespace> exec "$WORKER" -- \
  python3 "$BENCH" controller --master-address "$SENDER_IP" --master-port 29600 \
  --system-url http://127.0.0.1:9090 \
  --manifest "$RUN.manifest.json" --trigger "$RUN.trigger" \
  --result "$RUN.controller.json" --warmup-cycles 1 --cycles 5

# Seven differentiators consume only the live paths shown above.
```

## Canonical artifact contract

Every accepted artifact identifies the exact real checkpoint:

```json
{
  "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "checkpoint": "<snapshot hash or immutable checkpoint ID>",
  "checkpoint_bytes": 61064245248,
  "tensor_source": "safetensors",
  "bytes": 61064245248
}
```

Receiver artifacts use `tensor_source: "received_safetensors"`. EP/TP artifacts
put the actual filtered transfer in `bytes` while retaining
`checkpoint_bytes: 61064245248`. Partial manifests put the same identity fields
beside their `tensors` array. The egress analyzer takes this metadata separately
with `--artifact`. Qwen3-4B, missing checkpoint IDs, shape-only/random tensors,
and byte counts from any checkpoint other than the 61,064,245,248-byte checkpoint
are rejected.

## Live direct-vs-tree fan-out

Set `HF_SNAPSHOT` to the immutable local snapshot directory, not a model name or
download target. The trainer reads every safetensors shard and copies the real
weights to GPU; seeds republish only bytes they received from that trainer.
Use N=13 when resources permit. For a smaller allocation, choose
`N=min(13, available_receiver_GPUs)` and analyze with `FANOUT_N=adaptive`.

Run each role in its assigned GPU/RDMA pod with the benchmark file on the shared
PVC. Use a fresh `RESULT_DIR` for each trial:

```bash
BENCH=/mnt/rl-workspace/bench/fanout_bench.py
SNAP=/mnt/rl-workspace/kavink/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/<hash>
N=13

# Direct trial: one trainer and N receivers.
HF_SNAPSHOT="$SNAP" RESULT_DIR=/mnt/rl-workspace/fanout/direct \
  EXPECTED_RECEIVERS="$N" python3 "$BENCH" trainer
HF_SNAPSHOT="$SNAP" RESULT_DIR=/mnt/rl-workspace/fanout/direct \
  PARENT=trainer python3 "$BENCH" receiver <0..N-1>

# Tree trial example for N=13: four seeds, with 4/3/3/3 children.
HF_SNAPSHOT="$SNAP" RESULT_DIR=/mnt/rl-workspace/fanout/tree \
  EXPECTED_RECEIVERS="$N" python3 "$BENCH" trainer
HF_SNAPSHOT="$SNAP" RESULT_DIR=/mnt/rl-workspace/fanout/tree \
  EXPECTED_RECEIVERS=<children-for-this-seed> python3 "$BENCH" seed <0..3>
HF_SNAPSHOT="$SNAP" RESULT_DIR=/mnt/rl-workspace/fanout/tree \
  PARENT=seed:<seed-id> python3 "$BENCH" receiver <0..N-1>

python3 differentiator_suite.py fanout \
  --direct /mnt/rl-workspace/fanout/direct \
  --tree /mnt/rl-workspace/fanout/tree --workers 13
```

The role commands are intentionally pod-local; launch them through the
deployment's normal Kubernetes exec/job mechanism. This harness does not create
or mutate cluster resources.

## Scenarios

- **S1 transport baseline** — NCCL broadcast vs MX pull, single- and multi-rail.
- **S2 refit-phase breakdown** — register/wire/translate/load/e2e via a `mx`↔`nccl` backend swap on the same model + step.
- **S3 elastic-join latency** — scale in a fresh worker, time it to the current version (NCCL needs a communicator rebuild; MX pulls from the catalog).
- **S4 straggler isolation** — one slow worker; healthy workers should complete independently under MX vs stalling on a collective barrier.
- **S5 partial / EP byte-pruning** — per-worker bytes-on-wire at EP>1 should be ~1/EP.

## Prerequisites for the live scenarios

- A trainer **actively publishing** versions to the MX server.
- For the EP number: the **EP>1 inference DGD** (`../examples/ep8_rollout_dgd.example.yaml`, TP1×DP8×EP8, `--enable-expert-parallel`, `--moe-backend triton`).
- For the NCCL arm: an NCCL weight-transfer path deployed for the same model.

Reference numbers (GB200, preliminary): transport NCCL ~375 Gbps single-rail vs
MX ~316 / ~506–529 (4-rail) / ~1.4 Tbps (intra-node NVLink); 30B registration
11.9s→0.16s (arena); MDL load 2.15s→0.55s; end-to-end refit ~6s→~1.5s.

## First-party single-purpose harnesses (2026-07-08, GB200, real Qwen3-30B-A3B)

Each runs inside a vLLM+MX pod (or a pair of RDMA pods); results verified on GB200.

| Harness | What it measures | Verified result |
|---|---|---|
| `mdl_correctness.py <model>` | corrupt→warm-reload→generate on a live vLLM engine | 4B + 30B MoE byte-identical (18,432 expert writes) |
| `ep_live_test.py <EP>` | live EP engine: placement parity + refit correctness | EP=4 placement matches `compute_local_expert_ids`; byte-identical |
| `wire_bench.py {publisher\|receiver}` | cross-host NIXL transport (2 RDMA pods, diff nodes) | full 61 GB in 0.54s ≈ 900 Gbps; `EP_SIZE=8` → 10.33 GB in 0.17s |
| `elastic_bench.py {publisher\|receiver}` | decoupled/elastic/straggler timelines (N receivers) | each worker independent ~850–940 Gbps; 20s-late worker unaffected |
| `reg_bench.py {pertensor\|arena}` | buffer registration cost | per-tensor 0.90s → arena 0.016s (~56×) |
| `fp8_h1_test.py <fp8-model>` | fp8 warm-path (H1) diagnosis | fp8 `load_weights` not re-entrant; loaderless maps 84%; scales/DeltaNet open |

The 2-pod / N-pod RDMA manifests use the workers' network annotations
(`networking.gke.io/interfaces` for rdma-0..3, `infiniband` hostPath, `IPC_LOCK`,
`MX_RDMA_NIC_PIN=stripe`, GPUDirect UCX env) with pod anti-affinity to force
cross-host placement. Generate them per-run from a worker's spec.
