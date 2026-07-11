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
| `run_differentiator_bench.sh` | Push-button orchestrator for all scenarios. Preflight-checks the cluster and **skips gracefully** when a prereq (publishing trainer, NCCL path, EP>1 deploy) is missing, so it's safe to run incrementally. |
| `mx_vs_nccl_refit_bench.py` | Deployment-agnostic driver: runs N refit cycles per backend (`mx` vs `nccl`) over the native weight-transfer API, times register/wire/translate/load/e2e, and `--compare`s the JSON. |
| `ep_gt1_byte_pruning.py` | Synthetic-but-real-planner proof that EP>1 pulls only 1/EP of expert bytes (EP=8 → 8.0×). Runnable now, no cluster. |
| `mdl_partial_smoke.py` | Runtime smoke for MDL incremental/partial-update (cold/warm/subset/incremental, byte-identical). No transport. |
| `configs/nixl_ep4_tp1_gb200_rc_debug.yaml` | Sanitized EP4→TP1 GB200 topology and UCX/NIXL config, including the TCP-fallback baseline and validated `^tcp` fix (12/12 pulls on RDMA, ~7× steady-state refit speedup). |
| `preflight_ep8_tp2.sh` | Fail-fast Kubernetes gate for the two 4-GPU EP8 trainer pods and 2-GPU TP2 rollout: GPU count, `rdma0..3`, `^tcp`, matching package versions, and native `mx` backend registration. |

## Run

```bash
# synthetic EP proof (no cluster):
MX_PY=<repo>/…/modelexpress_client/python python3 ep_gt1_byte_pruning.py

# full differentiator matrix (once GPUs/deploy are up):
NS=<namespace> FRONTEND=http://<dgd-frontend>:<port> \
  MX_PY=<…/modelexpress_client/python> \
  bash run_differentiator_bench.sh
```

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
