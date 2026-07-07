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
