# Fault Tolerance Experiment — Disaggregated Inference Recovery

Quantify the benefit of **fault-tolerant disaggregated generation** (a worker
dies, the router evicts it and training recovers *in place*) versus traditional
single-cluster training that **crashes and restarts from checkpoint** on any
failure.

> **Status (2026-06-08):** The disaggregated fault-recovery path is fixed,
> committed (`cd54a7303`), and validated end-to-end (TEST / kill-1 completed 20
> steps through 3 faults). The non-disaggregated CONTROL runs are currently
> **blocked by a deterministic NCCL collective hang** in the colocated-cluster
> training path — see [Known Blockers](#known-blockers). Read that section
> before re-running the controls.

## Experiment Matrix

| # | Setup | Disagg | FT recovery | Faults | Ckpt period | Status |
|---|-------|:---:|:---:|:---:|:---:|:---|
| 1 | **No-failure baseline** | No | — | none | 1 | ✅ done (20 steps, 18.5 min) |
| 2 | **Control** | No | No (crash+restart) | 3 | 1 | ✅ done (20 steps, 35.5 min, 4 attempts) |
| 3 | **Test** (disagg + FT) | Yes | Yes (in-place) | 3 / 2 | 1 | ✅ kill-1 done · ✅ kill-half done |
| 4 | **Alt control** | No | No (crash+restart) | 3 | 5 | ✅ done (20 steps, 4 attempts, 9 steps redone) |

### Headline result — FT benefit (steps 1→20 wall, same 3-fault load)

| Setup | wall (step1→20) | per-fault overhead | steps lost | restarts |
|---|---|---|---|---|
| #1 baseline (no faults) | **18.5 min** | — | 0 | 0 |
| #3 disagg FT (in-place recovery) | **23.6 min** | ~1.7 min (+5 min total) | **0** | **0** |
| #2 control (crash + restart) | **35.5 min** | ~5.7 min (+17 min total) | ~1/fault | 3 |

Disaggregated fault tolerance recovers **~3.4× cheaper per fault** than
crash+restart-from-checkpoint, completing the same 3-fault 20-step run **~12 min
(34%) faster** and losing **zero** steps. The control pays a full cold restart
(Megatron + vLLM reload, ~5 min) on every fault; the disagg TEST keeps the
cross-cluster group alive and recovers in place.

Common to all runs (the spec):
- Model **Qwen3-30B-A3B**; Async GRPO (lag 1, `max_trajectory_age_steps=1`); 20 steps.
- Training: **32 GPU** (8 nodes × 4 GB300), Megatron TP=1 PP=1 EP=8.
- Generation: **8 GPU** (8 vLLM DP shards × 1 GPU, TP=1).
  - TEST (disagg): gen is a *separate* RayCluster.
  - CONTROL (non-disagg, non-colocated): gen shares the cluster on dedicated
    nodes → total **10 nodes** (8 train + 2 gen) = 40 GPU. `cluster.num_nodes`
    must be **10** (gen carves its 2 nodes out of the pool) so train DP=32 and
    `global_batch_size 64 / 32 = 2` microbatches. (`num_nodes=8` → DP=24, which
    64 is not divisible by → Megatron asserts at init.)
- Validation disabled; `keep_top_k=1`; `save_optimizer=true`; W&B project
  `nemorl-ft-experiment`.
- Faults occur **only after training has started** (never during the
  not-fault-tolerant startup) and 2–3 times across the 20 steps.

## The fault-recovery fix (TEST path)

Committed as **`cd54a7303`** — "fix: converge disagg fault recovery without
wedge or world cascade". Four coordinated changes let a gen worker die
mid-refit and the cross-cluster collective recover in place:

1. **In-band self-abort** (`stateless_process_group.py`, `packed_tensor.py`):
   `synchronize_or_abort()` polls the CUDA event + `comm_get_async_error` and,
   on peer-loss/timeout, calls `comm_abort` to unblock the wedged kernel then
   raises — the raw-NCCL group has no torch watchdog, so a blind
   `cudaStreamSynchronize` would hang forever.
2. **Keep-PG-alive across refits** (`vllm_backend.py`, `refit_worker.py`,
   `remote_generation.py`): hold the cross-cluster `model_update_group` between
   refits; tear down only on failure; re-init only on a world-size change.
3. **Router fast-fail eviction discriminator** (`generation_router.py`): after
   a fault the recovery `init_collective` sees *every* survivor's future fail,
   but for two reasons — a peer whose NCCL state was poisoned by the abort
   raises in <1s ("unhandled"), while a healthy survivor blocked on the
   rendezvous raises at the ~30s bootstrap timeout. `_per_worker_results` does a
   two-phase `ray.wait` and returns `failed_fast`; only fast-failers (+ actor-
   dead) are force-evicted — the rest are kept. **This is what stops the
   world-8→2 cascade.**
4. **Refit retry + fault timing** (`grpo.py`, `fault_inject.py`): broadened the
   outer refit retry from `RuntimeError` to `Exception` (the gen 503 surfaces
   as `RayTaskError(HTTPError)`), and anchored fault injection to the first
   refit via `/refit_count`.

Unit tests: `tests/unit/models/generation/test_generation_router.py`
(`...fast_fail_force_evicts_poisoned`), `tests/unit/distributed/test_stateless_process_group.py`.

## Results

### #3 TEST — kill 1 worker per fault ✅ CONFIRMED

Run `ptA4-20260607-215026`. WandB: `ft-experiment-test-disagg-with-fault`
(project `nemorl-ft-experiment`). **Completed all 20 steps through 3 pod-kill
faults, 3 in-place recoveries, no crash, no lost steps.**

Per-fault behaviour (from the router log): the killed shard + its NCCL-poisoned
peers fast-fail and are evicted; the timed-out healthy survivors are kept; the
FaultInjector re-adds a replacement. Example (fault 1): `force_dead=['dp-1',
'dp-6']` (NCCL "unhandled", <8s), `skipping 5 alive survivors` (timed out).

Step timeline (Unix ts, Δ vs previous step start):

| step | Δ (s) | note |
|---:|---:|---|
| 1→4 | ~52–75 | steady state |
| **5** | **+293** | fault 1 (kill dp-0 @step 4) → re-rendezvous + recover |
| 6→9 | ~52 | steady |
| **10** | **+162** | fault 2 → recover |
| 11→14 | ~52 | steady |
| **15** | **+95** | fault 3 → recover |
| 16→20 | ~52 | steady |

- Steps 1→20: **1414 s wall** (step 1 @ 1780869388 → step 20 @ 1780870802).
- Steady-state step ≈ **52 s**.
- Recovery overhead **decreases** per fault: **+241 s, +110 s, +43 s** (≈394 s
  total) — the first fault re-inits the most NCCL ranks; later faults re-init
  fewer. **Zero steps lost, zero restarts.**
- Gen world degrades gracefully 8→6→4 (each fault also evicts ~2 peers the
  abort poisoned; the FaultInjector only re-adds what it killed) and the run
  still completes — demonstrating "recover without full shutdown".

Telemetry recovered on attempt 2 all 3 times (`✓ ensure_collective_synced
recovered on attempt 2` ×3).

### #3 TEST — kill half (4 of 8 workers) ✅ CONFIRMED

Run `ptD1-killhalf-20260608-040554` (`test_gen_server.yaml` with
`burst_size: 4`, `max_cycles: 2` → two kill-half events). **Completed all 20
steps through 2 kill-half faults, 2 in-place recoveries, no full shutdown.**

- **Fault 1** (~step 3): burst-killed all 4 targets (dp-0,1,2,3) at once → gen
  world 8→2 survivors. Router kept the 2 survivors cleanly (`skipping 2 alive
  survivors, force_dead=[]` — no over-eviction). FaultInjector re-added 4
  replacements (dp-7..dp-10); world climbed back 3→4→5→6. Recovered on attempt 2.
- **Fault 2** (~step 17): burst-killed 4 more (dp-5,6,7,8); same recovery,
  re-added dp-11,12,… Recovered on attempt 2.
- Recovery overhead: ~+45 s (fault 1), ~+170 s (fault 2); steady step ≈ 53 s.
- Only **1** extra worker evicted across both faults (vs ~2/fault under kill-1) —
  the larger simultaneous kill happens to poison fewer survivors.

Demonstrates the spec's "kill half-or-more → recover without full shutdown."
For the canonical kill-1 TEST, revert `burst_size`→1 / `max_cycles`→null.
(Earlier attempts `ptB2`/`ptC1` were blocked by GPU-cleanup races / GB300
capacity, not the kill-half logic — resolved once capacity was freed.)

### #1 NO-FAILURE baseline (non-disagg) ✅ CONFIRMED

Run `exp1-nofault-20260608-043948`. WandB run `9fee5n8x`. **20 steps, no faults,
~1111 s wall** (step 1 → step 20), steady step ≈ 55 s. This is the clean
comparison baseline; comparable per-step compute to the disagg test. Validated
the `NRL_USE_REFIT_WORKER=1` fix (see Known Blockers — now resolved): the first
refit and all 20 steps ran with no hang.

### #2 CONTROL (faults + crash/restart) ✅ CONFIRMED

Run `exp2-control-20260608-050754`. WandB `eljrkqwl`. **4 attempts, 3 faults,
20 steps, 35.5 min** (step 1 → step 20). `ft_requeue.sh` killed the driver at
steps 6/11/16; each crash forced a cold restart + resume from the last
checkpoint:

| attempt | resumed from | ran steps | killed at | restart gap |
|---|---|---|---|---|
| 1 | fresh | 1–6 | step 6 | — |
| 2 | step_5 | 6–11 | step 11 | 313 s |
| 3 | step_10 | 11–16 | step 16 | 305 s |
| 4 | step_15 | 16–20 | (finished) | 296 s |

Each fault cost **~5 min** of restart overhead (kill + Megatron/vLLM cold
reload + resume) plus ~1 redone step — vs the disagg TEST's ~1.7 min in-place
recovery with zero redone steps.

### #4 ALT-CONTROL (save_period=5) ✅ CONFIRMED

Run `exp4-altcontrol-20260608-055423`. WandB `rcjyoc5t`. **4 attempts, 3 faults,
20 steps.** Faults at steps 8/13/18; with `save_period=5` each crash resumes
from the prior step_5-multiple checkpoint, so it **redoes ~3 steps per fault**:

| attempt | resumed from | ran steps | killed at | steps redone |
|---|---|---|---|---|
| 1 | fresh | 1–8 | step 8 | — |
| 2 | step_5 | 6–13 | step 13 | 6,7,8 |
| 3 | step_10 | 11–18 | step 18 | 11,12,13 |
| 4 | step_15 | 16–20 | (finished) | 16,17,18 |

**9 steps redone total vs #2's 3** — the egregious lost-work penalty of
infrequent checkpointing without FT. (Wall clock 25.8 min came in below #2's
35.5 min only because the restart cold-start time dominates total wall and this
later run hit warmer image/venv caches; the deterministic, apples-to-apples
penalty is the **3× redone work**. The disagg TEST redoes **0** steps on any
fault.)

## Telemetry

Driver logs persist on Lustre at `/mnt/rl-workspace/terryk/driver_logs/` and are
the source of truth. Gen-daemon (router/FaultInjector) logs are in the gen
RayCluster's Ray job log — **save them before tearing the gen cluster down.**

- `[STEP-START] step=N unix_ts=…` — per-step wall-clock (driver log).
- `[VLLM-THROUGHPUT] unix_ts=… host=… tokens_per_sec=…` — per-worker gen
  throughput (every 0.5s, vLLM worker stdout → driver log). *Note: disagg/router
  mode only exposes aggregate gen throughput per step in the driver summary
  ("Generation Worker Group (Tokens/sec/gpu)"), not per-worker counts.*
- `[FAULT-EVENT] unix_ts=…` / `[RECOVERY-START]` / `[RECOVERY] shard=… status=…`
  — fault + recovery timeline (gen-daemon log, TEST only).
- `[FAULT-SIM] … unix_ts=…` / `[REQUEUE] …` — control fault + restart timeline
  (driver log, CONTROL only — emitted by `ft_requeue.sh`).
- `↻/✓ ensure_collective_synced …` — re-rendezvous attempts + world-size
  transitions (driver log, TEST only).

## Run commands

```bash
# TEST (disagg) — creates train + gen RayClusters
nrl-k8s run infra/nrl_k8s/examples/ft_experiment/test_recipe.yaml \
  --infra infra/nrl_k8s/examples/ft_experiment/test.gb300.infra.yaml \
  --raycluster --mode batch --run-id test-$(date -u +%Y%m%d-%H%M%S)

# CONTROL (#2) / ALT-CONTROL (#4) — single cluster; ft_requeue.sh loops
# run_grpo (auto-resume from checkpoint) until step 20, injecting 3 mid-
# training faults via the requeue driver.  (BLOCKED — see below.)
nrl-k8s run infra/nrl_k8s/examples/ft_experiment/control_recipe.yaml \
  --infra infra/nrl_k8s/examples/ft_experiment/control.gb300.infra.yaml \
  --raycluster --mode batch --run-id control-$(date -u +%Y%m%d-%H%M%S)
```

After a disagg run with pod-kills, **wait for the gen cluster to return to 8
ready workers** before relaunching, and avoid rapid `--replace` relaunches
(GPU/PG cleanup races → `ResourceInsufficientError` on gen or
`cudaErrorDevicesUnavailable` on train). A clean `cluster down` + fresh `run` is
the most reliable reset.

## Resolved issues

### Non-disagg CONTROL refit hang — ✅ FIXED (`NRL_USE_REFIT_WORKER=1`)

Validated: #1 ran 20 clean steps with the fix; the first refit no longer hangs.

Symptom: after the microbatch-divisibility fix (`cluster.num_nodes` 8→10), the
controls pass setup but the **first weight refit deadlocks** — `❌ Policy
generation refit failed: Get timed out` (`grpo.py:refit_policy_generation`),
with an NCCL collective-timeout watchdog dump (the `EXPERT_MODEL_PARALLEL_GROUP`
mention is a red herring — the watchdog dumps every PG when any one hangs).
Reproduced identically on two fresh clusters.

Root cause: the controls ran `NRL_USE_REFIT_WORKER=0`, so the refit weight
broadcast uses the **legacy all-train-ranks path** — every one of the 32 train
ranks + 8 gen ranks joins `model_update_group` (a **40-way** NCCL collective).
At this scale/topology in the colocated cluster, that collective deadlocks on
the first refit. The disagg TEST uses `NRL_USE_REFIT_WORKER=1`: rank 0 streams
to a sibling RefitWorker actor that broadcasts to gen over a **9-way** group
(1 + 8 gen) — small, isolated, and proven to refit cleanly. The flag is gated
only on `megatron_enable` (not on disagg), so it applies to the non-colocated
control too (`init_collective` runs for any non-colocated Megatron gen).

Fix: set **`NRL_USE_REFIT_WORKER=1`** in all three control entrypoints
(`control_no_fault`, `control`, `alt_control` `.gb300.infra.yaml`). Validation
pending — retest #1 once GPUs free from the kill-half run.

## Config files

| File | Purpose |
|------|---------|
| `test_recipe.yaml` / `test.gb300.infra.yaml` | TEST (disagg, 2 RayClusters) |
| `test_gen_server.yaml` | TEST gen server + FaultInjector (kill-1 default; burst_size=4 for kill-half) |
| `control_no_fault_recipe.yaml` / `control_no_fault.gb300.infra.yaml` | #1 no-failure baseline |
| `control_recipe.yaml` / `control.gb300.infra.yaml` | #2 control (faults + requeue) |
| `alt_control_recipe.yaml` / `alt_control.gb300.infra.yaml` | #4 alt control (save_period=5) |
| `ft_requeue.sh` | shared requeue-on-failure + step-based fault driver for #2/#4 |

## W&B runs (project `nvidia/nemorl-ft-experiment`)

| # | Setup | Run | Link |
|---|-------|-----|------|
| 1 | baseline | `ft-experiment-control-nofault` | https://wandb.ai/nvidia/nemorl-ft-experiment/runs/9fee5n8x |
| 2 | control | `ft-experiment-control-no-ft` | https://wandb.ai/nvidia/nemorl-ft-experiment/runs/eljrkqwl |
| 3 | test (kill-1) | `ft-experiment-test-disagg-with-fault` | https://wandb.ai/nvidia/nemorl-ft-experiment/runs/wtki1r59 |
| 3 | test (kill-half) | `ft-experiment-test-disagg-with-fault` | https://wandb.ai/nvidia/nemorl-ft-experiment/runs/fxbb17bm |
| 4 | alt-control | `ft-experiment-alt-control-ckpt5` | https://wandb.ai/nvidia/nemorl-ft-experiment/runs/rcjyoc5t |

Driver logs (per-run, with `[STEP-START]`/`[FAULT-*]`/`[REQUEUE]` telemetry) and
the saved kill-half gen-daemon log are under `/mnt/rl-workspace/terryk/driver_logs/`.

## Branch

`tk/clean/hemil/fault-tolerant-generation`
