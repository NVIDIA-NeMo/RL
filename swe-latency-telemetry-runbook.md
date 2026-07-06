# SWE Latency Telemetry — Work Log & Run Guide

Status: **2026-06-08**, branch `harness-layer5`. Working tree (uncommitted) — see *Change set* below.

This documents the generation/agent **latency telemetry** added to the SWE GRPO harness
(prefill/decode + per-call/tool durations), the W&B workspace tooling, the launch recipes,
and how to run, monitor, and validate them.

---

## 1. What we added

Two latency families, plus workspace/recipe tooling.

### A. Agent-side per-call durations — **exact** percentiles ✅ validated
Pooled from the raw duration lists the SWE agent (OpenHands fork) writes per trajectory
(`model_call_durations`, `command_exec_durations`), surfaced by the rollout reducer.

- **W&B keys**
  - `train/swe_agents_train/tool_call_exec_durations_s/p{50,95,99}` — per-tool-exec time
  - `train/swe_agents_train/generation_call_durations_s/p{50,95,99}` — agent-measured per-LLM-call time (incl. HTTP/transport)
- **Validated** on run `5z95mmwe`: tool p99 ≈ **30 s**, model-call p99 ≈ **17 s**.
- **Why pooled-raw, not per-group percentiles:** in async GRPO, per-prompt-group `rollout_metrics`
  are aggregated by **mean** for scalars (`grpo.py` ~`rollout_metrics = {k: sum(v)/len(v) ...}`).
  Computing a percentile per group then letting grpo average them **dilutes the tail**. So the
  reducer stashes the raw values under `{base}/_values` (a list — survives aggregation as a
  list-of-lists) and `reduce_pooled_latency_percentiles()` pools across groups and reduces **once**.

### B. Engine-side prefill/decode latency — mean + bucketed percentiles ⚠️ live-unvalidated
From vLLM's Prometheus **histograms** (`vllm:request_prefill_time_seconds`,
`vllm:request_decode_time_seconds`) read via `get_metrics_snapshot()` in the metrics-logger thread.
This is the **only** source that captures the SWE agent's calls, because the agent generates over
vLLM's **OpenAI HTTP server** (`expose_http_server: true`), which **bypasses** the worker
`generate()` loops — so a per-request hook there sees nothing (we tried and removed it).

- **W&B keys**: `generation_metrics/latency/request_{prefill,decode}_time_seconds/{mean,p50,p95,p99}`
  - `/mean` exact (cumulative `sum`/`count` delta per step).
  - `/p50,/p95,/p99` via Prometheus-style `histogram_quantile` over per-step bucket deltas.
- **Probe-confirmed**: both histograms are present in the snapshot, and buckets are fine-grained
  into the tail (`…30, 40, 50, 60, 120, 240, …, 7680, +Inf`), so percentiles resolve the long tail
  (a 30 s p99 lands in the 30–40 s bucket, not clipped at `+Inf`).
- **Not yet validated live** — see *Status / known issues*.

### C. W&B workspace builder
`make_swe_wandb_workspace.py` builds a saved grouped view that collapses the ~150–238
auto-generated per-suffix panels (`mean/median/min/max/stddev/p50/p90/p95/p99/histogram` ×
~24 metrics) into ~20–44 grouped panels (one multi-line `LinePlot` per metric) plus
`Generation / vLLM` and `Performance / throughput` sections. Uses `auto_generate_panels=False`.

> Lesson: per-step `wandb.Histogram` media **don't render** as panels here (async logs them as a
> list-of-N per step → no panel), and `wandb-workspaces` 0.4.1 has **no histogram panel class** —
> hence the grouped *scalar/percentile* approach instead of histograms.

### D. Recipes
- `run_grpo_qwen3_30b_thinking_swe2.sh` — full async-GRPO training (16 nodes; 8 policy DP2 + 8 gen),
  3-step smoke (`grpo.max_num_steps=3`, checkpointing off). Now supports `EXCLUDE_NODES`.
- `run_collect_trajectories_qwen3_30b_swe.sh` — **NEW** budget-minimized collect-only:
  `++env.nemo_gym.is_trajectory_collection=true` (no training), **1 vLLM replica**
  (`NUM_GEN_NODES=1`, `VLLM_TP=8`), **9 nodes** (8 policy DP2 + 1 gen), 2 h cap. Supports
  `EXCLUDE_NODES`. Collect mode logs `full_result` to jsonl and now logs the same scalar
  latency metrics to W&B as the train loop.

---

## 2. Change set (files)

| File | Change |
|------|--------|
| `nemo_rl/experience/rollouts.py` | `reduce_pooled_latency_percentiles()` + `_values` stash in `run_async_nemo_gym_rollout` |
| `nemo_rl/algorithms/utils.py` | `log_request_latency_metrics_to_wandb()` + `_percentile_from_bucket_deltas()` |
| `nemo_rl/models/generation/vllm/vllm_worker_async.py` | metrics-logger thread reads the prefill/decode histograms (first/last snapshot per step) + **temporary** `[latency-probe]` |
| `nemo_rl/models/generation/vllm/vllm_generation.py` | gathers `request_latency_hist` per dp leader |
| `nemo_rl/algorithms/grpo.py` | calls `log_request_latency_metrics_to_wandb` at both (sync+async) log sites |
| `make_swe_wandb_workspace.py` | grouped-view builder (+ Generation/Performance sections) |
| `harness-instrumentation-metrics.md` | metrics reference (Layer 1a engine latency, Layer 5 per-call durations) |
| `run_grpo_qwen3_30b_thinking_swe2.sh` | `EXCLUDE_NODES` support |
| `run_collect_trajectories_qwen3_30b_swe.sh` | **new** collect-only recipe |
| `tests/unit/algorithms/test_utils.py` | `TestLogRequestLatencyMetrics` (mean + percentile, both bucket layouts) |
| `tests/unit/experience/test_rollouts.py` | `TestReducePooledLatencyPercentiles` (pooled p50/p95/p99) |

---

## 3. How to run

### Prereqs (in your shell)
```bash
# Tokens are inherited from your shell; the recipe fail-fasts if unset.
: "${HF_TOKEN:?}"; : "${WANDB_API_KEY:?}"
# The recipe sets a persistent Lustre UV cache itself; no action needed.
```
All paths (HF_HOME, UV_CACHE_DIR, caches, logs, checkpoints) are already repointed to
joyang's writable dirs inside the recipes.

### Full e2e training run
```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/RL
# Exclude known-bad nodes (see Status). Comma-separated.
EXCLUDE_NODES=pool0-01684,pool0-01047 bash run_grpo_qwen3_30b_thinking_swe2.sh
```
- 16 nodes, async GRPO, 3-step smoke. ~25–30 min cold start, ~55 min total when healthy.
- Logs **both** latency families to W&B. Job id → `latest_thinking_swe2_job_id.txt`.

### Cheap collect-only run (availability check / trajectory dump + W&B metric validation)
```bash
EXCLUDE_NODES=pool0-01684,pool0-01047 bash run_collect_trajectories_qwen3_30b_swe.sh
```
- 9 nodes, 1 vLLM replica, no training. Writes trajectories to `trajectory_collection.jsonl`.
- The smoke recipe uses `grpo.val_batch_size=${PPS}` and `grpo.max_num_steps=3`, so it logs up
  to 3 small collect batches rather than waiting for the full validation set.
- Logs the W&B latency keys listed above. Job id → `latest_collect_traj_job_id.txt`.

### Useful overrides
```bash
NUM_NODES=9 NUM_GEN_NODES=1 MODEL_PATH=/path/to/ckpt EXP_SUFFIX=my-exp bash run_collect_trajectories_qwen3_30b_swe.sh
```

---

## 4. Monitor & validate

```bash
JOB=$(cat latest_thinking_swe2_job_id.txt)
DL=logs/slurm/$JOB-logs/ray-driver.log

squeue -j "$JOB" -o "%.12i %.8T %.10M %.6D %R"          # state
sacct  -j "$JOB" --format=JobID,State,ExitCode,Elapsed  # final state
grep -rhoE 'https://wandb\.ai/[^ )]+/runs/[A-Za-z0-9]+' "$DL" | sort -u | head   # W&B URL
grep -aoE 'Step [0-9]+' "$DL" | sort -u                 # training progress
# early-failure sniff (the recurring engine deaths):
grep -acE 'EngineDeadError|unspecified launch failure|Engine core initialization failed' "$DL"
```

### Validate the keys in W&B (after a run logs ≥1 step)
```bash
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/uv_cache
uv run --no-sync python - <<'PY'
import wandb
run = wandb.Api(timeout=60).run("nvidia/swe-benchmark-harness/<RUN_ID>")
s = run.summary
for base in ("generation_metrics/latency/request_prefill_time_seconds",
             "generation_metrics/latency/request_decode_time_seconds"):
    print(base, {k: s.get(f"{base}/{k}") for k in ("mean","p50","p95","p99")})
for base in ("train/swe_agents_train/tool_call_exec_durations_s",
             "train/swe_agents_train/generation_call_durations_s"):
    print(base, {f"p{p}": s.get(f"{base}/p{p}") for p in (50,95,99)})
PY
```

### Build the grouped W&B view (declutters ~238 → ~44 panels)
Run against a RUN_ID that has the metrics (login node lacks py3.13 / writable default cache):
```bash
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/uv_cache
SYSPY=$(command -v python3.11 || command -v python3.10 || command -v python3)
uv run --no-project --python "$SYSPY" --with "wandb<0.19" --with wandb-workspaces \
    python make_swe_wandb_workspace.py <RUN_ID>
```
(`wandb<0.19` is required — newer wandb drops the `wandb_gql` module wandb-workspaces imports.)
Then open the printed `?nw=...` view and select your run. Latest built view: `nw=wxvmejdhbya`.

---

## 5. Status / known issues

- **Validated:** agent-side exact percentiles (tool/model-call) on run `5z95mmwe`.
- **Pending live validation:** engine prefill/decode keys. Code is unit + functionally tested
  (both bucket layouts), and prior probe output confirmed availability + fine buckets. The
  collect-only smoke now logs these keys without waiting for a full train step.
- **Cluster engine deaths (the blocker):** 4 consecutive training runs died with vLLM
  `EngineDeadError` / `cudaErrorLaunchFailure`, each on a **different** node
  (`pool0-01684` ×2, `pool0-01047`, a `10.65.1.x` set). The crashes are in the **EngineCore
  subprocess** with no logged CUDA/OOM/NCCL error (hard process kill — GPU Xid / OS OOM-kill),
  **independent of this telemetry code** (it runs in the worker process, reads a snapshot
  read-only, and its exceptions are caught). Mitigate with `EXCLUDE_NODES`; raise the node
  instability with cluster ops.
- **TEMP probe removed:** `[latency-probe]` one-shot print in `vllm_worker_async.py` served its
  purpose and was removed; validate through W&B scalar keys instead.
- **Tip — fresh `EXP_SUFFIX` for smoke tests** so a prior checkpoint isn't auto-resumed with a
  mismatched step schedule (Megatron `OptimizerParamScheduler` abort). Checkpointing is off in
  these recipes, so not an issue here, but keep in mind if you enable it.

---

## 6. Reference

- **W&B project:** `nvidia/swe-benchmark-harness`
- **Validated run:** `5z95mmwe` (tool/model-call). Grouped views built this session:
  `i2x4u7ku08c`, `w5uxf1aux3m`, `wxvmejdhbya` (latest).
- **Metrics reference:** `harness-instrumentation-metrics.md` (per-layer key catalog).
- **Recipe science (keep byte-for-byte):** container = nliang's vLLM-0.17.1 image; model = bihu's
  SWE1 `step_230_hf`; `env.nemo_gym.skip_venv_if_present: true`; parallelism TP4·EP8·CP4·PP2.
  See the `reproduce-recipe` skill for the enroot-image-vs-vLLM-pin check.
