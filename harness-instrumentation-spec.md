# Generation-telemetry instrumentation spec (work order)

**Goal:** expose enough telemetry to fully break down a SWE async-GRPO step into
generation / tool / idle time, and to diagnose GPU-utilization bottlenecks.

**CRITICAL — implement against THIS repo's current code, not the spec's wording:**
- This spec is *design intent*, not a patch. File paths / function names below are
  hints; **locate the real code in this checkout.**
- **Audit first.** This branch may already implement some items (e.g. explicit
  `idle/buffer_starvation` / `idle/refit_bubble`, per-trajectory SWE timing such as
  `openhands_run_time` / `total_model_call_time` / `total_command_exec_time`). For each
  item below, first check whether it already exists; **implement only the gaps.**
- **One layer per PR.** Create a branch, implement, open a PR, **do not merge**.
- Verify upstream API usage (vLLM `vllm.v1.metrics.reader`, nemo-gym agent) against the
  actually-installed versions. Follow repo conventions and add/adjust tests.

## Layer 1 — scrape vLLM per-request latency histograms (highest value)
- **Where:** the vLLM async generation worker's metrics sampler loop (the thread that
  calls `get_metrics_snapshot()` from `vllm.v1.metrics.reader`). Today it likely handles
  only `Gauge`/`Counter` and drops `Histogram`.
- **Add:** an `isinstance(m, Histogram)` branch capturing `(count, sum[, buckets])` for:
  `vllm:request_queue_time_seconds`, `request_prefill_time_seconds`,
  `request_decode_time_seconds`, `request_inference_time_seconds`,
  `time_to_first_token_seconds`, `time_per_output_token_seconds`,
  `e2e_request_latency_seconds` (if present), and the `vllm:prompt_tokens` counter.
- **Aggregate:** window-diff `Δsum/Δcount` between refits → mean; buckets → p50/p95/p99.
- **Acceptance:** per-step scalars for queue / prefill / decode / TTFT / TPOT latency.
  Enables splitting agent-perceived model-call time into queue vs decode vs transport.

## Layer 2 — engine utilization scalars (data already sampled, just log it)
- **Where:** the timeline aggregation that today renders per-worker `inflight_batch_sizes`
  to a plot.
- **Add scalars:** `engine_idle_ratio` / `engine_busy_ratio` (mean+max over workers, from
  the `inflight==0` fraction), `engine_idle_time_s` / `engine_busy_time_s` /
  `active_time_s`, and **`per_instance_concurrency = GBS / num_vllm_instances`** where
  `num_vllm_instances = generation_GPUs / (vllm_tp * vllm_pp)`.
- **Acceptance:** `per_instance_concurrency` is the headline GPU-utilization diagnostic
  (low concurrency ⇒ high engine idle). Prefer native W&B series over static image panels.

## Layer 3 — derived metrics + bug fixes
- Derive `transport_overhead = total_model_call_time(agent) − engine_e2e(layer 1)`.
- Log an **output-only** generated-token rate (masked / vLLM `generation_tokens` counter),
  distinct from the full-sequence `total_num_tokens` (which includes prompt + tool-result).
- **Bug:** `policy_and_reference_logprobs_tokens_per_sec_per_gpu` divides by a
  near-zero logprobs time when logprobs are skipped → astronomical value. Guard `time>eps`.
- **Bug:** `training_worker_idle_time_ratio` guard looks inverted
  (`0 if exposed_generation > 0.1 else …`) → reports 0 idle exactly when idle exists.

## Layer 4 — per-step percentiles (smallest, do first; kills the need to download tables)
- **Where:** the per-sample metric reducer for rollout / per-agent metrics (the helper
  that emits `/mean,/median,/min,/max,/stddev,/histogram`).
- **Add:** `p50/p90/p95/p99` to every per-agent metric (openhands_run_time,
  total_model_call_time, total_command_exec_time, turns_per_sample, gen_tokens_per_sample).
- **Acceptance:** per-step tail latencies queryable directly (no `full_result` table download).

## Layer 5 — SWE agent / nemo-gym (deepest)
- **Per-tool-call duration list** → single-tool-call p50/p95/p99 (today only the
  per-trajectory sum `total_command_exec_time` exists).
- **Per-turn timing** (model-call + tool-exec per turn) → per-turn e2e distribution.
- **Tool-call counts + types** (`num_tool_calls`, per-tool-name), with canonical
  name normalization (e.g. `execute_bash`/`bash`/`shell_command` → shell).
- **Per-call token usage** (some responses log `usage=None`).
- **Bug:** `generation_apptainer_spinup_time` is seeded with `-time.time()` + an
  unreliable sandbox timestamp → negative garbage; fix or drop.

## Suggested order
Layer 4 (tiny, safe) → Layer 2 (mostly logging existing data) → Layer 1 (the real
diagnostic win) → Layer 3 bug-fixes → Layer 5 (agent-side, largest).
