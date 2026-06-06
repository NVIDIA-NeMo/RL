# Harness Instrumentation Metrics Reference

All metrics appear in the W&B project **swe-benchmark-harness**.

---

## Layer 4 — Per-Sample Percentiles

**Source:** `nemo_rl/experience/rollouts.py` · `_calculate_single_metric()`

Added to **every** per-sample and per-agent scalar metric (reward, token counts, latencies, etc.). The existing `mean/median/min/max/stddev/histogram` suffixes are unchanged; four new percentile suffixes are added.

**W&B key pattern:** `{metric_prefix}/{metric_name}/p{N}`

Where `{metric_prefix}` is e.g. `swe_agents_train` or `swe_agents_val`, and `{metric_name}` is any per-sample scalar (e.g. `total_model_call_time`, `total_reward`, `num_tokens`).

| Suffix | Meaning |
|--------|---------|
| `/p50` | 50th percentile (median) across samples in the step |
| `/p90` | 90th percentile |
| `/p95` | 95th percentile |
| `/p99` | 99th percentile |

These complement `/mean` and `/max` to reveal tail behaviour without downloading the full rollout table. All four collapse to the single sample value when the batch contains exactly one trajectory.

---

## Layer 2 — vLLM Engine Utilization

**Source:** `nemo_rl/algorithms/utils.py` · `print_performance_metrics()`

**W&B key prefix:** `performance/`

Computed from the `inflight_batch_sizes` time-series (sample count in each vLLM engine's queue, polled every `vllm_metrics_logger_interval` seconds). Only emitted in async (non-colocated) mode when the metrics logger is enabled.

| W&B key | Meaning |
|---------|---------|
| `performance/engine_idle_ratio` | Mean fraction of polling intervals across all vLLM workers where the engine had zero in-flight requests. Values near 1 = generation GPUs are underutilised. |
| `performance/engine_idle_ratio_max` | Worst-case idle ratio across workers — highlights a single overloaded or underloaded instance. |
| `performance/engine_busy_ratio` | `1 - engine_idle_ratio`. Fraction of time engines were actively serving requests. |
| `performance/engine_idle_time_s` | Mean idle seconds per engine over the step. |
| `performance/engine_busy_time_s` | Mean busy seconds per engine over the step. |
| `performance/per_instance_concurrency` | `GBS / (generation_GPUs / (TP × PP))` — average number of concurrent trajectories queued per vLLM instance. Low values indicate under-subscription; high values indicate queuing pressure. |

---

## Layer 1 — vLLM Request Latency Scalars and Timelines

**Source:** `nemo_rl/models/generation/vllm/vllm_worker_async.py` · `VllmAsyncGenerationWorker`  
**Source:** `nemo_rl/algorithms/utils.py` · `log_generation_metrics_to_wandb()`

### 1a. Per-step latency scalars

**W&B key pattern:** `generation_metrics/latency/{histogram_name}/{stat}`

One row per histogram metric × stat combination. `{stat}` is one of `mean`, `p50`, `p95`, `p99` (computed from per-step delta counts/sums/buckets; `count` is omitted from W&B to reduce noise).

| Histogram name | Meaning |
|----------------|---------|
| `vllm:e2e_request_latency_seconds` | Wall-clock time from request submission to last output token returned |
| `vllm:time_to_first_token_seconds` | Prefill latency — time until the first token is streamed back |
| `vllm:time_per_output_token_seconds` | Mean per-token decode latency (TPOT) |
| `vllm:request_queue_time_seconds` | Time a request spent waiting in the vLLM scheduler queue before any GPU work |
| `vllm:request_prefill_time_seconds` | GPU time spent on the prefill pass |
| `vllm:request_decode_time_seconds` | GPU time spent on all decode steps |
| `vllm:request_inference_time_seconds` | Total GPU time (prefill + decode) |

### 1b. Per-worker timeline plots

**W&B key pattern:** `generation_metrics/{name}` (custom timeline chart, one line per worker)

| Name | Meaning |
|------|---------|
| `generation_metrics/inflight_batch_sizes` | Number of in-flight requests in each vLLM worker's queue over time. 0 = engine idle. |
| `generation_metrics/num_pending_samples` | Number of samples still waiting to be scheduled (backpressure indicator). |
| `generation_metrics/kv_cache_usage_perc` | KV-cache occupancy percentage per worker over time. Near 100% = memory pressure / potential OOM. |
| `generation_metrics/generation_tokens` | Cumulative output tokens generated per worker (monotonically increasing counter). |

---

## Layer 3 — Pipeline Efficiency

**Source:** `nemo_rl/algorithms/utils.py` · `print_performance_metrics()`

**W&B key prefix:** `performance/`

These metrics diagnose the balance between training and generation in async GRPO. All are per-step.

| W&B key | Meaning |
|---------|---------|
| `performance/training_worker_idle_time_ratio` | Fraction of the total step wall-time during which training workers were stalled waiting for generation to finish. 0 = training is always the bottleneck; approaching 1 = generation is far too slow. Only emitted in async mode; set to 0 when `exposed_generation_time ≤ 0.1s` (generation is negligibly small). |
| `performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu` | Token throughput (total sequence tokens / logprobs time / training GPUs) during the joint policy + reference logprob forward pass. Guards against near-zero logprob time with a 1ms floor to avoid division by zero. |
| `performance/output_only_tokens_per_sec_per_gpu` | Output-only token throughput measured from the vLLM `generation_tokens` counter (excludes prompt/tool-result tokens). More faithful to "new tokens generated" than the logprob-based rate. `None`/absent when the counter isn't available. |
| `performance/transport_overhead_s` | `agent_model_call_time − engine_e2e_latency`. The extra latency added by HTTP serialisation, network round-trips, and the OpenHands tool-call loop compared to the raw engine time. Only emitted when both Layer 1 (engine e2e mean) and rollout data (agent model call mean) are available. |

---

## Layer 5 — Per-Trajectory Turn, Token, and Tool Statistics

**Source:** `3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/app.py` · `SWEBenchWrapper._extract_completion_metrics()`

Written into the per-instance `metrics.json` file and surfaced through W&B via the rollout reducer (Layer 4 percentiles apply). Derived by scanning all OpenHands `llm_completions/*.json` files after each trajectory completes.

### Static fields (always present when completions exist)

| W&B key (after reducer) | Meaning |
|-------------------------|---------|
| `swe_agents_train/num_turns/mean` (+ `p50/p90/p95/p99/max`) | Number of LLM turns (= completion files) per trajectory. A turn is one model call / assistant response. |
| `swe_agents_train/num_tool_calls/mean` (+ percentiles) | Total tool calls executed per trajectory across all turns. |
| `swe_agents_train/mean_turn_duration_s/mean` (+ percentiles) | Mean inter-completion interval: `(last_timestamp − first_timestamp) / (turns − 1)`. Approximates average time per turn including tool execution. Absent for single-turn trajectories. |

### Per-call duration histograms

W&B histogram showing the distribution of individual call latencies, pooled across **all trajectories** in the batch. Logged as a single chart rather than scattered scalar percentiles, which keeps the W&B workspace uncluttered.

**Source:** `_extract_duration_distributions()` reads raw lists from `NEMO_GYM_METRICS_FPATH`; the rollout reducer pools them and calls `wandb.Histogram`.

| W&B key | Meaning |
|---------|---------|
| `swe_agents_train/generation_call_durations_s` | Distribution of per-LLM-call latencies (seconds) — timed by `nemo_gym_client._update_model_call_time`. A long tail here → model generation is the bottleneck. |
| `swe_agents_train/tool_call_exec_durations_s` | Distribution of per-Action execution times (seconds) — timed by `runtime.base.on_event`. A long tail here → tool/environment execution is the bottleneck. |

### Token fields (present only when `response.usage` is populated)

| W&B key | Meaning |
|---------|---------|
| `swe_agents_train/total_prompt_tokens/mean` (+ percentiles) | Sum of `prompt_tokens` across all turns in the trajectory. Grows each turn as the context window accumulates prior messages. |
| `swe_agents_train/total_completion_tokens/mean` (+ percentiles) | Sum of `completion_tokens` across all turns. Measures total output tokens generated to resolve the issue. |

### Dynamic per-tool-name counts

**W&B key pattern:** `swe_agents_train/tool_calls_{name}/mean` (+ percentiles)

One metric per distinct tool name seen in the run (e.g. `tool_calls_execute_bash`, `tool_calls_str_replace_based_edit_tool`, `tool_calls_read_file`). Counts how many times each tool was invoked per trajectory.

### Spinup time fixes (existing fields, corrected)

| W&B key | Fix applied |
|---------|-------------|
| `swe_agents_train/generation_apptainer_spinup_time` | Previously could be negative when the container's Unix timestamp file was stale or clocks skewed. Now clamped to `None` (omitted from W&B) when the computed value is negative. |
| `swe_agents_train/final_eval_apptainer_spinup_time` | Same fix applied symmetrically. |

---

## Notes

- All `swe_agents_train/` keys have a `swe_agents_val/` counterpart emitted during validation steps.
- Layer 4 percentiles apply to **all** per-sample scalar metrics including the Layer 5 trajectory stats above, so e.g. `swe_agents_train/num_turns/p99` shows the 99th-percentile trajectory length per step.
- `performance/` metrics are scalars logged once per training step.
- `generation_metrics/` timeline plots appear as W&B custom charts with one trace per vLLM worker.
