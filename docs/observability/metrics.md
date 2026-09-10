# Metrics

NeMo-RL emits two namespaces of metrics: async efficiency metrics (`rl.efficiency.*`) and vLLM generation metrics (`gen_ai.*`, following the OTel GenAI semantic conventions).

Metrics are emitted **only when telemetry is exporting** — the driver always exports, so the `rl.*` series come from the driver's metrics logger. For the general instrument pattern (per-Meter caching, `None`-skipping), see [lens: metrics](https://github.com/NVIDIA-NeMo/Lens).

Training scalars — reward, loss, KL, grad norm, learning rate, throughput — are **not** mirrored to OTel. nemo-lens declares `record_rl_metrics` gauges for most of them, plus `rl.generation.duration_ms` and `rl.rollout.duration_ms` histograms, but NeMo-RL emits none of them: mapping its logger keys onto lens's fixed fields is still being settled with the lens owners. Read those scalars from W&B / TensorBoard, and phase durations from the spans.

## Async efficiency metrics (`rl.efficiency.*`)

Async GRPO measures where wall time goes with a `Timer` and logs the result as `efficiency/*` scalars (`print_efficiency_summary` in `nemo_rl/algorithms/utils.py`). Those same values are teed to OTel as one **dimensioned** gauge rather than one instrument per category, so adding a category needs no instrument change.

The tee lives outside the algorithm code: `nemo_rl/telemetry/metrics.py` hooks `nemo_rl.utils.logger.Logger.log_metrics`, so after `log_metrics` fans a step out to the file / W&B / MLflow backends it calls `tee_rl_metrics_to_otel(metrics, prefix)`. It is best-effort — only the driver's `train` dicts (`prefix in ("train", "")`) carry the efficiency scalars, so other prefixes are skipped, non-scalar values are ignored, and the whole path is a no-op unless telemetry is actively exporting. The efficiency numbers you already see in W&B are therefore the same series you get in your OTLP backend, with no double bookkeeping.

| Metric | Type | Attributes | Description |
|---|---|---|---|
| `rl.efficiency.seconds` | Gauge (`s`) | `rl.efficiency.category`, `rl.efficiency.measurement`, `rl.efficiency.window`, `rl.bucket` | Time attributed to one efficiency category |
| `rl.efficiency.pct` | Gauge (`%`) | `rl.efficiency.measurement`, `rl.efficiency.window` | Productive share of one step's driver-side wall clock |

These instruments are defined in `nemo_rl/telemetry/metrics.py` rather than in lens, because they are keyed by NeMo-RL's own efficiency-category labels and there is no fixed lens field for them.

### Always filter on `rl.efficiency.measurement`

Some categories are measured on the driver against wall time; others are summed across concurrent collector threads and **can exceed the wall time they happened in**.

| `rl.efficiency.measurement` | Recorded on | Categories | Safe to sum against elapsed driver time? |
|---|---|---|---|
| `wall_clock` | driver, sequentially | `init/total`, `idle/buffer_starvation`, `idle/refit_bubble`, `idle/validation` | yes |
| `collector_wall_clock` | collector's collection-loop thread, sequentially | `idle/refit_event_wait`, `idle/generation_limit_pause` | no — real durations, but on a timeline that runs concurrently with the driver's |
| `thread_seconds` | collector's batch-worker threads, concurrently | `idle/buffer_full_backoff`, `wasted/failed_trajectory` | no — not durations at all |

Eight rollout threads each backing off for 10s during the same 10s window produce a `thread_seconds` value of 80, not 10. Summing `rl.efficiency.seconds` by `rl.bucket` without filtering therefore overstates idle time — a wrong answer that looks like a real one. Filter to `rl.efficiency.measurement="wall_clock"` before comparing against elapsed time; read the other two per-phase, `thread_seconds` as a saturation signal.

The non-`wall_clock` values also carry `rl.bucket` so all three share one vocabulary with the spans; the `measurement` attribute is what keeps a bucket rollup honest.

Two deliberate metric/span disagreements to know about before comparing a metric rollup against a trace rollup:

- **The two collector-loop categories** carry `rl.bucket="idle"` as metrics and rely on you filtering by `measurement`, but carry no bucket at all as spans, since a trace has no equivalent filter to rely on.
- **`idle/validation`** is `idle` as a metric and `overhead` on every span covering the same seconds. Both are true of different fleets: the training GPUs are idle, which is what the driver's timer measures, while the generation GPUs are doing necessary non-training work, which is what `bucket_scope(Bucket.OVERHEAD)` in `validate()` tags. Attributing this phase properly needs per-fleet accounting; until then, do not expect the two rollups to agree on a validation step. See [span groups — why `idle/validation` is not a span](span-groups.md#why-idlevalidation-is-not-a-span).

### `rl.efficiency.window`: what a value covers in time

`measurement` says whether values may be summed *against each other*; `window` says whether one may be summed *across steps*. They are independent, and the second is the easier one to get silently wrong.

| `rl.efficiency.window` | Categories | Meaning |
|---|---|---|
| `step` | `idle/buffer_starvation`, `idle/refit_bubble`, `idle/validation` | per-step delta — the driver resets its `Timer` every step, so these sum across steps |
| `run` | `init/total`, and all four collector-side categories | cumulative since the process started — consecutive points already contain each other, so summing across steps multiplies by the step count |

`init/total` is the driver-side exception: it is measured once, waiting for the first buffer fill before the step loop, then republished unchanged every step so it does not disappear from a dashboard after step 1. Read it as a constant. The collector's `Timer` is never reset, which is why everything from it is `run`.

`rl.efficiency.pct` is tagged `window="step"` for the same reason its numerator is: the three `step`-window idle categories over that step's wall time. `init/total` is deliberately excluded — it is a run constant, so folding it in would charge the whole startup cost to every step — and so are the collector's categories, which are on another clock. Against the run's elapsed time the ratio would climb toward 100% as the run lengthened no matter what the idle time did, which is why the denominator is one step and not the run.

## vLLM generation metrics (`gen_ai.*`)

The driver-side vLLM generation path records token and latency metrics through lens's `record_inference_metrics` with `provider_name="vllm"`, following the [OTel GenAI metrics spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/).

| Metric | Type | Description |
|---|---|---|
| `gen_ai.client.token.usage` | Histogram | Tokens per request, split by `gen_ai.token.type` (`input` / `output`) |
| `gen_ai.server.request.duration` | Histogram | End-to-end generation request latency |

These ride the normal `http/protobuf` OTLP path and reach the same backend as everything else. They are distinct from vLLM's **native** engine metrics (opt-in, gRPC-only) — see [vLLM Tracing](vllm-tracing.md).

## Metric vs span tag vs resource attribute

The one rule that trips people up. Classify each value before you emit it:

| Kind | Use | Example |
|---|---|---|
| **Metric** | numerical value that changes over time | per-category efficiency seconds → `rl.efficiency.seconds` |
| **Span tag** | categorical per-span context for filtering | `rl.iteration`, `rl.bucket`, `rl.num_generations_per_prompt`, `rl.weight_version` |
| **Resource attribute** | stable for the whole run | `rl.algorithm`, `rl.model`, `dl.tensor_parallel.size` |

Do **not** put a time-series number (loss, reward) on a span attribute — it produces no useful series in your backend and wastes storage. Do **not** put a per-step categorical (iteration number) on a metric label — that is unbounded cardinality. See [lens: metrics — metric vs span attribute vs resource attribute](https://github.com/NVIDIA-NeMo/Lens).

### Goodput (monitor-derived)

NeMo-RL does **not** emit `rl.goodput` or `rl.bucket.*` rollup metrics.
Leaf spans carry `rl.bucket` ∈ `{productive, overhead, idle, wasted}`;
umbrella spans (`job` / `step` / `rollout`) omit it. `rl.efficiency.seconds`
carries the same `rl.bucket` tokens, but it is a per-category duration, not a
rollup — and it needs the `rl.efficiency.measurement` filter described above.
Offline monitors (e.g. wandb-monitor) SUM span / phase GPU-time by `rl.bucket`
and compute:

```text
rl_goodput = productive_gpu_s / (productive + overhead + idle + wasted)_gpu_s
```

See [Span groups — goodput buckets](span-groups.md) and `nemo_rl/telemetry/instrumentation.py`.

Metric names use the **application scope** (`rl.*`); attribute names use the **shared namespace** (`rl.*`, `dl.*`) defined in lens's `semconv.py`.

## Filtering across runs

Every `rl.*` data point carries the `run_id` resource attribute. Use it to isolate or compare runs in your backend (Grafana/Prometheus, or any OTLP-compatible backend). See [Configuration — Run identification](configuration.md#run-identification).
