# Metrics

NeMo-RL emits five namespaces of metrics: async efficiency metrics (`rl.efficiency.*`), startup phase durations (`rl.setup.duration`), training scalars (`rl.reward.*`, `rl.policy.*`, …), vLLM engine metrics (`rl.vllm.*`), and driver-side vLLM generation metrics (`gen_ai.*`, following the OTel GenAI semantic conventions).

Metrics are emitted **only when telemetry is exporting** — the driver always exports, so the `rl.*` series come from the driver's metrics logger.

Every `rl.*` series is **declared by NeMo-RL**, not by lens. lens provides a consumer-driven metric registry: NeMo-RL calls `register_metric_group("rl", [...])` once per process with a `MetricSpec` per series, then records against it with `record_metrics(meter, "rl", ...)`. The declarations live in one table in `nemo_rl/telemetry/metrics.py`, so adding a series needs no lens release and no agreement on field names. `record_metrics` never raises into the caller and skips `None`.

## The tee

The tee lives outside the algorithm code: `nemo_rl/telemetry/metrics.py` hooks `nemo_rl.utils.logger.Logger.log_metrics`, so after `log_metrics` fans a step out to the file / W&B / MLflow backends it calls `tee_rl_metrics_to_otel(metrics, prefix)`. It is best-effort — two prefixes are read, the driver's per-step `train` dicts (`prefix in ("train", "")`) and the one-off `timing/setup` dict at step 0, so other prefixes are skipped, non-scalar values are ignored, and the whole path is a no-op unless telemetry is actively exporting. The numbers you already see in W&B are therefore the same series you get in your OTLP backend, with no double bookkeeping.

Each family has its own exception handler, so a bad value in one cannot silence the others.

## Training scalars

Mirrored from the keys the algorithms already log, one row per series in `_TRAIN_SCALARS`:

| Metric | Type | Logger key |
|---|---|---|
| `rl.reward.mean` | Gauge | `reward` |
| `rl.kl.divergence` | Gauge | `kl_penalty` |
| `rl.policy.loss` | Gauge | `loss` |
| `rl.value.loss` | Gauge | `critic/loss` (PPO) |
| `rl.entropy` | Gauge | `approx_entropy` |
| `rl.response.length.mean` | Gauge (`{token}`) | `mean_gen_tokens_per_sample` |
| `rl.grad_norm` | Gauge | `grad_norm` |
| `rl.learning_rate` | Gauge | `lr` |

`kl_penalty` is named for the penalty but holds the divergence — the coefficient is divided back out in `loss_functions.py` — which is why the metric is named `rl.kl.divergence`.

A test (`tests/unit/telemetry/test_source_drift.py`) parses the sources and fails the build if a declared logger key stops being emitted, so a renamed key surfaces as a build failure rather than a gauge that silently reports nothing.

## Startup phases

Every algorithm's `setup()` already timed its startup phases and logged them under the `timing/setup` prefix at step 0 (`SetupTimingMetrics` in `nemo_rl/algorithms/metric_utils.py`). Those are teed as one dimensioned gauge:

| Metric | Type | Attributes | Description |
|---|---|---|---|
| `rl.setup.duration` | Gauge (`s`) | `rl.setup.phase` | Wall clock spent in one startup phase |

Dimensioned rather than one series per phase because the phase set is not knowable at declaration time: alongside its declared fields `SetupTimingMetrics` carries a free-form `extras` dict, and the sparse-refit transports add `vllm_<transport>_sparse_init_time_s` to it at runtime. A row per phase would silently drop whatever it had not been taught about.

The phase name is the logger key with its `_time_s` / `_s` suffix stripped, so `generation_init_time_s` becomes `phase=generation_init`. Requiring one of those suffixes is also the filter that keeps non-durations out — `parallel_init_enabled` is a flag, not a phase.

Representative phases: `generation_init` (and its `generation_init_reserve` / `generation_init_load` parts), `policy_init`, `value_init`, `nemo_gym_init`, `collective_init`, `weight_sync`, `teacher_model_init`, `parallel_wall`, `worker_setup`, `other_setup`, `total_setup`.

### `rl.setup.duration` carries no `rl.bucket`

Unlike `rl.efficiency.seconds`, these phases overlap each other by construction: `total_setup` contains the rest, `parallel_wall` covers the generation and policy builds running concurrently, and `generation_init_reserve` / `generation_init_load` are parts of `generation_init`. Anything summing them by bucket would count startup several times over.

Read one phase at a time. For "what did startup cost", use `phase=total_setup` — that is the one value that cannot double-count. The `rl.setup.*` spans give the same phases their shape in a trace; see [Span groups — Startup](span-groups.md#startup-what-happens-before-the-first-step).

## Async efficiency metrics (`rl.efficiency.*`)

Async GRPO measures where wall time goes with a `Timer` and logs the result as `efficiency/*` scalars (`print_efficiency_summary` in `nemo_rl/algorithms/utils.py`). Those same values are teed to OTel as one **dimensioned** gauge rather than one instrument per category, so adding a category needs no instrument change.

| Metric | Type | Attributes | Description |
|---|---|---|---|
| `rl.efficiency.seconds` | Gauge (`s`) | `rl.efficiency.category`, `rl.efficiency.measurement`, `rl.efficiency.window`, `rl.bucket` | Time attributed to one efficiency category |
| `rl.efficiency.pct` | Gauge (`%`) | `rl.efficiency.measurement`, `rl.efficiency.window` | Productive share of one step's driver-side wall clock |

The single-controller path reuses two of these category names — `idle/buffer_starvation` for its `exposed_generation` phase and `idle/refit_bubble` for `weight_sync` — so a goodput rollup reads the same vocabulary on both paths.

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

## vLLM engine metrics (`rl.vllm.*`)

Read from the vLLM engine's own Prometheus registry, delta'd per step. `snapshot_step_metrics()` takes a baseline before generation and `get_step_metrics()` a second reading after, both fanned out to the DP-leader workers; the same snapshot pair also feeds the spec-decode metrics, so the wider coverage costs no extra RPC.

| Metric | Type | Source series |
|---|---|---|
| `rl.vllm.prompt_tokens` | Counter (`{token}`) | `vllm:prompt_tokens` |
| `rl.vllm.generation_tokens` | Counter (`{token}`) | `vllm:generation_tokens` |
| `rl.vllm.prompt_length.mean` | Gauge (`{token}`) | `vllm:request_prompt_tokens` histogram |
| `rl.vllm.generation_length.mean` | Gauge (`{token}`) | `vllm:request_generation_tokens` histogram |
| `rl.vllm.generations.ok` | Counter (`{generation}`) | `vllm:request_success{finished_reason}` |
| `rl.vllm.generations.failed` | Counter (`{generation}`) | `vllm:request_success{finished_reason}` |

The counters carry the step's delta, so summing them across steps reconstructs the run total. The means come from each histogram's `sum`/`count` delta, so they are exact rather than bucket-interpolated; a step in which the engine served no request omits them rather than reporting a zero-length sequence.

**What counts as failed.** vLLM's finish reasons are `stop`, `length`, `abort`, `error` and `repetition`. Only `stop` and `length` leave a usable sample behind, and `length` is a normal RL outcome — a rollout routinely runs to `max_tokens` — so those two are `ok` and *everything else* is `failed`. Counting the remainder rather than an explicit deny-list keeps `ok + failed` equal to the engine's own total even if a future vLLM adds a reason we have not heard of.

vLLM renames these series occasionally, so each is looked up against a short candidate list (the same approach as the alias lists in `nemo_rl/models/generation/dynamo/metrics.py`). A series that matches nothing is **omitted**, leaving a visible gap in the dashboard rather than a plausible-looking zero.

## Driver-side generation metrics (`gen_ai.*`)

The driver-side vLLM generation path records token and latency metrics through lens's `record_inference_metrics` with `provider_name="vllm"`, following the [OTel GenAI metrics spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/).

| Metric | Type | Description |
|---|---|---|
| `gen_ai.client.token.usage` | Histogram | Tokens per request, split by `gen_ai.token.type` (`input` / `output`) |
| `gen_ai.server.request.duration` | Histogram | End-to-end generation request latency |

These overlap with `rl.vllm.*` on token counts but are not redundant: `gen_ai.*` is derived from the tensors a `generate()` call returns, so it measures what the driver received, while `rl.vllm.*` is the engine's own accounting. When the two disagree, the gap is work the engine did that never reached the driver — the aborted requests, which appear in no returned tensor at all.

These ride the normal `http/protobuf` OTLP path and reach the same backend as everything else. They are distinct from vLLM's **native** engine tracing (opt-in, gRPC-only) — see [vLLM Tracing](vllm-tracing.md).

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
