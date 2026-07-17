# Metrics

NeMo-RL emits two namespaces of metrics: RL training metrics (`rl.*`) and vLLM generation metrics (`gen_ai.*`, following the OTel GenAI semantic conventions).

Metrics are emitted **only when telemetry is exporting** — the driver always exports, so the `rl.*` series come from the driver's metrics logger. For the general instrument pattern (per-Meter caching, `None`-skipping), see [lens: metrics](https://github.com/NVIDIA-NeMo/Lens).

## RL training metrics (`rl.*`)

Training has no OTel standard, so NeMo-RL uses a project-specific application scope. These are recorded via `nemo.lens.instruments.rl.record_rl_metrics` from the driver.

| Metric | Type | Description |
|---|---|---|
| `rl.reward.mean` | Gauge | Mean reward over the batch |
| `rl.kl_divergence` | Gauge | KL divergence from the reference policy |
| `rl.policy_loss` | Gauge | Policy (actor) loss |
| `rl.value_loss` | Gauge | Value (critic) loss — PPO |
| `rl.entropy` | Gauge | Policy entropy |
| `rl.response_length.mean` | Gauge | Mean generated response length (tokens) |
| `rl.grad_norm` | Gauge | Global gradient norm |
| `rl.learning_rate` | Gauge | Current learning rate |
| `rl.tokens_per_sec` | Gauge | Training throughput (tokens/sec) |
| `rl.generation.duration_ms` | Histogram | Generation latency (ms) |
| `rl.rollout.duration_ms` | Histogram | Rollout-collection latency (ms) |

Loss, reward, KL, grad norm, learning rate, and throughput are **Gauges** (point-in-time value per log step), not Histograms — semantically correct for a value that changes every log interval.

## How the `rl.*` gauges are populated (the Logger tee)

NeMo-RL does not sprinkle `record_rl_metrics()` calls through the algorithm code. Instead, `nemo_rl/telemetry/metrics.py` **tees** the scalar metrics that already flow through `nemo_rl.utils.logger.Logger.log_metrics` into nemo-lens: after `log_metrics` fans out to the file / W&B / MLflow backends, it calls `tee_rl_metrics_to_otel(metrics, prefix)`.

Only the driver's **`train`-prefix** metrics are teed (`prefix in ("train", "")`); other prefixes are skipped. The tee is best-effort — a raw metrics dict is matched against a fixed key map, the first present candidate key wins, and unknown keys or non-scalar values are silently skipped. It is a no-op unless telemetry is actively exporting.

| Logger metric key (first match wins) | Emitted metric |
|---|---|
| `reward` / `reward_mean` / `mean_reward` | `rl.reward.mean` |
| `kl` / `kl_divergence` / `mean_kl` | `rl.kl_divergence` |
| `loss` / `policy_loss` | `rl.policy_loss` |
| `value_loss` / `critic_loss` | `rl.value_loss` |
| `entropy` | `rl.entropy` |
| `mean_gen_tokens_per_sample` / `response_length_mean` | `rl.response_length.mean` |
| `grad_norm` | `rl.grad_norm` |
| `lr` / `learning_rate` | `rl.learning_rate` |
| `valid_tokens_per_sec_per_gpu` / `tokens_per_sec` | `rl.tokens_per_sec` |

This means the metrics you already log to W&B are the same series you get in your OTLP backend — no double bookkeeping. If an algorithm logs a scalar under a key not in this map, add a candidate to `_RL_OTEL_METRIC_MAP` (see [Extending](extending.md)).

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
| **Metric** | numerical value that changes over time | reward, loss, KL, grad norm, throughput → `rl.*` |
| **Span tag** | categorical per-span context for filtering | `rl.iteration`, `rl.backend`, `rl.num_generations_per_prompt` |
| **Resource attribute** | stable for the whole run | `rl.algorithm`, `rl.model`, `dl.tensor_parallel.size` |

Do **not** put a time-series number (loss, reward) on a span attribute — it produces no useful series in your backend and wastes storage. Do **not** put a per-step categorical (iteration number) on a metric label — that is unbounded cardinality. See [lens: metrics — metric vs span attribute vs resource attribute](https://github.com/NVIDIA-NeMo/Lens).

Metric names use the **application scope** (`rl.*`); attribute names use the **shared namespace** (`rl.*`, `dl.*`) defined in lens's `semconv.py`.

## Filtering across runs

Every `rl.*` data point carries the `run_id` resource attribute. Use it to isolate or compare runs in your backend (Grafana/Prometheus, or any OTLP-compatible backend). See [Configuration — Run identification](configuration.md#run-identification).
