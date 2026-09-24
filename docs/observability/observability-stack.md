# Exporting to an OTLP backend

NeMo-RL's telemetry is a standard OpenTelemetry OTLP exporter: enable it, point it at an OTLP endpoint, and run your training. It works with **any OTLP-compatible backend or an OpenTelemetry Collector** — there is nothing NeMo-RL-specific about the backend, and no bundled Jaeger / Prometheus / Grafana.

Choosing an observability solution — retention, scale, auth, dashboards — is your decision, driven by your organisation's existing stack (e.g. Jaeger, Grafana Tempo, or an OpenTelemetry Collector that fans out to your backend of choice). For backend-specific guidance, see [lens: backends](https://github.com/NVIDIA-NeMo/Lens).

## Turn it on

What you measure goes in your run config, so a run's telemetry settings stay recoverable from the file that describes the run:

```yaml
telemetry:
  enabled: true
  span_groups: default        # start coarse; raise to per_step / all as needed
  metrics_enabled: true
  logs_enabled: true
  vllm_native_tracing: false  # gRPC by default, one span per request; leave off outside debugging
```

Where you send it goes in the environment, since it describes the machine rather than the run. These are the standard OTel SDK variables, read by the SDK directly:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc          # grpc (collector/Jaeger on :4317) or http/protobuf (SaaS OTLP on :443)
# OTEL_EXPORTER_OTLP_HEADERS=<header>=<value>   # optional auth headers, comma-separated
```

All three signals (traces, metrics, logs) ship over OTLP to the endpoint you set; on the `http/protobuf` path the SDK appends `/v1/traces`, `/v1/metrics`, `/v1/logs` per signal. Pick the protocol to match your backend: a local collector or Jaeger typically speaks gRPC on `:4317`; a direct-to-SaaS OTLP endpoint typically speaks `http/protobuf` on `:443`.

Raise `span_groups` to `per_step` (or `all`) for per-step traces. To name the run instead of taking the auto-generated id, set `NEMO_RL_OTEL_RUN_ID` (and optional `NEMO_RL_OTEL_USER_ID`) — the two settings with no `telemetry:` equivalent, since a job scheduler usually supplies them.

## Console / JSON output (no backend)

To confirm spans and metrics are produced without standing up any backend, use the `console` exporter. Hydra-style CLI overrides keep a one-off run in the config record — they are echoed into the run's log — so prefer them over exporting a variable:

```bash
uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml \
  ++telemetry.enabled=true ++telemetry.exporter=console
```

Each span and metric prints to stdout as **JSON** (`ConsoleSpanExporter` uses `span.to_json()`), so you can capture it to a file:

```bash
uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml \
  ++telemetry.enabled=true ++telemetry.exporter=console > telemetry.json 2>&1
```

`console` (set via `telemetry.exporter`) is the only backend-free JSON option nemo-lens exposes. For structured JSON-lines *files*, export OTLP to an OpenTelemetry Collector with a `file` exporter (nemo-lens ships a collector-file config) and point `OTEL_EXPORTER_OTLP_ENDPOINT` at the collector.

## vLLM native tracing chooses its protocol separately

vLLM's **native** OTLP tracing (`telemetry.vllm_native_tracing: true`) builds its own exporter rather than reusing lens's, and reads the protocol from `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` alone — defaulting to `grpc` and ignoring the generic `OTEL_EXPORTER_OTLP_PROTOCOL`. On an `http/protobuf` stack it would otherwise speak gRPC at an HTTP port, so either set `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` or add an OTLP/gRPC receiver (an OTel Collector on `:4317`) that forwards to your backend. The driver-side `rl.vllm.*` spans and `gen_ai.*` metrics (Layer 1) reach your backend regardless.

Keep `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` **unset** and configure the generic `OTEL_EXPORTER_OTLP_ENDPOINT` instead: vLLM's worker processes enable tracing on the mere presence of the traces-specific variable, which the Ray `runtime_env` spreads cluster-wide, so exporting it switches on per-request worker spans even with `vllm_native_tracing: false`. See [vLLM Tracing](vllm-tracing.md).
