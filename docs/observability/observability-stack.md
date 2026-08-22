# Exporting to an OTLP backend

NeMo-RL's telemetry is a standard OpenTelemetry OTLP exporter: enable it, point it at an OTLP endpoint, and run your training. It works with **any OTLP-compatible backend or an OpenTelemetry Collector** — there is nothing NeMo-RL-specific about the backend, and no bundled Jaeger / Prometheus / Grafana.

Choosing an observability solution — retention, scale, auth, dashboards — is your decision, driven by your organisation's existing stack (e.g. Jaeger, Grafana Tempo, or an OpenTelemetry Collector that fans out to your backend of choice). For backend-specific guidance, see [lens: backends](https://github.com/NVIDIA-NeMo/Lens).

## Turn it on

Set these on the process that runs training (the driver, and via Ray its workers inherit them):

```bash
NEMO_RL_OTEL_ENABLED=1
NEMO_RL_OTEL_SPAN_GROUPS=default          # start coarse; raise to per_step / all as needed
NEMO_RL_OTEL_METRICS_ENABLED=1
NEMO_RL_OTEL_LOGS_ENABLED=1
NEMO_RL_OTEL_VLLM_NATIVE_TRACING=0        # gRPC-only; leave OFF on an http/protobuf path

# OTLP target — set these for your backend or collector:
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc          # grpc (collector/Jaeger on :4317) or http/protobuf (SaaS OTLP on :443)
# OTEL_EXPORTER_OTLP_HEADERS=<header>=<value>   # optional auth headers, comma-separated
```

All three signals (traces, metrics, logs) ship over OTLP to the endpoint you set; on the `http/protobuf` path the SDK appends `/v1/traces`, `/v1/metrics`, `/v1/logs` per signal. Pick the protocol to match your backend: a local collector or Jaeger typically speaks gRPC on `:4317`; a direct-to-SaaS OTLP endpoint typically speaks `http/protobuf` on `:443`.

Raise `NEMO_RL_OTEL_SPAN_GROUPS` to `per_step` (or `all`) for per-step traces. Set an explicit `NEMO_RL_OTEL_RUN_ID` (and optional `NEMO_RL_OTEL_USER_ID`) to name the run instead of taking the auto-generated id.

## Console / JSON output (no backend)

To confirm spans and metrics are produced without standing up any backend, use the `console` exporter:

```bash
NEMO_RL_OTEL_ENABLED=1 NEMO_RL_OTEL_EXPORTER=console \
  uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml
```

Each span and metric prints to stdout as **JSON** (`ConsoleSpanExporter` uses `span.to_json()`), so you can capture it to a file:

```bash
NEMO_RL_OTEL_ENABLED=1 NEMO_RL_OTEL_EXPORTER=console \
  uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml > telemetry.json 2>&1
```

`console` (set via `telemetry.exporter` / `NEMO_RL_OTEL_EXPORTER`) is the only backend-free JSON option nemo-lens exposes. For structured JSON-lines *files*, export OTLP to an OpenTelemetry Collector with a `file` exporter (nemo-lens ships a collector-file config) and point `OTEL_EXPORTER_OTLP_ENDPOINT` at the collector.

## vLLM native tracing needs a gRPC endpoint

vLLM's **native** OTLP tracing (`NEMO_RL_OTEL_VLLM_NATIVE_TRACING=1`) uses a gRPC-only exporter, so it will not ride an `http/protobuf` OTLP endpoint. To capture vLLM's native engine spans, add an OTLP/gRPC receiver (an OTel Collector on `:4317`, or a gRPC-capable backend) that forwards to your backend, and point `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) at it. The driver-side `rl.vllm.*` spans and `gen_ai.*` metrics (Layer 1) reach your backend regardless. See [vLLM Tracing](vllm-tracing.md).
