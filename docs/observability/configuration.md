# Configuration

Telemetry can be configured two ways, which compose:

1. A `telemetry:` block in your run config (YAML).
2. `NEMO_RL_OTEL_*` and standard `OTEL_*` environment variables.

**Raw environment variables always win over the YAML block.** On the driver, the `telemetry:` block is translated into `NEMO_RL_OTEL_*` env vars with `os.environ.setdefault` *before* `init_ray()` — so anything already present in the environment is left untouched, and the resulting environment is snapshotted into the Ray `runtime_env` and inherited by every worker.

## The `telemetry:` config block

`telemetry:` is an optional top-level field of every algorithm's `MasterConfig`. It is **documented here, not baked into the exemplar configs** — add it to your own run config, or configure purely via env vars.

```yaml
telemetry:
  enabled: false              # master switch; when false, every site is a ~0-cost no-op
  service_name: nemo-rl       # service.name reported to the backend
  span_groups: default        # preset (default | per_step | all) or a comma-separated group list
  export_strategy: single_rank # single_rank | all_ranks | sampled | first_rank_per_node
  export_rank: -1             # for single_rank: which rank exports (-1 = last rank)
  traces_enabled: true        # emit trace spans
  metrics_enabled: true       # emit the rl.* metric instruments
  logs_enabled: false         # bridge Python logging to OTel logs (trace-correlated)
  exporter: otlp              # otlp | console
  vllm_native_tracing: false  # opt in to vLLM's own OTLP tracing (gRPC-only — see vllm-tracing.md)
```

The defaults above are the field defaults of `TelemetryConfig` (`nemo_rl/telemetry/config.py`). The endpoint, headers, and protocol are **not** in this block — they come from the standard `OTEL_EXPORTER_OTLP_*` env vars (see below).

The driver always exports (it hosts the training loop and the metrics logger); `export_strategy` / `export_rank` govern the Ray **worker** ranks.

## NeMo-RL environment variables

Each `NEMO_RL_OTEL_*` variable maps onto a `NemoLensConfig` field. Lens reads `NEMO_RL_OTEL_<KEY>` first and falls back to `NEMO_LENS_<KEY>`, so you can set a shared `NEMO_LENS_*` default and override it per-run with the RL-scoped prefix.

| Variable | Maps to | Default |
|---|---|---|
| `NEMO_RL_OTEL_ENABLED` | `enabled` | `0` |
| `NEMO_RL_OTEL_SPAN_GROUPS` | `span_groups` | `default` |
| `NEMO_RL_OTEL_EXPORT_STRATEGY` | `export_strategy` | `single_rank` |
| `NEMO_RL_OTEL_EXPORT_RANK` | `export_rank` | `-1` |
| `NEMO_RL_OTEL_TRACES_ENABLED` | `traces_enabled` | `1` |
| `NEMO_RL_OTEL_METRICS_ENABLED` | `metrics_enabled` | `1` |
| `NEMO_RL_OTEL_LOGS_ENABLED` | `logs_enabled` | `0` |
| `NEMO_RL_OTEL_EXPORTER` | `exporter` | `otlp` |
| `NEMO_RL_OTEL_VLLM_NATIVE_TRACING` | `vllm_native_tracing` | `0` |
| `NEMO_RL_OTEL_RUN_ID` | run identifier | (auto) |
| `NEMO_RL_OTEL_USER_ID` | optional user/team label | (empty) |

`service_name` maps onto the standard `OTEL_SERVICE_NAME` (lens reads it unprefixed).

For the full config model, field semantics, and validation rules, see [lens: configuration](https://github.com/NVIDIA-NeMo/Lens).

## Standard OTel SDK variables

Endpoint, protocol, and headers are honoured by the OTel SDK directly:

| Variable | Example |
|---|---|
| `OTEL_SERVICE_NAME` | `nemo-rl` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` or `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `<header>=<value>,<header>=<value>` (e.g. auth headers your backend requires) |

Pick the protocol to match your backend: a local collector or Jaeger typically speaks gRPC on `:4317`; a direct-to-SaaS OTLP endpoint typically speaks `http/protobuf` on `:443`. See [Observability Stack](observability-stack.md).

## Export strategy

`export_strategy` controls which **worker** ranks actually send telemetry:

- `single_rank` (default) — only the rank named by `export_rank` (`-1` = last rank).
- `all_ranks` — every worker exports.
- `sampled` / `first_rank_per_node` — sample a subset.

The driver is independent of this — it always exports. Non-exporting ranks get an empty (`frozenset()`) span-group set, so `is_span_group_enabled()` is `False` everywhere and no span objects are created at all. See [lens: sampling](https://github.com/NVIDIA-NeMo/Lens) for the detailed semantics.

## Run identification

Every run gets a `run_id` that flows to all backends as a resource attribute and is shared by the driver and every worker.

**Priority order:**

1. `NEMO_RL_OTEL_RUN_ID` (explicit, highest priority).
2. `SLURM_JOB_ID` (auto-detected on SLURM clusters).
3. Auto-generated 12-character hex id (fallback).

The `run_id` is written to the environment on the driver **before** `init_ray()`, so every worker inherits the same value and correlates to the same run. This is also how vLLM's native spans are correlated back to the RL run — see [vLLM Tracing](vllm-tracing.md).

Filter by `run_id` in your backend to isolate a specific run.

## Resource attributes

`init_telemetry_driver` sets stable-for-the-run values on the OTel `Resource`, so they appear on every span/metric as backend "Process" tags:

| Attribute | Source |
|---|---|
| `rl.algorithm` | the `algorithm="<algo>"` passed to `init_telemetry_driver` |
| `rl.model` | `policy.model_name` |
| `nemo.precision` | `policy.precision` |
| `dl.tensor_parallel.size` | `policy.megatron_cfg` / `dtensor_cfg` TP size |
| `dl.pipeline_parallel.size` | `policy.megatron_cfg` PP size |
| `dl.rank`, `dl.world_size` | set automatically by lens |

Attribute construction is best-effort: a missing config key simply omits that attribute; it never raises. Plus auto-detected host / GPU / SLURM / Kubernetes attributes from lens's resource detection.

## Typical configurations

### Console exporter (no backend)

```bash
export NEMO_RL_OTEL_ENABLED=1
export NEMO_RL_OTEL_EXPORTER=console
uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml
```

Spans and metrics print to stdout — a quick dry run with no backend to stand up.

### Direct to an OTLP backend (http/protobuf)

```bash
export NEMO_RL_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=https://<your-otlp-endpoint>:443
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_HEADERS="<header>=<value>"   # any auth headers your backend requires
uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml
```

See [Observability Stack](observability-stack.md) for the full backend-export setup.

### Per-step granularity

```bash
export NEMO_RL_OTEL_ENABLED=1
export NEMO_RL_OTEL_SPAN_GROUPS=per_step
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

`per_step` makes each training step its own root trace (rollout, generation, reward, advantage, policy update). See [Span Groups](span-groups.md).
