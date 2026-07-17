# NeMo-RL OpenTelemetry Instrumentation

This module contains NeMo-RL's OpenTelemetry integration, built on top of [`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens).

It emits **traces** at RL-algorithm boundaries (rollout, generation, reward, advantage, policy update, checkpoint, evaluate) and **metrics** (`rl.*`: reward, loss, KL, throughput, generation/rollout latency) that export to any OTLP-compatible backend.

Telemetry is **optional**: it activates only when `enabled` is true *and* nemo-lens is installed. When either is absent, every instrumentation site degrades to a ~0-cost no-op.

## Contents

```
nemo_rl/telemetry/
├── config.py       — TelemetryConfig: the telemetry: config block
├── setup.py        — init_telemetry_driver / init_telemetry_worker / get_telemetry / shutdown_telemetry
├── span_groups.py  — RLSpanGroup: RL-specific span groups + presets
├── metrics.py      — tees Logger.log_metrics scalars into the rl.* instruments
├── _fallbacks.py   — no-op shims for when nemo-lens is not installed
└── __init__.py
```

Metric instruments, resource detection, and the instrumentation primitives themselves live in `nemo-lens`. This module is a thin integration layer.

## Wiring

Each `examples/run_<algo>.py` calls `init_telemetry_driver(config, algorithm="<algo>")` **before** `init_ray()` (so `NEMO_RL_OTEL_*` is snapshotted into the Ray `runtime_env` and inherited by workers) and `shutdown_telemetry()` at the end of `main()`. `get_telemetry()` returns the process-global `TelemetryHandle`; `init_telemetry_worker()` sets up telemetry inside a Ray actor.

## Install

```bash
uv sync --extra telemetry             # or, to add just the SDK: uv pip install 'nemo-lens[sdk]'
```

## Quick start

```bash
export NEMO_RL_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export NEMO_RL_OTEL_SPAN_GROUPS=default

uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml
```

## Full documentation

See `docs/observability/` in this repository:

| Topic | Doc |
|---|---|
| Overview | [docs/observability/index.md](../../docs/observability/index.md) |
| Configuration (`telemetry:` block, env vars) | [docs/observability/configuration.md](../../docs/observability/configuration.md) |
| Span groups and per-algorithm span names | [docs/observability/span-groups.md](../../docs/observability/span-groups.md) |
| `rl.*` metrics and the Logger tee | [docs/observability/metrics.md](../../docs/observability/metrics.md) |
| vLLM tracing (driver spans + native OTLP) | [docs/observability/vllm-tracing.md](../../docs/observability/vllm-tracing.md) |
| Exporting to an OTLP backend | [docs/observability/observability-stack.md](../../docs/observability/observability-stack.md) |
| Adding new instrumentation | [docs/observability/extending.md](../../docs/observability/extending.md) |

For the generic `nemo-lens` documentation (configuration model, instrumentation primitives, custom exporters, design decisions), see the lens docs at <https://github.com/NVIDIA-NeMo/Lens>.
