# NeMo-RL OpenTelemetry Instrumentation

This module contains NeMo-RL's OpenTelemetry integration, built on top of [`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens).

It emits **traces** at RL-algorithm boundaries (rollout, generation, reward, advantage, policy update, checkpoint, evaluate) and **metrics** (`rl.*`: reward, loss, KL, grad norm, learning rate, throughput) that export to any OTLP-compatible backend.

Telemetry is **optional**: it activates only when `enabled` is true *and* nemo-lens is installed. When either is absent, every instrumentation site degrades to a ~0-cost no-op.

## Contents

```
nemo_rl/telemetry/
├── config.py       — TelemetryConfig: the telemetry: config block
├── setup.py        — init_telemetry_driver / init_telemetry_worker / get_telemetry_handle / shutdown_telemetry
├── span_groups.py  — RLSpanGroup: RL-specific span groups + presets
├── instrumentation.py — managed_span/trace_fn wrappers + phase/group → rl.bucket map (monitor derives goodput)
├── metrics.py      — tees Logger.log_metrics scalars into the rl.* instruments
└── __init__.py
```

Metric instruments, resource detection, and the instrumentation primitives themselves live in `nemo-lens`. This module is a thin integration layer.

## Wiring

Each `examples/run_<algo>.py` calls `init_telemetry_driver(config, algorithm="<algo>")` **before** `init_ray()` (so `NEMO_RL_OTEL_*` is snapshotted into the Ray `runtime_env` and inherited by workers) and `shutdown_telemetry()` from a `finally` block wrapping the whole run, so buffered spans are flushed on the failure path too. `get_telemetry_handle()` returns the process-global `TelemetryHandle`.

OTel providers are process-global, so each Ray actor sets up its own: the policy, value and vLLM generation workers call `init_telemetry_worker()` from `__init__` and `shutdown_telemetry()` from `shutdown` (the latter matters — span/metric processors buffer in the background, and an actor that exits without flushing drops whatever it had not exported). Worker ranks come from the `RANK` / `WORLD_SIZE` env vars, and `RayWorkerGroup` also exports `NRL_WORKER_GROUP` so a worker's spans carry `rl.worker_group` — `RANK` is group-local, so it alone cannot distinguish `lm_policy` rank 3 from `vllm_policy` rank 3.

The async trajectory collector is a singleton actor rather than a group member, so it passes `rank=0, world_size=1, always_export=True`: an `export_strategy` selecting among a group's ranks cannot meaningfully be applied to a synthetic rank, and would otherwise mute every rollout span in the run. It flushes on demand via `flush_telemetry()` because the driver reaps it with `ray.kill`, which runs no `atexit` handler.

Trace context does not cross the Ray call boundary on its own, so a worker's spans are roots of their own traces, correlated to the driver by `run_id` rather than parented to it. The one exception is the async trajectory collector: the driver hands it a W3C carrier at construction and it reattaches that context per thread, so its rollout spans nest under `rl.grpo.job`. See [span groups — getting the collector into one waterfall](../../docs/observability/span-groups.md#getting-the-collector-into-one-waterfall).

Not yet wired:

| Gap | Effect |
|---|---|
| SGLang, TRT-LLM and Megatron generation workers | no `init_telemetry_worker`, no generation spans — only vLLM is instrumented |
| `grpo_sync.py`, `single_controller.py` | no spans at all; `examples/run_grpo_single_controller.py` never calls `init_telemetry_driver`, so that entrypoint emits no telemetry |
| `run_vlm_grpo.py`, `run_grpo_sliding_puzzle.py`, `run_xtoken_off_policy_distillation.py`, `run_eval.py` | no `init_telemetry_driver`, so a `telemetry:` block in those configs parses and the run succeeds while emitting nothing, driver *and* worker |
| `VllmGeneration.generate_async` | no `rl.vllm.generate` span, so async rollouts and async validation show `rl.grpo.generation` / `rl.grpo.evaluate` with no generate breakdown inside |
| `SyncRolloutActor` | the sync data-plane counterpart of the collector; no `init_telemetry_worker`, no rollout spans |
| Worker `shutdown()` on non-async trainers | only `async_grpo_train` calls `policy.shutdown()` / `policy_generation.shutdown()`; elsewhere Ray reaps the actors, so the worker's final flush never runs and its telemetry depends on the periodic export |
| Trace context into ranked workers | policy/value/vLLM worker spans stay separate traces (see above) |

## Install

Nothing to install. `nemo-lens[sdk]` is a base dependency, which is what gets it into the *worker* interpreters: Ray actors run under the `PY_EXECUTABLES` entries in `nemo_rl/distributed/virtual_cluster.py` (`uv run --locked --extra vllm`, `--extra mcore`, ...), and those resolve the base dependencies plus one backend extra.

## Quick start

```yaml
# in your run config
telemetry:
  enabled: true
  span_groups: default
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

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
