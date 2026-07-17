# Observability

NeMo RL is instrumented with [OpenTelemetry](https://opentelemetry.io/) via the [`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens) library. It emits **traces** at RL-algorithm boundaries (rollout, generation, reward, advantage, policy update, ...) and **metrics** for reward, loss, KL, throughput, and more.

Telemetry exports OTLP and works with any OTLP-compatible backend or an OpenTelemetry Collector (e.g. Jaeger, Grafana Tempo, or an OpenTelemetry Collector that fans out to your backend of choice).

Telemetry is **entirely optional**. When nemo-lens is not installed or telemetry is disabled, every instrumentation site is a ~0-cost no-op — see `nemo_rl/telemetry/_fallbacks.py`.

## What's in this section

```{toctree}
:maxdepth: 1

configuration
span-groups
metrics
vllm-tracing
observability-stack
extending
```

## Scope

This documentation covers **NeMo-RL-specific** usage: the `telemetry:` config block, the `NEMO_RL_OTEL_*` environment variables, RL span names, `rl.*` metric names, and the two-layer vLLM tracing integration.

For general concepts — the span-group mechanism, instrumentation primitives, the configuration model, custom exporters, resource detection — see the [lens documentation](https://github.com/NVIDIA-NeMo/Lens). This section links to lens docs when relevant rather than duplicating them.

| Concern | Owned by |
|---|---|
| `telemetry:` YAML block, `NEMO_RL_OTEL_*` env vars | NeMo-RL (this section) |
| `RLSpanGroup` groups + presets, `rl.*` span/metric names | NeMo-RL (this section) |
| Driver/worker telemetry lifecycle, vLLM two-layer tracing | NeMo-RL (this section) |
| `managed_span` / `trace_fn` / `span_cm`, config model, exporters, resource detection | [lens](https://github.com/NVIDIA-NeMo/Lens) |

## Install

Telemetry needs the nemo-lens SDK (pulled from PyPI by the `telemetry` extra):

```bash
uv sync --extra telemetry          # or, to add just the SDK: uv pip install 'nemo-lens[sdk]'
```

## Quick start

```bash
export NEMO_RL_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # your OTLP backend / collector
export NEMO_RL_OTEL_SPAN_GROUPS=default                    # coarse-grained; safe for production

uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml
```

With `default` span groups, NeMo-RL emits a handful of coarse spans (job, checkpoint, evaluate) plus a steady stream of `rl.*` metrics. Switch to `per_step` for per-step traces (rollout/generation/reward/...), or `all` for everything.

You do not have to touch the config file: telemetry can be driven purely by env vars, or by adding a `telemetry:` block to your run config. Raw env vars always win over YAML. See [Configuration](configuration.md).

## What gets instrumented

Each algorithm's `examples/run_<algo>.py` calls `init_telemetry_driver(config, algorithm="<algo>")` **before** `init_ray()` (so the resolved `NEMO_RL_OTEL_*` settings are snapshotted into the Ray `runtime_env` and inherited by every worker) and `shutdown_telemetry()` at the end of `main()`.

| Algorithm | Entry point | Representative spans |
|---|---|---|
| GRPO (sync + async) | `examples/run_grpo.py` | `rl.grpo.step`, `rl.grpo.collect_rollouts`, `rl.grpo.compute_rewards`, `rl.grpo.compute_logprobs`, `rl.grpo.compute_advantages`, `rl.grpo.policy_update` |
| PPO | `examples/run_ppo.py` | `rl.ppo.step`, `rl.ppo.collect_rollouts`, `rl.ppo.compute_rewards`, `rl.ppo.compute_advantages`, `rl.ppo.policy_update`, `rl.ppo.value_update` |
| SFT | `examples/run_sft.py` | `rl.sft.step`, `rl.sft.data_processing`, `rl.sft.policy_update` |
| DPO | `examples/run_dpo.py` | `rl.dpo.step`, `rl.dpo.policy_update` |
| RM | `examples/run_rm.py` | `rl.rm.step` |
| Distillation | `examples/run_distillation.py` | `rl.distillation.step`, `rl.distillation.collect_rollouts`, `rl.distillation.teacher_logprobs`, `rl.distillation.policy_update` |
| vLLM generation | `nemo_rl/models/generation/vllm/vllm_generation.py` | `rl.vllm.generate`, `rl.vllm.generate_text` |

Each span belongs to a **span group** that controls whether it is emitted at runtime. See [Span Groups](span-groups.md) for the full per-algorithm span table.

## What gets exported

- **Traces**: any OTLP-compatible backend (Jaeger, Grafana Tempo, an OpenTelemetry Collector, ...) via OTLP.
- **Metrics**: the `rl.*` catalog (reward, loss, KL, grad norm, throughput, generation/rollout latency) teed from the driver's metrics logger — see [Metrics](metrics.md).
- **Logs** (optional): via the OTel log bridge when `NEMO_RL_OTEL_LOGS_ENABLED=1` — correlates Python `logging` records with the active span's trace ID.

By default, only **one rank** exports (`single_rank`, last rank). The driver always exports (it hosts the training loop and the metrics logger). See [Configuration — Export strategy](configuration.md#export-strategy).

## Related

- Exporting to an OTLP backend: [Observability Stack](observability-stack.md)
- vLLM tracing (driver spans + native OTLP): [vLLM Tracing](vllm-tracing.md)
- Adding new spans / metrics: [Extending Instrumentation](extending.md)
- Lens configuration model and env vars: [lens: configuration](https://github.com/NVIDIA-NeMo/Lens)
- Instrumentation primitives (`managed_span`, `trace_fn`, `span_cm`): [lens: instrumentation](https://github.com/NVIDIA-NeMo/Lens)
