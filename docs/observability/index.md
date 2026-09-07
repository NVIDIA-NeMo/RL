# Observability

NeMo RL is instrumented with [OpenTelemetry](https://opentelemetry.io/) via the [`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens) library. It emits **traces** at RL-algorithm boundaries (rollout, generation, reward, advantage, policy update, ...) and **metrics** for async efficiency accounting and vLLM generation.

Telemetry exports OTLP and works with any OTLP-compatible backend or an OpenTelemetry Collector (e.g. Jaeger, Grafana Tempo, or an OpenTelemetry Collector that fans out to your backend of choice).

Telemetry is **off by default**. nemo-lens ships as a base dependency, so it reaches every worker venv, and `telemetry.enabled` is the single switch: while it is false every instrumentation site is a ~0-cost no-op.

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

This documentation covers **NeMo-RL-specific** usage: the `telemetry:` config block, RL span names, `rl.*` metric names, and the two-layer vLLM tracing integration.

For general concepts — the span-group mechanism, instrumentation primitives, the configuration model, custom exporters, resource detection — see the [lens documentation](https://github.com/NVIDIA-NeMo/Lens). This section links to lens docs when relevant rather than duplicating them.

| Concern | Owned by |
|---|---|
| `telemetry:` YAML block | NeMo-RL (this section) |
| `RLSpanGroup` groups + presets, `rl.*` span/metric names | NeMo-RL (this section) |
| Driver/worker telemetry lifecycle, vLLM two-layer tracing | NeMo-RL (this section) |
| `managed_span` / `trace_fn` / `span_cm`, config model, exporters, resource detection | [lens](https://github.com/NVIDIA-NeMo/Lens) |

## Install

Nothing to install: `nemo-lens[sdk]` is a base dependency, so a normal `uv sync` covers the driver and every worker venv.

## Quick start

Add a `telemetry:` block to your run config:

```yaml
telemetry:
  enabled: true
  span_groups: default   # coarse-grained; safe for production
```

Point it at a backend and run:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # your OTLP backend / collector

uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml
```

With `default` span groups, NeMo-RL emits a handful of coarse spans (job, checkpoint, evaluate) plus whatever `rl.*` metrics the driver's logger produces. Switch to `per_step` for per-step traces (rollout/generation/reward/...), or `all` for everything.

Keeping the settings in the config file is what makes a run's telemetry reproducible from the file alone. The endpoint is the exception: `OTEL_EXPORTER_OTLP_*` are the standard OpenTelemetry variables, and they belong in the environment because they describe where you are running, not what you are measuring. See [Configuration](configuration.md).

## What gets instrumented

Each algorithm's `examples/run_<algo>.py` calls `init_telemetry_driver(config, algorithm="<algo>")` **before** `init_ray()` (so the resolved `NEMO_RL_OTEL_*` settings are snapshotted into the Ray `runtime_env` and inherited by every worker) and `shutdown_telemetry()` from a `finally` block wrapping the whole run, so buffered spans are flushed even when the run fails.

| Algorithm | Entry point | Representative spans |
|---|---|---|
| GRPO (sync + async) | `examples/run_grpo.py` | `rl.grpo.step`, `rl.grpo.generation`, `rl.grpo.reward_calculation`, `rl.grpo.policy_and_reference_logprobs`, `rl.grpo.advantage_calculation`, `rl.grpo.policy_training` |
| PPO | `examples/run_ppo.py` | `rl.ppo.step`, `rl.ppo.generation`, `rl.ppo.reward_calculation`, `rl.ppo.advantage_calculation`, `rl.ppo.policy_training`, `rl.ppo.value_training` |
| SFT | `examples/run_sft.py` | `rl.sft.step`, `rl.sft.data_processing`, `rl.sft.policy_training` |
| DPO | `examples/run_dpo.py` | `rl.dpo.step`, `rl.dpo.policy_training` |
| RM | `examples/run_rm.py` | `rl.rm.step` |
| Distillation | `examples/run_distillation.py` | `rl.distillation.step`, `rl.distillation.generation`, `rl.distillation.teacher_logprob_inference`, `rl.distillation.policy_training` |
| vLLM generation | `nemo_rl/models/generation/vllm/vllm_generation.py` | `rl.vllm.generate`, `rl.vllm.generate_text` |

Each span belongs to a **span group** that controls whether it is emitted at runtime. See [Span Groups](span-groups.md) for the full per-algorithm span table.

## What gets exported

- **Traces**: any OTLP-compatible backend (Jaeger, Grafana Tempo, an OpenTelemetry Collector, ...) via OTLP.
- **Metrics**: the `rl.efficiency.*` async accounting teed from the driver's metrics logger, plus the vLLM `gen_ai.*` series — see [Metrics](metrics.md).
- **Logs** (optional): via the OTel log bridge when `telemetry.logs_enabled` is true — correlates Python `logging` records with the active span's trace ID.

By default, only **one rank** exports (`single_rank`, last rank). The driver always exports (it hosts the training loop and the metrics logger). See [Configuration — Export strategy](configuration.md#export-strategy).

## Related

- Exporting to an OTLP backend: [Observability Stack](observability-stack.md)
- vLLM tracing (driver spans + native OTLP): [vLLM Tracing](vllm-tracing.md)
- Adding new spans / metrics: [Extending Instrumentation](extending.md)
- Lens configuration model and env vars: [lens: configuration](https://github.com/NVIDIA-NeMo/Lens)
- Instrumentation primitives (`managed_span`, `trace_fn`, `span_cm`): [lens: instrumentation](https://github.com/NVIDIA-NeMo/Lens)
