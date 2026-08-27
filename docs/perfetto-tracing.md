# Profile PPO with Perfetto

NeMo RL can write an opt-in CPU workload trace for synchronous and asynchronous
PPO. The output uses the Chrome JSON trace-event format and can be opened
directly in [Perfetto](https://ui.perfetto.dev/).

This trace complements Nsight Systems. Use it to understand the end-to-end PPO
driver schedule, rollout concurrency, replay-buffer stalls, critic and actor PPO
epochs, refit bubbles, validation, and checkpointing. Use Nsight Systems when
you need GPU kernels, CUDA streams, or NCCL detail.

## Enable tracing

Enable the optional Perfetto logger in the PPO master config:

```yaml
logger:
  perfetto:
    enable: true
    name: ppo_trace.json
```

`name` may be a filename or an absolute path. Relative names are written under
the resolved experiment `logger.log_dir`; absolute paths can target shared
persistent storage directly. The PPO base config defaults to:

```yaml
logger:
  perfetto:
    enable: false
    name: ppo_perfetto.json
```

This also makes command-line enablement available without adding new keys:

```bash
uv run examples/run_ppo.py --config <ppo-recipe.yaml> \
  logger.perfetto.enable=true \
  logger.perfetto.name=profile.json
```

Tracing is disabled by default. When disabled, the tracer stores no events and
the existing timer behavior is unchanged. The resolved logger configuration is
passed to the async collector actor, so sync and async PPO use the same switch.

## Read the trace

Open the JSON file in Perfetto and inspect these process groups:

- **driver**: the full synchronous or asynchronous PPO schedule. Existing PPO
  timer regions appear as nested spans, including generation, replay-buffer
  starvation, value inference and training, policy/reference logprobs, GAE,
  each actor update, refit, validation, and checkpointing.
- **trajectory_collector_actor**: asynchronous collector work and wait states.
- **ppo_sync_rollouts** or **ppo_async_rollouts**: one virtual track per native
  rollout sample. Each track contains the full sample rollout, turns,
  generation calls, reward calculation, and environment tokenization.

`ppo_step_start`, `ppo_epoch_start`, and `ppo_step_complete` events carry the
step and PPO-epoch numbers. Async step-complete events also include the replay
buffer size and average trajectory age. A sample's outer span includes token
counts, termination/truncation state, and total reward.

The async driver aligns the collector actor's monotonic clock with its own using
a midpoint round-trip estimate before merging both event streams into one JSON
file.

## Scope and limitations

- Per-sample timing is available when PPO uses the native asynchronous rollout
  path. Synchronous batched rollout backends still appear accurately as a
  driver-side generation span, but they do not expose independent sample
  timing.
- The driver keeps trace events in memory until shutdown. Prefer short profiling
  runs and representative batch sizes when sample-level tracing is enabled.
- Graceful completion and handled training failures write the trace. A hard
  process kill, node failure, or Slurm `SIGKILL` can prevent the final file from
  being written.
- Trace collection and file-writing failures are warnings and do not change PPO
  training behavior.
