# Profile PPO and GRPO with Perfetto

NeMo RL can write an opt-in CPU workload trace for synchronous and asynchronous
PPO and GRPO. The output uses the Chrome JSON trace-event format and can be
opened directly in [Perfetto](https://ui.perfetto.dev/).

This trace complements Nsight Systems. Use it to understand the end-to-end PPO
or GRPO driver schedule, rollout concurrency, replay-buffer stalls, model
updates, refit bubbles, validation, and checkpointing. PPO traces additionally
show critic work and PPO epochs. Use Nsight Systems when you need GPU kernels,
CUDA streams, or NCCL detail.

## Enable tracing

Enable the optional Perfetto logger in the master config:

```yaml
logger:
  perfetto:
    enable: true
    name: rl_trace.json
```

`name` may be a filename or an absolute path. Relative names are written under
the resolved experiment `logger.log_dir`; absolute paths can target shared
persistent storage directly. The PPO and GRPO base configs default to disabled
tracing with algorithm-specific filenames:

```yaml
logger:
  perfetto:
    enable: false
    name: ppo_perfetto.json  # grpo_perfetto.json in the GRPO base config
```

This also makes command-line enablement available without adding new keys:

```bash
uv run examples/run_ppo.py --config <ppo-recipe.yaml> \
  logger.perfetto.enable=true \
  logger.perfetto.name=profile.json

uv run examples/run_grpo.py --config <grpo-recipe.yaml> \
  logger.perfetto.enable=true \
  logger.perfetto.name=profile.json
```

Tracing is disabled by default. When disabled, the tracer stores no events and
the existing timer behavior is unchanged. The resolved logger configuration is
passed to rollout actors, so sync and async PPO and GRPO use the same switch.

## Read the trace

Open the JSON file in Perfetto and inspect these process groups:

- **driver**: the full synchronous or asynchronous training schedule. Existing
  timer regions appear as nested spans, including generation, replay-buffer
  starvation, reward and advantage calculation, policy/reference logprobs,
  policy training, refit, validation, and checkpointing. PPO also includes
  value-model inference/training, GAE, and individual PPO epochs.
- **trajectory_collector_actor**: asynchronous collector work and wait states.
- **sync_rollout_actor**: the TransferQueue GRPO rollout actor, when that sync
  data-plane path is selected.
- **ppo_sync_rollouts**, **ppo_async_rollouts**, **grpo_sync_rollouts**, or
  **grpo_async_rollouts**: one virtual track per native rollout sample. Each
  track contains the full sample rollout, turns, generation calls, reward
  calculation, and environment tokenization.

`ppo_*` and `grpo_*` step events carry the algorithm, mode, step, and available
epoch or weight-version metadata. Async step-complete events also include the
replay-buffer size and average trajectory age. A sample's outer span includes
token counts, termination/truncation state, and total reward.

The async driver aligns the collector actor's monotonic clock with its own using
a midpoint round-trip estimate before merging both event streams into one JSON
file.

## Scope and limitations

- Per-sample timing is available when PPO or GRPO uses the native asynchronous
  rollout path. Synchronous batched and NeMo Gym rollout backends still appear
  accurately as driver-side generation spans, but they do not expose
  independent native sample timing.
- The driver keeps trace events in memory until shutdown. Prefer short profiling
  runs and representative batch sizes when sample-level tracing is enabled.
- Graceful completion and handled training failures write the trace. A hard
  process kill, node failure, or Slurm `SIGKILL` can prevent the final file from
  being written.
- Trace collection and file-writing failures are warnings and do not change PPO
  training behavior.
