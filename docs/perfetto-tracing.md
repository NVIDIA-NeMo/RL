# Profile PPO with Perfetto

NeMo RL can write an opt-in CPU workload trace for synchronous and asynchronous
PPO. The output uses the Chrome JSON trace-event format and can be opened
directly in [Perfetto](https://ui.perfetto.dev/).

This trace complements Nsight Systems. Use it to understand the end-to-end PPO
driver schedule, rollout concurrency, replay-buffer stalls, critic and actor PPO
epochs, refit bubbles, validation, and checkpointing. Use Nsight Systems when
you need GPU kernels, CUDA streams, or NCCL detail.

## Enable tracing

Set both environment variables on the PPO driver:

```bash
export NEMORL_TRACE_ENABLED=1
export NEMORL_TRACE_FILE=/shared/path/ppo_trace.json

uv run examples/run_ppo.py --config <ppo-recipe.yaml>
```

`NEMORL_TRACE_FILE` defaults to `nemo_rl_perfetto_trace.json` in the driver's
working directory. For Slurm runs, choose a path on shared persistent storage:

```bash
NEMORL_TRACE_ENABLED=1 \
NEMORL_TRACE_FILE=/lustre/my-run/ppo_trace.json \
COMMAND="uv run examples/run_ppo.py --config <ppo-recipe.yaml>" \
CONTAINER=<container.sqsh> \
MOUNTS=/lustre:/lustre \
sbatch --time=00:20:00 ray.sub
```

Tracing is disabled by default. When disabled, the tracer stores no events and
the existing timer behavior is unchanged.

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
