# Fault-Tolerant Generation

Fault-tolerant generation automatically detects and recovers from crashed or
unresponsive vLLM DP shards without stopping training. When a shard dies, the
`GenerationRouter` removes it from the rotation, shrinks the NCCL world, and
spawns a replacement that rejoins on the next weight refit.

Fault tolerance applies only to **non-colocated** vLLM generation
(`colocated.enabled: false`). It is supported for GRPO and async GRPO; PPO
requires colocated generation and is not supported.

## Enable

```yaml
policy:
  generation:
    colocated:
      enabled: false
      resources:
        num_nodes: 2        # dedicated gen nodes
        gpus_per_node: 8
    vllm_cfg:
      async_engine: true    # recommended; see note below
    fault_tolerance:
      enabled: true
      auto_recover: true    # spawn a replacement shard automatically
```

`async_engine: true` is strongly recommended. With `async_engine: false` the
sync generation path dispatches to all live shards including newly-joined ones
that still hold random weights, which can corrupt gradient signal. The async
path skips joining shards until their first weight broadcast completes.

## How Recovery Works

1. **Detection** — a background health poller probes each shard's HTTP
   `/openapi.json` endpoint (or Ray actor liveness when no HTTP server is
   exposed). After `failure_threshold` consecutive failures the shard is
   cordoned and its actors are killed.
2. **Shrink** — the dead shard is removed from the shard table and the next
   refit rendezvouses at the reduced world size.
3. **Replacement** — `auto_recover: true` spawns a new vLLM worker on the
   same SLURM node and placement group as the dead shard. The replacement reuses
   the dead shard's `VLLM_CACHE_ROOT` for a warm `torch.compile` cache.
4. **Rejoin** — once the replacement passes health probes and a warm-up age
   gate, `ensure_collective_synced` grows the NCCL world and the next weight
   refit pushes fresh weights to the shard, promoting it from `joining` to
   `ready`.

## Configuration Reference

All keys live under `policy.generation.fault_tolerance`.

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable fault-tolerant generation. |
| `auto_recover` | `true` | Spawn a replacement shard when one is lost. No-op when `enabled` is `false`. |

The following environment variables tune recovery timing:

| Variable | Default | Description |
|---|---|---|
| `NRL_JOINABLE_MIN_AGE_S` | `90` | Seconds a new shard must be alive before it counts as joinable. Prevents a cold route from timing out the NCCL rendezvous. Proven shards (previously rejoined successfully) bypass this gate. |
| `NRL_REJOIN_DEBOUNCE_S` | `45` | How long the joinable set must be stable before the trainer grows the NCCL world. Coalesces replacements that arrive close together. |
| `NRL_COLLECTIVE_SYNC_MAX_ATTEMPTS` | `5` | Maximum retries for `init_collective` before giving up on a refit. |

## Constraints

- Non-colocated vLLM only. SGLang and colocated inference are not supported.
- GRPO and async GRPO only. PPO requires colocated generation.
- The synchronous `generate()` path (`async_engine: false`) dispatches to all
  live shards including joining ones. Use `async_engine: true` to avoid stale
  weight generation during recovery.
- Distillation: the plumbing is wired but has not been validated end-to-end.
