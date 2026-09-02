# Generation Fault Tolerance (SingleController)

A generation shard can die or wedge mid-run. Without help, the next weight sync blocks
forever inside NCCL — a collective needs every rank, and it does not care that you stopped
sending the dead one traffic. This feature notices the loss, rebuilds the refit
communicator over whoever is left, and keeps training.

It is **off by default**. Set `async_rl.generation_fleet_health.enabled: true` to turn it on.

## What is supported

Know these limits before enabling it.

**Weight-sync transports** — only the two that own an NCCL world can rebuild one:

| `policy.generation.refit_transport` | Recovers a lost shard |
|---|---|
| `collective` (packed broadcast) | ✅ full |
| `nccl_reshard` | ⚠️ only between syncs — any fault mid-transfer ends the run, fast and with a named cause. See [A mid-sync fault on nccl_reshard](#a-mid-sync-fault-on-nccl_reshard-is-different) |
| colocated IPC, HTTP, checkpoint-engine, `vllm_remote_sparse` | ❌ no communicator to rebuild |

**Generation backends** — only vLLM implements the fleet-health hooks:

| `policy.generation.backend` | Supported |
|---|---|
| `vllm` | ✅ |
| `sglang` | ❌ refuses at setup with `NotImplementedError` |
| Megatron, TensorRT-LLM, Dynamo | ❌ not accepted by SingleController |

Trainer ranks are **never** excluded. Losing one changes the parallelism the model was
sharded for, so there is nothing to recover onto. This feature is about generation shards.

## Who watches what

One ledger, `GenerationFleetHealth`, holds a state per shard. It lives in the
SingleController process, and no GPU worker ever writes to it.

![The fleet-health ledger: what writes it, what reads it, and the two states that surprise people](../assets/sc-fault-tolerance-ledger.png)

## What happens when a shard fails

Two things decide the outcome: what the ledger already thought of the shard when the sync
began, and how the sync then went.

![Shard state before the sync crossed with how the sync went, showing which combinations recover and which end the run](../assets/sc-fault-tolerance-grid.png)

| Shard state | Refit | Outcome | Why |
|---|---|---|---|
| `HEALTHY` | succeeds | ✅ continues | The ordinary step. |
| `HEALTHY` | hangs | 🛑 run ends | Every process is alive and nothing was suspect, so there is nobody to drop. Guessing would condemn a healthy shard and leave the real culprit in the group. |
| `HEALTHY` | crashes | ⚠️ depends | The crash names the shard: mark it dead, rebuild without it, retry once. Works on `collective`; ends the run on `nccl_reshard`. |
| `SUSPECT` | succeeds | ✅ continues | A wedged engine can still do NCCL. It keeps failing generations and reaches `DEAD` on its own. |
| `SUSPECT` | hangs | ⚠️ depends | The one suspect is condemned, the group shrinks, retry once. Works on `collective`; ends the run on `nccl_reshard`. |
| `SUSPECT` | crashes | ⚠️ depends | The dead process is visible, so no attribution is needed. Same split: works on `collective`; ends the run on `nccl_reshard`. |
| `DEAD` | any | ✅ continues | Already excluded before the collective started, so it cannot affect this refit at all. |

Recovery is **rebuild, then retry once** — and the rebuild needs a shard to drop. That is
why a crash can recover and a hang only recovers when the ledger already held a suspect.
Retrying without shrinking would rebuild over the same fleet and hang on the same silent
rank. On `nccl_reshard` there is a second requirement: the fault must land outside the
bulk transfer — see below.

A second failure during the retry ends the run: at that point it is a fault, not a
membership problem.

### A mid-sync fault on `nccl_reshard` is different

On `nccl_reshard`, any fault while the bulk transfer is in flight — a crash *or* a wedge —
cannot be recovered in-process. Aborting a communicator does not retire CUDA work already
queued on a stream, and on that transport the orphaned kernels sit on the **trainers'**
devices — so nothing done to the failed shard retires them, and no communicator can be
rebuilt on a device in that state. The run detects this (the abort carries a dedicated
marker) and ends in seconds with a message naming the limit, rather than stalling or
wedging in a rebuild that cannot complete.

A fault that lands **between** syncs is fine on both transports: the shard is dropped
before the next collective starts, which is the `DEAD` row above. On `collective` the
mid-sync cases recover too.

## Configuration

```yaml
async_rl:
  generation_fleet_health:
    enabled: true                 # off by default
    probe_interval_s: 5.0         # how often each shard is probed
    probe_timeout_s: 2.0          # must be < probe_interval_s
    unhealthy_threshold: 3        # consecutive failures before a shard is condemned
    min_healthy_shards: 1         # below this the run stops
    refit_timeout_s: 300.0        # deadline for one refit collective
```

`refit_timeout_s` is what makes recovery possible at all when a shard fails **during** a
refit: it is the only thing that can release ranks already blocked inside NCCL, and a
blocked worker cannot service the rebuild call either. Setting it to `null` disarms the
watchdog and leaves the refit path unchanged from before this feature existed. A healthy
refit takes seconds, so the default leaves a wide margin.

## Related

- [Single-Controller (Async GRPO)](../guides/single-controller.md) — the surrounding execution model
- [Weight Refit](../guides/refit.md) — choosing a refit transport
