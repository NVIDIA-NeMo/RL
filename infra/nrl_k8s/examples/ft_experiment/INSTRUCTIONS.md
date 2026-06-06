# Fault Tolerance Experiment — Disaggregated Inference Recovery

Experiment to quantify the benefit of fault-tolerant disaggregated generation
compared to traditional single-cluster training that crashes on failure.

## Experiment Matrix

| Setup | Disaggregated | FT Recovery | Checkpoint Period | Fault Interval |
|-------|:---:|:---:|:---:|:---:|
| **Test** | Yes | Yes | 1 step | 25 min |
| **Control** | No | No | 1 step | 25 min |
| **Alt Control** | No | No | 5 steps | 25 min |

All use:
- Model: Qwen3-30B-A3B
- Algorithm: Async GRPO (lag 1, `max_trajectory_age_steps=1`)
- Training: 8 nodes x 4 GPUs = 32 GPUs (Megatron TP=1, PP=1, EP=8, DP=4)
- Generation: 8 vLLM DP shards x 1 GPU each (TP=1)
- Steps: 20
- Batch: 8 prompts x 8 generations = 64 samples/step
- Sequence length: 2048
- W&B project: `nemorl-ft-experiment`

## Run Commands

### Test (disaggregated + fault recovery)

```bash
nrl-k8s run \
  infra/nrl_k8s/examples/ft_experiment/test_recipe.yaml \
  --infra infra/nrl_k8s/examples/ft_experiment/test.gb300.infra.yaml \
  --raycluster
```

- FaultInjector kills 1 vLLM shard every 25 min, immediate recovery via router `add_shard`
- Training continues without interruption
- Expected: completes in a single run

### Control (single cluster, no fault recovery)

```bash
nrl-k8s run \
  infra/nrl_k8s/examples/ft_experiment/control_recipe.yaml \
  --infra infra/nrl_k8s/examples/ft_experiment/control.gb300.infra.yaml \
  --raycluster
```

- Fault simulation: background timer kills training driver after 25 min (SIGTERM)
- Training crashes, must restart from last checkpoint
- Checkpoints every step: loses at most 1 step + restart overhead
- **Must re-run until 20 steps are reached**

### Alternate Control (checkpoint every 5 steps)

```bash
nrl-k8s run \
  infra/nrl_k8s/examples/ft_experiment/alt_control_recipe.yaml \
  --infra infra/nrl_k8s/examples/ft_experiment/alt_control.gb300.infra.yaml \
  --raycluster
```

- Same as control but checkpoints every 5 steps
- Loses up to 5 steps on each failure + restart overhead
- Shows worst-case scenario for infrequent checkpointing

## Telemetry

All logs include structured events parseable by `tools/generate_traces.py`:

- `[STEP-START]` — Unix timestamp at each training step start
- `[VLLM-THROUGHPUT]` — Per-worker gen tokens/sec with unix timestamps (every 0.5s)
- `[FAULT-EVENT]` — Exact time of fault injection
- `[FAULT-SIM]` — Exact time of simulated infrastructure failure (control only)
- `[RECOVERY-START]` — When recovery begins (test only)
- `[RECOVERY] shard=X status=joining` — When shard starts rejoining (test only)
- `[RECOVERY] shard=X status=ready` — When shard is ready to serve (test only)
- Timer events (`INFO:nemo_rl.utils.timer`) — Full init/train/idle timing breakdown

### Generating Chrome Traces

```bash
python tools/generate_traces.py /mnt/rl-workspace/terryk/driver_logs/<log-file>.log -o traces/
# Open traces/nemorl_trace.html in browser, or drag .json into ui.perfetto.dev
```

## Restarting Failed Runs (Control setups)

When the control/alt-control training is killed by the fault timer, re-run
the same command. Checkpointing with `save_optimizer=true` ensures the
optimizer state is preserved. The training will auto-resume from the latest
checkpoint in the checkpoint directory.

```bash
# Re-run the same command — it picks up from the last checkpoint
nrl-k8s run \
  infra/nrl_k8s/examples/ft_experiment/control_recipe.yaml \
  --infra infra/nrl_k8s/examples/ft_experiment/control.gb300.infra.yaml \
  --raycluster --replace
```

## Config Files

| File | Purpose |
|------|---------|
| `test_recipe.yaml` | Test recipe (disagg + FT, async GRPO, ckpt every 1 step) |
| `test_gen_server.yaml` | Test gen server config (8 shards, fault injection every 25min) |
| `test.gb300.infra.yaml` | Test K8s infra (2 RayClusters: train + gen) |
| `control_recipe.yaml` | Control recipe (same-cluster, async GRPO, ckpt every 1 step) |
| `control.gb300.infra.yaml` | Control K8s infra (1 RayCluster, 10 workers, 40 GPUs) |
| `alt_control_recipe.yaml` | Alt control recipe (ckpt every 5 steps) |
| `alt_control.gb300.infra.yaml` | Alt control K8s infra (same as control) |

## W&B Runs

| Setup | W&B Run Name | Link |
|-------|-------------|------|
| Test | `ft-experiment-test-disagg` | TBD |
| Control | `ft-experiment-control-no-ft` | TBD |
| Alt Control | `ft-experiment-alt-control-ckpt5` | TBD |

## Key Metrics to Compare

1. **Total wall time to 20 steps** — test should be significantly faster
2. **Steps lost per failure** — control: ~1 step, alt-control: up to 5 steps
3. **Restart overhead** — control/alt-control: time to restart (cluster setup + checkpoint load + buffer warmup)
4. **Throughput during recovery** — test: visible dip then recovery in `[VLLM-THROUGHPUT]` logs
5. **Number of restarts needed** — control/alt-control vs test (zero restarts)

## Branch

All configs and telemetry on branch: `tk/clean/hemil/fault-tolerant-generation`
