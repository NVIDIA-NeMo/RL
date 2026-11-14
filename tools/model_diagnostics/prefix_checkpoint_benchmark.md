# Four-node prefix checkpoint benchmark (#5176)

This branch adds measurement tools only, on PR #5176 head
`981376a3d01e1c4fc1c3313879d0c34bc3856b20`. It does not change the runtime.
Use the Gym commit pinned by that checkout, not the Gym from the older
`7865dd080` integration stack.

## What it runs

Four homogeneous nodes with four GPUs each: one driver, two independent TP=4
generation engines, and one SimpleStorage node. Driver/storage GPUs are allocated
but unused. The workload uses synthetic math prompts and real model inference,
not CPU fake generation or the full Gym agent/trainer graph.

Each engine keeps 630 captured calls outstanding, with `max_num_seqs=512`.
Two cuts are triggered at matched per-engine token-volume targets:
2,450,792 and 8,912,407 generated tokens; the second also requires 6,461,615
new staging positions. Polling and ongoing decoding can overshoot targets.
The first chunk includes prompt positions. Use actual staged positions in the
summary when comparing runs, not only the trigger target.

Each engine sends 39 disjoint inventories. Both use batches of 256 by default;
`--batch-sizes 1 256 --counterbalance` reproduces the earlier within-run
batch-size comparison. Only use that for a deliberate batch-size experiment.

## Prepare a checkout and container

```bash
git switch amahishi/pr5176-prefix-checkpoint-benchmark
git submodule update --init 3rdparty/Gym-workspace/Gym
```

Use a compatible RL/Gym image with the generation-worker venv, vLLM, TQ Simple
checkpoint support, and the pinned Gym APIs. The earlier four-node runs used
`rl-gym.81216245.sqsh` (vLLM 0.29, Ray 2.58, Torch 2.13). This new harness has
not yet been GPU-validated. A newer nightly is not automatically compatible:
the in-container preflight checks imports and the generation configuration
before starting actors. No venvs are rebuilt; version-mismatch bypass is
intentional for a benchmark-only branch. Reuse the same image for ablations.

## Submit

Supply your cluster's paths/account; all paths must be visible at the same
locations on all four nodes and inside the container. Slurm/Pyxis/Enroot and the
repository's `ray.sub` must work on the destination cluster.

```bash
export BENCH_IMAGE=/shared/images/rl-gym.sqsh
export BENCH_MODEL=/shared/models/nano35-hf
export BENCH_RESULTS=/shared/results
export BENCH_ACCOUNT=your_account
export BENCH_MOUNTS=/shared:/shared  # Also cover the checkout if outside /shared.
export BENCH_PARTITION=batch
export BENCH_QOS=short              # Optional; omit if not applicable.
export BENCH_WALLTIME=01:00:00
LAUNCH=tools/model_diagnostics/launch_prefix_checkpoint_benchmark.sh

bash "$LAUNCH" --dry-run --mode writer --router-replay
bash "$LAUNCH" --mode writer --router-replay
```

Run these separately, keeping the model, image, targets, outstanding calls,
batch sizes and max_num_seqs identical:

| Mode | Direct column encoder | Pipeline/coalescing | TQ writer actor |
| --- | --- | --- | --- |
| `--mode baseline` | Off | Off | Off |
| `--mode encoder` | On | Off | Off |
| `--mode pipeline` | On | On | Off |
| `--mode writer` (default) | On | On | On |

Repeat each mode with `--router-replay` and `--no-router-replay`.
Repeat runs to distinguish an optimization from workload/node variance.
To inspect CLI options without vLLM, Ray initialization, or Slurm submission:

```bash
uv run --no-project --python /path/to/python python \
  -m tools.model_diagnostics.prefix_checkpoint_benchmark \
  --model /shared/models/nano35-hf --output /shared/results/unused --dry-run
```

W&B is disabled unless `--project PROJECT` is supplied; optionally add
`--entity ENTITY`. The harness preserves the caller's W&B endpoint and credentials.
It does not source or copy any private environment files. Set `BENCH_PYTHON`
if the generation-worker venv has a different path in your image.

## Results and limitations

The results directory contains Slurm/Ray logs and `run/`:

- `plan.json`: workload, selected options, versions, RL and Gym commits.
- `summary.json`: HTTP cut wall time, admitted-writer drain, per-engine staged
  positions, stage wall/thread CPU time, inventory queue/preparation timings,
  cumulative native pipeline/writer counters, and TQ persistence time/bytes.
- `receipts-N.json`, `diagnostics-N.json`: per-call dispositions and raw events.
- `snapshot-N/`: TQ dumps and persisted referenced-prefix-row checks.

Inventory elapsed time includes time suspended for writes; preparation CPU
time measures only generator advancement, not other inventories while suspended.
Thread CPU is not process CPU. Summed overlapping spans are not critical-path
wall time. Native writer/pipeline counters are cumulative; subtract consecutive
rounds for per-cut values. Inspect failed-call dispositions as well as latency.

HTTP cancellation does not cancel admitted cuts; the benchmark drains those
cuts before persisting. Transport failures fail the run rather than reporting
a successful snapshot. Per-call cut failures remain visible in receipts; a
finished run is not a claim that all calls cut successfully.

This is a prefix-cut/persistence proxy, **not a restorable rollout checkpoint**.
There is no Gym prepare/commit, recovery ledger, trainer anchor or generation
fence. Generation and terminal writes continue during TQ persistence; no obsolete
prefix cleanup runs until the next round. Referenced prefix keys are checked in
the saved controller/payload, but this does not establish consistency of every
concurrent terminal row or validate restore. Use the full recipe for recovery
correctness, Gym admission queueing, trainer blocking and end-to-end latency.
