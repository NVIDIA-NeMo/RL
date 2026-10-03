# Temporary reference-placement measurements

This draft sits above the reference-placement PR. It is validation tooling to
discard after collecting results, rather than a production monitoring feature.

Run in the same exclusive Ray allocation as the approved recipe:
Use an empty output directory for each run; populated directories are rejected.
Cancelling the collector stops the driver's process group before releasing samplers.

```bash
uv run --script tools/rl1610_profile.py record --output "$NRL_ARTIFACT_DIR/profile" -- \
  uv run examples/run_grpo_single_controller.py --config <approved-recipe> \
  grpo.max_num_steps=25 logger.log_dir="$NRL_TRAIN_LOG_DIR" \
  checkpointing.checkpoint_dir="$NRL_CHECKPOINT_DIR"
```

The sampler creates one zero-GPU, zero-CPU-resource Ray actor per alive node,
including the head. It queries the GCS actor table, then samples Linux processes
by PID on their actual node once per second. Each process belongs to its nearest
actor ancestor, so a Ray actor nested beneath another actor is not counted twice.
New or reused PIDs start a new CPU baseline. Actor start time rejects a reused
root PID. Missing processes, unreadable memory, and failed collection stay visible.

`actors.jsonl` records actor IDs/names/classes, job IDs, root and child processes,
CPU cores consumed over each sample interval, RSS and PSS, node OS memory totals,
sampler PSS, and collection duration. RSS sums can count shared pages repeatedly;
use PSS for actor-tree memory comparisons. PSS divides shared resident pages among
all processes mapping them, including processes outside the measured tree. Node
OS memory includes the Ray control plane, page cache, and unrelated processes;
it is not the sum of actor PSS. GB200 OS totals may also include exposed GPU memory,
so label these as OS totals rather than claiming they measure only host DRAM.
Short-lived children and memory spikes between samples can be missed.

The temporary controller hook writes unrounded step times and valid-token counts
to `steps.jsonl`. Plot and summarize the measured window:

```bash
uv run --script tools/rl1610_profile.py plot --output <profile-dir> --warmup-steps 5
```

This saves `actor-resources.png` and `summary.json` with mean, standard deviation,
minimum and maximum step times. Keep every repeated steady-state step in the report;
there is no invented pass/fail tolerance. Inspect the warm-up window before using it.
Repeat each condition and keep the policy/generation configuration, initial weights,
seed, sample budget, head mode, and image fixed. Use two compute nodes for the
baseline and three for separate reference placement on each approved GPU SKU.

Run a matching uninstrumented control to measure overhead. Report collection time
and sampler memory alongside the measurements. Profiling failures do not establish
memory savings; incomplete PSS samples must not be interpreted as zero memory.
