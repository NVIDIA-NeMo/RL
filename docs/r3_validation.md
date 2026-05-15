# R3 Validation

## How To Use

Enable the default-off R3 trace:

```bash
export NRL_R3_TRACE=1
export NRL_R3_TRACE_STEPS=1
export NRL_R3_TRACE_SAMPLES=2
export NRL_R3_TRACE_MICROBATCHES=2
export NRL_R3_TRACE_DIR=logs/${RUN_TAG}/r3_trace
```

For the stronger check that verifies Megatron actually replays the installed
expert ids inside `RouterReplay.get_replay_topk(...)`, also set:

```bash
export NRL_R3_TRACE_VERIFY_FORWARD=1
```

Run an R3-enabled TQ recipe, for example:

```bash
+policy.router_replay.enabled=true
+data_plane.enabled=true
+data_plane.impl=transfer_queue
+data_plane.backend=simple
policy.sequence_packing.enabled=true
policy.megatron_cfg.context_parallel_size=2
```

Then validate the trace:

```bash
python tools/check_r3_trace.py "${NRL_R3_TRACE_DIR}" \
  --require-forward-verify \
  --require-cp-identity
```

For a run without `NRL_R3_TRACE_VERIFY_FORWARD=1`, omit the two `--require-*`
flags.

Expected final line:

```text
PASS: producer routed_experts matched TQ fetches, and replay was set.
```

## Tested Setup

We tested the debug path with a 1-step Moonlight MoE run.

Experiment setup:

- model: `moonshotai/Moonlight-16B-A3B-Instruct`;
- nodes: 2;
- TP=2, PP=2, CP=2, EP=2;
- sequence packing enabled;
- TQ enabled with simple backend;
- R3 enabled;
- max steps: 1;
- trace env: `NRL_R3_TRACE=1`, `NRL_R3_TRACE_VERIFY_FORWARD=1`,
  `NRL_R3_TRACE_STEPS=1`, `NRL_R3_TRACE_SAMPLES=2`,
  `NRL_R3_TRACE_MICROBATCHES=2`.

Observed run:

- Slurm job: `11771077`;
- W&B run: `https://wandb.ai/joc/grpo-r3-debug-0513/runs/j83txp4m`;
- result: completed `Step 1/1`;
- `Generation KL Error: 0.0002`;
- checker result: pass;
- 16 rollout samples checked;
- 26 unique MoE layers checked for both `prev-logprob` and `train`;
- 1,040 Megatron `RouterReplay.get_replay_topk(...)` verifier records;
- 0 replayed-topk mismatches;
- 48 CP identity records;
- 84,112 CP-local token rows verified.
