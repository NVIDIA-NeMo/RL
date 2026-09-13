# Deferred (offline) evaluation — implementation note

Ports the offline-evaluation rules for qwen35_397b_grpo
(mlcommons/training_policies PR 596) into this reference stack, as a minimal
extension of the existing launcher and MLPerf logger. Derived from
dl/mlperf/optimized MR 8012, adapted to the reference's direct logger calls.

## How it works

Training side (`DEFERRED_OFFLINE_EVAL=1`, matched by
`grpo.deferred_evaluation.enabled` in the recipe):

- `configure_deferred_evaluation()` (in
  `nemo_rl/algorithms/mlperf_grpo_deferred.py`) rewrites the run config before
  setup: inline validation off (`val_period=0`, `val_at_start/end=False`),
  native checkpointing on, weights only (`save_optimizer=False`), and the
  MLPerf logger gets `defer_run_stop`.
- The async loop saves a checkpoint after **every** step from
  `grpo.val_start_at` through the stop step (rule: "a checkpoint after every
  individual step until training stops"), and records
  `training_step_end_time_ms` (end of the step's `policy.train`) into each
  checkpoint's `training_info.json`.
- Training runs at full capacity through the final step; nothing about rollout
  collection, trajectory age, or the optimizer changes.

Handoff and evaluation (same allocation, same Ray cluster):

- The trainer epilogue stops its actors as usual; the driver then shuts down
  the two `RayVirtualCluster`s and waits for the GPUs to be released
  (`teardown_run_resources`). The MLPerf logger closes the train block with
  the final weight-update timestamp and emits no `run_stop`
  (`defer_run_stop`).
- `run.sub` runs a second foreground step on the head node:
  `run_and_time.sh --deferred-eval` → `python -m
  nemo_rl.algorithms.mlperf_grpo_deferred --checkpoint-root /checkpoint/<i>
  --log-dir /logs/deferred-evaluation`.
- For each checkpoint in step order: restore the MCore policy without
  optimizer state (`policy_factory=policy_without_optimizer`), refit vLLM,
  invalidate the KV cache, and run the recipe's validation (`validate()`,
  pass@k per `num_val_generations_per_prompt`). Stop at the first checkpoint
  meeting `grpo.deferred_evaluation.threshold` (the run's
  `logger.mlperf.target_accuracy`).
- The evaluator appends ordinary `eval_start`/`eval_accuracy`/`eval_stop`
  events to the run's mllog and emits the single `run_stop` with
  `time_ms` backdated to the passing checkpoint's weight-update timestamp;
  if none passes, `status=aborted` at the final checkpoint's timestamp.
  Operational failures fail the job (no terminal event is fabricated).

The checkpoint view handed to `setup()` is a single-symlink directory, so the
stock restore path (`get_latest_checkpoint_path` → `get_resume_paths`) loads
the requested endpoint unchanged. MCore saves on this stack are synchronous
(`async_save=False`), so the final checkpoint is fully written before the
trainer exits.

Also fixed here (ported from optimized fc0426345d): the trainers log a step's
validation before its train metrics, and a target-reaching eval used to emit
`run_stop` first and drop the final step's `tracked_stats`. Evaluations
started while a train block is open are now held and flushed after the step's
train metrics land (order: train `tracked_stats`, `BLOCK_STOP`, `EVAL_START`,
validation-time `tracked_stats`, `EVAL_ACCURACY`, `EVAL_STOP`, `RUN_STOP`);
`BLOCK_STOP`/`EVAL_START` are backdated by the validation duration.

## Verification (oci-jhb, 16-node dev shape)

Test scenario: 3 steps, validation from step 2, pass@1
(`num_val_generations_per_prompt=1`), GBS 64 (8x8), threshold 0.0, the
optimized oci-jhb datafiles. Deferred run (Slurm job 317727, exit 0):
checkpoints at steps 2 and 3, each with a valid weight-update timestamp;
step 2 evaluated at pass@1 = 0.3108 >= 0.0; step 3 never evaluated. mllog
(both phases append to one file):

```
/results file: (oci-jhb)
/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/jpiotrowski/offline-eval-results/qwen35-deferred-t3-260911133623/260911133648331398729_1_mllog.log

run_start      t=1789159556257
block_start    step=0 samples=192
tracked_stats  step=1..3 (train + timing)
block_stop     step=3 samples=192 t=1789160694963   <- step-3 weight update
eval_start     step=2 samples=128 t=1789161315374   <- wall time, after training
eval_accuracy  value=0.31075698137283325 samples=128
eval_stop      step=2
run_stop       status=success samples=128 t=1789160364612  <- step-2 weight update
```

The companion inline-validation run (job 318646, exit 0) shows the last-step
fix on the real trainer: the target-reaching step's train `tracked_stats`
precede its backdated `BLOCK_STOP`/`EVAL_START`, and `run_stop` is last
(`.../qwen35-inline-t4-260911143424/260911143448170274894_1_mllog.log`).

A 6.1.0 compliance-checker dry run over the deferred console log reports no
structural failures; the only failures are the deliberate dev-shape deviations
(GBS 64 vs the 256 formula, LR/clip defaults, 0.69 target, eval_samples 251
vs 256 at this checker pin).

Unit tests: `tests/unit/algorithms/test_mlperf_grpo_deferred.py` and
`test_mlperf_grpo_logging.py` (15 tests, in-container pass).
