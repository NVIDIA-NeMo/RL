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

## Why `training_step_end_time_ms` is its own timestamp

The rules score a run at the passing checkpoint's *weight-update* time and
explicitly exclude checkpoint-write time from the score. That instant is when
`policy.train()` returns in the training loop, so the loop stamps
`grpo_save_state["training_step_end_time_ms"]` right there, and the checkpoint
carries it in `training_info.json`.

It is deliberately not the timestamp of the step's `tracked_stats`
POINT_IN_TIME mllog event. That event is emitted at the step's logging call at
the end of the loop iteration — after the refit drain and weight sync, the
checkpoint write itself, and the per-step jsonl dump. Measured on the spec
run's step 18 (the gap between its weight update and its event was 285.2s):
230.7s rollout drain before refit (`exposed_generation`), 13.9s refit
(`weight_sync`), 34.2s checkpoint write (`timing/train/checkpointing`), ~6s of
metrics reduction and jsonl dump. Reusing the event time would charge the
checkpoint-write time the rule excludes; the event could move before the
checkpoint block, but it would still sit after the ~245s refit tail and would
still not be the weight-update instant — and the evaluator, a separate later
process, needs the value stored with the checkpoint regardless. The mllog
timestamps are observability artifacts tied to logging call sites;
`training_info.json` is written atomically with the checkpoint, so the
timestamp travels with the weights it describes.

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

## Full-spec run (oci-jhb, qualified 64-node shape)

Job 367663 (exit 0): GBS 256 (16x16), pass@4, `VAL_START_AT=18`,
`MAX_STEPS=19`, target 0.69. Training ran 19 steps at full capacity (no inline
validation), wrote `step_18` and `step_19` (741G weights-only each). The
deferred phase restored step 18, refit vLLM, and measured pass@4 = **0.7410**
(>= 0.69) on the 251-task validation set, so step 19 was never evaluated.

```
/results file: (oci-jhb)
/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/jpiotrowski/offline-eval-results/qwen35-deferred-spec3-260913134041/260913135524191054819_1_mllog.log

run_start      t=1789333589530
block_start    step=0 samples=4864
tracked_stats  steps=1..19 (train + timing each)
block_stop     step=19 samples=4864 t=1789340490400  <- step-19 weight update
eval_start     step=18 samples=4608 t=1789341041466  <- wall time, after training
eval_accuracy  value=0.7410358786582947 samples=4608
eval_stop      step=18
run_stop       status=success samples=4608 t=1789340105384  <- step-18 weight update
```

4608 = 256 * ceil(2.5 + 3840/256) is exactly the rules formula's first
evaluation sample, and the 6.1.0 compliance checker (logging pin 23787ba4)
passes the log's structure end to end; the only failing check is
`eval_samples == 256` versus the qualified 251-task set, which is the pending
upstream checker update (mlcommons/logging PR 475), not an implementation
deviation. Measured score (run_start -> run_stop): 108.6 minutes.

## Review outcomes (grok-4.6 review, addressed in-tree)

- **Retention**: `keep_top_k=None` in deferred mode (the earlier
  `max(..., 2)` floor could prune required endpoints), and the evaluator
  rejects a checkpoint series that does not start at `val_start_at` or has
  gaps, instead of scoring a partial window.
- **Inline-mode timing**: with the held-eval fix, a target-reaching inline
  `run_stop` is emitted when the step's logging completes, i.e. after the
  (non-default) checkpoint write rather than before it. This matches the
  accepted optimized fix (fc0426345d); the qualified inline recipe does not
  checkpoint, and the residual delay is seconds.
- **Timestamp placement**: `training_step_end_time_ms` is stamped at the end
  of `policy.train()` — the weight update itself. The refit that follows
  publishes the weights to vLLM but does not update them; the rules exclude
  checkpoint-write time, not refit time.
- **`save_period=val_start_at`**: intentional; it suppresses the recipe's
  periodic pre-window saves while the every-step-from-`val_start_at` rule in
  the async loop covers the window. (`save_period=1` would checkpoint the
  whole run.)
- Dataloader/replay-buffer state stays in the deferred checkpoints: native
  contents keep them resumable training checkpoints, and the extra write is
  bounded (one or two window saves).
- The held-eval logger machinery is the requested last-step-mllog fix (the
  reference dropped the final step's train stats on target hit), not
  offline-eval machinery; offline eval itself never triggers it
  (`val_period=0`).

## Design alternatives considered

**Async checkpoint saving — possible, not worth it.** Saves on this stack are
synchronous: `_create_checkpoint_config` hardcodes `async_save=False` and the
recipe's `megatron_cfg.checkpoint.async_strategy` is unwired; bridge async
saves would additionally need `GlobalState.initialize_async_checkpoint_worker()`
(never called here). Completeness is published by the `tmp_step_N -> step_N`
rename after `policy.save_checkpoint` returns, so async saving would require
delaying the rename until the async finalize callbacks run, plus a blocking
drain of the final save before the policy actors shut down (worker `shutdown`
does not finalize async saves). The only score-relevant win is overlapping the
step-H write with step-H+1 training (one save in the recommended H/H+1 stop);
the final checkpoint's write time is already excluded from the score by the
backdated `run_stop`. Not worth the shared-checkpoint-path blast radius for
this flow.

**Existing lifecycle hooks — reused where they fit.** The implementation rides
on: the stock checkpoint path (`init_tmp_checkpoint`/`finalize_checkpoint`),
`setup()`'s native restore (the single-symlink checkpoint view, no custom
loader), the existing `validate()`, the trainer epilogue's actor shutdown,
`mlperf_logger.finalize()` (gated by `defer_run_stop`), and run.sub's
per-experiment subshell for the second driver. Hooks deliberately not reused:
inline `val_at_end` validation (on the training clock, does not consume
checkpoints), `checkpoint_must_save_by`/`TimeoutChecker` (wall-clock, wrong
dimension), and top-k pruning (violates the every-step window). The genuinely
new machinery is minimal: the config switch, the per-checkpoint weight-update
timestamp, the every-step save predicate, the second-process evaluator, and
cluster/PG release between phases (no existing hook releases
`RayVirtualCluster`s after GRPO).
