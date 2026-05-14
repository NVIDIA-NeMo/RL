# OPD Pre-Generated Teacher Mixed Generation Work Plan

Status: Active work plan

Related:

- [OPD Pre-Generated Teacher and Mixed Generation Plan](opd-pregen-mixed-generation-plan.md)
- [OPD Pre-Generated Teacher and Mixed Generation Implementation Plan](opd-pregen-mixed-generation-implementation-plan.md)

## Objective

Deliver and validate mixed QAT OPD for Nano3 using pre-generated teacher
rollouts plus live student rollouts.

Target real experiment shape:

- Teacher generation sampling: `temperature=0.6`, `top_p=0.95`, `top_k=null`
- OPD step shape: `num_prompts_per_step=128`
- Per prompt: `teacher_generations_per_prompt=1`,
  `student_generations_per_prompt=3`
- Total generations per prompt per step: 4
- Real teacher generation target: enough teacher records for 1000 OPD steps
  (`1000 * 128 * 1 = 128000` teacher rollout records)

This work plan is the current execution plan and intentionally supersedes the
earlier parity-only `temperature=1`, `top_p=1` first-run suggestion in older
drafts. `T=1`, `top_p=1` remains a diagnostic baseline, but the first mixed
execution follows the current promising Nano3 `T=0.6`, `top_p=0.95` setup.

## Phase 1: Implement and Review

Implement the feature in the order from the implementation plan:

1. Serialization utilities and round-trip unit tests.
2. Generation-only NeMo-Gym writer for OPD-ready `final_batch` records.
3. Teacher rollout store and deterministic selection.
4. Mixed batch assembly with post-mix `STREAM_METADATA_KEYS`.
5. Metadata seam tests, including manifest build, conservation checks, and fake
   teacher-annotation alignment.
6. Disabled-by-default integration into `distillation_train()`.
7. Source count and source length metrics.
8. Optional source-specific SFT masking only after KL-only mixed mode is stable.

Use multiple agents during implementation:

- Coder agents own disjoint files or phases.
- Reviewer agents review each substantial patch before moving to Slurm smoke.
- At minimum, run one architecture/data-flow review and one test/integration
  review after the code path is wired.

Required local verification before Slurm:

- Unit tests for serialization, loader, selection, and mixer.
- Disabled-mode regression test.
- 3:1, 4:0, and 0:4 source-layout tests.
- Manifest/conservation/fake-teacher-annotation alignment test.
- Dataset identity tests:
  - generation writer records `dataset_namespace` and namespaced `prompt_uid`
  - training reconstructs the same namespace from the active data config
  - raw `dataset_index` collisions across namespaces do not collide
  - rollout-file namespace mismatch fails before rollout or training
- Negative and resume tests:
  - corrupted or partial JSONL records fail with actionable errors
  - insufficient teacher records for the configured schedule fails before
    training
  - duplicate `(prompt_uid, teacher_generation_id)` records are rejected
  - deterministic selection is unchanged after index rebuild from permuted file
    order
  - checkpoint/resume keeps the same teacher selection for the same step/seed
- Control-flow tests:
  - `mixed_generation.enabled=false`
  - student-only 4:0
  - teacher-only 0:4, asserting training-rollout generation prep/call/finish is
    not invoked

Run these tests in the same container/runtime environment used by OPD jobs when
the test requires Ray, NeMo-Gym, vLLM, or Megatron dependencies.

### Phase 1A: Writer Contract Smoke

Immediately after the generation-only writer patch lands, before downstream
store/mixer/training integration continues, run a tiny write/load smoke.

Required checks:

- A tiny teacher JSONL is generated through `run_async_nemo_gym_rollout()`.
- `teacher_generations_per_prompt=1` and `teacher_generations_per_prompt=4`
  both produce the expected per-prompt record counts.
- `prompt_uid`, `dataset_namespace`, `teacher_generation_id`, and token ids are
  present and valid.
- The generated file loads through `TeacherRolloutStore`.
- Rebuilding the store index from a permuted copy selects the same records for
  a fixed `(prompt_uid, step, seed)`.

### Phase 1B: Local End-to-End Gate

Before any Slurm smoke, run a local or mocked end-to-end integration using a
tiny fixture teacher JSONL.

Pass criteria:

- The mixed batch contains exact per-prompt 3 student + 1 teacher composition.
- Source counts sum exactly to the expected global source counts.
- `sft_weight=0` KL-only path runs at least 2 training steps with finite `loss`
  and `kl_loss`.
- Disabled mode produces baseline-equivalent behavior.
- Required metrics are emitted with non-null values:
  - `loss`
  - `kl_loss`
  - `rollout/source/student/count`
  - `rollout/source/teacher/count`
  - `rollout/source/student/gen_tokens_mean`
  - `rollout/source/teacher/gen_tokens_mean`
  - `rollout/source/student/capped_ratio`
  - `rollout/source/teacher/capped_ratio`
  - `rollout/source/student/reward_mean`
  - `rollout/source/teacher/reward_mean`
  - `opd/data/driver_rss_bytes`

## Phase 2: Smoke Test

Run a small end-to-end smoke in two parts.

### 2A: Small Teacher Generation Smoke

Use the generation-only entry point to generate a small OPD-ready teacher rollout
file.

Smoke settings:

- `temperature=0.6`
- `top_p=0.95`
- `top_k=null`
- `teacher_generations_per_prompt=1`
- small prompt count, enough for at least 3 mixed OPD smoke steps:
  `smoke_num_records >= smoke_num_steps * smoke_num_prompts_per_step`
- short Slurm time, around 30 minutes or less

Checks:

- JSONL exists and has the expected number of records.
- Record count equals `smoke_num_prompts * teacher_generations_per_prompt`.
- Every `(prompt_uid, teacher_generation_id)` pair is unique.
- `dataset_namespace` matches the active smoke data config.
- Records load through `TeacherRolloutStore`.
- Token ids round-trip exactly.
- `length`, `truncated`, `total_reward`, and assistant-token lengths are sane.
- Sampling metadata matches the smoke config.
- Re-running/resuming the output JSONL does not duplicate complete records and safely
  handles a partial final line.

### 2B: Small Mixed QAT OPD Smoke

Run a short mixed OPD smoke using the generated teacher rollout file.

Smoke settings:

- `student_generations_per_prompt=3`
- `teacher_generations_per_prompt=1`
- `num_prompts_per_step=128` if resource/time allows; otherwise use the nearest
  smaller shape that still exercises mixed batch assembly
- KL-only first: `sft_weight=0`
- reverse KL
- generation sampling: `temperature=0.6`, `top_p=0.95`, `top_k=null`
- short Slurm time

Monitor:

- Startup config: source counts must be 3 student and 1 teacher per prompt.
- `rollout/source/student/count` and `rollout/source/teacher/count`.
- Generation lengths and capped ratio by source.
- Reward range by source.
- KL loss and total loss for the first few steps.
- Driver RSS and OPD data memory metrics.
- Step time versus the current gen4 OPD baseline.

Pass criteria:

- No driver OOM or obvious memory growth across steps.
- `loss` and `kl_loss` are finite for at least the first 3 training steps.
- Global source counts equal `3 * prompt_count` student and `1 * prompt_count`
  teacher on every checked step.
- Every prompt has exactly 3 student and 1 teacher samples.
- No prompt receives 0 or more than 1 teacher sample in 3:1 mode.
- Source counts sum exactly to the global source counts.
- All required metrics from Phase 1B are emitted with non-null values.
- Driver RSS does not grow by more than 20% or 20 GiB, whichever is larger,
  across the first 3 training steps.
- Median step time after the first step is not more than 2x the current gen4
  smoke baseline unless logs identify a one-time startup or validation cost.
- Initial KL is compared against all-student gen4 and teacher-context smoke
  baselines and recorded; unexplained NaN/Inf or order-of-magnitude jumps block
  progression.
- Source counts stay fixed across steps.
- Reward mean, generation length mean, and capped ratio by source are recorded.
  Block progression if capped ratio is more than +0.15 absolute above the
  corresponding source baseline or if mean generation length is more than 2x
  the corresponding smoke baseline without a known sampling/config reason.

Only proceed if both smoke phases pass.

Write a smoke summary artifact before moving on. It must include the exact job
script or command, resolved config, teacher rollout file path, source-count
table, first-step and last-step metrics, memory/step-time summary, and the
baseline values used for comparison.

## Phase 3: Real Teacher Generation Job

Submit a real teacher generation job using the validated generation-only entry
point.

Generation target:

- `temperature=0.6`
- `top_p=0.95`
- `top_k=null`
- `teacher_generations_per_prompt=1`
- target OPD coverage: 1000 steps
- `num_prompts_per_step=128`
- total target records: 128000

Job requirements:

- Use a self-contained `sbatch` script.
- Container workload starts with `srun`.
- Put Pyxis container args on `srun`, including `--no-container-mount-home`.
- Do not mount `/home`.
- The script hardcodes:
  - container image path/version
  - repo mount and container workdir, for example repo mounted at
    `/opt/nemo-rl`
  - model, data, output, cache, and log paths
  - `HF_HOME`/HF cache paths under shared storage
  - all generation parameters and the output JSONL path
- `srun` carries the complete Pyxis contract:
  - `--container-image`
  - `--container-mounts`
  - `--container-workdir`
  - `--no-container-mount-home`
- Store output JSONL, sentinels, logs, and metadata under shared project/data
  paths.
- The script must hardcode the parameters above; do not submit with external
  env overrides.
- Use collision-safe resumability:
  - the generation job writes one explicit output JSONL and a `.done` sentinel
  - the active writer uses a `.inprogress` marker containing `SLURM_JOB_ID`
  - resume truncates any partial final line before continuing
  - training reads only a finalized JSONL with its `.done` sentinel, never a file
    with a live `.inprogress` marker
  - sharded generation is deferred until a separate merge/finalize tool exists

Completion checks:

- The output JSONL and its `.done` sentinel exist.
- The rollout file contains exactly 128000 valid teacher records.
- Unique `(prompt_uid, teacher_generation_id)` count is exactly 128000.
- No duplicate rows exist after merge.
- Every prompt needed by the planned 1000-step schedule has exactly one teacher
  record.
- Rebuilding the index from permuted file order gives the same
  `select_for_step(..., step, seed)` output for sampled validation steps.
- Sampling metadata is uniform and matches `T=0.6`, `top_p=0.95`, `top_k=null`.
- A loader/index build and namespace validation succeed before launching real
  OPD training.

## Phase 4: Real Mixed QAT OPD Job

Submit the real mixed OPD job using the generated teacher data.

Training shape:

- `num_prompts_per_step=128`
- `student_generations_per_prompt=3`
- `teacher_generations_per_prompt=1`
- `num_generations_per_prompt=4`
- teacher rollout path: output from Phase 3
- KL-only first unless explicitly switching to source-aware SFT later
- generation sampling for live student rollout: `temperature=0.6`,
  `top_p=0.95`, `top_k=null`

Job requirements:

- Use a self-contained `sbatch` script.
- Use `srun` with Pyxis container args and `--no-container-mount-home`.
- Do not mount `/home`.
- The script hardcodes:
  - container image path/version
  - repo, model, data, teacher-rollout, checkpoint, cache, and log paths
  - container workdir and mount map
  - all OPD source counts, batch sizes, sampling config, validation cadence, and
    checkpoint cadence
- Use singleton follow-on jobs for continuous resumable training.
- Use the same job name for follow-on singleton jobs and distinct output/checkpoint
  directory for this experiment.
- Before submitting a follow-on, check previous scheduler state, latest
  checkpoint, and log freshness.
- Keep the job name and output directory distinct from previous OPD experiments.
- Hardcode all experimental parameters in the script.

Monitor until stable:

- Scheduler state and log freshness:
  - check `squeue` every 10-15 minutes until the first training steps finish
  - use `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,Start,End -P`
    for terminal jobs
  - treat no new driver log for more than 30 minutes during active training as
    a debug trigger unless the scheduler shows the job is pending or launching
- First successful teacher rollout load/index.
- First mixed batch source counts.
- First 5 successful training steps after the first mixed batch.
- KL loss, total loss, reward, generation length, capped ratio.
- Driver RSS and OPD data memory metrics.
- Checkpoint save and resume behavior if a follow-on singleton starts.

Success criteria:

- The job reaches at least 5 real training steps without memory failure.
- Source counts and metadata remain consistent.
- `loss` and `kl_loss` are finite for those steps.
- Per-prompt source composition remains exactly 3 student and 1 teacher.
- Driver RSS growth across the first 5 training steps is at most 20% or 20 GiB,
  whichever is larger.
- Median training step time after warmup is not more than 1.5x the mixed smoke
  median or 2x the prior gen4 OPD baseline without an identified cause.
- Reward mean, generation length mean, and capped ratio are within the smoke
  gates defined above.
- A checkpoint is written, and if a follow-on singleton starts, it resumes from
  that checkpoint without changing the step's deterministic teacher selection.

## Stop Conditions

Stop and debug before the next phase if any of these occur:

- Teacher rollout records fail round-trip validation.
- Mixed metadata fails manifest/conservation/fake-annotation alignment.
- Any required metric is missing or null in smoke.
- `loss` or `kl_loss` is NaN/Inf on any checked smoke or first-real-job step.
- Initial KL has an unexplained order-of-magnitude jump relative to the relevant
  all-student or mixed smoke baseline.
- Source counts differ from 3 student / 1 teacher per prompt.
- Any prompt has 0 teacher samples or more than 1 teacher sample in 3:1 mode.
- Driver RSS exceeds the 20% / 20 GiB growth gate.
- Generation length mean exceeds 2x the corresponding smoke baseline without a
  known sampling/config reason.
- Capped ratio is more than +0.15 absolute above the corresponding source
  baseline.
- Step time exceeds the defined smoke or real-job regression gates.
- The job OOMs, stalls by log-freshness criteria, or scheduler state/logs
  disagree about progress.
