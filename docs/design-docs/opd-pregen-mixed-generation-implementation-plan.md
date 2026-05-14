# OPD Pre-Generated Teacher and Mixed Generation Implementation Plan

Status: Draft implementation plan

Related:

- [OPD Pre-Generated Teacher and Mixed Generation Plan](opd-pregen-mixed-generation-plan.md)
- [On-policy Distillation Data Pipeline](opd-data-pipeline.md)
- [OPD Data Pipeline Implementation Plan](opd-data-pipeline-implementation-plan.md)

## Objective

Implement mixed OPD generation where each training step can combine live
student NeMo-Gym rollouts with pre-generated teacher NeMo-Gym rollouts. The
implementation must preserve the current on-the-fly OPD training semantics by
entering the existing data path at the `run_async_nemo_gym_rollout().final_batch`
boundary.

Initial target experiment:

- `num_prompts_per_step=128`
- `num_generations_per_prompt=4`
- per prompt: 3 live student generations plus 1 pre-generated teacher generation
- KL-only first: `temperature=0.6`, `top_p=0.95`, `top_k=null`, `sft_weight=0`,
  reverse KL

## Non-Goals

- Do not change the disabled/default OPD path.
- Do not train directly from NeMo-Gym `full_result` trajectory logs.
- Do not add online BF16 teacher vLLM generation in the training loop for the
  first implementation.
- Do not add NeMo-Gym `top_k`; the current NeMo-Gym path rejects `top_k`.
- Do not replace online teacher top-k annotation. Pre-generated rollouts provide
  contexts/tokens, not teacher logits.

## Compatibility Contract

The durable on-disk record is a serialized single-sample form of the current
post-rollout `final_batch`.

Required JSONL row shape:

```json
{
  "schema_version": 1,
  "source": "teacher",
  "dataset_index": 0,
  "prompt_uid": "train:train.jsonl:0",
  "teacher_generation_id": 0,
  "agent_ref": {"type": "...", "name": "..."},
  "message_log": [
    {"role": "user", "content": "", "token_ids": [1, 2, 3]},
    {"role": "assistant", "content": "", "token_ids": [4, 5],
     "generation_logprobs": [-0.1, -0.2]}
  ],
  "length": 3,
  "loss_multiplier": 1.0,
  "total_reward": 0.0,
  "truncated": false,
  "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": null},
  "model": {"name_or_path": "...", "tokenizer": "..."},
  "extra_env_info": {}
}
```

Loader output must reconstruct:

```python
BatchedDataDict({
    "agent_ref": list[dict],
    "message_log": list[list[dict]],
    "length": torch.Tensor,
    "loss_multiplier": torch.Tensor,
    "total_reward": torch.Tensor,
    "truncated": torch.BoolTensor,
})
```

`message_log[*].token_ids` is canonical. `content` remains empty to match
current NeMo-Gym OPD postprocessing.

`prompt_uid` is the durable on-disk identity. It should include enough namespace
to avoid train/validation or multi-dataset collisions. `dataset_index` stores
the NeMo-RL `idx` value from the processed dataset and is the primary runtime
join key when the training config and rollout file were produced from the same
dataset namespace. The loader must validate that `prompt_uid` and
`dataset_index` are consistent for the configured dataset namespace.

The implementation should make this namespace explicit. The generation writer
records a `dataset_namespace` derived from split plus dataset identity, for
example `train:/abs/path/train.jsonl` or a stable hash of that value. Training
reconstructs the same namespace from the active train dataset config during
setup, builds runtime prompt identities from `(dataset_namespace, batch["idx"])`,
and passes those identities to the teacher rollout store. `prompt_uid` can then
be derived as `f"{dataset_namespace}:{idx}"`. Raw `dataset_index` alone is never
used across namespaces.

## Implementation Slices

### Phase 0: Contracts and Round-Trip Utilities

Goal: introduce serialization and validation helpers with no training behavior
change.

Target files:

- `nemo_rl/algorithms/distillation_mixed_generation.py` or
  `nemo_rl/algorithms/distillation_rollout_store.py`
- `tests/unit/algorithms/test_distillation_mixed_generation.py`

Tasks:

1. Define `TeacherRolloutRecord` as a typed dictionary or dataclass.
2. Add `serialize_final_batch_sample(final_batch, index, metadata)` that converts
   one `final_batch` sample into a JSON-serializable record.
3. Add `deserialize_teacher_rollout_record(record)` that validates fields and
   converts token lists/logprob lists back to tensors.
4. Add `records_to_final_batch(records)` that builds the exact
   `BatchedDataDict` contract above.
5. Add explicit validation for:
   - required fields
   - matching `generation_logprobs` length when present
   - `length` equals the first user message token length
   - supported `schema_version`
   - `source == "teacher"`
   - unique `(prompt_uid, teacher_generation_id)` pairs inside a rollout file

Acceptance:

- Unit tests prove tensor -> JSON -> tensor round trip preserves token ids.
- Unit tests prove invalid records fail with actionable errors.
- No import of vLLM, Ray, or NeMo-Gym in the utility module.

### Phase 1: Generation-Only Entry Point

Goal: create a reusable generation-only script that writes OPD-ready teacher
records through the same NeMo-Gym postprocess path used by training.

Target files:

- `examples/nemo_gym/generate_nemo_gym_rollouts.py`
- optional small config example under `examples/nemo_gym/`

Tasks:

1. Reuse setup flow from `examples/nemo_gym/run_grpo_nemo_gym.py`:
   config load, hydra overrides, tokenizer setup, `configure_generation_config`,
   `setup_nemo_gym_config`, `setup_response_data`, Ray init, policy/generation
   setup, and NeMo-Gym env creation.
2. Support explicit dataset split:
   - `--split train`
   - `--split validation`
3. Support explicit multi-sample teacher generation:
   - `--num-generations-per-prompt`
   - the writer must either repeat each prompt before one rollout call, matching
     OPD's live `repeat_interleave()` semantics, or run deterministic passes
     over the selected split
   - assign `teacher_generation_id` from the repeated slot/pass that actually
     produced the sample
4. Support output path and resumability:
   - `--output`
   - `--resume` for append-only continuation with an explicit `--output`
   - done sentinel for the finalized JSONL
   - in-progress marker while the JSONL is being written
   - partial-line truncation or equivalent protection before resume
   - stable traversal independent of physical JSONL append order
5. For each dataloader batch:
   - call `run_async_nemo_gym_rollout()`
   - serialize `nemo_gym_rollout_result.final_batch`
   - add `dataset_index`, `prompt_uid`, and `teacher_generation_id`
   - write JSONL incrementally
6. Record config metadata:
   - generation temperature/top_p/top_k
   - model path
   - tokenizer path/config
   - NeMo-Gym config paths if available

Acceptance:

- A tiny run writes records that can be loaded by Phase 0 utilities.
- `--num-generations-per-prompt=4` creates four records per prompt with
  distinct `teacher_generation_id` values.
- The script never writes `rollout_metrics["*/full_result"]` as the training
  artifact.
- Resuming a completed output is idempotent.
- Sharded generation and merge/finalize are deferred until a dedicated merge
  tool can enforce finalized inputs; training consumes one finalized JSONL.

### Phase 2: Teacher Rollout Store

Goal: select teacher records deterministically for each training step.

Target files:

- `nemo_rl/algorithms/distillation_mixed_generation.py`
- `tests/unit/algorithms/test_distillation_mixed_generation.py`

Tasks:

1. Add `TeacherRolloutStore(path)` that builds a compact JSONL offset index by
   namespaced `prompt_uid`, with `dataset_index` retained as runtime join
   metadata. The store must not tensorize or retain full `message_log` payloads
   for the whole file; selected teacher rows are materialized lazily.
2. Add `select_for_step(prompt_identities, teacher_generations_per_prompt, step,
   seed)` returning one or more records per prompt. `prompt_identities` are
   namespaced values reconstructed in training from dataset namespace plus
   `batch["idx"]`; raw dataset indices are only consistency metadata.
3. Preserve deterministic behavior across resume:
   - selection must depend on stable prompt identity, step, seed, and teacher
     slot, not dataloader worker timing
   - candidate records for each prompt must be sorted deterministically before
     any hash/selection logic, so physical append order cannot perturb training
   - mismatched `dataset_index` and `prompt_uid` for the configured dataset
     namespace must fail loudly
4. Add optional sampling-config enforcement:
   - if `require_sampling_match=true`, compare configured generation sampling to
     record sampling and raise on mismatch
5. Add provenance enforcement:
   - validate the configured dataset namespace against each record `prompt_uid`
   - validate teacher `model.name_or_path` and tokenizer metadata when expected
     values are supplied by training

Acceptance:

- Unit tests prove deterministic selection.
- Unit tests prove selection changes only when seed/step/slot changes.
- Unit tests cover resumed training with permuted batch order.
- Unit tests cover duplicate raw `dataset_index` values across different
  namespaces and verify they do not collide.
- Missing prompt records produce a clear error listing the missing identifiers.
- Loading a full teacher rollout file has O(number of rows) compact index memory,
  not O(total generated tokens) driver memory.

### Phase 3: Mixed Batch Assembly

Goal: produce one final `repeated_batch` with both sources and consistent stream
metadata.

Target files:

- `nemo_rl/algorithms/distillation_mixed_generation.py`
- `tests/unit/algorithms/test_distillation_mixed_generation.py`

Tasks:

1. Add config parsing:

   ```yaml
   distillation:
     mixed_generation:
       enabled: false
       teacher_rollout_path: null
       student_generations_per_prompt: null
       teacher_generations_per_prompt: null
       source_layout: interleave
       require_sampling_match: true
       log_source_metrics: true
   ```

2. Validate source counts before any rollout or store selection:
   - `student_generations_per_prompt >= 0`
   - `teacher_generations_per_prompt >= 0`
   - `student_generations_per_prompt + teacher_generations_per_prompt ==
     distillation.num_generations_per_prompt`
   - if `teacher_generations_per_prompt > 0`, `teacher_rollout_path` is set
   - if `student_generations_per_prompt == 0`, generation prep/finish will be
     skipped for training rollouts
3. Add `build_source_plan(prompt_count, num_generations_per_prompt, config)`.
   Initial supported layout:

   ```text
   prompt i: student gen 0, student gen 1, student gen 2, teacher gen 3
   ```

4. Add `build_student_rollout_input(batch, source_plan)` that repeats only live
   student slots before calling `run_async_nemo_gym_rollout()`.
   - Do not attach final `STREAM_METADATA_KEYS` to this reduced live-rollout
     batch.
   - Preserve a side mapping from live rollout output index to final mixed-batch
     slot.
5. Add `mix_rollout_batches(student_final_batch, teacher_final_batch,
   source_plan, full_step_manifest)` that:
   - places samples in final training order
   - concatenates standard final-batch keys
   - adds `rollout_source`
   - derives all `STREAM_METADATA_KEYS` from the final full-step manifest and
     source plan after mixing
6. Validate final metadata:
   - batch size equals `prompt_count * num_generations_per_prompt`
   - `sample_ids` unique
   - `global_batch_slot` contiguous
   - `prompt_ids` grouped correctly
   - `generation_ids` unique within each prompt
   - `sample_order` equals canonical final mixed-batch order
   - `update_group` exactly matches the full-step manifest for every final slot

Acceptance:

- Unit tests cover 3:1, 0:4, and 4:0 source layouts.
- Mixed final batch can be flattened by `batched_message_log_to_flat_message()`.
- Mixed final batch can build a manifest through
  `build_batch_manifest_from_train_data()`.
- Mixed final batch passes the existing stream conservation checks, including
  `ConservationOracle.validate_student_boundary()`.
- A metadata alignment test mixes known student/teacher rows, builds the
  manifest, runs a fake teacher annotator that emits row-identifying top-k
  tensors, and asserts the attached teacher annotations align with the intended
  final `sample_ids`, `prompt_ids`, `generation_ids`, and `rollout_source`.

### Phase 4: Distillation Training Integration

Goal: insert mixed generation into `distillation_train()` behind the disabled
default config.

Target files:

- `nemo_rl/algorithms/distillation.py`
- `nemo_rl/algorithms/distillation_mixed_generation.py`
- `examples/nemo_gym/run_distillation_nemo_gym.py` only if setup plumbing is
  needed

Tasks:

1. Parse `distillation.mixed_generation` near the existing data-pipeline config.
2. During setup, reconstruct the configured training `dataset_namespace` and
   initialize `TeacherRolloutStore` only when enabled.
3. In the training loop:
   - build the full step manifest as today
   - build the source plan
   - build namespaced prompt identities from `dataset_namespace` plus
     `batch["idx"]`
   - generate live student slots through `run_async_nemo_gym_rollout()`
   - load teacher slots through `TeacherRolloutStore`
   - call `mix_rollout_batches()`
   - continue with existing data processing, teacher top-k annotation, and
     student training
4. Preserve current path when disabled by keeping the existing repeated-batch
   code path intact.
5. Make generation prep/finish conditional on
   `student_generations_per_prompt > 0`.
6. Skip student rollout generation for all-teacher mode, except validation.
   The first implementation may still create the NeMo-Gym environment at setup
   time if avoiding that requires broader runner changes; the training rollout
   path itself must not refit/prepare/call/finish student generation when there
   are no student slots.

Acceptance:

- Disabled mode produces the same batch shape and metadata as before.
- Teacher-fraction-zero mode matches current live generation behavior.
- Teacher-only mode trains without calling training-rollout student generation.
- Existing OPD unit tests pass.
- A no-behavior-change patch before this phase must lock down the metadata seam:
  student-slot rollout mapping, post-mix metadata rebuild, manifest build, and
  conservation tests.

### Phase 5: Source-Aware Metrics and Optional Source Masks

Goal: make experiments interpretable without changing default loss behavior.

Source-specific SFT/KL masks are not part of the first KL-only mixed-generation
implementation. Do not expose `sft_on_sources` or `kl_on_sources` as active
config until this phase is implemented and tested. Before any mixed SFT
experiment, implement source-aware masking before or during token-mask creation,
not as a post-hoc zero multiplier after SFT loss computation.

Target files:

- `nemo_rl/algorithms/distillation.py`
- `nemo_rl/algorithms/loss/loss_functions.py`
- `nemo_rl/algorithms/loss/utils.py`
- tests for loss masking only if source-specific SFT/KL is implemented

Tasks:

1. Log source counts and lengths before training:
   - `rollout/source/student/count`
   - `rollout/source/teacher/count`
   - `rollout/source/student/gen_tokens_mean`
   - `rollout/source/teacher/gen_tokens_mean`
2. Add per-source training metrics if existing loss plumbing can support it
   without large refactors.
3. Add optional config only with implementation:

   ```yaml
   distillation:
     mixed_generation:
       sft_on_sources: ["teacher"]
       kl_on_sources: ["student", "teacher"]
   ```

4. If implementing `sft_on_sources`, project `rollout_source` to token masks
   before or immediately after `add_loss_mask_to_message_log()` and before the
   flattened `token_mask` becomes training input. Disallowed sources must skip
   SFT token loss computation, not merely receive zero weight afterward.
5. Keep KL applied to all sources for the first experiment.

Acceptance:

- KL-only mixed mode logs source counts and sequence stats.
- SFT source masking has unit tests proving skipped sources do not execute SFT
  token loss calculation.
- Unit tests cover source-to-token-mask projection.
- An integration test verifies excluded-source rows produce zero SFT token mask
  while KL rows remain active.
- A regression test proves non-mixed SFT masking is unchanged.
- Existing non-mixed KL/SFT metrics are unchanged.

### Phase 6: Smoke Tests and Experiment Scripts

Goal: prove the end-to-end path works before real Nano3 jobs.

Target files:

- Root workspace Slurm scripts under `/lustre/.../qwq-32b/scripts/`
- no debug logs committed into the repo

Tasks:

1. Add a short teacher pre-generation smoke script:
   - uses the same container and NeMo-Gym config
   - short Slurm time
   - self-contained sbatch script
   - `srun` carries Pyxis container args and `--no-container-mount-home`
2. Generate a small OPD-ready teacher rollout JSONL.
3. Add fast local or mocked integration smoke coverage before Nano3 Slurm:
   - `mixed_generation.enabled=false`
   - 3:1 mixed
   - 4:0 student-only
   - 0:4 teacher-only
   - JSONL resume/idempotence
   - sampling mismatch enforcement
4. Run a small mixed OPD smoke:
   - 4 nodes for Nano3 shape
   - short time
   - 3:1 source plan
   - KL-only setup first
5. Compare smoke metrics:
   - source counts
   - generation lengths
   - reward
   - initial KL
   - driver RSS / OPD data memory metrics
6. Only after smoke passes, create full singleton jobs.

Acceptance:

- Generation-only smoke writes valid records.
- Mixed OPD smoke reaches at least a few training steps.
- Initial KL difference versus all-student gen4 baseline is explainable.
- No driver OOM or abnormal step-time regression.

## Suggested Patch Order

1. Serialization utilities and unit tests.
2. Generation-only entry point and tiny smoke.
3. Teacher rollout store and selection tests.
4. Mixer and metadata tests.
5. Metadata-seam conservation tests for post-mix manifest construction.
6. Disabled-by-default integration into `distillation_train()`.
7. Source metrics.
8. Optional source-specific SFT masking.
9. Nano3 smoke scripts and real-job scripts.

Each patch should be reviewable independently. The first five patches should not
change training behavior.

## Review Checklist

- Does every pre-generated sample re-enter at the `final_batch` boundary?
- Are token ids preserved exactly through disk round trip?
- Are `STREAM_METADATA_KEYS` valid after mixing?
- Is selection deterministic after checkpoint/resume?
- Does disabled config preserve current behavior?
- Are SFT-zero and source-skipped SFT implemented as skipped computation, not
  post-hoc zero weighting?
- Are smoke scripts short, self-contained, and using the expected Slurm/Pyxis
  pattern?
