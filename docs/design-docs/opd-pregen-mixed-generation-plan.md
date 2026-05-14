# OPD Pre-Generated Teacher and Mixed Generation Plan

Status: Draft plan

Related:

- [On-policy Distillation Data Pipeline](opd-data-pipeline.md)
- [OPD Data Pipeline Implementation Plan](opd-data-pipeline-implementation-plan.md)
- [NeMo-Gym Integration](nemo-gym-integration.md)

## Goal

Support OPD batches that mix live student rollouts with pre-generated teacher
rollouts while keeping the downstream training path identical to the current
on-the-fly NeMo-Gym OPD path.

The core rule is that pre-generated teacher samples must re-enter OPD at the
same boundary as live NeMo-Gym rollouts: the postprocessed NeMo-RL
`final_batch` shape returned by `run_async_nemo_gym_rollout()`. We should not
train directly from the current `trajectory_collection.jsonl` full-result logs,
because those logs are for inspection and drop token ids from the raw NeMo-Gym
payload.

## Current OPD Flow

Current NeMo-Gym OPD uses this path:

1. `NemoGymDataset` stores each JSONL row as raw `extra_env_info`.
2. `nemo_gym_data_processor()` parses `extra_env_info` and emits a fake empty
   `message_log` for compatibility.
3. `distillation_train()` repeats prompts with
   `batch.repeat_interleave(num_generations_per_prompt)`.
4. `attach_step_metadata()` adds `sample_ids`, `sample_order`,
   `update_group`, `global_batch_slot`, `prompt_ids`, and `generation_ids`.
5. `run_async_nemo_gym_rollout()` sends `extra_env_info` to NeMo-Gym and
   applies the current `temperature` and `top_p`.
6. `NemoGymEnvironment._postprocess_nemo_gym_to_nemo_rl_result()` converts raw
   NeMo-Gym outputs into `message_log` entries:

   ```python
   {"role": "user", "content": "", "token_ids": prompt_delta_tokens}
   {"role": "assistant", "content": "", "token_ids": generation_token_ids,
    "generation_logprobs": generation_log_probs}
   ```

7. `run_async_nemo_gym_rollout()` returns `final_batch`:

   ```python
   {
     "agent_ref": ...,
     "message_log": ...,
     "length": ...,
     "loss_multiplier": ...,
     "total_reward": ...,
     "truncated": ...,
   }
   ```

8. OPD replaces `repeated_batch` with `final_batch`, reattaches the saved stream
   metadata, flattens `message_log`, adds assistant-token loss masks, annotates
   teacher top-k, and trains the student.

The pre-generation path must produce records that can be loaded back into this
same step-8 input shape.

## Proposed User-Facing Modes

Add an opt-in config block under `distillation`:

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

Initial defaults should preserve existing behavior. When disabled, no behavior
changes.

The first implementation supports explicit source counts and the `interleave`
layout only. Fraction-based configuration, `prefix`/`suffix` layouts, and
source-specific SFT/KL masks are later extensions. Do not expose
`sft_on_sources` or `kl_on_sources` until the source-aware loss masking phase is
implemented and tested.

For the first Nano3 experiment, use a fixed 3:1 layout:

- `num_prompts_per_step=128`
- `num_generations_per_prompt=4`
- 3 live student samples per prompt
- 1 pre-generated teacher sample per prompt
- global batch remains 512

## Serialized Teacher Rollout Format

Use JSONL. Each row is one generated sample, not one prompt.

Required fields:

```json
{
  "schema_version": 1,
  "source": "teacher",
  "prompt_uid": "...",
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

`message_log[*].token_ids` is the source of truth. `content` should remain empty
to match current NeMo-Gym OPD. `generation_logprobs` is optional for OPD loss,
but preserving it makes parity checks and debugging easier.

Do not rely on `full_result.response.output[*].generation_str` for training.
That text is useful for inspection but does not preserve exact tokenization and
terminal tokens.

## Phase 1: Generation-Only Writer

Goal: add a reusable data-generation entry point that writes OPD-ready teacher
rollout records.

Implementation options:

1. Extend `examples/nemo_gym/run_grpo_nemo_gym.py` trajectory collection with a
   new output mode that serializes `nemo_gym_rollout_result.final_batch`.
2. Prefer a dedicated entry point if the collection mode becomes cluttered, for
   example `examples/nemo_gym/generate_nemo_gym_rollouts.py`.

The entry point should:

- Use the same setup path as `run_grpo_nemo_gym.py`: config loading,
  `setup_nemo_gym_config()`, data loading, vLLM generation setup, NeMo-Gym env
  creation.
- Accept train or validation data selection explicitly.
- Run `run_async_nemo_gym_rollout()` over the selected dataloader.
- Serialize `final_batch`, not `rollout_metrics["*/full_result"]`.
- Convert tensors to JSON lists/scalars.
- Write a resumable JSONL with `.inprogress` and `.done` sentinels if used
  under Slurm; shard merge/finalize is out of scope for the first implementation.
- Record the exact generation config and model/tokenizer metadata.

Acceptance:

- A tiny generation-only smoke test creates JSONL records whose `message_log`
  can be converted back through `batched_message_log_to_flat_message()`.
- The serialized token lengths match `length`, `total_reward`, and `truncated`.
- A deterministic small fixture round-trips through save/load without changing
  token ids.

## Phase 2: Teacher Rollout Dataset and Loader

Goal: load pre-generated teacher rollout records into a batched structure that
matches `run_async_nemo_gym_rollout().final_batch`.

Add a lightweight loader that:

- Reads JSONL records once to build a compact line-offset index.
- Does not retain all `message_log` token payloads in driver memory; selected
  rows are re-read and converted lazily.
- Converts selected `message_log[*].token_ids` and optional
  `generation_logprobs` back to tensors.
- Validates all required fields.
- Groups records by namespaced `prompt_uid`.
- Keeps raw `dataset_index` only as consistency metadata; raw indices are never
  used across dataset namespaces.
- Supports repeatable selection by `(step, prompt_uid, teacher_slot)`.
- Validates sampling, model, tokenizer, and dataset namespace provenance when
  the training config supplies expected values.
- Emits a `BatchedDataDict` with:

  ```python
  {
    "agent_ref": list[dict],
    "message_log": list[list[dict]],
    "length": torch.Tensor,
    "loss_multiplier": torch.Tensor,
    "total_reward": torch.Tensor,
    "truncated": torch.BoolTensor,
  }
  ```

Acceptance:

- Unit tests cover tensor restoration, missing-field errors, and deterministic
  sample selection.
- Loader does not initialize vLLM or NeMo-Gym.
- Loader preserves exact token ids from disk.

## Phase 3: Mixed Batch Assembly

Goal: combine live student `final_batch` and pre-generated teacher `final_batch`
before teacher top-k annotation.

Add a helper with a narrow contract:

```python
mixed_batch = mix_rollout_batches(
    student_batch=student_final_batch,
    teacher_batch=teacher_final_batch,
    source_layout=...,
    step_manifest=...,
)
```

The helper should:

- Concatenate `message_log`, `agent_ref`, `length`, `loss_multiplier`,
  `total_reward`, and `truncated`.
- Add `rollout_source` as a per-sample string/list or encoded tensor.
- Rebuild or reassign `STREAM_METADATA_KEYS` so final batch order is exactly
  the order seen by training.
- Preserve `prompt_ids` semantics: all samples generated from the same prompt
  should share the same `prompt_id`.
- Preserve `generation_ids` semantics: generation ids should be unique within a
  prompt and should reflect the mixed layout.
- Keep `global_batch_slot` contiguous from 0 to global batch size - 1.

Initial source layout:

```text
prompt 0: student gen 0, student gen 1, student gen 2, teacher gen 3
prompt 1: student gen 0, student gen 1, student gen 2, teacher gen 3
...
```

This preserves current grouping assumptions better than appending all teacher
samples at the end.

Acceptance:

- For `prompt_count=128` and `num_generations_per_prompt=4`, output size is 512.
- `sample_ids` are unique.
- `global_batch_slot` is contiguous.
- `update_group` remains compatible with `train_global_batch_size`.
- Flattening and `build_batch_manifest_from_train_data()` work unchanged.

## Phase 4: OPD Training Integration

Goal: insert mixed generation into `distillation_train()` without disturbing the
existing disabled path.

Training step changes:

1. Build the full step manifest for the final mixed batch.
2. Decide per prompt how many live student generations and teacher generations
   are needed.
3. For student generations, create a repeated batch with only the student
   slots and call `run_async_nemo_gym_rollout()` as today.
4. Load matching teacher rollout records for the teacher slots.
5. Assemble the mixed `repeated_batch`.
6. Continue with existing data processing, teacher top-k annotation, and student
   training.

The first implementation can support a static integer split, for example
`student_generations_per_prompt=3` and `teacher_generations_per_prompt=1`.
General fractions can come later.

Acceptance:

- With `mixed_generation.enabled=false`, the training loop is byte-for-byte
  equivalent except for harmless config parsing.
- With teacher fraction 0, behavior matches current live student rollout.
- With teacher fraction 1, OPD trains on pre-generated teacher rollouts without
  initializing student generation for training rollout, except validation.
- Mixed mode logs per-source counts and per-source sequence statistics.

## Phase 5: Source-Aware Loss and Metrics

Goal: make mixed data interpretable and allow experiments where SFT applies only
to teacher-source samples.

Add source-aware metrics:

- `rollout/source/student/count`
- `rollout/source/teacher/count`
- `rollout/source/student/gen_tokens_mean`
- `rollout/source/teacher/gen_tokens_mean`
- `train/source/student/kl_loss`
- `train/source/teacher/kl_loss`
- `train/source/student/sft_loss`
- `train/source/teacher/sft_loss`

Add source masks to the loss path only if needed. The first KL-only experiment
can apply KL to all sources. For SFT experiments, we likely want:

```yaml
distillation:
  mixed_generation:
    sft_on_sources: ["teacher"]
    kl_on_sources: ["student", "teacher"]
```

This avoids self-imitation SFT on student-generated samples when the purpose is
to make teacher rollouts directly shape behavior.

Acceptance:

- KL-only mixed run reports source-split KL metrics.
- SFT-on-teacher-only run reports zero valid SFT tokens for student source.
- Existing non-mixed SFT and KL metrics are unchanged.

## Phase 6: Smoke and Parity Tests

Run tests in increasing cost:

1. Unit tests for serialization, loader, and mixer.
2. CPU/Ray-light test that constructs fake teacher and student final batches
   and feeds the mixed batch into `build_token_stream_from_rollout_batch()`.
3. Short NeMo-Gym generation-only smoke test that writes a few teacher samples.
4. Short OPD mixed smoke test:
   - Nano3 config shape.
   - 4 nodes.
   - short Slurm time.
   - small `max_num_steps`.
   - no SFT first: `T=0.6`, `top_p=0.95`, `top_k=null`, reverse KL.
5. Compare initial KL against:
   - all-student on-the-fly baseline
   - all-teacher pregen mode
   - mixed 3:1 mode

Success criteria:

- No driver OOM or large unexpected driver RSS jump.
- Initial KL is explainable from source mix and sampled contexts.
- Reward and generation length are in the expected range.
- Capped ratio is not worse than the corresponding source distribution.

## Key Consistency Requirements

- Pre-generated samples must use the same tokenizer and NeMo-Gym postprocess
  semantics as live generation.
- `message_log` token ids are canonical. Text is optional metadata.
- Generation `top_k` remains unsupported in the NeMo-Gym path.
- Sampling config must be recorded and optionally enforced at load time.
- Mixed mode must not reorder samples after `STREAM_METADATA_KEYS` are assigned,
  unless it rebuilds those metadata tensors consistently.
- Teacher top-k annotation remains computed online by the teacher policy over
  the final mixed contexts. Pre-generated teacher rollout tokens are not a
  substitute for teacher top-k logits.

## Risks

- The current trajectory collection path writes `full_result`, not OPD-ready
  `final_batch`; using it directly would silently lose token-id fidelity.
- Prompt identity is not currently a first-class stable field in every
  NeMo-Gym dataset. We may need to derive `prompt_uid` from original row index
  plus dataset path, or add it during generation.
- If teacher rollouts are generated with a different prompt template, tokenizer,
  max model length, or NeMo-Gym resource config, KL differences will be
  difficult to interpret.
- If teacher samples are appended instead of interleaved by prompt, update-group
  and per-prompt generation semantics become harder to reason about.
- Source-aware SFT requires care so a zero source weight skips computation where
  intended, matching the earlier SFT-weight=0 requirement.

## Recommended First Experiment

Implement only the minimum for this experiment:

- Generate OPD-ready teacher rollout JSONL through the NeMo-Gym rollout path.
- Load teacher samples deterministically by prompt index.
- Mix 3 student + 1 teacher sample per prompt.
- Train KL-only first:
  - `T=0.6`
  - `top_p=0.95`
  - `top_k=null`
  - `sft_weight=0`
  - reverse KL
- Compare against the current gen4 KL-only OPD-data baseline.

The earlier `T=1`, `top_p=1` setting remains useful as a parity diagnostic, but
the first mixed-generation execution follows the current promising Nano3
`T=0.6`, `top_p=0.95` setup.

Only after KL-only behavior is understood should we add SFT-on-teacher-source
experiments.
