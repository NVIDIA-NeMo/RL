# OPD Data Pipeline Implementation Plan

Status: Draft implementation plan

Related design: [On-policy Distillation Data Pipeline](opd-data-pipeline.md)

This plan turns the OPD data-pipeline design into reviewable implementation
steps. The first objective is to remove the teacher top-k gather from the Ray
driver without changing OPD training semantics. Later phases move rollout
normalization off the driver and replace dense teacher annotations with sparse
loss-active annotations.

## Constraints

- Keep the legacy OPD path as the default until the stream path passes parity
  and scale tests.
- Preserve serialized OPD stage execution in v1: rollout, teacher annotation,
  teacher offload, then student training.
- Preserve optimizer update grouping. Do not call `Policy.train()` once per
  chunk if that changes one optimizer update into multiple updates.
- Preserve current backend capability limits. Do not use this work to add new
  CP, PP, packing, dynamic batching, or multimodal combinations.
- Phase 0 containment is mandatory before any stream path is enabled.
- The driver may hold manifests and stream proxies, but it must not call
  `ray.get()` on token payloads or teacher top-k payloads in the stream path.

## Proposed Config Surface

Add an opt-in config block with legacy defaults:

```yaml
distillation:
  data_pipeline:
    mode: legacy  # legacy, stream_teacher, stream_rollout, sparse_loss
    max_chunk_tokens: 262144
    max_chunk_bytes: null
    max_chunk_loss_positions: null
    log_memory_metrics: true
```

`legacy` keeps the current behavior. Each later mode includes the earlier modes.

## Phase 0: Mandatory Containment

Goal: remove avoidable replicated materialization in the current path and add
metrics that prove where memory is going.

### Code Changes

Target files:

- `nemo_rl/distributed/worker_groups.py`
- `nemo_rl/models/policy/lm_policy.py`
- `nemo_rl/models/policy/workers/dtensor_policy_worker.py`
- `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- `nemo_rl/algorithms/distillation.py`

Tasks:

1. Change `MultiWorkerFuture.get_results()` so the non-generator path builds the
   selected object-ref list before `ray.get()` when `return_from_workers` is set.
   The selection must respect `called_workers`.
2. Preserve the existing `return_generators_as_proxies=True` behavior.
3. Add unit coverage that monkeypatches `ray.get` and proves unselected refs are
   not fetched.
4. For top-k worker methods, avoid CPU-copying and returning large top-k tensors
   on ranks that will not be returned to the driver. If a collective requires all
   ranks to participate, only the elected output rank should materialize the CPU
   payload after the collective.
5. Add teacher top-k byte-count and driver memory metrics around the legacy OPD
   teacher annotation step.

### Acceptance Criteria

- Existing top-k unit tests still pass for DTensor and Megatron.
- A new worker-group test proves selected-result fetching happens before
  `ray.get()`.
- OPD legacy mode still produces identical outputs on a small deterministic
  test.
- The legacy path logs teacher annotation bytes and driver RSS.

### Rollback

Revert only Phase 0 commits. No config migration is needed because the legacy
public API remains unchanged.

## Phase 1: Manifest, Contracts, and Planning

Goal: add stable step identity, stream data structures, validation, and reshard
planning without changing the OPD training path.

### Code Changes

Target files:

- `nemo_rl/algorithms/distillation_streaming.py` (new)
- `nemo_rl/algorithms/distillation.py`
- `nemo_rl/models/policy/interfaces.py`
- `nemo_rl/models/policy/lm_policy.py`
- `examples/configs/distillation_math.yaml`
- recipe configs that opt into OPD streaming later

Tasks:

1. Add the core dataclasses:
   - `SpanTable`
   - `StepManifest`
   - `BatchManifest`
   - `StageLayout`
   - `TokenChunk`
   - `TeacherTopKChunk`
   - `AnnotatedTokenChunk`
   - `ShardedBatchStream`
   Keep OPD-specific manifests in `nemo_rl/algorithms/distillation_streaming.py`.
   If a generic transport helper is needed later, put only that helper under
   `nemo_rl/distributed/`.
2. Add validators for:
   - sample ID uniqueness within a step
   - mandatory `sample_order`, `update_group`, and `global_batch_slot`
   - loss-span bounds relative to `input_lengths`
   - chunk byte/token budget
   - EOS marker shape
3. Add a `StepManifest` builder before `repeat_interleave()` and propagate
   `sample_id`, `sample_order`, `update_group`, and `global_batch_slot` through
   `repeated_batch`. Legacy mode may ignore the fields, but they must survive
   rollout and post-rollout flattening.
4. Add a `BatchManifest` builder from the current post-rollout flattened
   `train_data`. This is the bridge used before rollout normalization moves off
   the driver.
5. Add a thin `Policy.stage_layout()` query derived from existing sharding
   annotations and batching config. Do not store independent layout state.
6. Add `DPReshardPlanner` with two balancing modes:
   - preserve current update-group placement
   - balance by estimated valid loss tokens within an update group
7. Add config parsing for `distillation.data_pipeline`, defaulting to `legacy`.
8. Add early capability validation that rejects unsupported v1 combinations
   before actor work is submitted. Keep backend-specific truth close to existing
   backend setup/config validation and expose only a thin query to distillation.
9. Keep `ShardedBatchStream` transport-agnostic in this phase. Do not commit to
   `ObjectRefGenerator`, queue actors, or a stream-manager actor until Phase 2A.

### Acceptance Criteria

- Unit tests cover dataclass validation, loss-span bounds, EOS markers, and
  `DPReshardPlanner` ordering.
- Tests prove `update_group` and `global_batch_slot` match current
  `shard_by_batch_size(..., batch_size=train_global_batch_size)` grouping for
  representative batch sizes.
- Add a step-wide conservation oracle that can be reused by later phases. For
  every stream mode it must verify:
  - multiset equality of `sample_id`s across rollout output, teacher
    annotations, and student consumption
  - canonical `sample_order` restoration at the student boundary
  - exact conservation of active `(sample_id, position)` pairs
  - duplicate chunk and missing chunk detection
- Add negative config tests that prove unsupported v1 combinations fail before
  actor work starts. Cover unsupported CP, PP, sequence packing, dynamic
  batching, and multimodal combinations for each backend that has a known limit.
- No production behavior changes when `mode: legacy`.

### Rollback

Disable the config block or leave it unused. Since no production path consumes
streams yet, rollback is limited to new files and interface stubs.

## Phase 2A: Teacher Annotation Stream Producer

Goal: replace the teacher-side full-batch top-k return with chunked teacher
annotation production. This phase proves teacher chunks can be produced without
driver-side `[B, S, K]` materialization, but it does not yet complete streamed
student training.

### Code Changes

Target files:

- `nemo_rl/algorithms/distillation.py`
- `nemo_rl/algorithms/distillation_streaming.py`
- `nemo_rl/models/policy/interfaces.py`
- `nemo_rl/models/policy/lm_policy.py`
- `nemo_rl/models/policy/workers/dtensor_policy_worker.py`
- `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- `nemo_rl/distributed/worker_groups.py`

Tasks:

1. Add `Policy.annotate_topk_stream(token_stream, ...)`.
2. Resolve the v1 stream transport choice. Either:
   - use `ObjectRefGenerator` plus a small lifecycle wrapper, or
   - use an explicit queue/stream-manager actor.
   The chosen transport must support bounded in-flight chunks, EOS, ack,
   cancel, and teardown.
3. Add a stream-preserving worker-group helper that:
   - returns stream proxies to the caller
   - filters selected return ranks consistently
   - propagates EOS markers
   - exposes cancellation/teardown hooks
4. Build a `TokenChunk` stream from the existing post-rollout `train_data`.
   This phase may still flatten rollout data on the driver.
5. Compute teacher top-k per chunk. The first implementation may use dense
   per-chunk teacher tensors internally, but the driver must only see
   `AnnotatedTokenChunk` envelopes and object refs.
6. Add a test-only chunk collector that can compare teacher stream output against
   legacy `get_topk_logits()` without becoming part of the production path.
7. Add stream memory metrics:
   - `opd/data/token_chunk_bytes`
   - `opd/data/teacher_annotation_bytes`
   - `opd/data/inflight_chunks`
   - `opd/data/inflight_bytes`
   - `opd/data/driver_rss_bytes`
   - `opd/data/ray_object_store_bytes`

### Acceptance Criteria

- Teacher stream chunks reconstruct the same top-k values as legacy
  `get_topk_logits()` on a small deterministic batch.
- Driver never materializes full `[B, S, K]` teacher annotations in the
  teacher-stream path.
- Ray object-store usage is bounded by configured in-flight chunk limits.
- Slow-consumer test proves producer backpressure works.
- Cancel/timeout test proves refs are released and EOS/cancel is handled.

### Rollback

Set `distillation.data_pipeline.mode=legacy`. The old
`teacher_policy.get_topk_logits(train_data)` path remains intact.

## Phase 2B: Stateful Student Stream Consumer

Goal: consume annotated teacher chunks without changing student optimizer
semantics. This is the phase that makes `mode=stream_teacher` an end-to-end OPD
training path.

### Code Changes

Target files:

- `nemo_rl/algorithms/distillation.py`
- `nemo_rl/algorithms/distillation_streaming.py`
- `nemo_rl/algorithms/loss/utils.py`
- `nemo_rl/distributed/model_utils.py`
- `nemo_rl/models/policy/interfaces.py`
- `nemo_rl/models/policy/lm_policy.py`
- `nemo_rl/models/policy/workers/dtensor_policy_worker.py`
- `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`

Tasks:

1. Add `Policy.train_distillation_stream(...)`.
2. Add a stateful student consumer that assembles or drains chunks by
   `update_group` and performs exactly one logical optimizer update per group.
3. Add a worker-local compatibility adapter from
   `TokenChunk + TeacherTopKChunk` to the existing dense distillation loss input.
   This adapter must densify only bounded microbatches or chunks inside the
   worker, never a full global batch on the driver.
4. Wire the adapter through `algorithms/loss/utils.py` or a nearby wrapper so
   Phase 2B does not duplicate distillation loss-prep logic.
5. Preserve update-group metric reductions, valid-token counts, and scheduler
   behavior.

### Acceptance Criteria

- A small deterministic OPD test shows legacy and end-to-end `stream_teacher`
  produce the same loss within tolerance.
- Optimizer-regression tests cover at least 2 to 3 uneven `update_group`s with
  chunk boundaries that split groups awkwardly. They must instrument and compare:
  - `optimizer.step()` count
  - `optimizer.zero_grad()` count
  - scheduler advancement count
  - parameter deltas after each update group
  - optimizer state and scheduler state after the step
  - valid-token and valid-sample metric reductions
- The step-wide conservation oracle proves sample IDs, sample order,
  update-group slots, and active sparse positions match legacy.
- Driver never materializes full `[B, S, K]` teacher annotations in
  end-to-end `stream_teacher`.

### Rollback

Set `distillation.data_pipeline.mode=legacy`. The old
`teacher_policy.get_topk_logits(train_data)` path remains intact.

## Phase 3: Stream Rollout Normalization

Goal: remove full rollout message logs and flattened token tensors from the
driver.

Implemented v1 status: partial. `mode=stream_rollout` uses a post-rollout
normalizer and trains the student from token/top-k refs, so the dense
post-rollout `train_data` is not passed into student training. Rollout message
logs still return to the driver, and normalization still happens on the driver;
moving that boundary into rollout workers remains future work.

### Code Changes

Target files:

- `nemo_rl/experience/rollouts.py`
- `nemo_rl/algorithms/distillation.py`
- `nemo_rl/algorithms/distillation_streaming.py`
- `nemo_rl/data/llm_message_utils.py`
- `nemo_rl/models/generation/vllm/vllm_generation.py` if rollout output needs
  backend support

Tasks:

1. Define a rollout-normalization interface that can emit token chunks or refs
   without changing the dense generation interface all at once.
2. Add a rollout-normalization actor or worker method that converts message logs
   to `TokenChunk` objects off the driver.
3. Build `loss_spans` during normalization from assistant-message role masks.
4. Return only `BatchManifest` metadata and `ShardedBatchStream[TokenChunk]` to
   the driver.
5. Adapt the synchronous rollout path.
6. Adapt the async rollout path separately. The async path must have independent
   parity and object-lifecycle tests because it duplicates normalization logic
   today.
7. Support multi-turn text logs by preserving assistant/tool/assistant role
   boundaries in loss spans.
8. Keep multimodal OPD disabled unless per-sample media refs and processor
   metadata are fully specified.

### Acceptance Criteria

- Legacy and `stream_rollout` match on a deterministic text-only OPD test.
- Sync rollout streaming matches legacy on deterministic text-only OPD.
- Async rollout streaming matches legacy on deterministic text-only OPD.
- Driver RSS no longer scales with full generated token tensors.
- Multi-turn masking tests prove sparse spans match dense `token_mask[:, 1:]`.
- `stream_rollout` plus `stream_teacher` can run a short text-only OPD smoke
  test.

V1 limitations:

- `stream_rollout` rejects async rollouts, multimodal OPD, dynamic batching, and
  sequence packing.
- Driver RSS can still scale with rollout message logs until rollout workers
  emit token chunks directly.

### Rollback

Set `mode=stream_teacher` or `mode=legacy`. The post-rollout driver flatten path
remains available until Phase 3 is stable.

## Phase 4: Sparse Student Loss

Goal: avoid dense `[B, S, K]` teacher tensors inside student workers.

Implemented v1 status: `mode=sparse_loss` reuses the stream-teacher transport
and attaches sparse teacher tensors in student workers:
`teacher_topk_sparse_logits`, `teacher_topk_sparse_indices`,
`teacher_topk_sparse_positions`, and `teacher_topk_sparse_mask`. The existing
`DistillationLossFn` is reused after sparse student top-k gather.

### Code Changes

Target files:

- `nemo_rl/distributed/model_utils.py`
- `nemo_rl/algorithms/loss/utils.py`
- `nemo_rl/algorithms/loss/loss_functions.py`
- `nemo_rl/models/policy/workers/dtensor_policy_worker.py`
- `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- `nemo_rl/models/megatron/data.py`
- `nemo_rl/models/automodel/data.py`

Tasks:

1. Add sparse distillation loss utilities that consume:
   - `(sample_id, position)` sparse coordinates
   - teacher top-k logprobs
   - teacher top-k indices
   - local pack maps or unpack maps
2. Gather student logprobs only at teacher top-k indices for active positions.
3. Preserve forward, reverse, and mixed KL semantics.
4. Preserve `zero_outside_topk` behavior, including entropy computation, without
   rebuilding full dense teacher tensors.
5. Add collective-count safety for packed and dynamic microbatch paths:
   - dummy microbatches, or
   - identical chunk groups, or
   - explicit flush/drain barriers

### Acceptance Criteria

- Sparse and dense distillation losses match within tolerance for:
  - plain batches
  - sequence-packed batches where supported
  - dynamic-batched batches where supported
  - CP-enabled batches where supported
- Sparse mask fixtures cover:
  - empty spans
  - all-prompt or no-loss samples
  - final-token exclusion
  - one-token assistant messages
  - multi-turn assistant/tool/assistant transitions
  - unsorted or overlapping spans rejected by validation
- Reverse KL and mixed KL tests include `zero_outside_topk=True` for sparse mask
  edge cases, not only for ordinary contiguous assistant spans.
- No student worker rebuilds full `[B, S, K]` teacher tensors in `sparse_loss`
  mode.
- Long-context smoke test shows student peak memory scales with chunk and sparse
  active-position budget, not full padded sequence length times K.

V1 limitations:

- `sparse_loss` rejects multimodal OPD, dynamic batching, sequence packing, and
  context parallelism.
- Sparse support is for the stream-teacher-style dense student batch path; the
  ref-only `stream_rollout` student path still uses the dense compatibility
  assembler.

### Rollback

Set `mode=stream_rollout` or `mode=stream_teacher`. The per-chunk dense
compatibility path remains until sparse loss is proven.

## Phase 5: Hardening and Cleanup

Goal: make the stream path production-ready and decide whether to replace the
legacy OPD path.

Tasks:

1. Add fault-injection tests:
   - teacher worker fails before chunk emit
   - teacher worker fails after chunk emit
   - student worker fails while consuming a chunk
   - driver cancels while refs are in flight
2. Add leak tests that assert Ray object-store bytes return near baseline after
   success, cancel, timeout, and worker failure.
3. Add step-level atomicity tests:
   - failed or cancelled steps commit zero optimizer updates
   - failed or cancelled steps do not advance scheduler state
   - retrying the same step produces the same first successful optimizer state
     as a clean baseline
   - stale chunk refs from a failed step cannot be consumed by the next step
   - duplicate chunk IDs are rejected or safely ignored
4. Add a short Slurm smoke recipe with a small time limit and text-only model.
5. Add scale validation for the target long-context configuration:
   - `num_prompts_per_step=128`
   - `num_generations_per_prompt=4`
   - `topk_logits_k=64`
6. Document unsupported v1 combinations in user-facing OPD docs.
7. Decide whether `stream_teacher` or `stream_rollout` should become the default
   for text-only OPD.

Short Slurm smoke recipe requirements:

- Use a self-contained `sbatch` script; do not submit by injecting caller-side
  environment overrides such as `KEY=value sbatch smoke.sh`.
- Use Pyxis/Pyxis-compatible container options with `--no-container-mount-home`
  and mount only shared project/model/data paths.
- Keep smoke time short, for example 30 minutes or less, and use a text-only
  model/config with dynamic batching, sequence packing, CP, multimodal inputs,
  and async rollouts disabled for streamed v1 modes.
- For resumable scale runs, submit follow-ons as singleton jobs instead of
  launching parallel duplicates of the same checkpoint directory.

### Acceptance Criteria

- No object-store leak across repeated stream OPD steps.
- No collective hangs in packed, dynamic, or CP-enabled supported tests.
- The target long-context smoke test reaches student training and completes a
  step with bounded driver memory.
- Logs include enough chunk/memory metrics to diagnose future OOMs.

### Numeric Memory Gates

Before any stream mode becomes the default, add explicit numeric pass/fail gates
to the scale tests. The exact constants should be chosen from cluster baseline
measurements in the implementation PR, but the gates must include:

- fixed upper bounds for driver RSS and Ray object-store bytes at the target
  long-context config
- slope checks across at least three global batch or token scales showing driver
  RSS is approximately flat after metadata is excluded
- plateau checks while varying chunk-size budgets, and later while varying
  `max_inflight_chunks` and `max_inflight_bytes` after a real backpressure
  transport is added
- per-phase thresholds derived from the design-doc memory budget table

## Suggested PR Boundaries

1. **PR 1: Phase 0 containment and metrics**
   - Smallest risky change.
   - No new stream abstractions.
2. **PR 2a: config schema, manifest builder, and metadata propagation**
   - Stable sample IDs, update groups, and early capability validation.
   - No production behavior change.
3. **PR 2b: planner and layout query**
   - `StageLayout`, `DPReshardPlanner`, validators, and conservation oracle.
   - Keep transport abstract.
4. **PR 3a: teacher-side non-gather stream production**
   - Teacher chunks can be produced and compared to legacy top-k.
   - Not yet an end-to-end training mode.
5. **PR 3b: stateful student stream consumption**
   - Adds loss adapter and update-group-preserving execution.
   - Enables end-to-end `mode=stream_teacher`.
6. **PR 4a: rollout normalization interface and sync rollout streaming**
   - Driver no longer owns full sync-rollout token tensors in opt-in mode.
7. **PR 4b: async rollout streaming**
   - Adapts async rollout separately with parity and lifecycle tests.
8. **PR 5: sparse student loss behind `mode=sparse_loss`**
   - Removes dense teacher annotation rebuild inside student workers.
9. **PR 6: hardening, docs, and default-mode decision**
   - Fault tests, smoke scripts, and cleanup.

## Test Commands

Run targeted tests after each PR:

```sh
uv run pytest tests/unit/distributed/test_worker_groups.py
uv run pytest tests/unit/distributed/test_batched_data_dict.py
uv run pytest tests/unit/algorithms/test_distillation.py
uv run pytest tests/unit/algorithms/test_loss_functions.py -k distillation
uv run pytest tests/unit/models/policy/test_dtensor_worker.py -k "topk or distillation"
uv run pytest tests/unit/models/policy/test_dtensor_worker_v2.py -k "topk or distillation"
uv run pytest tests/unit/models/policy/test_megatron_worker.py -k "topk or distillation"
```

Add new focused tests as the stream modules land:

```sh
uv run pytest tests/unit/algorithms/test_distillation_streaming.py
uv run pytest tests/unit/distributed/test_opd_streaming.py
```

For cluster validation, use short self-contained `sbatch` scripts with Pyxis
`--no-container-mount-home` and singleton scheduling for resumable training
runs.

## Go/No-go Gates

Do not move to the next phase until:

- Legacy mode remains unchanged.
- New mode has deterministic small-batch parity with legacy mode.
- Driver memory and Ray object-store metrics are collected.
- Object refs are released after success and cancel.
- Unsupported backend combinations fail early with clear errors.
- Optimizer update grouping is preserved.
