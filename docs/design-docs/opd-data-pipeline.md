# On-policy Distillation Data Pipeline

Status: Draft design

Implementation plan: [OPD Data Pipeline Implementation Plan](opd-data-pipeline-implementation-plan.md)

This document proposes a scalable data pipeline for on-policy distillation (OPD).
The goal is to remove driver-side collection and repartitioning of rollout,
teacher, and student training data while preserving support for different data
parallel (DP), context parallel (CP), tensor parallel (TP), and pipeline parallel
(PP) layouts across rollout, teacher inference, and student training.

## Problem

The current OPD pipeline is driver-centric. In `nemo_rl/algorithms/distillation.py`
the driver:

1. Repeats prompts with `batch.repeat_interleave(num_generations_per_prompt)`.
2. Runs rollout and receives the updated full message logs.
3. Flattens the full message logs with `batched_message_log_to_flat_message`.
4. Builds a full `BatchedDataDict` containing `input_ids`, `input_lengths`,
   `token_mask`, `sample_mask`, and multimodal payloads.
5. Calls `teacher_policy.get_topk_logits(train_data)`.
6. Stores `teacher_topk_logits` and `teacher_topk_indices` back into
   `train_data`.
7. Calls `student_policy.train(train_data)`, which shards the full batch again.

`Policy.get_topk_logits()` shards data by the teacher DP size, dispatches to
workers with `run_all_workers_sharded_data()`, collects all selected worker
results with `get_all_worker_results()`, concatenates all `[B, S, K]` top-k
tensors on the driver, optionally reorders them, then returns a full
`BatchedDataDict`.

This design works for small batches, but it scales poorly for long-context OPD:

- The driver owns full rollout message logs, flattened tensors, and teacher
  annotations at the same step.
- Ray object-store memory and driver RSS grow with global batch size and maximum
  sequence length, not with per-worker microbatch size.
- Teacher top-k output is replicated across CP/TP/PP dimensions before being
  filtered to one result per replicated group.
- The full teacher output is gathered and concatenated before student training
  can consume any part of it.
- The student then repartitions the same tensors into a different layout.

For a batch with 512 samples, sequence length 49152, and `topk_logits_k=64`, raw
teacher annotations are already about 18 GiB if stored as float32 logits plus
int64 indices:

```text
512 * 49152 * 64 * (4 bytes logits + 8 bytes indices) = 18.0 GiB
```

This excludes Ray copies, object-store overhead, concatenation intermediates,
Python object overhead, message logs, padding, and replicated model-parallel
outputs. Smaller microbatch sizes do not fix this because the OOM happens at the
inter-stage data boundary, not only inside a model forward microbatch.

## Goals

1. Keep the Ray driver on the control plane. The driver may own manifests,
   scheduling decisions, and object references, but not full token tensors or
   teacher top-k tensors.
2. Allow rollout, teacher, and student to use different DP, CP, TP, and PP
   layouts.
3. Bound memory with chunked streaming and backpressure.
4. Preserve deterministic sample order and loss alignment.
5. Preserve existing sequence packing, dynamic batching, and
   tokenizer-in-worker behavior where the backend already supports them.
6. Keep the first implementation simple enough to land incrementally.

## Non-goals

- This design does not require a new generation backend.
- This design does not require direct GPU-to-GPU transfer in the first version.
  Ray object references are acceptable initially if the driver does not call
  `ray.get()` on large payloads.
- This design does not require changing OPD loss semantics.
- This design does not require a fully fault-tolerant mid-step replay system.
  Step-level checkpointing remains the recovery boundary unless explicitly added.

## Design Principles

### Driver as Control Plane

The driver should create the step plan, call actor methods, hold object
references, collect scalar metrics, and enforce stage ordering. It should not
materialize full rollout batches, top-k teacher tensors, or student training
microbatches.

### Explicit Data Plane

Large tensors move as chunk references between producer and consumer actors. A
chunk is the unit of memory accounting, transfer, retry, and backpressure. The
chunk size should be bounded by either `max_chunk_tokens` or `max_chunk_bytes`.

### Full-sequence DP-sharded Stage Boundary

The default inter-stage layout is:

```text
layout = dp_fullseq
```

Each chunk owns complete token sequences for a set of samples. CP, TP, and PP are
internal implementation details of the stage that consumes the chunk.

This is intentionally conservative. A full-sequence boundary makes different CP
sizes easy to support because each consumer can apply its own CP split, sequence
packing, and dynamic batching locally. A CP-sharded boundary can be added later
as an optimization, but it should not be the first API boundary.

### Sparse Loss-active Annotations

Teacher annotations should be emitted only for positions that can contribute to
the distillation loss. Current OPD computes the per-token loss with
`token_mask[:, 1:]`, so the sparse positions are logit positions `t` where the
next token at `t + 1` is loss-active.

This avoids teacher top-k payload for prompt tokens, tool/environment tokens,
padding, and the final non-predictive position.

### Stable Sample IDs

Every generated sample gets a stable ID:

```text
sample_id = (step, prompt_index, generation_index)
```

All chunks carry `sample_id`, `sample_order`, `update_group`, and
`global_batch_slot` metadata. This is the contract that allows different stages
to reorder for efficiency while preserving final sample order, logging, and
optimizer-step grouping.

## Proposed Pipeline

The new pipeline splits metadata from large tensors:

```text
Driver
  |
  | StepManifest, BatchManifest, and stage calls
  v
Rollout actors
  |
  | ShardedBatchStream[TokenChunk]
  v
Teacher actors
  |
  | ShardedBatchStream[AnnotatedTokenChunk]
  v
Student actors
  |
  | scalar metrics
  v
Driver
```

The driver never performs a blocking `ray.get()` on token chunks or teacher top-k
chunks. It only waits for completion handles and scalar metric references.

### Step and Batch Manifests

There are two manifest levels:

- `StepManifest` is known before rollout. It describes prompts, generation
  counts, sampling settings, and stable sample IDs.
- `BatchManifest` is known after rollout normalization. It describes generated
  sequence lengths, sample masks, and loss spans.

Both manifests are small enough for the driver to own and serialize. The driver
must not depend on generated token tensors to construct either one. Generated
lengths and loss spans may be returned as small metadata objects from rollout
normalization, while full `input_ids` remain in stream chunks.

```python
from dataclasses import dataclass
from typing import Any, Literal

import torch
import ray


@dataclass(frozen=True)
class SpanTable:
    # CSR-style loss-active logit spans. For sample i, spans live in
    # [offsets[i], offsets[i + 1]). Span values are half-open intervals over
    # unpadded logit positions, not target token positions.
    offsets: torch.Tensor       # int64, shape [num_samples + 1]
    starts: torch.Tensor        # int32/int64, shape [num_spans]
    ends: torch.Tensor          # int32/int64, shape [num_spans]


@dataclass(frozen=True)
class StepManifest:
    batch_id: str
    step: int
    sample_ids: torch.Tensor    # int64, shape [B]
    sample_order: torch.Tensor  # int64, shape [B], canonical global order
    update_group: torch.Tensor  # int64, shape [B], optimizer update group
    global_batch_slot: torch.Tensor # int64, shape [B], slot within update group
    prompt_ids: torch.Tensor    # int64, shape [B]
    generation_ids: torch.Tensor # int16/int32, shape [B]
    max_sequence_length: int
    tokenizer_name_or_path: str
    tokenizer_config: dict[str, Any]
    processor_config: dict[str, Any] | None = None
    sampling_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class BatchManifest:
    batch_id: str
    step: int
    sample_ids: torch.Tensor    # int64, shape [B]
    sample_order: torch.Tensor  # int64, shape [B], canonical global order
    update_group: torch.Tensor  # int64, shape [B], optimizer update group
    global_batch_slot: torch.Tensor # int64, shape [B], slot within update group
    prompt_ids: torch.Tensor    # int64, shape [B]
    generation_ids: torch.Tensor # int16/int32, shape [B]
    input_lengths: torch.Tensor # int32, shape [B]
    sample_mask: torch.Tensor   # bool/float, shape [B]
    loss_spans: SpanTable       # next-token loss-active spans
    max_sequence_length: int
    tokenizer_name_or_path: str
    tokenizer_config: dict[str, Any]
    processor_config: dict[str, Any] | None = None
    multimodal_manifest_ref: ray.ObjectRef | None = None
```

`loss_spans` is preferred over a dense `token_mask` at the boundary. A dense mask
can still be materialized inside a worker when the existing backend needs it.
`update_group` and `global_batch_slot` preserve the grouping semantics currently
created by `shard_by_batch_size(..., batch_size=train_global_batch_size)`. The
reshard planner may change physical chunk placement, but it must not silently
change which samples belong to the same optimizer update.

### Token Chunks

`TokenChunk` carries the generated full sequences for a bounded set of samples.
It may be produced by rollout workers or by a normalization stage that flattens
message logs into tokens.

```python
@dataclass(frozen=True)
class TokenChunk:
    batch_id: str
    chunk_id: int
    sample_ids: torch.Tensor
    sample_order: torch.Tensor
    update_group: torch.Tensor
    global_batch_slot: torch.Tensor
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    sample_mask: torch.Tensor
    loss_spans: SpanTable
    multimodal_ref: ray.ObjectRef | None = None
```

`input_ids` remains right-padded within the chunk. Chunk-local padding is allowed
to differ across chunks.

### Teacher Annotation Chunks

Teacher output keeps the same top-k logits contract as the legacy dense path in
the first streaming implementation. The loss already owns normalization, and
keeping logits in Phase 2A preserves exact current semantics while changing only
the transport boundary.

```python
@dataclass(frozen=True)
class TeacherTopKChunk:
    batch_id: str
    chunk_id: int
    sample_ids: torch.Tensor
    sample_order: torch.Tensor
    update_group: torch.Tensor
    global_batch_slot: torch.Tensor
    # Flattened over loss-active positions, not over padded sequence length.
    position_offsets: torch.Tensor # int64, shape [num_samples + 1]
    positions: torch.Tensor        # int32/int64, shape [num_positions]
    topk_logits: torch.Tensor      # fp16/bf16/fp32, shape [num_positions, K]
    topk_indices: torch.Tensor     # int32/int64, shape [num_positions, K]
```

Later phases may normalize on the teacher worker, but that should be a separate
loss-contract change with parity checks for forward, reverse, and mixed KL.

Position schema:

- `positions` are sample-local, unpadded logit positions.
- Position `t` means the model logits at sequence position `t`, used to predict
  target token `input_ids[t + 1]`.
- Valid positions satisfy `0 <= t < input_lengths[i] - 1`.
- `positions` must correspond exactly to dense `token_mask[:, 1:] == 1`.
- For sample `i` in `sample_ids` order, positions live in
  `positions[position_offsets[i]:position_offsets[i + 1]]`.
- If a student worker packs or dynamically batches sequences, it must create a
  pack map from `(sample_id, position)` to local packed coordinates before
  gathering student logprobs.

The student still needs the original token chunk. The annotated stream should
therefore pair teacher annotations with the token chunk that produced them,
rather than replacing the token stream:

```python
@dataclass(frozen=True)
class AnnotatedTokenChunk:
    batch_id: str
    chunk_id: int
    token_chunk_ref: ray.ObjectRef
    teacher_topk_ref: ray.ObjectRef
    end_of_stream: bool = False
```

The teacher stage should avoid copying `input_ids` into its output. It should
forward references to the original token chunks and emit only annotation payloads.

### Sharded Batch Stream

The stream object is a lightweight handle that names the layout and provides one
bounded stream per DP shard.

```python
@dataclass(frozen=True)
class ShardedBatchStream:
    batch_id: str
    layout: Literal["dp_fullseq"]
    dp_size: int
    manifest: BatchManifest
    shard_streams: list[ray.ObjectRefGenerator]
```

`shard_streams[i]` yields chunk object references for DP shard `i`. A consumer
can either consume the stream directly or reshard it into its target DP layout.

## Handling Different Parallel Layouts

### Consumer-side Normalization Contract

The stream boundary carries semantic sequences, not a stage-specific padded
layout. Every consumer must normalize a chunk before model execution:

- Treat `input_lengths` and loss spans as authoritative.
- Treat `input_ids` padding as chunk-local transport padding only.
- Re-pad to the consumer stage's required multiple before forward passes. This
  may differ between teacher and student because CP/TP/PP and packed-sequence
  requirements differ.
- Rebuild attention masks, position IDs, packed-sequence metadata, and dynamic
  microbatch metadata after repadding.
- Never infer valid tokens from chunk-local padded shape.

Each stream consumer should advertise a small layout contract:

```python
@dataclass(frozen=True)
class StageLayout:
    dp: int
    cp: int
    tp: int
    pp: int
    pad_multiple: int
    supports_sequence_packing: bool
    supports_dynamic_batching: bool
    supports_multimodal: bool
```

The v1 implementation should explicitly inherit current backend limits rather
than claiming new combinations are supported. For example, if a backend rejects
CP with sequence packing, dynamic batching with PP, or multimodal inputs with CP
or sequence packing, the stream path should reject the same configuration before
submitting actor work. The data-pipeline change removes the driver gather; it
does not by itself expand backend parallelism support.

Initial v1 capability matrix:

| Capability | V1 support |
| --- | --- |
| Text-only OPD | Supported target |
| Rollout, teacher, and student with different DP | Supported through sample-axis resharding |
| Different teacher and student CP | Supported through full-sequence chunk boundary |
| Sequence packing | Supported only where the existing backend already supports it |
| Dynamic batching | Supported only where the existing backend already supports it |
| Multimodal OPD | Not a v1 target unless exact per-sample media refs and processor metadata are added |
| CP with multimodal or sequence packing | Follow existing backend restrictions |
| Dynamic batching with PP | Follow existing backend restrictions |

### DP Mismatch

DP mismatch is handled by sample-axis resharding. A `DPReshardPlanner` maps
sample IDs from source DP shards to destination DP shards. The planner should
balance by estimated valid loss tokens, not only by sample count, because OPD
sequence lengths vary widely.

```python
class DPReshardPlanner:
    def plan(
        self,
        manifest: BatchManifest,
        src_dp_size: int,
        dst_dp_size: int,
        max_chunk_tokens: int,
    ) -> list[list["ChunkAssignment"]]:
        ...
```

The driver may compute the plan because it is metadata-only. Source actors then
emit per-destination chunks directly. The driver keeps the resulting object
references or stream proxies but does not materialize chunk payloads.

`sample_order` is mandatory. The planner may rebalance physical chunks, but it
must preserve a canonical global order for logging, deterministic comparisons,
and optimizer-step grouping decisions. Any intentional change to current
global-batch grouping semantics should be a separate, measured behavior change.

### CP Mismatch

The first implementation should not expose CP-sharded tensors at the stage
boundary. Each chunk contains full sequences. The consumer stage applies its own
CP transform after it receives the chunk:

- Teacher CP is used only inside teacher top-k computation.
- Student CP is used only inside student training.
- Sequence packing and dynamic batching happen inside the consuming stage.

This avoids coupling rollout, teacher, and student to one CP degree or one
packed-sequence layout.

### Collective Count Safety

Any worker group that uses TP, CP, PP, or vocab-parallel collectives must execute
the same collective sequence on every participating rank. Streaming cannot let
one rank finish early, skip a chunk, or enter the next phase while another rank
is still executing collectives for the previous chunk.

The implementation must choose one of these policies per stage:

- Equalize per-rank microbatch counts with dummy microbatches, matching the
  existing sequence-packing behavior.
- Build chunk groups that have identical collective counts across all ranks.
- Add an explicit flush/drain barrier before a group advances to the next
  collective phase.

The reshard planner may balance valid tokens, but it must also emit the
microbatch-count metadata needed to enforce one of these policies.

### TP and PP Replication

TP and PP should not multiply returned CPU tensors.

For teacher top-k:

- TP ranks may cooperate to compute global top-k.
- CP ranks may cooperate to gather sequence positions needed by the output
  chunk.
- PP should only return from the final stage or from one elected rank per DP
  shard.
- Replicated axes should be filtered before CPU transfer when possible, not only
  after `ray.get()`.

As an immediate containment change, `MultiWorkerFuture.get_results()` should
avoid `ray.get()` on futures that are not in `return_from_workers`. This does not
solve the full data-pipeline problem, but it removes one avoidable replicated
materialization path. It is not sufficient by itself: non-return ranks must also
avoid returning large tensors into Ray object storage. They should return `None`
or a small sentinel, or the worker group should avoid calling them for the output
path.

## Execution Flow

The driver loop becomes:

```python
step_manifest = step_planner.make_step_manifest(
    batch,
    num_generations_per_prompt,
)

rollout_stream = student_generation.rollout_stream(
    step_manifest,
    greedy=False,
)

token_stream = rollout_normalizer.to_token_stream(
    rollout_stream,
    max_sequence_length=policy_cfg["max_total_sequence_length"],
)

annotated_stream = teacher_policy.annotate_topk_stream(
    token_stream,
    k=distillation_cfg["topk_logits_k"],
    target_layout=student_policy.stage_layout(),
)

train_metrics = student_policy.train_distillation_stream(
    annotated_stream,
    loss_fn=loss_fn,
)
```

`rollout_normalizer.to_token_stream()` may initially run inside rollout workers
or as a small actor pool. Long term, rollout should emit token chunks directly so
message logs do not return to the driver.

## API Additions

### Policy Interface

Add stream-oriented methods next to the existing batch APIs.

```python
class PolicyInterface:
    def annotate_topk_stream(
        self,
        token_stream: ShardedBatchStream,
        *,
        k: int,
        target_layout: StageLayout,
        max_chunk_tokens: int | None = None,
        timer: Timer | None = None,
    ) -> ShardedBatchStream:
        ...

    def train_distillation_stream(
        self,
        annotated_stream: ShardedBatchStream,
        loss_fn: DistillationLossFn,
        *,
        gbs: int | None = None,
        mbs: int | None = None,
        timer: Timer | None = None,
    ) -> dict[str, Any]:
        ...
```

Existing `get_topk_logits()` and `train()` remain for compatibility. The stream
implementation can initially wrap existing per-worker code, then remove the
driver gather path incrementally.

Implemented v1 currently exposes three non-legacy modes:

- `stream_teacher` streams teacher top-k refs into a student dense-batch
  compatibility path. It does not support multimodal OPD, teacher dynamic
  batching, or teacher sequence packing.
- `stream_rollout` adds a post-rollout token normalizer and a ref-only student
  consumer. It still receives rollout message logs on the driver in v1; it
  avoids passing the dense post-rollout train batch into student training. It
  rejects async rollouts, multimodal OPD, dynamic batching, and sequence
  packing.
- `sparse_loss` reuses the stream-teacher transport and attaches sparse
  teacher top-k tensors in student workers. It avoids dense `[B, S, K]` teacher
  tensors in that mode, and rejects multimodal OPD, dynamic batching, sequence
  packing, and context parallelism in v1.

`stage_layout()` is a proposed public helper. It should be derived from the
existing sharding annotations and batching-mode config; the stream API should not
duplicate independent layout state that can drift from the policy worker group.

### Worker Group Utilities

`RayWorkerGroup` should expose a stream-preserving path:

```python
class RayWorkerGroup:
    def run_all_workers_sharded_stream(
        self,
        method_name: str,
        *,
        data_stream: ShardedBatchStream,
        in_sharded_axes: list[str],
        output_layout: str,
        common_kwargs: dict[str, Any] | None = None,
    ) -> ShardedBatchStream:
        ...
```

When worker methods return `ObjectRefGenerator`, the worker group should pass
stream proxies to the next stage without consuming them on the driver.

`MultiWorkerFuture.get_results(..., return_generators_as_proxies=True)` already
provides part of this behavior. A dedicated stream method is still useful because
the OPD path needs stage layout metadata, selected-return-rank filtering,
end-of-stream handling, and cancellation/teardown semantics in one place instead
of relying on each caller to use the proxy path correctly.

### Stage Execution

V1 should preserve the current serialized stage lifecycle, especially for
colocated clusters:

1. Finish rollout and generation cleanup.
2. Run teacher annotation.
3. Offload or release teacher inference state.
4. Run student training.

Chunked streams bound memory within each stage, but they should not overlap
teacher inference and student training on the same colocated resources until a
separate scheduler proves that the overlap is safe.

## Backpressure and Memory Bounds

Every stream-producing stage should enforce chunk-size bounds in v1:

- `max_chunk_tokens`
- `max_chunk_bytes`
- optional `max_chunk_loss_positions`

Future queue or stream-manager transports should add real backpressure knobs
such as `max_inflight_chunks` and `max_inflight_bytes` once the producer/consumer
ack protocol is implemented. They are not part of the v1 public config because
the current stream transport does not throttle by in-flight object count.

The consumer acknowledges chunks after it has transferred or trained on them.
The producer can then release local CPU tensors and Ray object references.

The chunk size should be large enough to avoid one Ray object per sample, but
small enough to keep a single chunk below object-store and driver scheduling
limits. A reasonable initial target is to chunk by 1 to 4 student training
microbatches or by a fixed valid-token budget.

### Ray Object Lifecycle

The stream protocol must define object ownership explicitly:

1. Producer creates a payload object and yields an `AnnotatedTokenChunk` or
   `TokenChunk` reference.
2. Driver may hold stream proxies and small chunk envelopes, but it must not call
   `ray.get()` on payload refs.
3. Consumer owns the payload ref while transferring, repadding, annotating, or
   training the chunk.
4. Consumer sends an ack after the payload is no longer needed.
5. Producer drops local references after ack. If a stream manager actor is used,
   it also drops bookkeeping refs at this point.
6. Producer emits an explicit EOS marker per DP shard.
7. On cancel, timeout, or worker failure, the driver broadcasts stream cancel to
   producers and consumers, waits for teardown, and drops all known refs before
   the next step.

The implementation should avoid long-lived driver-owned lists of payload refs.
Such lists pin objects in Ray object storage even if no worker is using them.
Failure tests should verify object-store bytes return near baseline after
success, cancel, and mid-stream worker death.

## Loss Integration

The existing loss path expects dense `teacher_topk_logits` and
`teacher_topk_indices` aligned with dense `input_ids`. The streaming path should
add a sparse-aware preprocessing step inside the student worker:

1. Receive `TokenChunk` and matching `TeacherTopKChunk`.
2. Build the student forward microbatch using local sequence packing or dynamic
   batching.
3. Gather student logprobs at the teacher top-k indices only for active
   positions.
4. Apply the same KL formula as `DistillationLossFn`.
5. Reduce scalar metrics locally, then all-reduce across student DP as needed.

For compatibility, the first version can densify teacher annotations inside the
student worker before calling the existing loss utility. That keeps correctness
simple while still removing driver gather. The intended final state is to avoid
dense `[B, S, K]` teacher tensors entirely.

If Phase 1 uses this compatibility path, densification must happen per chunk or
per microbatch after local packing decisions. It must not rebuild a full student
DP shard of dense teacher annotations, or the OOM simply moves from the driver to
student workers.

## Migration Plan

### Phase 0: Mandatory Containment

- Change `MultiWorkerFuture.get_results()` so it only calls `ray.get()` for
  selected `return_from_workers`.
- Avoid CPU-copying top-k tensors on non-return ranks when the output is
  replicated across TP/CP/PP.
- Add memory metrics around teacher top-k output size and driver RSS.

This phase is a prerequisite for the stream API. It is low risk and reduces
replicated payloads, but it does not remove the driver gather.

### Phase 1: Stream Teacher Annotations

- Add `ShardedBatchStream`, `BatchManifest`, `TokenChunk`,
  `TeacherTopKChunk`, and `AnnotatedTokenChunk`.
- Keep rollout flattening unchanged initially.
- Replace `teacher_policy.get_topk_logits(train_data)` in OPD with
  `teacher_policy.annotate_topk_stream(token_stream, ...)`.
- Pass annotation chunk refs directly into student training.

This is the first phase that attacks the largest teacher-annotation driver OOM.
It may not be sufficient for every 512-sample long-context run because rollout
flattening still happens on the driver until Phase 2, and backend-internal CP/PP
top-k paths can still create dense temporary tensors.

### Phase 2: Stream Rollout Output

- Make rollout return token chunks or message-log chunks as stream refs.
- Move `batched_message_log_to_flat_message` off the driver.
- Build `loss_spans` during rollout normalization.

This removes the other large driver-owned tensor boundary.

### Phase 3: Sparse Student Loss

- Avoid densifying teacher annotations inside student workers.
- Compute student top-k logprobs and KL on sparse active positions.
- Add sparse metrics for valid tokens, skipped positions, and top-k coverage.

### Phase 4: Transport Optimizations

- Consider GPU-aware transport, colocated actor handoff, or NCCL all-to-all for
  specific same-node layouts.
- Consider CP-sharded stage boundaries only after the full-sequence stream API is
  stable.

### Memory Budget by Phase

| Phase | Driver RSS | Ray object store | Teacher worker peak | Student worker peak |
| --- | --- | --- | --- | --- |
| Current | Full rollout batch plus full dense teacher top-k | Full dense worker returns and concat inputs | Dense `[B_shard, S, K]`, with CP/PP replicated outputs possible | Dense teacher top-k after reshard |
| Phase 0 | Same semantic payloads, fewer replicated `ray.get` results | Fewer non-return-rank results if workers also suppress payloads | Lower only if non-return ranks avoid CPU payloads | Same as current |
| Phase 1 | No full teacher top-k on driver; rollout flattening may remain | Bounded teacher annotation chunks | Bounded by chunk size, but internal CP/PP dense temps remain | Bounded only if densification is per microbatch/chunk |
| Phase 2 | No full rollout tokens or teacher top-k on driver | Bounded rollout and teacher chunks | Same as Phase 1 | Same as Phase 1 |
| Phase 3 | Metadata and scalar metrics only | Sparse annotation chunks | Sparse active-position output | Sparse loss path, no dense `[B, S, K]` rebuild |

Each phase should log the metrics in this document before claiming scale
improvement.

## Verification Plan

Correctness tests:

- Compare stream and non-stream OPD losses on a small deterministic batch.
- Test `DPReshardPlanner` for `src_dp != dst_dp`, uneven sequence lengths, and
  stable sample ordering.
- Test `update_group` and `global_batch_slot` preservation against the existing
  `shard_by_batch_size(..., batch_size=train_global_batch_size)` grouping.
- Test sparse loss-active positions against dense `token_mask[:, 1:]`.
- Test teacher top-k logprob output against the existing dense `topk_logits`
  path as a versioned new contract, not as a reinterpretation of the old one.
- Test sparse versus dense OPD loss for plain, sequence-packed, dynamic-batched,
  and CP-enabled batches where each backend supports the configuration.
- Test pack-map and unpack-map correctness for `(sample_id, position)` lookups.
- Test sequence packing and dynamic batching consumers with streamed chunks,
  including dummy microbatch insertion or another collective-count safety policy.

Scale tests:

- Run a short OPD smoke test with `num_prompts_per_step=128`,
  `num_generations_per_prompt=4`, long context, and `topk_logits_k=64`.
- Track driver RSS, Ray object-store bytes, in-flight chunk count, chunk bytes,
  and teacher annotation bytes.
- Verify that peak driver memory is approximately independent of global batch
  token count once chunk metadata is excluded.

Failure tests:

- Force a teacher worker failure before and after emitting a chunk.
- Verify that chunk IDs make duplicate detection possible.
- Verify that a failed step can be retried from the step boundary without stale
  chunk refs leaking into the next step.
- Verify stream EOS handling for every DP shard.
- Verify cancel and timeout teardown release Ray object-store memory.
- Verify slow consumers apply backpressure instead of allowing unbounded producer
  chunk creation.

## Metrics

Add per-step metrics:

- `opd/data/token_chunk_bytes`
- `opd/data/teacher_annotation_bytes`
- `opd/data/inflight_chunks`
- `opd/data/inflight_bytes`
- `opd/data/sparse_position_ratio`
- `opd/data/driver_rss_bytes`
- `opd/data/ray_object_store_bytes`
- `opd/data/reshard_src_dp`
- `opd/data/reshard_dst_dp`
- `timing/opd/rollout_stream`
- `timing/opd/teacher_annotation_stream`
- `timing/opd/student_train_stream`

These metrics should be logged before and after each migration phase so the
memory improvement is measurable.

## Open Questions

1. Should a later sparse-loss phase switch teacher payloads from logits to
   log-probabilities? Doing so would reduce repeated normalization but must
   validate numerical drift for forward, reverse, and mixed KL.
2. Should the first stream implementation use Ray `ObjectRefGenerator`, explicit
   actor queues, or a small stream manager actor? `ObjectRefGenerator` is the
   lightest API, but queue actors may make backpressure and cancellation clearer.
3. How much mid-step fault tolerance is needed? If step-boundary replay is
   enough, stream state can stay simple.
4. Should rollout normalization produce token chunks directly, or should it first
   stream message-log chunks for easier compatibility with multi-turn
   environments?
5. How should multimodal payloads be chunked for VLM OPD? The manifest should
   keep large image/audio tensors behind refs, but the exact ownership model
   needs validation.
6. Should the public layout API be a new `StageLayout` object or a wrapper around
   existing `NamedSharding` plus batching-mode config?
7. Should v1 reject unsupported backend combinations at config validation time or
   at stream construction time? Config-time rejection is preferable, but some
   constraints are backend-specific.

## Expected Outcome

The new pipeline should make OPD memory scale with chunk size and per-stage
worker layout rather than with the full global batch on the driver. It should
also make DP and CP mismatch explicit, allowing rollout, teacher, and student to
use the layouts that fit their workloads without forcing an all-data gather and
repartition on the Ray driver.
