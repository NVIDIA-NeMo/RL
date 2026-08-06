# Multimodal payload deduplication

Multimodal GRPO repeats each prompt for multiple generations. Without
deduplication, the driver, Ray object store, and replay buffer may store the
same image, video, or audio payload once per logical generation even though
the model must eventually receive the same logical batch.

The deduplication feature keeps one physical copy of verified-equivalent media
and preserves a logical row-to-segment mapping until the common policy-worker
materialization boundary. It does not reduce the logical model batch or media
encoder compute.

## Configuration and supported transport

The schema defaults both options to disabled when a recipe omits them:

```yaml
grpo:
  deduplicate_multimodal_data: false
  debug_payload_metrics: false
```

The shared VLM roots declare these disabled defaults. Qualified Nemotron Omni
recipes opt in to deduplication so they continuously exercise that path; other
VLM and text-only recipes remain opt-in. Payload metrics stay disabled because
protocol-5 sizing is diagnostic work, and qualification runs enable it
explicitly.

`deduplicate_multimodal_data=true` is supported for vLLM generation with the
legacy Ray argument transport (`data_plane.enabled=false`). Configuration
validation fails early for other generation backends and for the TransferQueue
data plane rather than silently running an unqualified path.

The implementation covers both synchronous and asynchronous GRPO, including
NeMo Gym image rollouts, DAPO cache concatenation, replay push/sample, replay
checkpoints, validation, log-probability calculation, policy training, and
KV-cache calibration.

## Representation and safety contract

`PackedTensor` has two representations:

- Legacy values store one physical tensor entry per logical row.
- Deduplicated values store physical media segments plus CSR-style
  `row_offsets` and `segment_indices`.

Each physical segment receives opaque provenance when deduplication is first
enabled. Provenance survives deepcopy, pickle, replay, slicing, and sharding.
Concatenation re-interns segments only when their provenance matches.

Prompt or problem identity is never evidence that media is equal. It may
narrow a search, but two trajectories from one prompt can receive different
media from a tool or environment. Those segments receive different provenance
and stay distinct.

The following invariants apply:

- Logical row count and order do not change.
- `PackedTensor.as_tensor()` reconstructs logical segment order before dynamic
  shape padding and model-owned sequence packing.
- Dtype, device, packing dimension, and `pad_to_max_shape` are preserved.
- Mutable row, conversation, tool, and Gym request containers are copied.
  Only explicit immutable media leaves are shared.
- Model-visible labels, masks, rewards, advantages, token-type tensors, and
  log probabilities are never deduplicated.
- Coupled inputs such as `pixel_values`/`image_grid_thw`,
  `pixel_values`/`imgs_sizes`, `pixel_values`/`num_frames`, and
  `pixel_values_videos`/`video_grid_thw` must have compatible ordered per-row
  logical segment counts before materialization.

Operations that change logical rows—slice, filter, reorder, chunk, dynamic
batching, sequence packing, ordinary sharding, and replay concatenation—remap
the CSR indices. Each final worker shard re-interns matching provenance
locally. Copies required on different Ray workers are not incorrectly counted
as avoidable.

## Where physical copies are removed

There are four related boundaries:

1. Prompt repeat copies row containers but shares immutable media leaves.
2. Conversation flattening merges ordered media segments without first
   materializing one full tensor per repeated row.
3. Replay/DAPO concatenation represents a missing media key as an explicit
   empty logical row when the dedup flag is enabled.
4. Final dynamic/sequence-packed worker shards re-intern provenance after
   sharding has scattered prompt groups.

For vLLM generation, native `vllm_content`, `vllm_images`, `vllm_videos`, and
`vllm_audios` remain the generation representation. Redundant policy-ready
`PackedTensor` media is omitted from a generation call only when every active
row has one of the image, video, or audio side channels that the vLLM formatter
actually consumes. Raw content alone and unconsumed path metadata are never
used to justify suppression. Policy media remains attached to the trajectory
for later training.

Image, video, and audio leaves use the same ownership and payload-measurement
rules in the generic non-Gym representation. Processor-produced model inputs
and metadata use the generic `PackedTensor` path; native vLLM side channels use
explicit key and typed content recognition. The implementation does not use
model-name branches.

## Gym and replay

NeMo Gym may return trajectory-specific images. Initial images are processed
once on the unrepeated prompt only when deduplication is enabled, then
reattached by user-turn ordinal after the Gym Ray call. New Gym images are left
untouched. This avoids resending the original large policy payload through Gym
and avoids substituting images across generations. NeMo Gym audio and video
lineage are not implemented or qualified by this change; those media remain
supported only by the generic non-Gym path.

Sparse text-only and multimodal trajectory groups are normalized only at the
GRPO replay/DAPO call sites. The default `BatchedDataDict.from_batches()`
missing-key behavior remains strict for other algorithms.

Replay checkpoints store the compact representation directly. Save and restore
run inside the replay actor, so a buffer-sized state dict is not copied through
the long-lived driver frame. `checkpointing.save_replay_buffer=false` can skip
that checkpoint entirely for especially large runs; resume then regenerates
trajectories. The mapping is self-describing. Async replay assembly detects
restored compact tensors even when the current flag is off, so sparse flag-on
checkpoints can be resumed flag-off; legacy checkpoints also remain readable
after enabling the flag. When legacy and compact batches later concatenate,
legacy segments receive fresh provenance and restored compact segments remain
shared.

## Policy backends and context parallelism

All policy backends use the existing common materialization call,
`get_multimodal_dict(as_tensors=True)`.

| Policy path | Dedup data-path status |
| --- | --- |
| AutoModel/DTensor, CP=1 | Supported by the common worker boundary; exact worker materialization is unit tested, and Qwen2.5-VL G=16 is exercised end to end. |
| Megatron, CP=1 | Supported by the common worker boundary and model-owned packing. |
| Megatron, CP=2 | Supported by the data representation; two-rank Nemotron model-ingress, logprob, loss, and gradient parity is tested. |
| AutoModel VLM, CP>1 | Not claimed; the upstream VLM worker currently rejects this topology independently of deduplication. |
| Megatron, CP>2 | Shared data operations are topology-independent, but model-level qualification is not yet claimed. |

Model-family support follows processor and backend support. Qwen, Nemotron, and
other VLMs do not need dedicated dedup branches, but a family is only
end-to-end qualified when its maintained recipe and checkpoint have been run
with the feature enabled. This change is exercised with the maintained
Nemotron NeMo Gym image recipe and a Qwen2.5-VL native-rollout recipe on the
Megatron policy backend. Qwen2.5-VL also exercises the AutoModel CP=1 data path
at G=16 with the expected payload reduction. AutoModel TMPE is not used as
correctness evidence for that qualification, and exact trajectory identity
after independently updating and refitting two policies is not claimed.
Concrete job IDs, W&B runs, parity results, and transport measurements belong
in the pull-request validation report so they remain tied to the exact code
revision that was run.

Gemma is excluded from this change's evaluation set. Its vLLM generation and
AutoModel policy paths have a pre-existing token-logprob mismatch with
deduplication both disabled and enabled, so that model cannot provide a valid
deduplication correctness signal. Resolving the Gemma backend mismatch is
separate work. Audio/video model runs remain unqualified; their shared data
primitives are covered by focused tests.

## Payload metrics

`debug_payload_metrics=true` emits stable lines beginning with
`▶ [PAYLOAD]`. Metrics include:

- physical and logical media bytes;
- physical and logical media segment counts;
- estimated saved bytes and physical-to-logical ratio;
- cloudpickle protocol-5 frame plus out-of-band buffer size;
- total and maximum serialized size for exact unique final DP-shard arguments.

The measured Python object is the exact object passed at that Ray boundary.
The protocol-5 serialized size is a serialization proxy, not a claim that it
is an exact Ray object-store allocation. Object-store qualification should
pair it with tracked object IDs or an isolated matched cluster delta. Totals
count each DP-shard object once; they do not multiply bytes for TP/CP replicas
that consume the same Ray object or reference.

When `debug_payload_metrics=false`, call sites return before walking media or
serializing payloads.

## Qualification expectations

Correctness tests compare dedup-on materialization against the legacy
representation exactly. Dynamic resolution, pack dimensions 0 and 1,
multi-turn divergence, missing media keys, replay checkpoints, native
image/video/audio leaves, dynamic batching, sequence packing, and shard-local
re-interning require focused coverage.

End-to-end qualification should record the exact commit, container, model
revision, recipe, overrides, hardware, and Ray version. Pre-shard savings are
compared with `1 - unique_physical/logical_occurrences`; worker savings use the
same formula independently on each final shard.
