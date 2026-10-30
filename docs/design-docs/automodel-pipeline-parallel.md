# Automodel Pipeline Parallelism

This design note describes pipeline parallelism (PP) in NeMo RL's Automodel policy
worker as implemented. Automodel's `AutoPipeline` splits the model into stages and
builds the PyTorch schedule (`torch.distributed.pipelining`), which runs the forward
and backward passes and the inter-stage communication. NeMo RL owns everything with
RL semantics: batch planning, the loss, gradient normalization and clipping, the
optimizer step, logprobs, the reference policy, checkpoints, and weight refit.

The schedule is 1F1B, which places one stage on each PP rank, as in NeMo RL's
Megatron backend. Interleaved and V-shaped schedules, which place several stages on
one rank, are not supported (Section 8).

## 1. Configuration

```yaml
policy:
  train_micro_batch_size: 2
  logprob_batch_size: 2              # must equal train_micro_batch_size
  sequence_packing:
    enabled: false
  dynamic_batching:
    enabled: false
  automodel_cfg:
    pipeline_parallel_size: 2
    defer_fsdp_grad_sync: false      # must be set explicitly; see Section 6
    pipeline:
      microbatches_per_step: 8       # required, >= pipeline_parallel_size
```

`pipeline_parallel_size` is optional in `AutomodelConfig`; an absent key means 1.
The `pipeline` block is the Pydantic model `AutomodelPipelineConfig`
(`nemo_rl/models/policy/__init__.py`):

| Field | Meaning |
| --- | --- |
| `microbatches_per_step` (M) | Microbatches of `train_micro_batch_size` samples per schedule call, at least `pipeline_parallel_size`. Required: M trades the pipeline bubble (about `(pp - 1) / (M + pp - 1)` of each call) against receive-buffer memory (one buffer per microbatch on every non-first stage) and padding (all M microbatches share one sequence length). |

The block rejects unknown keys, so schedule settings such as `pp_schedule` or
`layers_per_stage` fail at startup.

The Ray worker layout is `[PP, DP, CP, TP]` and Automodel's mesh is
`(pp, dp_replicate, dp_shard, cp, tp)`, so ranks line up. Data is sharded over DP and
replicated within each PP group; DP size is `world_size / (pp * cp * tp)`.

Expert parallelism composes with PP: Automodel builds the expert mesh inside each
pipeline stage, so `expert_parallel_size` must divide the `dp * cp * tp` ranks of one
stage (for example EP8 with PP2 on 16 GPUs). Multi-token-prediction heads (such as
Qwen3.5-MoE's MTP layer) are not placed on any stage; RL training does not use them.

### 1.1 Validation

`validate_automodel_pipeline_config` (`nemo_rl/models/policy/utils.py`) runs in
`Policy.__init__` before any worker is created and rejects:

- A missing `pipeline` block, an implicit `defer_fsdp_grad_sync`, or
  `logprob_batch_size != train_micro_batch_size`.
- `M < pp_size`: 1F1B needs at least one microbatch per stage.
- Sequence packing, dynamic batching, sequence parallelism, CPU offload,
  LoRA, draft co-training, router replay, reward models, ModelOpt quantization,
  multimodal (VLM) models, non-colocated generation, and any `refit_transport`.
- Models whose embedding and LM head share one weight (`tie_word_embeddings`).

SFT additionally requires `sft.val_micro_batch_size == policy.train_micro_batch_size`.
Top-k logits, full logits, and reward scoring raise `NotImplementedError` in the worker.

## 2. Pipeline Stage

`setup_model_and_optimizer` passes a `PipelineConfig` built by `build_pipeline_config`
to `DistributedSetup`, and Automodel's `from_pretrained` returns an `AutoPipeline`. The
NeMo RL-owned fields are:

| Field | Value | Why |
| --- | --- | --- |
| `pp_schedule` | `1F1B` | One stage per PP rank. |
| `pp_microbatch_size`, `pp_batch_size` | `mbs`, `mbs * M` | One schedule call runs M microbatches. |
| `scale_grads_in_schedule` | `False` | The loss adapter already applies the non-PP backward scale (Section 4). |
| `pp_seq_len` | `max_total_sequence_length` | Stage shapes are precomputed, so the schedule uses static stage metadata instead of shape inference. |
| `dtype` | FSDP `output_dtype` (fp32) | Activations between stages carry the mixed-precision output dtype, not the parameter dtype. |
| `loss_fn` | `PipelineOutputAdapter()` | The RL loss runs on the last stage. |

Under 1F1B each rank owns one stage, `AutoPipeline.parts[0]`. Setup treats that stage
as the model, so `worker.model`, `worker.optimizer`, and `worker.scheduler` are the
stage, its optimizer, and its LR scheduler, as in the Megatron worker, and
`worker.pipeline` holds the `AutoPipeline` (`None` without PP). Lifecycle code (mode
switches, zero-grad, clipping, optimizer and scheduler steps, offload, reference swap)
is the same with and without PP; only execution (Section 4), logprobs, refit, and the
checkpoint's optimizer namespace (Section 5) branch on `worker.pipeline`. Worker
features with no PP implementation either raise in the worker or are rejected by
validation (Section 1.1), so no non-PP code path runs on a single stage.

## 3. Training Update Lifecycle

`train()` runs each global batch as one optimizer update in three phases:

```text
process_global_batch          global valid sequence/token counts, summed over DP
_open_update                  zero_grad; record loss post-processor and counts
_run_update_chunk(batch)      PP=1: automodel_forward_backward; PP>1: pipeline schedule
_close_update                 clip (PP-aware) -> optimizer.step -> broadcast metrics
...                           (next global batch)
_finish_train_call            zero_grad, scheduler.step, empty_cache, aggregate metrics
```

The normalization counts are always known before the first chunk (`GlobalCounts`);
`_open_update(global_counts=None)` is reserved for streaming updates whose counts are
known only at close and raises `NotImplementedError`. `_abort_update` drops an open
update's gradients and metrics. `_run_update_chunk` does not assume one chunk is the
whole update, and nothing in it changes when an update runs several chunks.

## 4. Pipeline Execution

### 4.1 Batch Planning

`plan_execution_batches` (`nemo_rl/models/automodel/pipeline_batch.py`) turns the data
of one chunk (or one logprob request) into execution batches of `mbs * M` samples. The
tail is padded with copies of the first sample whose `sample_mask` and `token_mask` are
zero, and the number of execution batches is all-reduced (max) over the DP group, so
every DP rank issues the same number of schedule calls and FSDP collectives. Every
execution batch keeps the data's padded sequence length.

Each row gets a sample ID (its position in the planned data). The schedule receives the
IDs as the loss `target` and splits them along dim 0 exactly like the model inputs.

With context parallelism, Automodel's `ContextParallelSharder` shards each execution
batch once for all of its microbatches (`num_chunks=M`), as Automodel's own pipeline
trainer does. The schedule call runs inside the sharder's context, which may shard the
inputs in place, so the stage shapes are set from the input length inside it: the local
shard for the generic round-robin layout, or the full length for models that shard in
their own forward and describe their local activations through
`get_pipeline_stage_metas`. The microbatches' canonical data stays unsharded, and the
loss adapter maps each microbatch's targets and logprobs through the same sharder, which
is valid because the layout depends only on the sequence position.

### 4.2 Loss Adapter

`PipelineOutputAdapter` (`nemo_rl/models/automodel/pipeline_parallel.py`) is the
schedule's `loss_fn`. The schedule calls it on the last stage once per microbatch as
`loss_fn(logits, sample_ids, context=PipelineCallContext)`. The adapter maps the IDs to
the microbatch they select and calls `post_process_logits`, the same temperature,
masking, logprob, and loss path the non-PP worker uses. Results are written by
microbatch index, so an extra call (for example during dynamic metadata inference)
overwrites instead of double-counting.

| Mode | Returned value | Recorded result |
| --- | --- | --- |
| `train` | `loss * dp_size * cp_size / cp_gradient_fanout` (the non-PP backward scale; zero for padding microbatches) | detached loss and metrics |
| `eval` | detached loss | loss and metrics; the logits storage is freed |
| `logprobs` | zero | token logprobs; the logits storage is freed |

Forward-only calls keep every microbatch's output in the stage's forward cache until
the next call, so the adapter frees the logits storage after use and the runner calls
`stage.clear_runtime_states()` after each forward-only call.

### 4.3 Numerics

- Global counts are reduced over the DP group only; PP ranks hold the same data.
- Gradients are clipped with `scale_grads_and_clip_grad_norm(pp_enabled=True,
  pp_axis_name="pp", num_label_tokens=None)`. `num_label_tokens=None` keeps the helper
  from dividing gradients by `num_label_tokens / dp_group_size`, which would scale them
  up by DP x CP.
- Losses and metrics are computed on the last stage and broadcast within the PP group
  after each update; `get_logprobs` broadcasts the logprobs tensor, so every PP rank
  returns the same output, as `Policy` expects.

### 4.4 Stage Re-initialization

Re-initializing a stage (on a sequence-length change or a train/eval switch) clears
FSDP gradients in the pinned PyTorch/Automodel versions. All execution batches of an
update therefore use the data's sequence length, and the worker never switches mode
within an update.

## 5. Logprobs, Reference Policy, Offload, Checkpoints, and Refit

- **Logprobs** run `schedule.eval()` under `torch.no_grad()` with the sample IDs as
  target; the reference policy swaps the weights of the local stage.
- **Offload.** `offload_before_refit` releases PP runtime state (forward cache and the
  per-microbatch activation and gradient receive buffers) before the rollout engine
  runs; the next schedule call reallocates them.
- **Checkpoints** save and load each rank's stage through the same path as the whole
  model; Automodel's checkpointer gets the PP rank and group, and optimizer state is
  namespaced by the stage index (`optimizer_part_ids`), as in Automodel's own pipeline
  trainer.
- **Refit.** `PipelineWeightExporter` (`nemo_rl/models/automodel/pipeline_refit.py`)
  makes every PP rank produce the full model's `(name, tensor)` stream: each stage lists
  its tensors (the same per-module stream as the non-PP refit: full tensors, Hugging
  Face names, refit dtypes), the lists are gathered in PP rank order (stage order under
  1F1B) once and cached with a check for duplicated names, and each tensor is broadcast
  from its owning stage within the PP group. The exporter replaces `automodel_params_generator` for
  colocated CUDA IPC (vLLM, TRT-LLM) and colocated SGLang refit, and its
  `refit_info()` replaces `prepare_refit_info`. It does not assume its consumer is
  same-GPU IPC, so enabling non-colocated refit later needs no exporter change.

## 6. FSDP Gradient Synchronization

By default PyTorch PP disables FSDP gradient sync and resharding for every microbatch
and reduce-scatters once at the end of the schedule call; unsynchronized gradients
accumulate in `reduce_dtype` (fp32), keeping unsharded parameters and fp32 gradients of
a stage resident for the whole call. Automodel maps `defer_fsdp_grad_sync=false` to a
per-microbatch reduce-scatter, which matches the memory profile of the non-PP path. PP
configurations must therefore set `defer_fsdp_grad_sync` explicitly; `false` is
recommended.

## 7. TransferQueue Coordinates

`_local_coords` includes `pipeline_parallel` and the replica group is the flattened
`(pp, cp, tp)` sub-mesh, so exactly one rank of each DP replica writes back to TQ.

## 8. Limitations and Follow-ups

- Streaming (SingleController) updates need unknown-N normalization and an upstream
  fix that preserves gradients across stage re-initialization.
- Inputs whose batch dimension is not dim 0, as reported by a model's
  `get_pipeline_kwargs_chunk_dims` hook (for example multimodal M-RoPE position IDs),
  are rejected when a batch is planned. Text-only inputs of such models run.
- Only 1F1B is supported. Interleaved 1F1B would roughly halve the bubble (about 3%
  instead of 6% of the training phase at PP=2, M=16; more at larger PP and smaller M)
  but places several stages on one rank, as do V-shaped schedules; zero-bubble
  schedules also split the backward pass. Supporting them needs per-rank lists of
  stages, optimizers, and schedulers again.
