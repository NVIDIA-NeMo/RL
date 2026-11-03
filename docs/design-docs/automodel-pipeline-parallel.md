# Automodel Pipeline Parallelism

This note describes pipeline parallelism (PP) in the Automodel policy worker. PP is
supported for SFT and GRPO with the 1F1B schedule, one stage per PP rank, as in the
Megatron backend.

The work is split between Automodel and NeMo RL:

| Owner | Responsibilities |
| --- | --- |
| Automodel | Splitting the model into stages (`AutoPipeline`), building the PyTorch schedule (`torch.distributed.pipelining`), forward and backward passes, inter-stage communication |
| NeMo RL | Batch planning, the RL loss, gradient normalization and clipping, the optimizer step, logprobs, the reference policy, checkpoints, and weight refit |

NeMo RL code lives in `nemo_rl/models/automodel/pipeline_parallel.py`; the worker
branches on `worker.pipeline`, which is `None` without PP.

## Configuration

```yaml
policy:
  train_micro_batch_size: 2
  logprob_batch_size: 2           # must equal train_micro_batch_size
  automodel_cfg:
    pipeline_parallel_size: 2     # omitted means 1
    defer_fsdp_grad_sync: false   # required with PP; false is recommended
    pipeline:
      microbatches_per_step: 8    # required; >= pipeline_parallel_size
```

| Key | Description |
| --- | --- |
| `pipeline.microbatches_per_step` (M) | Microbatches of `train_micro_batch_size` samples per schedule call. Larger M shrinks the bubble, about `(pp - 1) / (M + pp - 1)` of each call, but costs one receive buffer per microbatch on every non-first stage and more padding, since all M microbatches share one sequence length. |
| `defer_fsdp_grad_sync` | `true` reduce-scatters gradients once per schedule call and keeps a stage's unsharded parameters and fp32 gradients resident for the whole call. `false` reduce-scatters after every microbatch, which matches the non-PP memory profile. |

The `pipeline` block rejects unknown keys, so schedule settings such as `pp_schedule`
fail at startup.

The Ray worker layout is `[PP, DP, CP, TP]` and Automodel's mesh is
`(pp, dp_replicate, dp_shard, cp, tp)`, so ranks line up. Data is sharded over DP and
replicated within a PP group. Expert parallelism is built inside each stage, so
`expert_parallel_size` must divide the `dp * cp * tp` ranks of one stage (for example,
EP8 with PP2 on 16 GPUs).

## Constraints

`validate_automodel_pipeline_config` (`nemo_rl/models/policy/utils.py`) runs in
`Policy.__init__` before workers start. With PP > 1 it rejects:

- A missing `pipeline` block or `defer_fsdp_grad_sync`, `M < pipeline_parallel_size`,
  or `logprob_batch_size != train_micro_batch_size`.
- Sequence packing, dynamic batching, sequence parallelism, CPU offload,
  `clear_cache_every_n_steps`, LoRA, draft co-training, router replay, reward models,
  ModelOpt quantization, VLMs, non-colocated generation, and `refit_transport`.
- Models with `tie_word_embeddings=true`.

SFT also requires `sft.val_micro_batch_size == policy.train_micro_batch_size`. DPO
rejects PP at setup because it runs microbatches of `2 * train_micro_batch_size`. The
Automodel value worker rejects PP. Top-k logits and full logits raise
`NotImplementedError` in the worker.

## Design

### Setup

`build_pipeline_config` (`nemo_rl/models/automodel/setup.py`) passes a `PipelineConfig`
to `DistributedSetup`, and `from_pretrained` returns an `AutoPipeline`:

| Field | Value | Reason |
| --- | --- | --- |
| `pp_schedule` | `1F1B` | One stage per rank. |
| `pp_microbatch_size`, `pp_batch_size` | `mbs`, `mbs * M` | One schedule call runs M microbatches. |
| `scale_grads_in_schedule` | `False` | `pipeline_loss_fn` applies the non-PP backward scale. |
| `pp_seq_len` | `max_total_sequence_length` | Static stage shapes; no shape inference. |
| `dtype` | FSDP `output_dtype` (fp32) | Dtype of activations sent between stages. |
| `loss_fn` | `pipeline_loss_fn` | The RL loss runs on the last stage. |

The rank's stage, `AutoPipeline.parts[0]`, becomes `worker.model`, so the optimizer,
scheduler, mode switches, clipping, offload, and reference-weight swap are the same
code as without PP.

### Batch Planning

`plan_execution_batches` splits a rank's data into execution batches of `mbs * M`
samples, one per schedule call:

- The call count is all-reduced (max) over DP, so every DP rank issues the same FSDP
  collectives. The tail is padded with copies of the first sample whose `sample_mask`
  and `token_mask` are zero.
- Models whose `forward` names a `padding_mask` parameter (Automodel's MoE models) also
  get a mask that is true on padding rows, so routers exclude them from the auxiliary
  loss and expert load.
- Each row's sample ID is passed as the schedule `target`. The schedule splits it with
  the inputs, and the loss function uses it to find the microbatch.
- With CP, `ContextParallelSharder(num_chunks=M)` shards the whole execution batch once.
  The loss function maps targets and logprobs through the same sharder.
- Every call uses the data's sequence length, because re-initializing a stage clears
  FSDP gradients.

### Loss Function

The schedule calls `pipeline_loss_fn(logits, sample_ids, context=PipelineCallContext)` on
the last stage for each microbatch. It calls `post_process_logits`, the same
temperature, masking, loss, and logprob path the non-PP worker uses, and records results
by microbatch index, so a repeated call overwrites instead of double-counting.

| Mode | Returns | Records |
| --- | --- | --- |
| `train` | `loss * dp * cp / cp_gradient_fanout`; zero for padding microbatches | loss and metrics |
| `eval` | detached loss | loss and metrics |
| `logprobs` | zero | token logprobs |

In `eval` and `logprobs` mode the logits storage is freed after use, and the runner
clears the stage's forward cache after each call.

### Training Step

`train()` keeps the non-PP loop, one optimizer update per global batch:

```text
process_global_batch            valid sequence and token counts, reduced over DP
plan_execution_batches
pipeline_forward_backward       M microbatches per schedule call; grads accumulate
clip_grad_norm                  norm reduced over the PP group
optimizer.step
...
broadcast losses and metrics from the last stage
```

Clipping uses `scale_grads_and_clip_grad_norm(pp_enabled=True, num_label_tokens=None)`.
A non-`None` `num_label_tokens` would rescale gradients the loss already normalized.

### Logprobs, Checkpoints, Refit

- **Logprobs** run `schedule.eval()`. The last stage broadcasts the result, so every PP
  rank returns the same tensor.
- **Checkpoints** pass the PP rank and group to Automodel's checkpointer and namespace
  optimizer state by stage index (`optimizer_part_ids`).
- **Refit.** `PipelineWeightExporter` gathers each stage's tensor list once, then
  broadcasts each tensor from its owning stage, so every PP rank yields the full model's
  `(name, tensor)` stream with Hugging Face names. It backs `prepare_refit_info` and the
  colocated CUDA IPC and SGLang refit paths. Every PP rank must consume the whole
  stream.
- **Offload.** `offload_before_refit` releases the stage's forward cache and receive
  buffers before rollout; the next schedule call reallocates them.
- **TransferQueue.** The replica group is the flattened `(pp, cp, tp)` sub-mesh, so one
  rank per DP replica writes back.

## Limitations

- Only 1F1B. Interleaved and V-shaped schedules place several stages on one rank and
  would need per-rank lists of stages, optimizers, and schedulers.
- Inputs that a model's `get_pipeline_kwargs_chunk_dims` hook splits along a non-batch
  dimension, such as multimodal M-RoPE `position_ids`, are rejected when a batch is
  planned.
- SingleController streaming updates need an upstream fix that preserves gradients
  across stage re-initialization.
