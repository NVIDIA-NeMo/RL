# AutoModel single-controller: full-batch implementation design

## Overview and scope

AutoModel integrates with the single-controller (SC) runtime through a split
training lifecycle: compute gradients, then commit or cancel the optimizer
step. This lets the shared controller own rollout admission, training dispatch,
and weight synchronization while AutoModel owns model execution and optimization.
The [single-controller guide](../guides/single-controller.md) describes the
controller, sampler, and rollout architecture shared with the Megatron backend.
This document describes the AutoModel worker contract and its full-batch execution.

Supported scope is text GRPO, full-parameter FSDP2, TP=CP=1, fixed
microbatches, disaggregated async vLLM, `in_order`, and lookahead zero. Dense and
MoE models use the same lifecycle. Expert parallelism uses AutoModel's existing
MoE mesh, dispatcher, gradient synchronization, and distributed clipping.
Configure expert parallelism and the dispatcher through the existing AutoModel
settings. Supported expert layouts, model architectures, and weight-refit formats
remain subject to the AutoModel and generation backends.

Packing, dynamic batching, HSDP, CPU offload, LoRA, draft training, VLM,
quantization, PPO, distillation, colocation, and streaming dispatches are outside
the supported scope. Configuration validation rejects unsupported combinations before allocation;
worker guards also protect direct lifecycle callers.

## Configuration entrypoint

Start from `examples/configs/grpo_math_1B_automodel_single_controller.yaml`:

```bash
uv run examples/run_grpo_single_controller.py \
  --config examples/configs/grpo_math_1B_automodel_single_controller.yaml
```

This exemplar inherits `grpo_math_1B.yaml` and supplies the supported AutoModel
SC settings: async vLLM on separate GPUs, TransferQueue, in-order sampling with
zero lookahead, fixed microbatches, and full-batch admission. The default layout
uses one training GPU and one rollout GPU. Global batch size, admission threshold,
and buffer capacities follow the configured prompt/generation counts.
Checkpointing is disabled in the base; enabling it retains optimizer and native
TQ replay state. Configure the run length, model, workload, and resources for
the intended experiment.

Model-specific examples inherit the same exemplar:

| Recipe | Training resources | Rollout resources |
| --- | --- | --- |
| [Gemma4 E2B IT](../../examples/configs/recipes/llm/grpo-gemma4-e2b-it-1n8g-automodel-single-controller-full-batch.yaml) | 4 GPUs, EP=1 | 4 GPUs |
| [Qwen3.5-35B-A3B Base](../../examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-base-3n8g-automodel-ep16-single-controller-full-batch.yaml) | 16 GPUs, EP=16 with HybridEP | 8 GPUs, TP=8 |

One rollout batch must supply exactly one global training batch:

```text
GBS = num_prompts_per_step × num_generations_per_prompt
GBS % (DP size × MBS) = 0
```

Full-batch admission waits for all prompt groups before dispatch. The worker then
accumulates gradients over its local microbatches. For example, GBS=32, DP=16,
and MBS=1 gives two forward/backward microbatches per rank and one optimizer
update. The full-batch restriction describes the dispatch boundary, not the
number of model forward/backward calls.

## Interfaces and ownership

`setup_single_controller` constructs `TQPolicy`. Its methods delegate through
`TQWorkerMixin` to the backend's `begin_train_step`, `train_microbatch`,
`finish_train_step`, and `abort_train_step`. These lifecycle interfaces and the
separation of backward from optimizer commit match the Megatron backend.

```mermaid
flowchart TD
    A[Rollout workers write samples to TQ] --> B[Controller selects a full batch]
    B --> C[TQPolicy dispatches policy and optional reference logprobs]
    C --> D[Advantage calculation]
    D --> E[begin_train_step]
    E --> F[TQPolicy shards training metadata across DP ranks]
    F --> G[TQWorkerMixin fetches and materializes each shard]
    G --> H[train_microbatch accumulates gradients]
    H --> I[finish_train_step commits one update]
    I --> J[Controller refits rollout weights and releases samples]
    H -. Cancel before commit .-> K[abort_train_step discards pending work]
```

Logprob calculation uses the existing AutoModel `get_logprobs` and reference
logprob methods through TQ's metadata wrappers. Workers fetch the required input
columns and write logprobs back to TQ for advantage and loss computation. This
phase is separate from the forward/backward calls that accumulate gradients.

An internal `_TrainStepState` holds the loss function, GBS/MBS, dispatch status,
completed metrics, and snapshots of mutable model state needed for cancellation.
The worker creates this state at begin and releases it at finish or abort. At
most one logical optimizer step can be open; the one-dispatch restriction is an
explicit full-batch execution constraint.

| Operation | Preconditions | Work and ownership |
| --- | --- | --- |
| Begin | Idle, valid capability/batch configuration | Snapshot buffers and router load counters; clear gradients; open state. |
| Dispatch | Open, no attempted dispatch | Check every DP rank has GBS/DP rows; compute all backend microbatches; retain gradients and metrics. |
| Finish | Dispatch completed | Clip, optimizer step, existing MoE gate-bias hook, scheduler step; return metrics; release step. |
| Abort | Open or idle | Restore pre-step buffers/router counters and discard gradients; no optimizer/scheduler update; idempotent. |

Mark dispatch attempted before execution. A failure requires abort before a new
step; a second dispatch cannot double-count gradients. DP row validation uses a
collective minimum/maximum before forward collectives. It rejects short and
oversized batches. Full-batch admission requires the streaming threshold to equal
prompts per step and zero prompt-drop budgets.

Commit failures are not transactional: optimizer updates cannot be rolled back.
The worker becomes unavailable for further SC steps after a failed commit and
requires recovery. Abort never clears that failure. Callers must not interleave
legacy `train()` with an open SC step. Abort does not rewind RNG or controller
rollout state.

## Numerical implementation

The synchronous `train()` method owns its complete training loop. SC implements
the split lifecycle using the same numerical primitives, so model execution,
loss normalization, gradient handling, and metrics follow the established
AutoModel implementation:

| Primitive | Responsibility |
| --- | --- |
| `check_sequence_dim` | Validate sequence axes. |
| `process_global_batch` | Select the complete local batch and collectively compute valid-token/sequence denominators. |
| `get_microbatch_iterator` | Existing batching and input preparation. |
| `LossPostProcessor` | Existing loss, precision-independent normalization, and metrics. |
| `automodel_forward_backward` | Forward/backward microbatches, autocast, FSDP and router-replay contexts. |
| `aggregate_training_statistics` | Existing metric reductions. |
| `scale_grads_and_clip_grad_norm` | Existing distributed clipping and norm calculation. |

The dispatch uses `num_global_batches=1` and known global denominators. It may
perform many forward/backward passes. Gradient clipping, optimizer and scheduler
each happen once at finish. Learning-rate metrics record the pre-update rate,
matching V1. Retained-gradient logprob phases are allowed; finish restores training
mode before the model's gate-bias update hook.

AutoModel's `Gate` accumulates `_cumulative_expert_load` outside registered
buffers. Snapshot and restore that counter explicitly, alongside model buffers,
so aborted samples do not affect the next successful bias update. Dispatcher
temporary tensors are released through the existing `release_moe_dispatcher_state`
utility after completion/cancellation. Additional model-owned mutable state
must be accounted for when extending architecture support; parameter rollback
after commit is not promised.

## Forward padding and expert parallelism

TQ stores variable-length token columns and reconstructs padded tensors when
the driver or workers fetch them. Each model dispatch carries a shared
`GLOBAL_FORWARD_PAD_SEQLEN` in its metadata. The target is computed from the
longest valid sequence and all applicable alignment requirements:

```python
alignment = lcm(
    policy["make_sequence_length_divisible_by"],
    producer_padding_multiple,
    dynamic_batching_sequence_round,
)
target_length = round_up(longest_sequence, alignment)
```

`TQDriverMixin._stamp_pad_seqlen` computes the target before DP sharding.
`TQPolicy._stamp_placed_pad_seqlen` applies the same constraints across metadata
batches already assigned to ranks. Driver reads and worker fetches use that
target; valid sequence lengths are preserved and added token-mask positions
are zero. Dispatch metadata is isolated per consuming model so one model's
alignment or an inherited stale target does not dictate another model's shape.

The Qwen3.5 HybridEP recipe sets `make_sequence_length_divisible_by: 64`.
This provides aligned input shapes for HybridEP's default 64-token chunks:
its combine kernel requires the configured per-rank token capacity to be a
multiple of the chunk size. For a longest sequence of 2336, the padded target
is 2368. TQ honors the configured alignment; it does not infer the dispatcher's
requirements or change its chunk size. The generic TQ calculation also handles
dynamic-rounding constraints, although AutoModel SC currently rejects dynamic
batching.

## Checkpoint and recovery boundary

Trainer checkpoints are taken at completed step boundaries. The worker's open
`_TrainStepState`, pending gradients, and cancellation snapshots are temporary;
they are not an independently resumable partial training step. Model and
optimizer restoration use the existing AutoModel checkpoint path.

The controller separately owns dataloader progress, sampler position, rollout
recovery, and the TQ snapshot. Enable optimizer and data-plane saving when
continuing training from a checkpoint. See
[Checkpointing and Replay Recovery](../guides/single-controller.md#checkpointing-and-replay-recovery)
for the shared checkpoint contents and replay behavior.

## Extension boundaries

Later streaming work replaces the one-dispatch guard with accumulation accounting
and defines cross-chunk normalization/FSDP synchronization. It retains the four
public lifecycle methods and existing numerical primitives. Lookahead and sampler
selection remain controller concerns. Packing/dynamic batching need a distinct
row-count/participation contract. Broader parallelism and model features receive
capability-specific tests rather than dense-versus-MoE implementations.

## Matched 100-update training curves

The following figures show dense and MoE training behavior using the Gemma4
E2B IT and Qwen3.5-35B-A3B Base EP=16 recipes above. Each figure covers 100
updates and links to the corresponding V1 and SC runs.

Each pair uses seed 42, global batch size 32, microbatch size 1, and matched
model, optimizer, and sampling settings. SC uses full-batch dispatch, in-order
sampling, and lookahead zero. V1 uses legacy Ray data transport; SC uses TQ.
Rollouts are sampled independently, so the curves describe training behavior
rather than exact numerical parity or a statistically established quality
difference. Identical-input worker replay provides the separate numerical
equivalence evidence.

All five panels use the original NeMo RL metric names: `train/reward`,
`train/mean_gen_tokens_per_sample`, `train/loss`, `train/gen_kl_error`, and
`train/approx_entropy`. Every metric has 100 finite values per run. Plotted
values come from the complete exported W&B histories and were checked against
the native TensorBoard records. Faint lines show every raw update; solid lines
show trailing 10-update means, starting at update 10. No updates are removed.

### Gemma4 E2B IT

Each run uses four training and four rollout H100 GPUs, with EP=1.
[V1 run](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/cf9668ea) and
[SC run](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/afadd4ca).

![Gemma4 E2B IT: V1 versus full-batch SC over 100 updates, showing reward, mean generated tokens per sample, loss, generation KL error, and approximate entropy.](../assets/automodel-sc/gemma4-e2b-v1-vs-sc-100steps.png)

### Qwen3.5-35B-A3B Base, EP=16

Each run uses sixteen training H100 GPUs with HybridEP and eight rollout H100
GPUs with TP=8. These histories include checkpoint/resume and contain updates
1–100 once. SC policy-version lag stayed zero.
[V1 run](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/2a121400) and
[SC run](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/c926b37d).

![Qwen3.5-35B-A3B Base at EP=16: V1 versus full-batch SC over 100 updates, showing reward, mean generated tokens per sample, loss, generation KL error, and approximate entropy.](../assets/automodel-sc/qwen3.5-35ba3b-ep16-v1-vs-sc-100steps.png)

For reference, the means over all 100 updates are:

| Native metric | Gemma4 V1 | Gemma4 SC | Qwen3.5 EP16 V1 | Qwen3.5 EP16 SC |
| --- | ---: | ---: | ---: | ---: |
| `train/reward` | 0.339375 | 0.352188 | 0.291875 | 0.276563 |
| `train/mean_gen_tokens_per_sample` | 1734.29 | 1710.88 | 1755.90 | 1801.75 |
| `train/loss` | 0.019315 | 0.017606 | 0.023737 | 0.021416 |
| `train/gen_kl_error` | 0.00038688 | 0.00038830 | 0.00091623 | 0.00090550 |
| `train/approx_entropy` | 0.177857 | 0.186579 | 0.478026 | 0.479710 |
