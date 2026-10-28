# AutoModel single-controller: PR1 implementation design

## Baseline and scope

Base: upstream `main` at `0e36f2e6a944a2e617fbc1941d57041715a4ac3f`.
PR1 adds full-batch single-controller (SC) execution to AutoModel. The existing
synchronous `AutomodelPolicyWorkerImpl.train()` remains byte-for-byte unchanged.
The previous experimental branch is preserved separately; its validation results
are not evidence for this revised implementation.

Supported scope is text GRPO, full-parameter FSDP2, TP=CP=EP=1, fixed
microbatches, disaggregated async vLLM, `in_order`, and lookahead zero. Dense and
MoE models use the same lifecycle. EP>1 is a parallelism extension, not a
prerequisite for MoE. Validation targets Gemma4 E2B and Gemma4 26B-A4B at EP=1.
Model-specific numerical/refit support remains subject to the existing backend.

Packing, dynamic batching, HSDP, CPU offload, LoRA, draft training, VLM,
quantization, PPO, distillation, colocation, and streaming dispatches are outside
PR1. Configuration validation rejects unsupported combinations before allocation;
worker guards also protect direct lifecycle callers.

## Interfaces and ownership

`setup_single_controller` already constructs `TQPolicy`. Its existing methods
delegate through `TQWorkerMixin` to the backend's `begin_train_step`,
`train_microbatch`, `finish_train_step`, and `abort_train_step`. Metadata is
sharded by DP, fetched, and materialized by the existing mixin. No new controller,
sampler, metadata schema, or transport API is introduced. These interface names
and the separation of backward from optimizer commit match the MCore path.

An internal `_TrainStepState` holds the loss function, GBS/MBS, dispatch status,
completed metrics, and snapshots of mutable model state needed for cancellation.
It is initialized in the worker constructor. It represents a logical optimizer
step; the one-dispatch restriction is an explicit PR1 execution constraint.

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
legacy `train()` with an open SC step; V1 is deliberately not modified to enforce
this new caller contract. Abort does not rewind RNG or controller rollout state.

## Numerical implementation

V2 calls established primitives directly, without a shared V1/V2 training wrapper:

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

The pinned AutoModel `Gate` accumulates `_cumulative_expert_load` outside registered
buffers. Snapshot and restore that counter explicitly, alongside model buffers,
so aborted samples do not affect the next successful bias update. Dispatcher
temporary tensors are released through the existing `release_moe_dispatcher_state`
utility after completion/cancellation. Gemma4's router has no cumulative load
counter. Arbitrary new model-owned mutable state must be audited when qualifying
additional architectures; parameter rollback after commit is not promised.

## Extension boundaries

Later streaming work replaces the one-dispatch guard with accumulation accounting
and defines cross-chunk normalization/FSDP synchronization. It retains the four
public lifecycle methods and existing numerical primitives. Lookahead and sampler
selection remain controller concerns. Packing/dynamic batching need a distinct
row-count/participation contract. Broader parallelism and model features receive
capability-specific tests rather than dense-versus-MoE implementations.

## Acceptance and evidence

1. Verify the V1 method text against the upstream base.
2. Unit tests cover transitions, failed backward, full-batch guards, commit-once,
   abort restoration, MoE hook ordering, and failed-commit recovery requirements.
3. Distributed fixed-input V1/V2 parity starts from identical model, optimizer,
   scheduler and RNG state. Compare pre-clip gradients, loss/metrics, norm,
   updated parameters/buffers and optimizer/scheduler state for token- and
   sequence-normalized losses over multiple updates. Exercise metadata fetch,
   retained-gradient logprob transitions, and abort. Use `rtol=1e-5`, `atol=1e-6`;
   report observed errors, not only a pass flag.
4. Run dense and MoE EP=1 initialization, generation/refit, loss/KL and checkpoint
   resume gates before matched 100-step integration runs. Record source hashes,
   resolved configuration, environment differences, Slurm status and W&B links.
5. Independent rollout curves establish integration behavior, not exact numerical
   parity or learning-quality equivalence. Disclose transport differences if the
   synchronous baseline requires a workaround.

Validation results for this revision are pending execution. Do not substitute
results from the earlier experimental implementation.
