# AutoModel single-controller: full-batch implementation design

## Baseline and scope

Base: upstream `main` at `0e36f2e6a944a2e617fbc1941d57041715a4ac3f`.
This design adds full-batch single-controller (SC) execution to AutoModel. The existing
synchronous `AutomodelPolicyWorkerImpl.train()` remains byte-for-byte unchanged.
The previous experimental branch is preserved separately; its validation results
are not evidence for this revised implementation.

Supported scope is text GRPO, full-parameter FSDP2, TP=CP=EP=1, fixed
microbatches, disaggregated async vLLM, `in_order`, and lookahead zero. Dense and
MoE models use the same lifecycle. EP>1 is a parallelism extension, not a
prerequisite for MoE. Validation targets Gemma4 E2B and Qwen3-30B-A3B at EP=1.
Model-specific numerical/refit support remains subject to the existing backend.

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

The Qwen full-batch regression recipe inherits this exemplar and supplies its
model/workload, checkpoint, and output settings. L1 and nightly continue to use
that recipe with their existing step/checkpoint overrides. Extracting the base
preserves the recipe's fully resolved configuration.

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
step; the one-dispatch restriction is an explicit full-batch execution constraint.

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

## Automated regression coverage

AutoModel follows the same test layers as MCore; shared controller and TQ
wrapper tests remain backend independent.

| Layer | AutoModel coverage | Discovery |
| --- | --- | --- |
| Worker unit tests | `tests/unit/models/policy/test_automodel_split_state.py`: lifecycle, guards, failure cleanup, buffer and MoE hook behavior | Existing AutoModel L0 policy shards |
| GPU numerical parity | `tests/unit/models/policy/test_automodel_split_parity.py`: tiny Llama, two FSDP ranks, two updates for each loss normalization, masks and gradient accumulation | Existing AutoModel L0 policy shards; HF-gated marker, like MCore's tiny-model fixture |
| Configuration | `tests/unit/single_controller/test_automodel_config.py`: base exemplar, prompt-count overrides, derived regression recipe, and unsupported configurations | Shared L0 controller suite |
| Live integration | `tests/functional/automodel_sc_full_batch_smoke.sh`: Qwen2.5-1.5B, one training and one rollout GPU, timeout checkpoint at step 1, resume through step 4 | `L1_Functional_Tests_Automodel_SingleController.sh`, including FAST |
| Longer regression | `grpo-qwen2.5-1.5b-1n2g-automodel-single-controller-full-batch` recipe and matching driver: 100 updates, finite metrics, zero lag, logprob health and final checkpoint | `tests/test_suites/nightly.txt` |

The L0 test reuses `tests/functional/automodel_sc_full_batch_parity.py`
with strict deterministic kernels and zero tolerances. It compares pre-clip
gradients, returned metrics, model parameters/buffers, optimizer and scheduler
state. It also checks retained gradients across logprob inference and rollback
after an aborted backward. Its local metadata transport isolates worker
numerics; L1 exercises Ray dispatch, network TransferQueue, async vLLM and refit.
The tiny model uses the Hugging Face implementation inside AutoModel. Native
dense/MoE architecture qualification remains separate from this small CI test.

L1 checks checkpoint artifacts and restore messages, and requires exactly step 1
before restart and steps 2–4 afterward. It tests save/resume mechanics, not
equivalence to uninterrupted training: the controller may drop in-flight
prompts on restart. W&B is optional for L1; the nightly uses the original NeMo RL
metric names in the `nemo-rl-automodel-sc` project.

Validation on 2026-10-30 passed 20 AutoModel worker tests (including the exact
two-GPU parity test), then 37 configuration/shared-wrapper tests. The first job
(`21044661`) passed the worker tests but exposed a config-access error in the
new recipe test assertion; that assertion was corrected. Job `21044947`
passed the 37 tests and the live four-step save/resume smoke, exiting 0.
The smoke had zero policy lag and zero logprob-error-masked sequences;
maximum generation KL was 0.000553 and maximum sequence probability error
was 1.04654. The 100-step nightly was registered and dry-run checked, but was
not executed in this validation pass.

## Validation status

The validation checkout is frozen at `56b4fc2d582c06204fa5f8effceaecb21dd5419c`.
Subsequent configuration type casts do not change runtime behavior; the worker
is identical. Gemma4 E2B passed four distributed comparisons on four H100
80 GB GPUs: two token-normalized and two sequence-normalized updates. Maximum
absolute differences were **0.0** for pre-clip gradients, metrics, updated
parameters/buffers, and optimizer state. Scheduler equality, gradients retained
through a logprob phase, and abort preservation also passed. Slurm job
`20956201` completed with exit code 0.

[W&B parity run](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4/runs/uwx3bgek)
and [six-panel comparison dashboard](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4?nw=vovwm7ey6oo).
The focused lifecycle suite passed 19 tests, configuration coverage passed 134,
and broader setup/controller/checkpoint regression coverage passed 537.
The fixed-input harness uses deterministic execution, standard AdamW, and a
StepLR schedule to compare complete optimizer state. Live integration runs use
the existing recipes' Transformer Engine FusedAdam and scheduler settings.
MoE live qualification uses sixteen training ranks and eight rollout GPUs
across three nodes, with the recipe's BF16 optimizer moments and GBS=32.
Four training ranks exhausted memory when saving BF16 moments as FP32 copies.
Eight ranks trained two updates and saved full checkpoints, but optimizer restore
exhausted an 80 GB GPU while PyTorch cast state tensors inside FusedAdam's
loader. Using FP32 moments on eight ranks also exhausted memory during restore;
the final sixteen-rank configuration retains the recipe's BF16 moments.
Six ranks failed before training because the native checkpoint adapter
could not load uneven expert shards. Those layouts are not qualified for the
complete training/checkpoint/resume workflow.

The dense three-path logprob gate included the HF reference. Its mean KL values
were 0.0 for HF versus AutoModel, 0.00286 for HF versus vLLM, and 0.00204 for
vLLM versus AutoModel (global mean threshold 0.01; four prompts, 115 tokens).
The live SC checkpoint gate saved at step 2, restored model/optimizer and
controller/data state, and advanced to step 3 with zero policy lag.
[Its W&B run](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4/runs/1f964f66)
contains all three steps and is marked finished.

Qwen3-30B-A3B Base passed the same four fixed-input comparisons on eight H100
GPUs at EP=1, again with maximum absolute error **0.0** for gradients, metrics,
parameters/buffers, and optimizer state. Scheduler, retained-gradient logprob,
and abort checks passed as well.
[MoE parity run](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4/runs/fntrn5ku)
and [six-panel comparison dashboard](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4?nw=ho8jhl0ldq5).

MoE's three-path logprob probe passed on sixteen fixed prompts, including HF,
using eight policy ranks and TP=4 rollout. Global mean KL was 0.00370 for HF
versus AutoModel, 0.00461 for HF versus vLLM, and 0.00511 for vLLM versus
AutoModel (threshold 0.01). This is a global-mean criterion, not a per-prompt
bound. An earlier TP=2 four-prompt probe failed at 0.01277; expanding that
probe to sixteen prompts lowered its global mean but did not change the
original subset's result. The final rollout configuration uses TP=4.

The sixteen-rank live checkpoint/resume gate completed with exit code 0
(Slurm job `20963367`). It saved at step 2, restored model/optimizer and
controller/data state, advanced to step 3, and saved again. Both retained
checkpoints contain sixteen weight and optimizer shards. All three W&B
training records are finite, include a nonzero gradient, and have zero policy
lag. The live generation-KL diagnostic ranged from 0.00147 to 0.00162.
[MoE checkpoint run](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4/runs/ab3b3610).

The matched 100-step MoE results are tracked in the
[PR validation record](https://github.com/NVIDIA-NeMo/RL/pull/4963).
The eight-rank scoring probe and sixteen-rank checkpoint gate have distinct
coverage; use the PR record for the completed long-run integration evidence.

The Gemma4 E2B upstream/SC pair completed 100 real GRPO updates, using four
training and four generation H100s per run, GBS=32, MBS=1, and seed 42.
Both runs have 100 finite training records, identical learning-rate sequences,
32 valid samples per update, and complete step-100 weight/optimizer checkpoints.
SC policy-version lag remained zero. Mean reward over these updates was 0.345
for upstream and 0.36875 for SC; a single seed does not establish a quality
difference. [Paired 12-panel dashboard](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4?nw=1tuwnil9b0a).

The original dense Slurm wrappers exited 2 **after** training completed because
the shared shell launcher was edited while they ran. These job failures are
preserved, not counted as clean exits. Immutable launcher bundles then restored
each step-100 checkpoint, completed one extra update, and saved step 101.
Both recovery jobs (`20959496`, `20959498`) completed with exit code 0.
The comparison dashboard is restricted to updates 1–100. Upstream uses legacy
Ray data transport because its synchronous TQ path requests absent reference
logprobs when KL is disabled; SC uses TQ. The upstream code is unmodified.

Gemma4 26B-A4B was initially considered for MoE qualification, but two existing
backend failures occurred before any SC call. With the native model at EP=1,
its checkpoint adapter returns local expert tensors and loses their global
shape (`128` experts in the checkpoint versus `16` on each of eight ranks).
With `force_hf=true`, loading succeeds but the existing logprob forward fails
with an FP32/BF16 mismatch at the LM head. Neither configuration is qualified
by this PR. Qwen3 MoE validation uses the existing recipe with the local Base
checkpoint, EP=1, the torch dispatcher, and auxiliary router loss disabled.

Validation uses the existing pipeline container with the pinned AutoModel source
`b916107a5ce8e77a3f18682f8b0eaa46a4b54389`, PyTorch `2.13.0+cu130`, and
Transformers `5.12.1`. The container receives the upstream-pinned Lens and
TransferQueue packages plus MSC `0.51.0` for newer imports; this is not a full
rebuild from the upstream lockfile. Dependency pins in this PR are unchanged.
For TP=4 MoE generation, the validation launcher starts one explicit Ray head
and exports `RAY_ADDRESS` to all children. Without that address, vLLM's spawned
EngineCore started a second cluster and failed to find the original placement
group. This launcher adjustment is shared by the upstream and SC MoE runs.
