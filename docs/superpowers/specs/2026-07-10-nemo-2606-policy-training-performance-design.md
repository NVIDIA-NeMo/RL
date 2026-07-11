# NeMo 26.06 Policy Training Performance Integration Design

**Source baseline:** NeMo-RL `main` at `c1868b139819e6364d65f862a843eb39489072a1`
**Primary model:** `Qwen/Qwen3-30B-A3B`
**Target hardware:** GB200 on OCI-HSG, Lyris, Pre-Tyche, and AWS-DFW

## Goal

Enable and validate the NeMo 26.06 MoE training performance stack in the NeMo-RL Megatron policy-training path:

1. CuTeDSL fused grouped MLP for MXFP8 experts.
2. Expert-parallel A2A overlap through the MCore schedule-plan interface.
3. Fixed-shape full-iteration CUDA Graph execution.
4. HybridEP fixed-capacity and paged-stash integration for the combined path.

The external `Policy.train(data, loss_fn, ...)` API remains unchanged. The work adds internal configuration, forward-step, runner, data-adapter, refit-export, and lifecycle interfaces.

## Scope Decomposition

The work is split into independently testable changes. Each feature must pass its own functional and performance gate before features are combined.

### Change 1: CuTeDSL and refit-safe GLU layout

- Expose the Transformer Engine op-fuser and GLU interleave settings.
- Install Cutlass DSL in the MCore-only environment.
- Enable the CuTeDSL kernel through the Ray policy-worker environment.
- Correct train-to-rollout export of block-interleaved expert FC1 weights.
- Verify native checkpoint and refit round trips.

### Change 2: PP=1 EP A2A overlap

- Add Bridge `CommOverlapConfig` lifecycle handling.
- Extend the NeMo-RL forward step with `return_schedule_plan=True` support.
- Preserve the NeMo-RL custom loss contract and MTP mask handling.
- Support synchronous `Policy.train()` with PP=1 first.
- Reject unsupported combinations before model construction.

### Change 3: Fixed-shape full-iteration CUDA Graph

- Inject a worker-owned forward/backward runner.
- Serialize `ProcessedMicrobatch`, `BatchedDataDict`, and `PackedSeqParams` into a graph-safe tensor tree.
- Maintain persistent scalar inputs and a graph signature.
- Reset graph state before offload, reload, refit, or storage changes.
- Graph synchronous training only; keep evaluation, logprob, and top-k paths eager.

### Change 4: Paged stash and combined stack

- Wrap `FullCudaGraphWrapper` inside `PagedStashRunner`.
- Add fixed-capacity HybridEP and paged-stash configuration.
- Verify both no-overflow replay and forced-overflow eager retry.
- Run CuTeDSL, A2A overlap, and full CG together.

## Architecture

### Existing execution path

```text
Policy.train
  -> MegatronPolicyWorker.train
  -> get_microbatch_iterator
  -> megatron_forward_backward
  -> get_forward_backward_func
  -> forward_with_post_processing_fn
  -> GPTModel.forward
```

### Target execution path

```text
Policy.train
  -> MegatronPolicyWorker.train
  -> graph-safe microbatch adapter when full CG is enabled
  -> megatron_forward_backward(runner=worker.training_runner)
  -> PagedStashRunner                         optional, outer
  -> FullCudaGraphWrapper                     optional, inner
  -> MCore forward/backward schedule
  -> forward_with_post_processing_fn
       eager: GPTModel.forward
       A2A:   GPTModel.build_schedule_plan
```

The worker owns the runner because it also owns model, optimizer, offload, refit, and teardown state.

## Interfaces

### CuTeDSL configuration

Extend the existing `MegatronConfig` with optional pass-through fields:

```python
use_transformer_engine_op_fuser: NotRequired[bool]
moe_mlp_glu_interleave_size: NotRequired[int | None]
```

The exemplar YAML owns defaults. A performance recipe explicitly selects:

```yaml
moe_grouped_gemm: true
use_transformer_engine_op_fuser: true
moe_mlp_glu_interleave_size: 32
expert_tensor_parallel_size: 1
fp8_cfg:
  enabled: true
  fp8_recipe: mxfp8
env_vars:
  NVTE_CUTEDSL_FUSED_GROUPED_MLP: "1"
```

### Refit export

The in-memory interleaved FC1 layout must be converted to contiguous HF gate/up layout without mutating training parameters. The preferred implementation is in Megatron-Bridge `GatedMLPMapping`, because layout conversion belongs to the Bridge conversion layer. If the Bridge change cannot land with the NeMo-RL change, NeMo-RL uses a temporary typed export adapter and tracks removal once Bridge contains the fix.

### A2A schedule-plan forward

Extend the internal forward step:

```python
def forward_with_post_processing_fn(
    data_iterator: Iterator[ProcessedMicrobatch],
    model: GPTModel,
    post_processing_fn: PostProcessingFunction,
    *,
    return_schedule_plan: bool = False,
    ...,
) -> tuple[torch.Tensor | AbstractSchedulePlan, Callable[..., Any]]:
    ...
```

The schedule branch returns `model.build_schedule_plan(...)` and the same bound NeMo-RL loss callable. Temperature scaling moves inside the callable because no output tensor exists when the plan is built.

### Full-CG runner injection

Extend the internal forward/backward entry point:

```python
def megatron_forward_backward(
    ...,
    forward_backward_func: ForwardBackwardFunc | None = None,
) -> Any:
    ...
```

The worker builds and caches the runner. Eager callers omit the argument and retain current behavior.

### Graph-safe microbatch

Add typed serialization helpers in `nemo_rl/models/megatron/data.py`:

```python
def serialize_processed_microbatch(
    microbatch: ProcessedMicrobatch,
) -> dict[str, Any]:
    ...

def deserialize_processed_microbatch(
    payload: Mapping[str, Any],
) -> ProcessedMicrobatch:
    ...
```

The serialized form contains only nested plain dictionaries, lists, tuples, tensors, and graph-invariant scalar metadata. Tensor fields inside `PackedSeqParams` are flattened and refreshed through static buffers.

## Initial Compatibility Boundary

The first supported combined configuration is deliberately narrow:

- `Qwen/Qwen3-30B-A3B`
- synchronous policy training
- PP=1, TP=1, ETP=1, EP=4 for one-node smoke
- PP=1, TP=1, ETP=1, EP=16 for four-node validation
- BF16 baseline followed by MXFP8
- fixed sequence length of 1024 or 2048
- fixed microbatch count of at least four
- dynamic batching disabled
- sequence packing disabled for the first full-CG validation
- router replay disabled
- draft-model hidden capture disabled
- fused-linear-logprob training disabled
- shared-expert overlap disabled
- full activation recompute disabled

PP+VPP, split training, variable-shape graph caches, value-model training, router replay, and draft training are follow-up scopes.

## Failure Handling

Unsupported combinations fail during setup with a specific error. The implementation must not silently fall back from a requested performance feature.

Examples:

- CuTeDSL requested without MXFP8, ETP=1, interleave 32, or required dimensions.
- A2A overlap requested with PP>1, full recompute, shared-expert overlap, or unsupported MTP count.
- Full CG requested with dynamic batching, variable graph signature, split training, or `empty_unused_memory_level > 0`.
- Interleaved FC1 export requested without a de-interleave-capable conversion path.

Paged-stash overflow is the exception: it follows the upstream synchronized eager retry path and records the fallback.

## Test Model Matrix

| Gate | Model and topology | Purpose |
|---|---|---|
| CPU unit | Tiny synthetic MoE tensors/config | Config, mapping, serialization, runner, guards |
| 1-node GPU | Qwen3-30B-A3B, 4 GB200, TP1/PP1/EP4 | Kernel and functional smoke |
| 4-node GPU | Qwen3-30B-A3B, 16 GB200, TP1/PP1/EP16 | HybridEP/A2A/paged-stash behavior |
| Short E2E RL | Qwen3-30B-A3B, 2–5 GRPO steps | Train, refit, rollout correctness |
| Scale follow-up | Qwen3-235B or DeepSeek-V3 | Large-model performance scalability |

## Functional Success Criteria

- Existing unit tests remain green when all new features are disabled.
- Eager and feature-enabled loss, gradient, and optimizer update match within precision-appropriate tolerances.
- Interleaved FC1 train-to-HF export reconstructs the original gate and up matrices.
- Refit produces rollout logits matching the eager/non-interleaved reference within the configured precision tolerance.
- A2A overlap completes without hang and preserves MTP=1 behavior when tested.
- Full CG shows warmup, capture, and at least two replay iterations.
- Offload/reload invalidates the graph and triggers successful recapture.
- Forced paged-stash overflow retries without dropped tokens.

## Performance Evidence

- CuTeDSL: Nsight or TE debug evidence identifies the CuTeDSL fused grouped MLP kernel rather than the basic op-fuser fallback.
- A2A: timeline evidence shows dispatch/combine communication overlapping expert compute.
- Full CG: CPU launch gaps decrease after warmup and replay.
- Every comparison records median policy-training step time after warmup, tokens/s, peak memory, and software versions.
- A feature is not declared a performance win from a single iteration. Use at least five measured policy updates or a dedicated repeated training microbenchmark.

## Cluster and Container Strategy

### Execution order

1. OCI-HSG: primary development and one-node smoke.
2. Lyris: GB200 NVL72 portability and four-node topology validation.
3. Pre-Tyche: independent GB200/ARM validation after account and Lustre path discovery.
4. AWS-DFW: final reproducibility run using GRES allocation.

### Container policy

Use the existing `nemo_rl_nightly_20260707.sqsh` for the first import and one-node smoke on OCI-HSG. OCI-HSG, Lyris, and AWS-DFW currently expose an image with this immutable filename.

Before cross-cluster performance comparison, record the actual SHA256 on each cluster through a scheduled lightweight job. Matching filenames are not sufficient provenance.

If the existing image fails because of a missing or incompatible runtime dependency:

1. Stage the current `nvcr.io/nvidian/nemo-rl:nightly` once on OCI-HSG.
2. Save it under an immutable filename with source tag, retrieval time, source commit, metadata, and SHA256.
3. Run a one-node import/GPU smoke.
4. Transfer the verified squashfs to the other clusters through an approved shared artifact path.
5. Compare performance only with identical image SHA256 values.

Image failures and code failures are reported separately. Registry, authentication, proxy, and import failures do not count as training failures.

## Experiment Records

Each run records:

- NeMo-RL, Megatron-Bridge, Megatron-LM, and Transformer Engine SHAs
- image path and SHA256
- CUDA, cuDNN, NCCL, Cutlass DSL, and PyTorch versions
- cluster, nodes, GPUs, account, partition, and topology segment
- model precision and TP/PP/CP/EP/ETP
- global batch, microbatch, microbatch count, and padded sequence length
- feature flags and environment variables
- job ID, log path, W&B URL when enabled, and Nsight artifact path

Job scripts and logs live under an experiment directory rather than the repository root.

## Delivery

The work is delivered as separate reviewable changes in this order:

1. CuTeDSL configuration, dependency, refit export, and tests.
2. PP=1 A2A schedule-plan support and tests.
3. Fixed-shape full-CG runner/data/lifecycle support and tests.
4. Paged-stash combined recipe and distributed validation.

No performance claim is made until the corresponding functional gate and profiler evidence both pass.
