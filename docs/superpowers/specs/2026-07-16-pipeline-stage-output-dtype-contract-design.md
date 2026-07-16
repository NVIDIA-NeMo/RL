# Pipeline Stage Output Dtype Contract Design

## Problem

NeMo RL configures a single `MixedPrecisionPolicy.output_dtype` for FSDP2. Under
pipeline parallelism, FSDP2 applies that dtype recursively to every floating-point
tensor returned by a stage root.

This is correct for stages whose outputs all share one activation dtype, but it is
incorrect for stages with heterogeneous output contracts. A non-final GLM-MoE-DSA
stage returns:

- hidden states in the configured activation dtype, normally `torch.bfloat16`; and
- IndexShare top-k indices transported as `torch.float32` so pipeline receive buffers
  can require gradients while preserving integer position values exactly.

With root `output_dtype=torch.bfloat16`, FSDP2 changes the top-k carry to BF16. The
pipeline metadata still correctly declares FP32, producing `PipeliningShapeError`.
Changing the metadata to BF16 would hide the exception but corrupt indices such as
257, 1023, and 4095.

## Goals

- Allow PP stages to preserve model-declared, per-output dtypes.
- Keep the existing uniform-output behavior unchanged by default.
- Keep child transformer FSDP units on the configured output dtype so hidden-state
  communication and mixed-precision behavior remain unchanged.
- Make the capability reusable by GLM and future models with heterogeneous PP
  outputs.
- Fail during setup when a model requests dtype preservation without declaring an
  explicit pipeline I/O contract.

## Non-goals

- Do not infer output dtypes from arbitrary parameter dtypes. Outputs such as GLM's
  top-k carry have no corresponding parameter.
- Do not add first-forward shape or dtype discovery.
- Do not patch PyTorch's `PipelineStage.forward_one_chunk` or private FSDP state.
- Do not change non-PP execution or the optimizer-dependent model load policy in
  NeMo RL.
- Do not automatically migrate other models unless they require heterogeneous
  floating-point outputs.

## Chosen Architecture

Add an optional model capability hook:

```python
def preserve_pipeline_stage_output_dtypes(self) -> bool:
    """Whether the PP stage root must preserve the dtypes returned by forward()."""
```

The default is absence of the hook, which preserves current behavior. Automodel's
default FSDP2 parallelization strategy consults this hook only when PP is enabled.

For a stage that opts in:

1. Recursive child FSDP wrapping receives the original mixed-precision policy.
2. The root FSDP wrapper receives a cloned policy with `output_dtype=None`.
3. The model's `forward()` remains responsible for returning each output in its
   semantic dtype.
4. The existing `get_pipeline_stage_metas()` hook remains the authoritative shape
   and dtype declaration used to allocate PP send/receive buffers.
5. PyTorch's existing metadata validation remains the runtime consistency check.

This produces the GLM data path:

```text
decoder child FSDP units (output_dtype=BF16)
    -> GLM stage forward returns (hidden BF16, top-k FP32)
    -> stage root FSDP (output_dtype=None) preserves both dtypes
    -> PP metadata declares (hidden BF16, top-k FP32)
    -> send/receive buffers match the actual values
```

## Components

### Mixed-precision policy cloning

Add a focused helper next to Automodel's existing `_mp_policy_with_param_dtype`
helper. It clones a `MixedPrecisionPolicy` and explicitly replaces only
`output_dtype`, including the meaningful `None` value. The original policy must not
be mutated because child FSDP units continue to use it.

### Root FSDP policy selection

In `DefaultParallelizationStrategy.parallelize`:

- determine whether PP is enabled from the device mesh;
- invoke `preserve_pipeline_stage_output_dtypes()` when present;
- require a callable `get_pipeline_stage_metas()` when preservation is requested;
- use the original policy for recursive child sharding; and
- use the cloned `output_dtype=None` policy only for the root `fully_shard` call.

The hook is ignored outside PP. A model without the hook follows the current path
byte-for-byte at the policy-selection level.

### GLM opt-in

`GlmMoeDsaForCausalLM` opts in only for non-final PP stages, identified by the
absence of `lm_head`. Those stages return `(hidden, topk_carry)` with heterogeneous
floating dtypes. The final stage returns only logits and keeps the existing root
output cast.

The current GLM metadata and forward contract remain unchanged:

- hidden meta uses the pipeline activation dtype;
- top-k meta uses FP32; and
- `carry_out` is explicitly created as FP32 after decoder layers have returned their
  integer top-k indices.

### Diagnostics

Log once per local stage when root dtype preservation is active. The message includes
the model class and states that child FSDP output casting remains enabled while root
output casting is disabled. Existing `PipeliningShapeError` validation remains the
authoritative runtime failure if a model's forward output disagrees with its metas.

## Error Handling

- If the preserve hook returns a non-boolean value, raise `TypeError` during
  parallelization.
- If preservation is requested without `get_pipeline_stage_metas`, raise `ValueError`
  before the root FSDP wrapper is installed.
- If PP is disabled, do not call or validate the hook.
- If `mp_policy` is `None`, retain Automodel's existing default-policy construction,
  then clone the resolved policy for the root.
- Do not silently fall back to the generic single-output metadata path for an
  opt-in model.

## Testing Strategy

### Policy unit tests

Test that the output-dtype helper:

- returns a distinct policy object;
- preserves `param_dtype`, `reduce_dtype`, and `cast_forward_inputs`;
- sets `output_dtype=None`; and
- does not mutate the child policy.

### Parallelizer unit tests

Using a mocked `fully_shard_fn`, verify:

- a normal PP model passes the original policy to both child and root wrappers;
- an opt-in PP model passes the original policy to children and a preserve policy to
  the root;
- the hook is ignored without PP;
- a missing metadata hook fails before root wrapping; and
- a non-boolean hook result fails with a clear error.

### GLM model tests

Extend the existing GLM IndexShare tests to verify:

- a non-final stage opts in;
- a final stage does not opt in;
- non-final metadata remains `(activation_dtype, torch.float32)`; and
- forward emits FP32 carry values without rounding representative indices such as
  257, 1023, 1025, and 4095.

### GPU integration tests

Run a minimal FSDP2 test demonstrating that child `output_dtype=BF16` plus root
`output_dtype=None` produces `(BF16, FP32)` and preserves large integer-valued carries.

Run the existing GLM PP2 GRPO recipe through at least:

- generation;
- `get_logprobs()` on all policy workers;
- one optimizer step; and
- the next rollout/logprob boundary if the smoke duration permits.

The run must have no `PipeliningShapeError`, and logs must show root preservation only
on non-final GLM stages.

## Compatibility and Rollout

- The change is opt-in, so existing models retain the existing root output cast.
- The NeMo RL optimizer/load-dtype logic remains unchanged and continues to select
  the activation dtype supplied to Automodel.
- GLM is the first consumer. Other models may opt in only after their forward outputs
  and `get_pipeline_stage_metas()` are verified to agree per tensor.
- The implementation belongs in the Automodel submodule; the parent NeMo RL change is
  limited to the updated submodule pointer and any integration-level regression test.

## Acceptance Criteria

- GLM PP2 completes policy logprob computation and at least one training step.
- A non-final GLM stage communicates hidden states as BF16 and top-k carries as FP32.
- Top-k values remain exact across the stage boundary for positions not exactly
  representable in BF16.
- Non-opt-in models retain their previous FSDP root policy.
- Automodel unit tests and the relevant NeMo RL tests pass.
