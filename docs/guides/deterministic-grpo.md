# Deterministic Megatron GRPO

`grpo.deterministic: true` enables deterministic training and generation settings
for fresh runs on a fixed software stack, GPU topology, parallel layout, and
training microbatch layout. It supports text-only Megatron training and generation.
End-to-end reproducibility also requires deterministic environment observations
and rewards.
The mode is opt-in; ordinary GRPO scheduling is unchanged.

## Fixed-lag asynchronous training

Enable `grpo.async_grpo.enabled: true` with separate training and inference GPUs.
A logical plan assigns each training batch its exact prompt-stream ordinals,
generation replicas, and inference policy version before completion order can
influence them. With Wk denoting the weights before update k:

| Training batch | Generation weights | Concurrent work |
| --- | --- | --- |
| B0 | W0 | Bootstrap B1 under W0 |
| B1 | W0 | Generate B2 under W1 while training B1 |
| B2 | W1 | Generate B3 under W2 while training B2 |

The first update trains B0 while B1 can still be generating. Each subsequent
update consumes a one-update-old batch. This schedule is deterministic but is a
different algorithm from synchronous on-policy training: matching training and
generation log-probs requires comparing the **same policy snapshot**. Async GRPO
still requires importance sampling correction.

The collector validates exact membership and version at dispatch; the trainer
validates replay results and restores prompt order before computing advantages
and training. Seeds depend on the run seed, prompt occurrence, replica, and turn.
Native Megatron `SamplingParams.seed` supplies request-local token sampling.

Completion notifications distinguish two dependencies:

- **Generation complete:** a trajectory cannot issue another model request.
  Refitting waits for all trajectories assigned to the old version, including
  lookahead that has not been dispatched yet.
- **Batch ready:** all required rewards have been acknowledged by replay.
  Training waits for this condition.

At a known final turn or truncation, generation releases its weight dependency
before terminal reward processing. Otherwise it retains that dependency through
intermediate environment calls. Terminal rewards can overlap refitting and the
next generation window. Waits use condition notifications from collector threads;
no blocking wait is placed on the replay actor that receives results. The final
step never schedules an unused lookahead batch. Validation runs between generation
windows, after the old policy has been released and before the next is admitted.

Failures stop strict execution. A failed or cancelled trajectory is never replaced
with the next available prompt or treated as successful cleanup. The initial
implementation does not recover failed rollouts or resume checkpoints.

## Example

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen2.5-0.5b-1n4g-megatron-deterministic-async.yaml
```

The recipe allocates two training and two inference GPUs on one four-GPU node.
A synchronous counterpart is provided in
`grpo-qwen2.5-0.5b-1n4g-megatron-deterministic.yaml`.

Strict execution resolves worker environments, Torch deterministic algorithms,
batch-invariant Megatron kernels, and isolated dataloader RNGs before training.
The current batch-invariant contract requires temperature 1, top-p 1, no top-k
filter, raw log-probs, and matching supported training/inference parallel settings.
Math verification disables wall-clock scoring deadlines, resets verifier RNG and
caches per response, and propagates verifier errors. External job supervision can
still terminate a stalled run; it must not convert a timeout into a training reward.

Other environments are allowed with a startup warning. The mode keeps training
and inference determinism enabled but does not modify those environments or
guarantee their reproducibility. Their observations and rewards must be repeatable
for the same inputs and seeds regardless of worker scheduling, external state,
and wall-clock timeouts. The built-in verifier controls apply only to the `math`
environment with `hf_math_verify` and the default `math` verifier type, including
when it is used alongside other environments. Passing configuration validation
does not establish end-to-end determinism for a custom environment.

## Draft scope and dependencies

- GPU-only optimizer; one-update lag; no tolerated
  generation failures or in-flight weight replacement.
- Async checkpointing and checkpoint resume are rejected explicitly. Exact resume
  needs the lookahead payload or its original policy snapshot, the input cursor,
  and training RNG/optimizer state. Ordinary frontier-aligned replay restoration
  alone does not provide this guarantee.
- Inference CUDA graphs, prefix caching, and chunked prefill retain their configured
  values. The recipes enable all three. Speculative decoding remains unsupported;
  training CUDA graphs remain disabled. Multiple resident policy versions and
  deeper lookahead are deferred.
- Batch-invariant Mamba prefix caching requires [Megatron-LM #8214](https://github.com/NVIDIA/Megatron-LM/pull/8214)
  for cache-state extraction and cache-aware chunk scheduling, FP32 SSM states, and KV blocks aligned to
  the model's SSM chunk alignment. Older Megatron versions reject this combination.
  Prefix cache entries must be scoped to the weight version across refits.
- Requires Megatron's request-local seed API from
  [Megatron-LM #7828](https://github.com/NVIDIA/Megatron-LM/pull/7828), compatible
  batch-invariant kernels, and matching Bridge/Transformer Engine dependencies.
  Dependency pins are not advanced by this PR.
- Focused scheduling, rollout, configuration, and sampler tests are provided.
  Full-model repeated-run and throughput validation of this revision remains
  required before enabling this mode for production. Tests of an earlier local
  prototype are not validation of this revision.
