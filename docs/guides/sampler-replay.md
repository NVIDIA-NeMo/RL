# Sampler Replay

With top-k or top-p sampling, the trainer by default recomputes the filters on
its own logits (see the [GRPO guide](grpo.md)). Sampler Replay instead
replays the sampler's kept set: the trainer renormalizes its logits over the
tokens the sampler actually drew from, read from the vLLM output captured during
rollout. Enable it with `policy.generation.sampler_replay.enabled: true`; it is
disabled by default.

## Why Replay

vLLM samples from the tokens its filters keep, `S_q`, renormalized over that
set. The default recompute builds the trainer's own kept set, `S_p`, from the
trainer's logits. When the two sets of logits differ slightly (bf16 rounding, or
a rollout model quantized differently from the trainer), `S_p` and `S_q` can
disagree at the filter boundary. A sampled token outside `S_p` gets a `-inf`
log-probability and is dropped from the actor loss.

Replay uses `S_q` itself, so:

- every sampled token is inside the support and `p / q` is defined at every
  token;
- trainer and sampler numerics cannot move the filter boundary.

## How It Works

1. Each rollout request asks vLLM for the top `K` log-probabilities per
   generated token, where `K` is `policy.generation.top_logprobs`. With
   `processed_logprobs`, these are computed after temperature, top-k, and
   top-p: kept tokens are renormalized, and tokens the filters removed come
   back as `-inf`.
2. Token capture stages the `K` IDs and log-probabilities per token in the data
   plane. With `K >= top_k`, the finite entries are `S_q`.
3. The trainer computes each log-probability as the token's logit minus the
   log-sum-exp of the logits over `S_q`. It gathers only those `K` logits: no
   vocabulary-wide softmax or top-k. Under tensor parallelism, each rank
   gathers the logits in its vocabulary shard and one all-reduce of `K` values
   combines them.

The actor loss and the prev-logprob pass both use this renormalization. The
reference KL stays on the full-vocabulary distribution.

## Configuration

```yaml
policy:
  generation:
    top_k: 64
    top_p: 0.9
    top_logprobs: 64  # >= top_k
    sampler_replay:
      enabled: true
    vllm_cfg:
      logprobs_mode: processed_logprobs
token_capture:
  enabled: true
```

Replay requires:

- SingleController with NeMo Gym and the vLLM generation backend. The legacy
  GRPO, PPO, and distillation trainers reject the option.
- Top-k or top-p sampling. Top-p requires `top_k`, which bounds `S_q`.
- `top_logprobs >= top_k`, with token capture enabled.
- Vocabulary logits: Megatron without fused linear logprobs, or Automodel
  with `tensor_parallel_size=1` and `context_parallel_size=1`.

## Cost

Replay needs the top-`K` head captured for every generated token, which adds
generation time that grows with `K`. On the trainer side, the log-probability
passes gather `K` logits per token instead of filtering the vocabulary, so they
are not slower than the default recompute.

## With Score Centering

[Score centering](score-centering.md) under top-k or top-p sampling requires
replay; see [Sampling filters](score-centering.md#sampling-filters) for how the
estimator uses the replayed support. Replay itself works with or without score
centering.
