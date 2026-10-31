# Score Centering

Score centering adds a correction to the policy-gradient objective when the
rollout sampler and training policy assign different probabilities to tokens.
Enable it with `loss_fn.score_centering: true`; it is disabled by default.

The current rollout integration requires SingleController, NeMo Gym, vLLM, and
token capture with a positive top-k logprob count. The functional test uses
Megatron training. This feature is experimental: distributed integration and
training convergence need validation for the intended model and layout.

## Objective

Let `p(v)` be the current training policy, `q(v)` the rollout sampler, and `H`
the sampler's top-k token set at one generated position. The sampler supplies
both token IDs and logprobs for `H`. The trainer computes its own logprobs at
those same IDs, normalized over the full vocabulary rather than just `H`.
With top-k/top-p sampling, `q` is the filtered sampler distribution and `p`
is the trainer's distribution renormalized over the same support (see
[Sampling filters](#sampling-filters)).

The estimator models the unobserved sampler tail as a rescaled training-policy
tail:

```text
rho = (1 - sum_H q(v)) / max(1 - sum_H p(v), 1e-6)
q(v) ≈ rho * p(v) outside H
```

The per-token correction is:

```text
C = sum_H stop_gradient(m(v) - alpha * p(v)) * log p(v)
L = L_base + A * C
```

Here `A` is the token advantage and `L_base` is the existing actor objective,
including its clipping and configured importance-sampling correction. The
coefficients depend on the importance-sampling mode:

| Mode | Head mass `m(v)` | Tail coefficient `alpha` |
|---|---|---|
| IS off | `q(v)` | `rho` |
| Untruncated token-level IS | `p(v)` | `1` |
| Token-level TIS with bounds `[lo, hi]` | `clamp(p(v), lo*q(v), hi*q(v))` | `clamp(1, lo*rho, hi*rho)` |
| Token-level ICEPOP with bounds `[lo, hi]` | `p(v)` if `p(v)/q(v)` is in bounds, otherwise `0` | `1` if `1/rho` is in bounds, otherwise `0` |

Only `log p(v)` carries gradients through this correction. The captured head
retains its original probability mass; it is not renormalized to one. The loss
applies the existing token/sample masks and normalization to `A * C`.
Masked logprobs are neutralized before exponentiation. An ICEPOP rejection of
the sampled token does not mask the head/tail correction. Under untruncated
IS, the head and tail coefficients cancel and the correction is zero.

This implementation adds a correction to the existing clipped objective; it
does not replace it with centered REINFORCE or exactly center every possible
PPO clipped gradient. In particular, the ordinary sampled-token IS ratio uses
the previous policy, while the head correction uses the current policy.
`force_on_policy_ratio: true` selects the detached current policy for the
ordinary ratio and requires one optimizer update per rollout batch.

## Configuration

Apply these settings to a SingleController NeMo Gym GRPO configuration with
the TransferQueue data plane and async vLLM serving already configured:

```yaml
loss_fn:
  score_centering: true
  token_level_loss: true
  sequence_level_importance_ratios: false
  use_importance_sampling_correction: false
  truncated_importance_sampling_type: null
  truncated_importance_sampling_ratio_min: null
  truncated_importance_sampling_ratio: null

token_capture:
  enabled: true

policy:
  sequence_packing:
    enabled: false
    fuse_loss: false
  logprob_chunk_size: 128
  megatron_cfg:
    enabled: true
    use_fused_linear_logprobs: false
  generation:
    backend: vllm
    temperature: 1.0
    top_p: 1.0
    top_k: null
    top_logprobs: 16
    vllm_cfg:
      async_engine: true
      logprobs_mode: processed_logprobs
```

`top_logprobs` controls how many candidates are recorded; it does not truncate
sampling. Setup raises vLLM's `max_logprobs` cap to at least that count when
needed; an explicit `-1` (unlimited) cap is preserved. A positive count requires
score centering and token capture. Requests must omit nonempty
`logprob_token_ids`: vLLM would return those specific IDs instead of its top-k
head, even if the list contains K IDs. Such requests are rejected.
Use positive temperature and
`processed_logprobs` to keep the reported sampler probabilities consistent
with the distribution being sampled. Startup validation rejects
nonpositive/nonfinite temperatures. Raw/default logprobs are accepted only at
temperature 1 without sampling filters; non-unit temperatures and top-k/top-p
filtering require processed logprobs. Other sampling processors must not change
the distribution when using raw logprobs.

### Sampling filters

Score centering under top-k or top-p sampling requires
[Sampler Replay](sampler-replay.md): set
`policy.generation.sampler_replay.enabled: true` and meet its requirements,
including `top_logprobs >= top_k`. Score centering with a filter and replay
disabled is rejected at setup.

With replay, `p` is the trainer's distribution renormalized over the sampler's
kept set `S_q`, the same support as `q`, and the captured head holds all of
`S_q`:

- Head slots the sampler filtered out (`-inf`) carry no coefficient.
- The head holds all of `p`, so the estimator has no tail (`rho = 0`) and is
  exact for every IS mode and for both token- and sequence-level loss.
- The sampled token and the head are scored in one pass over `S_q`.

To enable TIS, replace the four IS settings with:

```yaml
loss_fn:
  use_importance_sampling_correction: true
  truncated_importance_sampling_type: tis
  truncated_importance_sampling_ratio_min: 0.5
  truncated_importance_sampling_ratio: 2.0
```

For untruncated IS, set `use_importance_sampling_correction: true` and leave
the truncation type and both bounds `null`. For ICEPOP, use the TIS example
with `truncated_importance_sampling_type: icepop`; out-of-bounds weights become
zero instead of being clamped. Both the head and modeled tail use this rule.
`seq-mask-tis` and sequence-level IS remain unsupported with score centering.

## Data Flow

1. NeMo-RL configures Gym to send `logprobs=true, top_logprobs=K` in each policy
   HTTP request. This applies to every request to the policy model server,
   including uncaptured calls such as a judge routed to `policy_model`.
   Uncaptured responses carry the top-k values over HTTP. Gym's final request preprocessing normalizes a missing/null
   count to zero when logprobs are enabled and preserves an explicit count.
   The capture adapter leaves that count unchanged.
2. vLLM resolves that request normally. The worker checks that the requested K
   matches its configured capture count and extracts the engine's top-k IDs
   and logprobs before response serialization.
   Tokens, processed media and sampler tensors are written in one staging
   operation. A failed write returns `capture_failed` and attempts to discard
   any partial row before acknowledgement.
3. The worker stages `sampler_topk_token_ids` as int32 and
   `sampler_topk_logprobs` as float32. It removes the arrays from the HTTP
   response, which retains completion content and capture acknowledgement.
4. The CPU finalizer checks fragment shapes, assembles token-aligned rollout
   columns, and fills prompt/tool positions with zeros.
5. The policy replica leader fetches the columns and broadcasts them to its
   model-parallel workers. The trainer gathers differentiable
   `policy_logprobs_at_sampler_topk`, aligned with the next-token predictions.
6. `ClippedPGLossFn` uses those trainer logprobs and the sampler logprobs to
   compute the correction.

This requires the Gym capture adapter version pinned by this feature. An older
adapter that overwrites `top_logprobs` with zero is rejected by the worker.
The fields above are separate from the teacher/student top-k tensors used by
on-policy distillation.

### Extending Candidate Collection

[The sampler adapter](../../nemo_rl/models/generation/score_centering.py)
keeps request preparation and extraction behind three standalone functions:

- `build_score_centering_request_overrides(num_candidates=K)` supplies the
  HTTP fields that Gym merges into each policy request.
- `validate_score_centering_request(request, num_candidates=K)` checks the
  collection settings on the serving worker before capture admission.
- `extract_score_centering_samples(output, num_candidates=K)` returns CPU
  `(sampler_ids, sampler_logprobs)` tensors with shape `[generated_len, K]`,
  int32 IDs, and float32 logprobs under the sampler distribution.

Today these functions implement top-k collection. A future candidate-selection
strategy can change request preparation, validation, and extraction together
while preserving the tensor pair and the existing staging fields. The loss
estimator must also match that strategy; this interface does not implement
sampling without replacement or change the current top-k estimator.

## Supported Layouts and Costs

- Megatron's ordinary loss preparation supports tensor/context parallel
  top-k gathering. Single-device logits are also supported by the helper.
- Internal sequence packing supports `sequence_packing.fuse_loss=true`:
  sampler IDs are packed and shifted per sequence, TP gathers the full-vocabulary
  logprobs, and CP shards are reconstructed separately for each padded sequence.
  Externally prepacked inputs are rejected. The example above uses the ordinary
  unpacked path; enable both packing and `fuse_loss` for the packed path.
- With sampling filters, the sampled token and the head are scored in one pass
  over the captured support: a gather of the `K` head logits and a logsumexp
  over the finite ones (under tensor parallelism, one all-reduce of those `K`
  logits per position), with no vocabulary top-k or softmax.
- Automodel context-parallel sharding and DTensor logits are unsupported.
- Full-vocabulary on-policy distillation and fused linear logprobs are
  rejected because this path needs ordinary vocabulary logits.
- Extraction checks the generated-token count, candidate count, unique IDs,
  nonnegative int32-compatible IDs, float32-representable logprobs (`-inf` is
  kept for candidates the sampler filtered out), logprobs no larger
  than `1e-5`, and head mass no larger than `1.0001`. Sampler logprobs
  are preserved without flooring or renormalization. These checks apply before
  staging; ignored prompt/tool positions are still zero-filled by the finalizer.

Storage and transport grow linearly with K. IDs and sampler logprobs alone use
`8 * T * K` bytes for T token positions, excluding copies, padding, metadata,
and trainer tensors. At 32K positions this is 32 MiB for K=128 or 256 MiB for
K=1024. Avoiding HTTP serialization removes JSON overhead; CPU staging,
finalization, data-plane transport, replica broadcast, and trainer gathers
still carry the top-k data. Large-K performance has not been benchmarked.

## Functional Test

In the configured NeMo-RL GPU environment, run:

```bash
uv run --no-sync bash tests/functional/grpo_score_centering_gym_single_controller.sh
```

The test adapts the existing SingleController Gym token-capture smoke. It
requires two GPUs, the Gym dependencies, model/data access, and `HF_TOKEN` for
the Workplace Assistant data preparation. It runs three training steps each
with IS off, untruncated IS, TIS, and ICEPOP, using Qwen3-0.6B, Megatron, and
async vLLM. TIS and ICEPOP enable internal packing with `fuse_loss=true`; the
other two modes use unpacked preparation.
It is registered in the full `L1_Functional_Tests_SingleController_3.sh` suite.

The checks require three loss/head-mass records, finite loss and gradient
norms, positive valid-token counts, head mass in `(0, 1.0001]`, finalizer
activity, and zero invalid-row and poisoned-rollout metrics. The
`train/score_centering_head_mass` metric proves the score-centering loss path
executed. This smoke does not establish convergence, large-K performance, or
TP/CP correctness. Separate packed TP/CP unit tests compare values and gradients
against an unpacked, full-vocabulary reference.

The L1 plan discovers `L1_Functional_Tests_ScoreCentering_Distributed.sh` for
packed and unpacked TP/CP checks. This entrypoint requires two visible GPUs
for TP=2/CP=1 and TP=1/CP=2 on NCCL. It also runs packed value and gradient
checks on CPU/Gloo with four processes for combined TP=2/CP=2, using the same
reference and assertions as the GPU test. Combined four-GPU NCCL cases run
when four GPUs are visible and are skipped on the two-GPU CI runners:

```bash
uv run --no-sync bash tests/functional/L1_Functional_Tests_ScoreCentering_Distributed.sh
```

## Implementation

- [Shared correction utility](../../nemo_rl/algorithms/loss/score_centering.py)
- [Clipped policy-gradient loss](../../nemo_rl/algorithms/loss/loss_functions.py)
- [Trainer top-k gather](../../nemo_rl/distributed/model_utils.py)
- [Functional test](../../tests/functional/grpo_score_centering_gym_single_controller.sh)
- [GRPO guide](grpo.md)
