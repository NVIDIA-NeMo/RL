# Hybrid AR + Diffusion GRPO

This guide covers `policy.logprob_estimation.type: hybrid_ar_diffusion`, an auxiliary
cross-entropy term for GRPO training of block-diffusion language models.

Block-diffusion models decode in two modes from one set of weights: autoregressively, or by
iteratively unmasking tokens within a block. RL alone optimizes whichever mode produced the
rollouts, and the diffusion mode drifts away from the objective it was pretrained for. The
hybrid loss anchors it by adding the masked-diffusion (MDM) pretraining objective back into
the RL update.

## The objective

Each sequence is laid out as `[noisy | clean]`:

- the **clean** half carries the standard clipped policy gradient;
- the **noisy** half is the same response with a random subset masked, and carries a plain
  masked cross-entropy term — no ratio, no clipping, no advantage weighting.

```
loss = pg_loss_weight * pg_loss + ce_loss_weight * ce_loss
```

Both halves are scored in a **single forward pass**. The batch builder bakes the
autoregressive shift into `hybrid_target_ids`, so one same-position gather serves both: the
noisy half predicts the token *at* each position, the clean half predicts the *next* one.
`hybrid_pg_mask` and `hybrid_ce_mask` select which positions feed which term; they occupy
disjoint halves and never overlap.

Because the clean half never attends to the noisy half, the mask can be redrawn every step
without invalidating `prev_logprobs`.

## Usage

```yaml
policy:
  worker_extension_cls_fqn: nemo_rl.models.policy.workers.hybrid_ar_diffusion_megatron_policy_worker.HybridARDiffusionMegatronPolicyWorker
  logprob_estimation:
    type: hybrid_ar_diffusion
    mask_token_id: 100          # must match the checkpoint's mask token
    ce_loss_weight: 0.1         # 0.0 recovers the RL-only baseline
    pg_loss_weight: 1.0         # 0.0 trains on CE alone
    mask_ratio_min: 0.2         # per-sample masking ratio t ~ U(min, max)
    mask_ratio_max: 0.8
    elbo_weight_ce: true        # scale CE by 1/t to recover the MDM ELBO
```

A reference config is at
`examples/configs/nemotron_labs_diffusion_3b_hybrid_ar_diffusion_ce.yaml`.

Two settings that are easy to get wrong:

- **`mask_token_id` must match the checkpoint.** A mismatch silently corrupts the CE term
  rather than erroring.
- **Halve `train_micro_batch_size` and `logprob_batch_size`** relative to a comparable AR run.
  The `[noisy | clean]` layout is roughly twice as wide.

The loss emits `pg_loss`, `ce_loss` and `weighted_ce_loss` separately. Log all three: their
relative magnitude is how `ce_loss_weight` is chosen. A reasonable target is for the weighted
terms to land within an order of magnitude of each other.

### Unsupported combinations

These are rejected at startup rather than silently ignored:

| option | reason |
| --- | --- |
| `loss_fn.reference_policy_kl_penalty != 0` | not implemented; route KL through the reward instead |
| `loss_fn.use_importance_sampling_correction` | not implemented |
| `loss_fn.truncated_importance_sampling_type` | no single per-token ratio to truncate |
| `policy.megatron_cfg.use_fused_linear_logprobs` | fused forward gathers at the next input token, not at `hybrid_target_ids` |
| sequence packing | the post-processor requires unpacked microbatches |

## Results

All results are on
[Nemotron-Labs-Diffusion-3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B).

**Setup.** Two GRPO runs on a 102,630-prompt multi-domain blend, 2,000 steps at 128
prompts/step and 8 generations/prompt (≈2.5 epochs), 16 GPUs. The two configs are identical
apart from `ce_loss_weight` (0.0 vs 0.1). Evaluation is a 15,594-row suite — AIME24/25/26,
HMMT, IFBench, MMLU-Pro, MMLU-Redux. Each column reports that run's best checkpoint by
overall accuracy.

### Ablation: does the CE term help?

Diffusion-mode decoding, single deterministic run at temperature 0.

| | base model | `ce_loss_weight=0.0` | `ce_loss_weight=0.1` |
| --- | ---: | ---: | ---: |
| | (no RL) | step 1200 | step 1600 |
| **overall** | 0.5882 | 0.6023 | **0.6376** |
| **AIME-90** | 0.1333 | 0.2111 | **0.3556** |
| aime24 | 0.200 | 0.267 | **0.367** |
| aime25 | 0.133 | 0.233 | **0.300** |
| aime26 | 0.067 | 0.133 | **0.400** |
| hmmt | 0.000 | 0.167 | 0.167 |
| ifbench | — | 0.737 | **0.776** |
| mmlu_pro | 0.563 | 0.582 | **0.617** |
| mmlu_redux | 0.710 | 0.679 | **0.715** |

Every dataset improves or ties against the CE-free run; none regresses. The gain concentrates
on math — AIME-90 nearly doubles.

The comparison against the base model is the more useful one. RL without the CE term moves
diffusion-mode accuracy 0.588 → 0.602 (**+1.4pp**); with it, 0.588 → 0.638 (**+5.0pp**). The
CE term is not a marginal tuning knob — without it, RL largely fails to transfer to diffusion
decoding.

### Decode modes on the CE-trained model

The same CE-trained weights can be decoded three ways. Each column is that mode's own best
checkpoint; all three happen to peak at or near step 1600.

| | Diffusion<br>(20-rep avg) | AR-mode<br>(5-rep avg) | Linear speculation<br>(single, greedy) |
| --- | ---: | ---: | ---: |
| best checkpoint | step_1600 | step_1400 | step_1600 |
| **overall** | 0.6297 | 0.6517 | **0.6624** |
| **AIME-90** | 0.3417 | 0.4133 | **0.4444** |
| aime24 | 0.377 | **0.460** | 0.400 |
| aime25 | 0.317 | 0.393 | **0.433** |
| aime26 | 0.332 | 0.387 | **0.500** |
| hmmt | 0.162 | 0.220 | **0.267** |
| ifbench | 0.758 | 0.752 | **0.775** |
| mmlu_pro | 0.608 | 0.630 | **0.641** |
| mmlu_redux | 0.710 | 0.735 | **0.743** |
| **tokens / forward pass** | 2.03 | 1.00 | **2.99** |

Linear self-speculative decoding — one bidirectional block draft verified by a causal pass,
accepting the matching prefix — is the strongest mode on this checkpoint on both axes: it
edges out AR-mode accuracy (+1.1pp overall, +3.1pp AIME-90) while committing ~3× as many
tokens per forward pass. Confidence-threshold diffusion decoding is the weakest of the three
on accuracy, and its throughput advantage over AR is roughly half what linear speculation
achieves.

This matters for the generation-cost trade-off below: the CE term's cost to
confidence-threshold decoding does not have to be paid, because a better decode mode exists
for the same weights.

### The effect is diffusion-specific

AR-mode decoding of the same checkpoints is unchanged — mean over steps 700–1900:

| | `ce_loss_weight=0.0` | `ce_loss_weight=0.1` |
| --- | ---: | ---: |
| AR-mode overall | 0.6282 | 0.6257 |

Within noise. This is expected: the CE term regularizes the diffusion objective, so it should
not move AR-mode behaviour. Do not read this guide as claiming a general quality improvement.

### Training cost

The two configs differ only in `ce_loss_weight`, so this is like-for-like. Medians over 1,595
and 1,999 steady-state steps, excluding restart-warmup and validation/checkpoint steps:

| | `ce=0.0` | `ce=0.1` | Δ |
| --- | ---: | ---: | ---: |
| s/step (6144 ctx, 16 GPU) | 299.1 | 300.0 | **+0.3%** |
| s/step (4096 ctx, 8 GPU) | 285.5 | 286.2 | **+0.2%** |
| `policy_training` (s) | 178.4 | 178.2 | −0.1% |

**Under 0.5% — within run-to-run noise.** This follows from the design: both halves are scored
in one forward pass, so the CE term adds masking and a second masked-mean, not a second pass.

### Generation-time trade-off

The CE-trained model produces longer reasoning chains and commits fewer tokens per forward
pass under confidence-threshold diffusion decoding:

| | base | `ce=0.0` | `ce=0.1` |
| --- | ---: | ---: | ---: |
| tokens / forward pass | 2.554 | 2.646 | 2.106 |
| mean generated tokens | 308 | 417 | 768 |

Combined, `ce=0.1` needs roughly 2.3× the forward passes per sample at generation time. The
mechanism is visible in the last row: longer reasoning means lower per-position confidence, so
fewer tokens clear the unmasking threshold each denoising step.

This cost is specific to confidence-threshold decoding and is recoverable. Linear
self-speculative decoding on the *same* CE-trained checkpoint reaches **2.93** tokens per
forward pass — above even the CE-free model's 2.65 — because its acceptance test is an
exact match against a causal verifier rather than a confidence threshold.

### Reading these numbers

- **The ablation table is single deterministic runs at temperature 0**, not averages. The
  decode-mode table is different: its diffusion column is a 20-rep average at temperature 1.0
  (overall σ ≈ 0.002 across reps) and its AR column a 5-rep average. Linear speculation is
  inherently greedy — it supports temperature 0 only — so it is a single run with no
  sampling noise to average.
- The two tables are therefore not directly comparable cell-to-cell: 0.6376 (ablation,
  temp 0) and 0.6297 (decode modes, temp 1.0 20-rep) are the same checkpoint measured two
  ways.
- AIME-90 pools 90 problems, so its standard deviation is ≈0.03. Treat the pooled figure as
  the reliable one and the per-AIME-dataset deltas as indicative.
- The base-model diffusion run has no IFBench entry, so its 0.5882 overall is computed over a
  slightly smaller row set than the RL numbers.

## Limitations

- Validated on one model family and one data blend.
- One `ce_loss_weight` value (0.1) against 0.0; no sweep over the weight.
- Requires a Megatron policy worker that can build the `[noisy | clean]` batch; the built-in
  workers cannot, which is why `worker_extension_cls_fqn` is mandatory and validated at
  startup.
- The model must expose asymmetric-AR attention hooks (`set_asymmetric_ar_metadata` and
  related). Checkpoints without them fail at the first microbatch with a clear error.
