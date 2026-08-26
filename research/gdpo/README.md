# GDPO: RL for Masked Diffusion Language Models

NeMo RL can train masked diffusion language models (dLLMs) such as
[LLaDA](https://github.com/ML-GSAI/LLaDA) with GRPO. This implements the setup of
[GDPO](https://openreview.net/forum?id=JaqvespRBP), *Improving Reasoning for
Diffusion Language Models via Group Diffusion Policy Optimization* (ICLR 2026).

```{note}
This is unrelated to `grpo.adv_estimator.name = "gdpo"`, which is the
multi-reward advantage estimator from
[arXiv:2601.05242](https://arxiv.org/abs/2601.05242). The diffusion support
described here is enabled with `policy.masked_diffusion.enabled`.
```

## Why dLLMs need a different likelihood

Policy-gradient methods need `log p_θ(y | q)`. An autoregressive model gives this
exactly, as a product of next-token probabilities. A masked diffusion model does
not: its training objective is a variational bound, and the sequence likelihood is
only available as an expectation over mask ratios,

```
log p_θ(y | q) >= ELBO = E_{t ~ U[0,1]} (1/t) * E_{y_t} sum_{i masked} log p_θ(y_i | y_t, q)
```

Estimating that expectation by naive double Monte Carlo is high-variance. GDPO
instead uses **Semi-deterministic Monte Carlo (SDMC)**: the integral over `t` is
evaluated with a fixed Gauss-Legendre quadrature rule, and only the mask draw at
each node stays stochastic. Two or three quadrature points with one mask draw
each are usually enough; the published recipe uses three.

The estimator is implemented in [`SdmcElboEstimator`](gdpo/elbo.py).
Crucially, the ELBO decomposes per position, so it occupies the same
`[batch, seq_len]` slot that autoregressive log probabilities do: the loss
functions, KL penalty, and advantage computation are unchanged.

## Enabling it

Add a `policy.masked_diffusion` block and select the `automodel` generation backend:

```yaml
policy:
  model_name: "GSAI-ML/LLaDA-8B-Instruct"
  masked_diffusion:
    enabled: true
    mask_id: null
    shift_targets: false
    likelihood:
      type: "sdmc"
      quadrature: "gauss-3"
      mc_samples: 1
  generation:
    backend: "automodel"
    max_new_tokens: 256
    denoise_cfg:
      type: "block"
      block_length: 32
      diffusion_steps: 128
      cfg_scale: 0.0
```

A ready-to-edit exemplar lives at `research/gdpo/configs/gdpo_llada_8b.yaml`, and a
runnable recipe at
`research/gdpo/configs/recipes/llm/gdpo-llada-8b-instruct-1n8g-fsdp2tp1.yaml`:

```sh
uv run research/gdpo/gdpo.py \
    --config research/gdpo/configs/recipes/llm/gdpo-llada-8b-instruct-1n8g-fsdp2tp1.yaml
```

### Cost

Each likelihood evaluation costs `len(quadrature points) * mc_samples` forward
passes. GDPO evaluates it for the old and current policy on each step;
`gauss-3` with `mc_samples: 1` therefore uses six model forwards per sequence
across scoring and training. Setting
`generation.denoise_cfg.cfg_scale > 0` doubles the cost
of *generation* on top of that.

### Prompt masking

`policy.masked_diffusion.likelihood.p_mask_prompt` optionally corrupts prompt
tokens as a regularizer. Leave it at `0.0` to reproduce GDPO. The setting is
inherited from the [d1](https://github.com/dllm-reasoning/d1) *diffu*-GRPO
trainer that GDPO builds on; corrupted prompt positions condition the model but
are never scored by the SDMC estimator.

### Adapters

The exemplar trains LoRA adapters (`policy.dtensor_cfg.lora_cfg`, r=128,
alpha=64), as the published recipe does. This is not just a memory
convenience: the ELBO keeps `quadrature * mc_samples` forward activations alive
for the backward pass, so full fine-tuning an 8B dLLM needs considerably more
room than the same model trained autoregressively.

### Mask token

The mask token id is read from the model config (LLaDA publishes `mask_token_id`).
Set `policy.masked_diffusion.mask_id` explicitly only for a model that does not publish one —
a wrong value silently corrupts the likelihood rather than failing.

## Generation

Masked diffusion models decode a fixed-width canvas by iteratively unmasking
positions, so there is no KV cache for the autoregressive engines to manage.

SGLang does serve some diffusion models: since 0.5.12 it ships `--dllm-algorithm`
with low-confidence and joint-threshold remasking, covering `LLaDA2MoeModelLM`
(LLaDA2.0) and `SDARForCausalLM`. It does not cover LLaDA-8B (`LLaDAModelLM`),
which is the model the GDPO recipe uses, and vLLM and TRT-LLM have no diffusion
support at all.

The `automodel` backend therefore denoises inside the training workers, which means:

- Generation is **colocated by construction** and reads the live training weights,
  so there is no refit step.
- Blocks of `generation.denoise_cfg.block_length` positions are denoised left to right; within a block,
  each step unmasks the highest-confidence positions.
- The response length is recovered after the fact by scanning the filled canvas for
  the first stop token, since the model fills the whole region regardless.

## Loss configuration

The ELBO is a *sequence* likelihood — only its masked sum is meaningful — so GDPO
uses a GSPO-shaped objective with one length-normalized importance ratio per
sequence:

```yaml
loss_fn:
  reference_policy_kl_penalty: 0.0
  sequence_level_importance_ratios: true
  token_level_loss: false
  use_importance_sampling_correction: false
  position_aligned_logprobs: true
```

`position_aligned_logprobs` reflects that a dLLM scores token `i` at position `i`,
with no autoregressive shift to undo. It must agree with `policy.masked_diffusion.shift_targets`.
The current KL estimators require token log probabilities, so both policy KL and
reward KL are unsupported for the sequence-level ELBO.

Generation-time log probabilities are likewise unavailable for iterative
denoising. Metrics that compare generation and policy log probabilities
(`token_mult_prob_error`, generation/policy KL, JSD, sampling importance ratio,
and approximate entropy) are logged as zero and sequence log-probability error
masking must remain disabled.

Advantages should be left **unnormalized**:

```yaml
grpo:
  normalize_rewards: false
  use_leave_one_out_baseline: false
```

Dividing by the group standard deviation amplifies ELBO estimator noise on
low-variance groups; GDPO follows
[Dr.GRPO](https://arxiv.org/abs/2503.20783) in using `A_g = R_g - mean(R)`.

## Unsupported combinations

`validate_gdpo_config` rejects these at startup rather than letting them silently
train against the wrong likelihood:

| Setting | Why |
| --- | --- |
| `sequence_packing`, `dynamic_batching` | Reorder or reuse tokens in ways that assume a causal factorization. |
| `megatron_cfg.enabled` | Only the DTensor backend implements the ELBO path. |
| `router_replay` | Assumes one stable token-to-expert map per rollout; a dLLM re-routes every position on every denoising step. |
| `dtensor_cfg.context_parallel_size > 1` | Shards the sequence, but the ELBO masks positions across the whole sequence. |
| Nonzero `reference_policy_kl_penalty`, `use_kl_in_reward` | Existing KL estimators require token log probabilities rather than per-position ELBO contributions. |
| `grpo.seq_logprob_error_threshold` | Denoising does not expose generation-time token log probabilities to compare against policy ELBOs. |
| `generation.stop_strings` | The denoiser fills a canvas rather than decoding incrementally; use `stop_token_ids`. |
| Autoregressive generation backends | The ELBO path needs the in-worker denoiser. SGLang can serve LLaDA2.0 and SDAR for inference, but wiring that into RL rollouts is not implemented yet. |

## References

- **GDPO** — Rojas, Lin, Rasul, Schneider, Nevmyvaka, Tao and Deng, *Improving
  Reasoning for Diffusion Language Models via Group Diffusion Policy
  Optimization*, ICLR 2026.
  [paper](https://openreview.net/forum?id=JaqvespRBP) ·
  [arXiv:2510.08554](https://arxiv.org/abs/2510.08554) ·
  [code](https://github.com/kevinrojas1499/GDPO).
  The SDMC ELBO estimator and the sequence-level objective implemented here.
- **d1** — Zhao, Gupta, Zheng and Grover, *d1: Scaling Reasoning in Diffusion
  Large Language Models via Reinforcement Learning*, NeurIPS 2025.
  [paper](https://openreview.net/forum?id=7ZVRlBFuEv) ·
  [code](https://github.com/dllm-reasoning/d1).
  Introduced *diffu*-GRPO and the prompt-masking regularizer; GDPO builds on its
  trainer, which is why some of its config keys survive into GDPO's recipes.
- **LLaDA** — [code](https://github.com/ML-GSAI/LLaDA). The block-wise
  denoising sampler and the low-confidence remasking rule.
- **Dr.GRPO** — Liu et al., *Understanding R1-Zero-Like Training: A Critical
  Perspective*. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783). The
  unnormalized advantages GDPO adopts.
