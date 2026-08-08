# RL for Masked Diffusion Language Models (dLLMs)

NeMo RL can train masked diffusion language models (dLLMs) such as
[LLaDA](https://github.com/ML-GSAI/LLaDA) with GRPO. This implements the setup of
[GDPO: Group Diffusion Policy Optimization](https://arxiv.org/abs/2510.08554).

```{note}
This is unrelated to `grpo.adv_estimator.name = "gdpo"`, which is the
multi-reward advantage estimator from
[arXiv:2601.05242](https://arxiv.org/abs/2601.05242). The diffusion support
described here is enabled with `policy.dllm.enabled`, and adds no new algorithm
name.
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
each node stays stochastic. Two quadrature points with one mask draw each are
usually enough.

The estimator is implemented in
{py:class}`SdmcElboEstimator <nemo_rl.models.policy.dllm.elbo.SdmcElboEstimator>`.
Crucially, the ELBO decomposes per position, so it occupies the same
`[batch, seq_len]` slot that autoregressive log probabilities do: the loss
functions, KL penalty, and advantage computation are unchanged.

## Enabling it

Add a `policy.dllm` block and select the `dllm` generation backend:

```yaml
policy:
  model_name: "GSAI-ML/LLaDA-8B-Instruct"
  dllm:
    enabled: true
    quadrature: "gauss-2"   # gauss-1..gauss-5, simpson, or mc
    mc_samples: 1           # mask draws per quadrature point
    block_length: 32        # denoising block width
    diffusion_steps: 128    # total denoising steps per rollout
    cfg_scale: 0.0          # unsupervised classifier-free guidance
  generation:
    backend: "dllm"
    max_new_tokens: 512     # must be a multiple of block_length
```

A ready-to-edit exemplar lives at `examples/configs/dllm_grpo_llada_8b.yaml`, and a
runnable recipe at
`examples/configs/recipes/llm/grpo-llada-8b-instruct-1n8g-fsdp2tp1-dllm.yaml`:

```sh
uv run examples/run_grpo.py \
    --config examples/configs/recipes/llm/grpo-llada-8b-instruct-1n8g-fsdp2tp1-dllm.yaml
```

### Cost

Each likelihood evaluation costs `len(quadrature points) * mc_samples` forward
passes, and GRPO evaluates the likelihood three times per step (current, previous,
and reference policy). `gauss-2` with `mc_samples: 1` therefore doubles the forward
cost relative to an autoregressive policy. Setting `cfg_scale > 0` doubles the cost
of *generation* on top of that.

### Mask token

The mask token id is read from the model config (LLaDA publishes `mask_token_id`).
Set `policy.dllm.mask_id` explicitly only for a model that does not publish one —
a wrong value silently corrupts the likelihood rather than failing.

## Generation

Masked diffusion models decode a fixed-width canvas by iteratively unmasking
positions, so there is no KV cache for the autoregressive engines to manage.

SGLang does serve some diffusion models: since 0.5.12 it ships `--dllm-algorithm`
with low-confidence and joint-threshold remasking, covering `LLaDA2MoeModelLM`
(LLaDA2.0) and `SDARForCausalLM`. It does not cover LLaDA-8B (`LLaDAModelLM`),
which is the model the GDPO recipe uses, and vLLM and TRT-LLM have no diffusion
support at all.

The `dllm` backend therefore denoises inside the training workers, which means:

- Generation is **colocated by construction** and reads the live training weights,
  so there is no refit step.
- Blocks of `block_length` positions are denoised left to right; within a block,
  each step unmasks the highest-confidence positions.
- The response length is recovered after the fact by scanning the filled canvas for
  the first stop token, since the model fills the whole region regardless.

## Loss configuration

The ELBO is a *sequence* likelihood — only its masked sum is meaningful — so GDPO
uses a GSPO-shaped objective with one length-normalized importance ratio per
sequence:

```yaml
loss_fn:
  sequence_level_importance_ratios: true
  token_level_loss: false
  use_importance_sampling_correction: false
  position_aligned_logprobs: true
```

`position_aligned_logprobs` reflects that a dLLM scores token `i` at position `i`,
with no autoregressive shift to undo. It must agree with `policy.dllm.shift_targets`.

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

`validate_dllm_policy` rejects these at startup rather than letting them silently
train against the wrong likelihood:

| Setting | Why |
| --- | --- |
| `sequence_packing`, `dynamic_batching` | Reorder or reuse tokens in ways that assume a causal factorization. |
| `megatron_cfg.enabled` | Only the DTensor backend implements the ELBO path. |
| `router_replay` | Assumes one stable token-to-expert map per rollout; a dLLM re-routes every position on every denoising step. |
| `dtensor_cfg.context_parallel_size > 1` | Shards the sequence, but the ELBO masks positions across the whole sequence. |
| `logprob_batch_size != train_micro_batch_size` | Masks are drawn from a seeded generator over the microbatch shape, so differing sizes would compare ELBOs taken at different masks. |
| Autoregressive generation backends | The ELBO path needs the in-worker denoiser. SGLang can serve LLaDA2.0 and SDAR for inference, but wiring that into RL rollouts is not implemented yet. |
