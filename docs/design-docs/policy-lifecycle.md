# Policy Lifecycle and Resource Preparation

> **TL;DR.** Every GPU-bound call on a `LMPolicy` (`train`, `get_logprobs`,
> `score`, `get_topk_logits`, `generate`) requires the model weights to
> be **on the GPU** at the moment of the call. You move them there by
> calling `prepare_for_training()` or `prepare_for_lp_inference()`. If
> you forget, you will see opaque CUDA errors — most often
> `RuntimeError: CUDA error: an illegal memory access was encountered`.

This document is aimed at users writing their own training loops (or
making large changes to existing ones) on top of NeMo RL. The built-in
algorithms (`grpo`, `sft`, `dpo`, `rm`, `distillation`) already call
these methods in the right places; you only need this if you are
replacing or extending them.

## The state machine

A policy worker can be in one of three logical states:

| State | Weights on | Optimizer on | What you can call |
| --- | --- | --- | --- |
| **Training** | GPU | GPU | `train`, `get_logprobs`, `score`, `save_checkpoint` |
| **Logprob / inference** | GPU | offloaded (CPU/disk) or absent | `get_logprobs`, `score`, `get_topk_logits`, `generate` |
| **Offloaded** | CPU/disk | CPU/disk | `prepare_*`, refit helpers — **nothing GPU-bound** |

The transitions are:

```text
            prepare_for_training()
   Offloaded ─────────────────────► Training
        ▲                              │
        │ offload_*                    │ prepare_for_lp_inference()
        │                              ▼
        └──────────────────────── Logprob/inference
```

The reason this matters: training mode pins the optimizer state on the
GPU, which can be large. RL pipelines that interleave training with
generation routinely **offload** the optimizer (and sometimes the
weights) to make room for the inference engine, then move them back
before the next training step.

## The two preparation methods

Both methods live on
[`PolicyInterface`](../../nemo_rl/models/policy/interfaces.py) and are
implemented by every policy worker backend (`dtensor_policy_worker`,
`dtensor_policy_worker_v2`, `megatron_policy_worker`).

### `prepare_for_training()`

Call this **before** any of:

- `policy.train(...)`
- `policy.save_checkpoint(...)`

This moves the optimizer state back to the GPU (if it was offloaded)
and switches the model to train mode. After this call the policy can
do backward passes and step the optimizer.

### `prepare_for_lp_inference()`

Call this **before** any of:

- `policy.get_logprobs(...)`
- `policy.score(...)`
- `policy.get_topk_logits(...)`
- `policy.generate(...)` (Megatron backend only)

This moves model weights to the GPU (if they were offloaded), switches
the model to eval mode, and frees optimizer state that is not needed
for inference. It is cheaper than `prepare_for_training()` and is what
you want between training steps when you only need forward passes (for
example to compute reference/policy logprobs against a rollout).

### Symmetry

There is no separate `finish_*` call you must make. The next
`prepare_*` or `offload_*` call handles the transition. Use
`offload_before_refit()` / `offload_after_refit()` explicitly when you
need to make room for an external inference engine (e.g. vLLM) during
refit. See `nemo_rl/algorithms/grpo.py` for the canonical
training-loop example.

## What happens if you skip the prepare step

If you call a GPU-bound API while the policy is in an offloaded state,
you will typically see one of:

- `RuntimeError: CUDA error: an illegal memory access was encountered`
- `RuntimeError: Expected all tensors to be on the same device`
- A silent crash with no Python traceback (the worker process dies)

The traceback usually points deep into the model forward pass and gives
no hint that the cause was a missing `prepare_for_*` call. If you see
any of these in a custom training loop, **first** check that the most
recent state transition was a `prepare_for_lp_inference()` or
`prepare_for_training()` matching the call you are about to make.

## A minimal custom training loop

```python
from nemo_rl.models.policy.lm_policy import LMPolicy

policy: LMPolicy = ...  # constructed by the example runners

for step in range(num_steps):
    # 1. Generate rollouts. Weights need to be on the GPU.
    policy.prepare_for_lp_inference()
    rollouts = my_rollout_fn(policy, batch)

    # 2. Compute reference logprobs (still inference mode).
    ref_logprobs = policy.get_logprobs(rollouts)

    # 3. Switch to training mode before the train step.
    policy.prepare_for_training()
    train_metrics = policy.train(rollouts, ref_logprobs, ...)

    # 4. Optionally offload to free GPU for an external generator.
    policy.offload_before_refit()
    refit_external_engine(policy)
    policy.offload_after_refit()
```

If you replace any of these steps but keep the surrounding ones, keep
the `prepare_*` calls in the same relative position — they are not
optional.

## Cross-references

- Interface: [`nemo_rl/models/policy/interfaces.py`](../../nemo_rl/models/policy/interfaces.py)
- Reference loop: [`nemo_rl/algorithms/grpo.py`](../../nemo_rl/algorithms/grpo.py)
- Related issues: [#1141](https://github.com/NVIDIA-NeMo/RL/issues/1141)
