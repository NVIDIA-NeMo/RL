# PPO on the DTensor Backend

This guide shows how to run PPO with the **DTensor V2** backend on the
`ppo-qwen2.5-1.5b-gsm8k-1n8g` recipe, the DTensor twin of the Megatron-only
recipe shipped with the upstream `ppo` branch.

For the underlying algorithm (GAE, value head, training loop, loss), see the
[PPO Guide](ppo.md). This page is **only about how to launch the DTensor
backend** and what to expect.

## Quick Start (single node, 8 GPUs)

```bash
uv run examples/run_ppo.py \
  --config examples/configs/recipes/llm/ppo-qwen2.5-1.5b-gsm8k-1n8g-dtensor.yaml \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=8 \
  policy.model_name=Qwen/Qwen2.5-1.5B-Instruct \
  ppo.seed=42 \
  ppo.max_num_steps=100
```

The Megatron A/B twin is the same command with the
`ppo-...-1n8g-megatron.yaml` config. Both write to the same `wandb` project so
the two curves render side-by-side.

**Reminder**: set `HF_HOME`, `WANDB_API_KEY`, `HF_DATASETS_CACHE` and run
`huggingface-cli login` for gated models.

## What This Recipe Pins

| Setting | Value | Why |
| --- | --- | --- |
| `policy.dtensor_cfg.enabled` | `true` | Selects DTensor V2 policy worker (vs. Megatron) |
| `policy.dtensor_cfg._v2` | `true` | Required to dispatch to `dtensor_*_worker_v2.py` |
| `value.dtensor_cfg.enabled` | `true` | Selects DTensor V2 value worker |
| `policy.optimizer.kwargs.lr` | `1.0e-6` | Matches the Megatron twin's effective policy LR |
| `value.optimizer.kwargs.lr` | `1.0e-5` | Matches the Megatron twin's value LR |
| `policy.dtensor_cfg.tensor_parallel_size` | `1` | 1n8g recipe runs pure DP=8 |

DTensor V2 has **no pipeline parallelism** — `policy.*.pipeline_*_size` is
not exposed for this backend.

## DTensor Value Worker: Standalone FP32 Head

To match the Megatron value worker's numerical behavior on PPO's narrow
critic signal, the DTensor value worker uses a **standalone fp32 `ValueHead`**
(`nn.Linear(hidden, 1)`) with its own `torch.optim.AdamW`, separate from the
backbone's FSDP2 optimizer. The HuggingFace `score` head is frozen and
bypassed; values are right-shifted by one position to align with the Megatron
`V(t) = V(before t)` convention.

The backbone itself stays on the standard FSDP2 `MixedPrecisionPolicy`
(`param_dtype=bf16, reduce_dtype=fp32`). Only the value head's 5121
parameters are upgraded to fp32 — this side-steps the bf16 Adam precision
floor that previously kept `value/loss` ~12× higher than Megatron without
touching the backbone's memory/speed budget.

Implementation: [`nemo_rl/models/value/workers/dtensor_value_worker_v2.py`](../../nemo_rl/models/value/workers/dtensor_value_worker_v2.py).

## Verification vs. Megatron A/B Twin (100 steps, GSM8K, Qwen2.5-1.5B, 1n8g)

End-of-training mean over steps 80–99:

| Metric | Megatron | DTensor | Relative diff |
| --- | ---: | ---: | ---: |
| `train/total_reward` | 0.876 | 0.873 – 0.882 | **< 1%** |
| `policy/loss` | -0.016 | -0.036 – -0.062 | ~3× larger but stable |
| `value/loss` | 0.008 | 0.030 | ~4× larger (bf16 backbone residual) |

The two `policy/loss` and `value/loss` magnitudes do not match exactly
because the DTensor backbone is still bf16 (vs. Megatron's fp32 master
weights). This residual difference does not propagate to reward thanks to
PPO's `advantage = R − V` formulation with `ppo.adv_estimator.normalize_advantages=true`,
which cancels out any constant value bias before the policy update.

## Known Limitations

- **vLLM sampling is not deterministic across launches.** The vLLM engine
  seed is derived from Ray `node_idx`, not from `ppo.seed`, so two runs of
  the same config on different physical nodes produce different generation
  trajectories. `train/total_reward` and `validation/accuracy` still
  converge to the same plateau (PPO's batch-mean target), but
  `mean_gen_tokens_per_sample` and `validation/avg_length` will differ.
  See [`vllm_worker.py`](../../nemo_rl/models/generation/vllm/vllm_worker.py)
  L78–94 for the seed derivation.
- **No pipeline parallelism** — DTensor V2 doesn't support PP. For larger
  models that need PP, use the Megatron backend.
- **Residual value-loss gap** vs. Megatron (~4×) — fixing this would require
  fp32 master weights for the backbone (e.g., torchao integration). Not
  required for reward alignment.

## Where to Look If Something Goes Wrong

- Generation tokens/length differ across runs → expected (see Known
  Limitations); confirm reward still aligns.
- `value/loss` stuck at ~12× Megatron → the standalone fp32 ValueHead is
  not active. Check `value.dtensor_cfg._v2: true` and that
  `dtensor_value_worker_v2.py:ValueHead` is being instantiated.
- `aten.embedding.default got mixed torch.Tensor and DTensor` →
  the backbone forward bypassed the FSDP wrapper. The value worker uses a
  `register_forward_hook` on the inner backbone to capture
  `last_hidden_state` while still routing through the wrapped top-level
  forward; do not bypass it.
