# YaRN Long-Context Training

[**YaRN** (Yet another RoPE extensioN)](https://arxiv.org/abs/2309.00071) extends a model's usable context window beyond the length it was pretrained on by rescaling RoPE frequencies. NeMo RL supports YaRN RoPE scaling for SFT, GRPO, DPO, RM, and distillation workflows, letting you fine-tune or RL-train models at sequence lengths much larger than their original pretraining context.

## Requirements

YaRN is only supported with the **Megatron backend**. The DTensor (Automodel) backend will raise an assertion error if `rope_scaling.rope_type=yarn` is set. Make sure:

1. Megatron submodules are initialized: `git submodule update --init --recursive`
2. Megatron backend is enabled: `policy.megatron_cfg.enabled=True` and `policy.dtensor_cfg.enabled=False`

## Enablement

YaRN is configured through `policy.hf_config_overrides.rope_scaling`. All YaRN fields are required — NeMo RL validates that none are missing before training starts.

```yaml
policy:
  max_total_sequence_length: 65536
  megatron_cfg:
    enabled: true
  dtensor_cfg:
    enabled: false
  hf_config_overrides:
    rope_scaling:
      rope_type: yarn
      rope_theta: 1000000
      factor: 1.6
      original_max_position_embeddings: 40960
      truncate: true
      beta_fast: 32
      beta_slow: 1
      mscale: 1
      mscale_all_dim: 0
```

### Required Fields

| Field | Description |
| --- | --- |
| `rope_type` | Must be `yarn` to enable YaRN. |
| `rope_theta` | RoPE base frequency used when recomputing scaled frequencies. |
| `factor` | Scaling factor — typically `max_total_sequence_length / original_max_position_embeddings`. |
| `original_max_position_embeddings` | The model's original pretraining context length. |
| `truncate` | Whether to truncate out-of-range positions. |
| `beta_fast` / `beta_slow` | YaRN interpolation thresholds on the fast and slow RoPE dimensions. |
| `mscale` / `mscale_all_dim` | Attention temperature scaling terms used by YaRN. |

You can also compute `factor` directly from other config values using the `div:` interpolation helper:

```yaml
factor: ${div:${policy.max_total_sequence_length},${policy.hf_config_overrides.rope_scaling.original_max_position_embeddings}}
```

> [!NOTE]
> YaRN only takes effect when `max_total_sequence_length` exceeds `original_max_position_embeddings`. If the two are equal, YaRN is a no-op.

## How Conversion Works

When `hf_config_overrides` are present, NeMo RL's Megatron setup:

1. Validates that every required YaRN field is specified.
2. Computes a stable hash over `hf_config_overrides` and appends it to the converted-checkpoint directory name (`<model>__hfovr_<hash>`). Different override sets therefore produce separate cached checkpoints and will not collide.
3. Re-imports the HF model into Megatron format if no cached checkpoint at that path exists.

The same `hf_config_overrides` are also propagated to vLLM during generation, so the rollout engine applies the identical YaRN settings as the trainer.

## Forcing a Fresh Conversion

If you change yarn parameters in `hf_config_overrides` after a conversion has been cached (for example, adjusting the YaRN `factor`), set:

```yaml
policy:
  megatron_cfg:
    force_reconvert_from_hf: true  # Default: false
```

This re-runs the HF → Megatron conversion and overwrites the cached checkpoint. It is equivalent to deleting the cached directory and rerunning. See also the [Training Backends design doc](../design-docs/training-backends.md#force-reconvert) for background on the checkpoint cache.

## Example Recipes

Two end-to-end YaRN recipes ship with NeMo RL:

- **SFT, 64K context**: [`examples/configs/recipes/llm/h100/sft-qwen3-0.6B-1n8g-megatron-yarn-64k.yaml`](../../examples/configs/recipes/llm/h100/sft-qwen3-0.6B-1n8g-megatron-yarn-64k.yaml) — Qwen3-0.6B fine-tuned to a 64K sequence length on Nemotron-Cascade-2-SFT-Math using `factor: 1.6` (64K / 40960).
- **GRPO, 256K context**: [`examples/configs/recipes/llm/h100/grpo-qwen2.5-1.5B-4n8g-megatron-yarn-256k.yaml`](../../examples/configs/recipes/llm/h100/grpo-qwen2.5-1.5B-4n8g-megatron-yarn-256k.yaml) — Qwen2.5-1.5B trained at 256K sequence length with `factor` derived from `max_total_sequence_length / original_max_position_embeddings`.

Launch them the same way as any other recipe:

```bash
uv run examples/run_sft.py \
    --config examples/configs/recipes/llm/h100/sft-qwen3-0.6B-1n8g-megatron-yarn-64k.yaml

uv run examples/run_grpo_math.py \
    --config examples/configs/recipes/llm/h100/grpo-qwen2.5-1.5B-4n8g-megatron-yarn-256k.yaml
```

## Practical Tips

- **Set context parallelism appropriately.** Long sequences typically require `policy.megatron_cfg.context_parallel_size > 1`. The 256K recipe uses `context_parallel_size: 32`; the 64K recipe uses `8`.
- **Check `make_sequence_length_divisible_by`.** Long-context recipes usually need a larger divisor (e.g. `64` at 256K) so sequences align with CP/TP shapes.
- **Keep override configs identical across trainer and generator.** Because `hf_config_overrides` flows into both Megatron and vLLM, editing only one side will cause mismatched RoPE behavior.
- **Reconvert after changing overrides.** Cached checkpoints are keyed by override hash, so a new hash produces a new cache entry — but if you deliberately mutate an existing cache path, use `force_reconvert_from_hf: true`.
