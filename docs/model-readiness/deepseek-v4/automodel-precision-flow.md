# Automodel SFT precision flow on DeepSeek-V4-Flash

How the upstream Automodel PR #2039 (`khazic/Automodel_lao @ feat/deepseek-v4-flash`,
commit `ab2d7a08`) actually trains DSV4-Flash. Captured from a successful 5-step
SFT smoke (job `11332097`, 2026-04-25) using
`examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml`.

## TL;DR

Automodel SFT runs in **bf16** end-to-end on DSV4-Flash.

The released checkpoint is *not* bf16 on disk — it's a mix of FP4-packed expert
weights, FP8-block-quantized attention, and a small bf16 tail. The
`dequantize_base_checkpoint=true` flag promotes everything to bf16 at load, and
FSDP2 keeps params bf16 throughout training.

## Flow chart

```
┌──────────────────────────── ON DISK ─────────────────────────────────┐
│ /lustre/.../models/deepseek-ai/DeepSeek-V4-Flash/                    │
│                                                                      │
│ config.json:                                                         │
│   torch_dtype: bfloat16                                              │
│   quantization_config:                                               │
│     quant_method: fp8, fmt: e4m3, weight_block_size: [128,128]       │
│                                                                      │
│ *.safetensors:                                                       │
│   ┌─────────────────────────────────┬──────────────────────────────┐ │
│   │ Routed experts (256 × 43 lyrs)  │ FP4 e2m1 PACKED (2-per-byte) │ │
│   │   .weight   int8     [out,in/2] │   + e8m0fnu scale [out,in/32]│ │
│   ├─────────────────────────────────┼──────────────────────────────┤ │
│   │ Attention / MLP / shared expert │ FP8 e4m3fn 128×128 block     │ │
│   │   .weight   float8_e4m3fn       │   + .scale  float32 block    │ │
│   ├─────────────────────────────────┼──────────────────────────────┤ │
│   │ Latent projections, embeds, LN  │ BF16 (no quantization)       │ │
│   └─────────────────────────────────┴──────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                            │
                            │  Checkpointer.load() with quantization=True
                            │  (auto-enabled via infrastructure.py:423-424
                            │   when config.quantization_config exists)
                            ▼
┌─────────────── DSV4StateDictAdapter._dequantize_state_dict ──────────┐
│   (3rdparty/.../models/deepseek_v4/state_dict_adapter.py)            │
│                                                                      │
│   for each tensor:                                                   │
│     if FP4-expert       →  _dequantize_expert_fp4 →  bf16 dense      │
│        (line 259)          unpack int8 nibbles, scale by e8m0,       │
│                            cast to bf16                              │
│                                                                      │
│     if FP8-block        →  _dequantize_fp8        →  bf16 dense      │
│        (line 240)          fp8_e4m3 × per-block fp32 scale, cast bf16│
│                                                                      │
│     if BF16             →  pass-through (line 632 comment)           │
└──────────────────────────────────────────────────────────────────────┘
                            │
                            │  state_dict loaded into model graph
                            ▼
┌─────────────── DeepseekV4ForCausalLM (model.py) ─────────────────────┐
│   model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)        │
│                       ↑                                              │
│                  reads "bfloat16" from config.json                   │
│                                                                      │
│   • Linear / embedding / lm_head:  dtype=bfloat16   (lines 302/312/  │
│                                                      334/527/534)    │
│   • RMSNorm:  backend.rms_norm = "torch_fp32"  → norm runs in fp32   │
│               (recipe yaml; the `_fp32` suffix forces upcast inside) │
│   • cast_model_to_dtype(self, bfloat16)  (line 596) — final forced   │
│     cast of every remaining param to bf16                            │
└──────────────────────────────────────────────────────────────────────┘
                            │
                            │  parallelize_module() wraps model
                            ▼
┌─────────────── FSDP2 MixedPrecisionPolicy ───────────────────────────┐
│   (parallelizer.py:286-287, 553-554, 592-593)                        │
│                                                                      │
│   MixedPrecisionPolicy(                                              │
│       param_dtype=torch.bfloat16,    ← all-gathered weights are bf16 │
│       reduce_dtype=torch.float32,    ← grad reductions in fp32       │
│   )                                                                  │
│                                                                      │
│   Sharding:  PP=4  →  EP=32 (experts)  →  FSDP_dim=remaining         │
│   Per-rank param footprint after sharding = 284B / 128 ≈ 2.2 GB bf16 │
└──────────────────────────────────────────────────────────────────────┘
                            │
                            │  AdamW.step()
                            ▼
┌─────────────── Optimizer state ──────────────────────────────────────┐
│   torch.optim.AdamW (recipe yaml line 119)                           │
│   • Master weights:  fp32 (default)                                  │
│   • exp_avg (m):     fp32                                            │
│   • exp_avg_sq (v):  fp32                                            │
│   ⇒ 12 bytes/param × 284B = 3.4 TB total, sharded across 128 ranks   │
│                                                                      │
│   Forward & backward:                                                │
│     activations bf16,  matmul bf16 in,  accumulate fp32 in tcgen,    │
│     RMSNorm fp32 (because rms_norm=torch_fp32)                       │
└──────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    step 0 .. step 4
                    loss/grad_norm shown
                    in slurm log
```

## What runs in what dtype

| Component | Storage on disk | After load | In forward |
|---|---|---|---|
| Routed-expert weights | FP4 e2m1 packed int8 + e8m0fnu scales | **bf16** | bf16 matmul |
| Attention / shared-expert / MLP weights | FP8 e4m3fn + fp32 128×128 block scales | **bf16** | bf16 matmul |
| Embeddings / lm_head / latent projections | bf16 | bf16 | bf16 |
| RMSNorm weights | bf16 | bf16 | **fp32** (recipe sets `rms_norm: torch_fp32`) |
| FSDP2 all-gather buffer | n/a | n/a | bf16 (`param_dtype`) |
| Gradient reduce-scatter | n/a | n/a | **fp32** (`reduce_dtype`) |
| Optimizer master / m / v | n/a | fp32 | n/a |

## How the auto-enable for dequantize works

`3rdparty/Automodel-workspace/Automodel/nemo_automodel/_transformers/infrastructure.py:423-424`:

```python
if checkpointer.config.dequantize_base_checkpoint is None:
    checkpointer.config.dequantize_base_checkpoint = hasattr(
        getattr(model, "config", None), "quantization_config"
    )
```

Even without `dequantize_base_checkpoint: true` in the recipe, the loader will
turn it on automatically because DSV4-Flash's `config.json` carries a
`quantization_config` block. The recipe sets it explicitly to be safe.

## Code references

- State-dict adapter (FP4 + FP8 dequant entry points):
  `3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/models/deepseek_v4/state_dict_adapter.py`
  - `_dequantize_expert_fp4` (line 259) — unpacks int8 nibbles, applies e8m0 scales, casts to bf16
  - `_dequantize_fp8` (line 240) — applies block scales, casts to bf16
  - Line 632 comment: "Latent projections are stored as BF16 in the V4 checkpoint (not FP8)"
- Auto-enable of dequant:
  `3rdparty/Automodel-workspace/Automodel/nemo_automodel/_transformers/infrastructure.py:423-424`
- DSV4 model dtype creation:
  `3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/models/deepseek_v4/model.py`
  - `model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)` (line 94)
  - `cast_model_to_dtype(self, dtype)` (line 596)
- FSDP2 mixed precision defaults:
  `3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/distributed/parallelizer.py`
  - `MixedPrecisionPolicy(param_dtype=torch.bfloat16, ...)` at lines 286-287, 553-554, 592-593
- Recipe YAML:
  `3rdparty/Automodel-workspace/Automodel/examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml`
  - `model.backend.rms_norm: torch_fp32`
  - `checkpoint.dequantize_base_checkpoint: true`
  - `optimizer._target_: torch.optim.AdamW` (default fp32 state)

## Memory check vs. observed (job 11332097)

Per-rank bf16 footprint after PP=4 + EP=32 + FSDP2:

- Params (bf16, sharded): ~2.2 GB
- Grad shard (bf16): ~2.2 GB
- Sharded fp32 optimizer state (AdamW, m+v+master): ~6.6 GB
- Activations + dequant scratch + NCCL buffers + frags: balance

Steady-state observed = **38.5 GiB/GPU** on H200 (141 GiB) — consistent with
full-bf16 training (no FP8 frozen weights, no QAT).

## Why this matters for the NeMo-RL GRPO recipe we still owe

When we move from upstream Automodel SFT to NeMo-RL GRPO, we have a precision
choice the PR didn't make for us:

1. **All-bf16 (this path).** Simple, matches what the PR validated. Cost: 3.4 TB
   optimizer state across the cluster. Required hardware ~roughly what we used here.
2. **Freeze experts as FP8/FP4, train attention + router only.** Much cheaper
   (most of the param count is in experts), but not what the PR's smoke validates,
   and changes RL semantics — the policy's expert mixture is fixed during training.
3. **QAT on experts** (FP8 expert weights kept, with fake-quant + straight-through
   gradient). Middle ground; needs scaffolding the PR doesn't ship.

That decision is exactly what's blocked on the architecture deep-dive in
`bring-up-status.md`'s "to do" section, and depends on the upstream Base loader
fix (vllm-project/vllm#40760) since RL starts policy from -Base, not -Flash.
