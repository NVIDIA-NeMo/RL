# DeepSeek-V4-Flash NeMo-RL recipe spec

Written 2026-04-25 from local DSV4 reference inference code (`models/deepseek-ai/DeepSeek-V4-Flash/inference/`) + Automodel PR #2039 source + on-disk `config.json` + `encoding/encoding_dsv4.py`. The tech report (`DeepSeek_V4.pdf`) couldn't be rendered locally (no poppler-utils on cluster host) — items marked **[TR-TODO]** need to be cross-referenced against the report or DSV4-Pro release notes once readable.

This spec covers the eventual real GRPO recipe. The integration smoke recipe (in flight, drafted by sub-agent) is intentionally degenerate — short seq, tiny batch, defaults — and does **not** consume this spec.

## 1. Architecture facts (from local sources, not the tech report)

| Property | Value | Source |
|---|---|---|
| Total params | 284B (after FP4→bf16 dequant) | meta-load smoke |
| Active params/token | 17.7B (8 experts × ~2B + shared expert + attention) | bring-up doc |
| Hidden layers | 43 | `config.json:num_hidden_layers` |
| MTP layers | 1 | `config.json:num_nextn_predict_layers` (drop for RL) |
| Hidden size | 4096 | `config.json:hidden_size` |
| Routed experts | 256 | `config.json:n_routed_experts` |
| Shared experts | 1 | `config.json:n_shared_experts` |
| Activated/token | 6 routed + 1 shared | `config.json:num_experts_per_tok` |
| Routing | sqrtsoftplus, no-aux-loss, biased top-k | `config.json:scoring_func`, `topk_method` |
| Hash-routed layers | 0..2 (deterministic via `tid2eid` lookup) | `inference/model.py:Gate.__init__` |
| Attention | MLA (q_lora=o_lora=1024, head_dim=512, 64 heads, 8 o-groups) | `config.json` |
| Per-layer compress | 0,0,4,128,4,128,...,4,0 (43 entries + 1 MTP) | `config.json:compress_ratios` |
| Sliding window | 128 tokens (always-on local attention) | `config.json:sliding_window` |
| Indexer | only on compress=4 layers, top-512 sparse positions | `inference/model.py:Indexer` |
| Hyper-connections | `hc_mult=4`, Sinkhorn-normalized mixing | `config.json:hc_mult`, `inference/model.py:Block.hc_pre/hc_post` |
| Position encoding | YaRN, factor=16, original=65536, max=1048576 | `config.json:rope_scaling` |
| Special compress RoPE | `compress_rope_theta=160000` (separate from main `rope_theta=10000`) | `config.json` |

### Layer-pattern intuition (from `compress_ratios`)

- Layers 0-1: full attention, no compression (warmup).
- Layers 2-41: alternate compress=4 (overlapping window with Indexer-selected sparse top-512) and compress=128 (large stride pooled summary, no Indexer).
- Layer 42: full attention again (final layer).
- Layer 43 (MTP head): full attention, hash-routed gate, runs over already-encoded hidden states + next-token embed.

This means per-layer KV memory is **non-uniform** — full-attention layers (0,1,42) cost the most; compress=128 layers cost ~1/128 the KV. Total per-rank KV ≈ `window×3 + Σ(seq/ratio for each compress)` per attention call.

## 2. Storage / runtime precision (verified, from `inference/model.py`)

| Tensor | On disk | Reference inference | Automodel (after dequantize) | Why for our recipe |
|---|---|---|---|---|
| Routed-expert weights (256×{w1,w2,w3}) | FP4 e2m1 packed int8, e8m0fnu scales | FP4 (default) or FP8 | bf16 | We dequantize, so we lose the ~0.5 byte/param savings |
| Shared-expert weights | FP8 e4m3fn | FP8 | bf16 | Same |
| Attention `wq_b`, `wkv` | FP8 | FP8 | bf16 | Same |
| Attention `wq_a` | FP8 | FP8 | bf16 | Same |
| Attention `wo_a` | FP8 | **forced bf16** (line 462 comment: "could do FP8 einsum here for better perf, but using BF16 for simplicity") | bf16 | Already matched between reference and us |
| Attention `wo_b` | FP8 | FP8 | bf16 | Same |
| RMSNorm weights | bf16 | **fp32** | fp32 (recipe sets `rms_norm: torch_fp32`) | Already matched |
| `Compressor.wkv`, `wgate`, `ape` | bf16 | **forced fp32** (line 297 comment) | bf16 (Automodel keeps as default model dtype) | **Mismatch!** Reference promotes these to fp32, Automodel doesn't. May need to override or accept slight drift |
| Hyper-connection params (`hc_*`) | bf16 | fp32 (set_dtype context) | bf16 by default? **need to verify** | **[CHECK]** Verify Automodel's DSV4 layers.py keeps hc_* in fp32 |
| Gate weight | bf16 | fp32 in scoring (line 565 `linear(x.float(), self.weight.float())`) | bf16 | Routing is in bf16 with us, fp32 in reference. Likely fine but might affect noaux_tc bias dynamics |
| `Indexer.weights_proj` | bf16 | bf16 | bf16 | Match |
| Latent projections (q-lora, kv_norm, q_norm) | bf16 | bf16 | bf16 | Match |
| Embeddings, lm_head | bf16 | bf16 (lm_head fp32 in reference for stability of softmax) | bf16 | Reference does `F.linear(x[:, -1].float(), self.weight)` for logits — we should ensure cross-entropy loss casts to fp32 (NeMo-RL default) |

### Activation-quant (QAT) — **lost in our path**

Reference inference does **on-the-fly activation quant** to match how the model was *trained*:
- Attention KV non-rope dims: `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)` — FP8 activations.
- Indexer Q and K: `fp4_act_quant(q, fp4_block_size, True)` — FP4 simulation.
- Compressor KV: `act_quant(kv[..., :-rd], 64, ...)` — FP8 activations.
- Comment in `Indexer.forward` (line 419): `"We performed QAT here, kv could also use fp8 format, though current implementation uses bf16"`.

**Implication for RL:** When we dequantize all weights to bf16 and run forward in bf16, we lose the QAT-style activation quant. Generated tokens during rollout via vLLM (which *does* use FP8 attention/FP4 experts in its production path) will not match the policy's bf16 forward exactly. Expect a small but measurable KL between policy logprob (bf16 dense) and generation logprob (FP8/FP4 quantized) — same class of issue we hit on Gemma4 (memory: `project_gemma4_kl_investigation`).

## 3. Reasoning modes & prompt format (from `encoding/encoding_dsv4.py`)

DSV4 has **two thinking modes** (not three as the bring-up doc suggested):
- `thinking_mode="chat"` — direct response, no `<think>...</think>` block. Format: `<|bos|>{system}<|User|>{prompt}<|Assistant|></think>{response}<|eos|>`. Note the dangling `</think>` even in chat mode.
- `thinking_mode="thinking"` — wrapped reasoning. Format: `...<|Assistant|><think>{reasoning}</think>{summary}<|eos|>`.
  - `reasoning_effort="max"` injects `REASONING_EFFORT_MAX` prefix at message 0 (verbose deliberation directive).
  - `reasoning_effort="high"` or `None` — default thinking.

**Tool calls** use a custom DSML format: `<｜DSML｜tool_calls>...<｜DSML｜invoke name="...">...</invoke>...</｜DSML｜tool_calls>`.

**Special tokens** (`tokenizer_config.json`):
- bos: `<｜begin▁of▁sentence｜>` (id 0)
- eos: `<｜end▁of▁sentence｜>` (id 1)
- pad: same as eos
- think: `<think>`, `</think>`
- DSML: `｜DSML｜`

### Recommended for RL recipe
- **GRPO on math/code (smoke + small-scale)**: `thinking_mode="thinking"`, `reasoning_effort=None`, drop_thinking=True (don't include earlier-turn reasoning in context).
- **GRPO on chat alignment**: `thinking_mode="chat"`.
- **For NeMo-RL integration**: NeMo-RL chat-template hook needs to call `encoding_dsv4.encode_messages(messages, thinking_mode=...)` rather than a raw HF tokenizer apply_chat_template (which doesn't know about `｜DSML｜` or DSV4 reasoning conventions). **Action**: write a `nemo_rl/data/templates/deepseek_v4.py` wrapper.

## 4. Parallelism layout for NeMo-RL on 16 nodes (128 H200)

### Constraint summary
- NeMo-RL `DTensorPolicyWorkerV2` does NOT support Automodel PP (project memory `feedback_nemorl_no_automodel_pp`). PP=1 is mandatory.
- TP within Automodel is supported, but DSV4 reference inference uses TP for `wq_b`, `wo_a`, `wo_b`, `experts` already.
- EP for routed experts is the natural sharding axis for DSV4 (256 experts → distribute).
- FSDP2 covers the rest (non-expert params, optimizer state).

### Recommended for the **smoke** (already in flight)
- PP=1, EP=64, TP=1, CP=1, FSDP_dim=2 (within each EP group)
- max_seq_len=4096, max_batch_size very small
- Co-located vLLM gen worker on same nodes (default NeMo-RL pattern)

### Recommended for the **real GRPO run** (post Base-unblock)
- PP=1, EP=128 (one expert pair per rank, 2 of 256 routed experts; aligned with DeepSeek's reference TP=128 for inference), TP=1, CP=1, FSDP_dim=1
- This minimizes inter-rank MoE all-to-all volume and matches how DeepSeek likely *served* it.
- max_seq_len: start at 8192, raise to 16384 if memory permits with rollout group size 8-16.
- vLLM gen layout: TP=8, EP=16 across the 8-GPU per-node units (vLLM PR #40760 default for Flash) [TR-TODO: confirm against tech report's serving config].

### Memory math at PP=1, EP=128, FSDP=1
Per-rank: 250B expert / 128 = 1.95B (3.9 GB bf16) + 34B non-expert / 128 (FSDP shards across all ranks) = 0.27B (0.54 GB bf16) → ~4.4 GB params. Grad shard same size. AdamW state 12 b/p × shards = 26 GB. Activations + dequant scratch + vLLM gen state + NCCL = the rest. Should fit in <90 GiB/GPU on H200 (141 GiB).

## 5. Training precision decisions

Three options. **Recommendation: option A for the integration smoke and the first real GRPO run; revisit B/C for cost-down if/when convergence is established.**

### Option A — All-bf16 (current Automodel SFT path; what job 11332097 used)
- Pros: simple, validated, matches what `dequantize_base_checkpoint=true` gives us automatically.
- Cons: 3.4 TB total fp32 optimizer state (across 128 ranks ≈ 26 GB/rank); large memory footprint.
- This is what the upstream PR's smoke proved works for SFT. Default for first RL pass.

### Option B — Freeze routed experts as FP8/FP4 + train attention/router/shared-expert in bf16
- Pros: ~6× cheaper optimizer state (only ~17B trainable params from attention + router + shared_expert + embeds).
- Cons: changes RL semantics — the policy's expert mixture is fixed during RL. For DSV4 the routing IS the policy's main capacity for behavior change, so freezing experts fights the RL objective.
- **Verdict**: do not use unless we're willing to interpret it as "freeze most of the model, fine-tune the orchestration."

### Option C — QAT (FP8 weight + FP8/FP4 activation, with fake-quant + STE gradient)
- Pros: matches how the model was trained; smallest train↔gen drift.
- Cons: requires scaffolding Automodel PR #2039 doesn't ship; significant new code.
- **Verdict**: park as "v2 recipe" once we know A works.

## 6. vLLM generation config

Validated working in our sqsh (job-11323713 sanity test):
- `tensor_parallel_size: 8`
- `enable_expert_parallel: True` (auto-derives EP within TP group)
- `max_model_len: 16384` (override from 1048576 to fit memory; YaRN already extends this)
- `gpu_memory_utilization: 0.85`
- `dtype: bfloat16` (model weights already FP8/FP4, dtype here means activation/non-quantized layer dtype)
- `kv_cache_dtype: auto` — **[TR-TODO]** confirm if FP8 KV cache is supported & beneficial for DSV4 specifically. vLLM's general FP8 KV path should work.

For NeMo-RL co-located rollout: same config, vLLM workers spawned by the actor pool.

## 7. GRPO hyperparameters

[TR-TODO: cross-reference DSV4 tech report's post-training section. The bring-up doc cited "two-stage domain-expert SFT+GRPO then on-policy distillation" — recover the actual HP table from the report.]

### Defaults to start (drawn from MiniMax M2 GRPO recipe + general practice, NOT from DSV4 paper)
- `rollout_group_size: 8` — DSV4 was likely trained with larger groups, but 8 is a sane starting point.
- `kl_coef: 0.001` — small; DSV4's paper formulation may use a different KL regularization (or none if pure GRPO).
- `lr: 1e-6` (AdamW), `lr_warmup: 50 steps`.
- `clip_ratio: 0.2` (PPO-style clip).
- `advantage_normalization: per-group` (GRPO's z-score over the group).
- `optimizer: AdamW(betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)` — matches upstream Automodel SFT recipe.

### What to lock down before launching real GRPO
1. KL formulation: hard-clip vs soft-penalty vs schedule.
2. Reasoning-mode for rollouts: chat vs thinking. For math RL, thinking-on with `reasoning_effort="high"` is the natural choice.
3. Reference policy: do we keep the SFT'd Base as ref, or use vanilla released Flash as ref? Probably Base-after-SFT.
4. Reward model: math-correctness checker for math; need a separate decision for general-purpose alignment runs.

## 8. MTP head handling for RL

DSV4-Flash ships with 1 MTP layer (`num_nextn_predict_layers: 1`).

Upstream Automodel SFT recipe sets `num_nextn_predict_layers: 0` — drops the MTP head entirely.

**Recommendation for our RL recipes**: drop MTP (`num_nextn_predict_layers: 0`).
- RL operates on a single next-token policy; MTP's multi-token prediction has no place in the standard GRPO formulation.
- Keeping it adds parameters that have no gradient signal during single-token GRPO, wasting memory.
- If we ever do speculative-decoding-style RL (rare), revisit.

## 9. Open questions (tech report needed)

[TR-TODO list]
1. DSV4-Pro post-training pipeline: precise SFT and GRPO HPs, dataset mix, KL formulation.
2. Two reasoning modes vs three: confirm the bring-up doc's "non-think / think-high / think-max" against the report. The encoding code only has chat/thinking; the paper may describe `think-max` as just "reasoning_effort=max" inside thinking mode (which is what the encoding suggests).
3. On-policy distillation: how much of the post-training was distillation vs GRPO; do we need to plan for that as a downstream stage?
4. Token mask / loss mask: are reasoning tokens included in loss or masked? (The encoding has `wo_eos` and per-message mask field — implies fine-grained masking is available.)
5. Native FP8 KV cache support specifically for MLA + Indexer paths.

## 10. Action items derived from this spec

Once Base unblocks upstream:

1. Write `nemo_rl/data/templates/deepseek_v4.py` that wraps `encoding_dsv4.encode_messages` for NeMo-RL's chat-template interface.
2. Draft `examples/configs/grpo_deepseek_v4_base.yaml` (real recipe, not smoke):
   - Inherits from the smoke recipe layout but with PP=1 / EP=128 / FSDP=1, max_seq_len=8192, max_steps=200+, real LR/KL/etc.
   - Sets `model.config.num_nextn_predict_layers=0` (drop MTP).
   - Points dataset and reward at a math GRPO benchmark (DAPO-math-17k cached locally).
3. Verify Compressor.wkv / wgate dtype — if Automodel keeps them bf16 instead of fp32 (reference behavior), decide whether to override; this is a small numerical issue that likely doesn't matter for RL, but worth noting.
4. Decide on KL formulation and reference policy (open question).
5. Implement reward function for math correctness if not already in repo.

## Source map

- `models/deepseek-ai/DeepSeek-V4-Flash/inference/model.py` — full reference implementation (Transformer, Block, Attention, MoE, Compressor, Indexer, MTPBlock, hyper-connections)
- `models/deepseek-ai/DeepSeek-V4-Flash/inference/generate.py` — sampling + generation loop
- `models/deepseek-ai/DeepSeek-V4-Flash/inference/kernel.py` — sparse_attn, fp4/fp8 GEMM, hc_split_sinkhorn, act_quant kernels (not read in detail; recipe-irrelevant)
- `models/deepseek-ai/DeepSeek-V4-Flash/encoding/encoding_dsv4.py` — chat encoding, reasoning modes, tool-call DSML
- `models/deepseek-ai/DeepSeek-V4-Flash/config.json` — official config
- `3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/models/deepseek_v4/{model,layers,state_dict_adapter}.py` — Automodel implementation we use
- `docs/model-readiness/deepseek-v4/automodel-precision-flow.md` — bf16-promotion flow we already documented
- `docs/model-readiness/deepseek-v4/bring-up-status.md` — upstream-blocker context
- `models/deepseek-ai/DeepSeek-V4-Flash/DeepSeek_V4.pdf` — **NOT YET READ** (no poppler on cluster); needs separate workflow
