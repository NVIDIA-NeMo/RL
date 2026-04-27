# Model Compatibility Report: deepseek-ai/DeepSeek-V4-Flash-Base

Scope: transformers + vLLM readiness + code-level requirements only.
Automodel backend is known not-ready and is intentionally not evaluated here.

## Model Info

- **Type**: LLM (MoE)
- **Architecture**: `DeepseekV4ForCausalLM`
- **Model type**: `deepseek_v4`
- **Config saved with**: `transformers_version: 4.57.1`
- **Torch dtype (config)**: `bfloat16`; **on-disk weights**: FP8 e4m3 block-quantized (128x128, ue8m0 scales, dynamic activation scheme)
- **Params (estimated)**: ~288B total / ~17.7B active (6 routed experts + 1 shared per token)
- **Architecture headline**:
  - 43 decoder layers, hidden=4096, head_dim=512, 64 attention heads, 1 KV head (MLA-style: q_lora_rank=1024, o_lora_rank=1024, qk_rope_head_dim=64)
  - MoE: 256 routed + 1 shared expert, `num_experts_per_tok=6`, `moe_intermediate_size=2048`, `topk_method=noaux_tc`, `scoring_func=sqrtsoftplus`, `routed_scaling_factor=1.5`
  - Per-layer `compress_ratios` mixing {0, 4, 128} over 43 layers → hybrid KV-compression; dual RoPE bases (`rope_theta=10000`, `compress_rope_theta=160000`)
  - YaRN scaling factor 16 from `original_max_position_embeddings=65536` → `max_position_embeddings=1,048,576`
  - `sliding_window=128`; `swiglu_limit=10.0`; `tie_word_embeddings=false`
  - MTP: `num_nextn_predict_layers=1`
  - Novel fields not in V3/V3.2 vocabulary: `num_hash_layers=3`, `hc_mult=4`, `hc_sinkhorn_iters=20`, `hc_eps=1e-6`, `index_head_dim=128`, `index_n_heads=64`, `index_topk=512`, `o_groups=8`, `o_lora_rank`
- **Vocabulary**: 129,280
- **HF repo contents**: `config.json`, `LICENSE`, `tokenizer_config.json`, `tokenizer.json`, 46 safetensor shards + index. **No `modeling_*.py`, no `configuration_*.py`, no `auto_map`** — the repo does NOT ship remote code, so native `transformers` support is required.

## transformers Readiness — BLOCKER

Current branch pins `transformers==5.3.0` (pyproject.toml + uv.lock).

| Source | `deepseek_v4` support? |
|---|---|
| transformers 5.3.0 (our pin) | No |
| transformers main (HEAD) | No |
| Any open PR (searched `deepseek_v4`, `DeepseekV4`) | None |
| HF model repo remote code (`auto_map` / `modeling_*.py`) | None |
| deepseek-ai GitHub org (`DeepSeek-V4` repo) | 404 |

**Verdict**: there is **no transformers implementation available anywhere** for this architecture. Any loader invoking `AutoConfig.from_pretrained` on `model_type=deepseek_v4` will raise `ValueError: The checkpoint you are trying to load has model type "deepseek_v4" but Transformers does not recognize this architecture`.

**Code-level requirement to unblock training via HF path**: write (or obtain from a DeepSeek-internal drop) the following files and register them in `transformers.models.auto`:

- `src/transformers/models/deepseek_v4/configuration_deepseek_v4.py` — extends `PretrainedConfig` with all DeepSeek-V3 fields plus V4 additions: `compress_ratios` (per-layer list), `compress_rope_theta`, `num_hash_layers`, `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`, `index_head_dim`, `index_n_heads`, `index_topk`, `o_lora_rank`, `o_groups`, `sliding_window`, `swiglu_limit`, `scoring_func=sqrtsoftplus`, `topk_method=noaux_tc`.
- `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py` — `DeepseekV4PreTrainedModel`, `DeepseekV4Model`, `DeepseekV4ForCausalLM`, plus:
  - `DeepseekV4Attention` with MLA (q_lora, kv_lora, o_lora, qk_rope_head_dim=64, head_dim=512) and per-layer `compress_ratios` branching between full / 4x / 128x paths
  - `DeepseekV4SparseMoeBlock` with 256 routed + 1 shared expert, `noaux_tc` routing, `sqrtsoftplus` scoring, Sinkhorn post-processing (`hc_*` fields, `num_hash_layers`)
  - Indexer head(s) (`index_head_dim`, `index_n_heads`, `index_topk`) — purpose to be confirmed from the weight layout
  - Dual-theta RoPE (base vs compressed)
  - Optional MTP head for `num_nextn_predict_layers=1`
  - FP8 block-quantized weight loader (either native FP8 kernels or dequantize-on-load to bf16; `compressed_tensors`/`fbgemm` paths exist in recent transformers for similar quant configs — needs verification)
- Entries in `src/transformers/models/auto/{configuration_auto.py, modeling_auto.py, tokenization_auto.py}` mapping `"deepseek_v4"` → config + `DeepseekV4ForCausalLM`.

**Upstream watch**: DeepSeek historically upstreamed V3 ~3 months after release. If no internal drop is available, expect transformers PR landing over the next weeks. Track by re-running `WebFetch` on `huggingface/transformers/tree/main/src/transformers/models` and on open PRs matching `deepseek_v4`.

## vLLM Readiness — BLOCKED-BUT-PATH-EXISTS

Current branch pins `vllm==0.17.1` (pyproject.toml + uv.lock). The vllm **submodule is NOT checked out** on `deepseek-v4-support` (no `3rdparty/vllm` in `.gitmodules`) — the pinned pypi wheel is what runs.

| Source | `DeepseekV4ForCausalLM` in registry? |
|---|---|
| vllm 0.17.1 (our pin) | No. Registry has `DeepseekForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`; V2/V3/V3.2 all share `deepseek_v2` module. |
| vllm `main` | No. |
| Open PR | **#40760** "[New Model] Support DeepseekV4" by `zyongye` (member) — branch `zyongye:dsv4`, opened 2026-04-24, 8 comments. Has merge conflicts, no approvals, awaits 26+ code-owner reviews. Uses fused kernels; example config notes `enforce_eager=True`. |

**Verdict**: vLLM generation will fail to initialize with `ValueError: Model architectures ['DeepseekV4ForCausalLM'] are not supported for now`.

**Code-level requirement to unblock vLLM generation**: cherry-pick / build vLLM from PR #40760. Concretely:

1. Replace the `vllm` pypi dependency with a submodule or a built wheel from `github.com/zyongye/vllm@dsv4`.
2. Verify PR #40760 transformers-version expectations — the PR summary does not spell them out, and modern vLLM requires transformers≥5.0; our 5.3.0 pin satisfies that but the V4 model code in the PR may import from a transformers module that does not exist in 5.3.0 (likely — see transformers BLOCKER above), so the PR effectively requires the transformers V4 support landing too.
3. Expect `enforce_eager=True` hint — plumb that into `policy.generation.vllm_cfg.enforce_eager: true` in the recipe.
4. Likely `trust_remote_code=true` is NOT needed (architecture is in vLLM's native registry once the PR lands), but if we build pre-merge from the feature branch, still set `trust_remote_code=true` defensively.
5. Watch for fused-kernel hardware requirements (the PR mentions "highly optimized" components) — the kernels may require specific Hopper/Blackwell capability and may not run on older GPUs.

## Known Quirks & Constraints (from config + ledger arch-check)

From the arch-check INSPECT list (`docs/model-readiness/deepseek-v4/arch-check-report.json`):

- **MoE gate double-norm risk** (ledger `gemma4-moe-double-norm-gate`): custom `sqrtsoftplus` + `noaux_tc` + Sinkhorn router is non-standard. Once the modeling file exists, trace `decoder → moe → gate` and verify the gate input is not re-normed. Symptom if broken: `gen_kl_error ~0.1`, elevated but not hard-failing.
- **Expert dtype under FSDP2 + ModuleDict**: with 256 experts on an EP mesh, the Gemma4 trap can recur if the modeling code wraps experts in `nn.ModuleDict` and automodel's `fully_shard_by_dtype` isn't registered. Not relevant for this pass (automodel is out of scope), but worth flagging for the env-build step.
- **MTP head (`num_nextn_predict_layers=1`)**: classic V3 trap. The NeMo-RL loader must drop or freeze the MTP head, or the optimizer state on the unused params will break ckpt resume. Whoever writes the HF modeling file must expose a clean knob for this (V3 handled it via `output_router_logits` / optional module).
- **Dual-theta RoPE + per-layer compress_ratios**: if the HF implementation wires only one `rope_theta`, compressed-attention layers will silently use the wrong base. Verify during implementation.
- **FP8 e4m3 / ue8m0 / block-128x128 on disk**: the HF loader must either (a) call `compressed_tensors` / native FP8 kernels, or (b) dequantize to bf16 on load. Training in FP8 is not what NeMo-RL does today; for RL we almost certainly want dequant-on-load to bf16 → then FSDP2-shard bf16. Disk footprint (275G → ~550G bf16) must fit the training sharded memory budget.
- **Sliding window 128 + YaRN factor 16 + max_pos 1M**: recipe seq length can NOT exceed the YaRN original window (65K) unless we intentionally want YaRN-extended context; most RL recipes should clamp `max_total_sequence_length` at 16K–32K.

## General NeMo-RL constraints (sanity-check, not model-specific)

- Automodel training: FSDP2 only (no TP/PP), EP for MoE. **N/A for this pass** per user note.
- vLLM generation TP must divide `num_key_value_heads`. This model has `num_key_value_heads=1` → vLLM TP is effectively **1**. (Expert parallel + DP inside vLLM is a separate dimension.) This is a real constraint that will bite in Phase 2 recipe sizing.
- CP + sequence_packing incompatible; CP + TP+SP incompatible; VLM/Gemma3 disallow CP. DeepSeek V4 is not VLM; CP itself should be fine once modeling lands.

## Recommended Config Direction (deferred, for Phase 2 reference)

Active params ~17.7B but total ~288B → sharded-state memory dominates. First-pass sizing once upstream support lands:

| Field | Suggested start | Rationale |
|---|---|---|
| Training parallelism | FSDP2 + EP=32 or EP=64 | 256 routed experts; EP divides cleanly |
| Activation checkpointing | **Yes** | ~288B total, MLA + sliding-window |
| Nodes | 16–32 (128–256 H100s) | similar to DeepSeek V3 671B scaled by active-param ratio |
| Micro batch size | 1 | start small; raise only after memory proof |
| vLLM TP | **1** (forced by `num_key_value_heads=1`) | MLA with single KV head |
| vLLM EP / DP-inside | raise to cover generation memory | relies on PR #40760's MoE sharding |
| Enforce eager (vLLM) | `true` | per PR #40760 hint |
| GPU memory utilization | 0.5 | MoE + long context |
| Learning rate | 3e-7 | large MoE, RL fine-tune |
| Max total seq length | 16384 | inside YaRN original window |
| `offload_optimizer_for_logprob` | `true` | standard for ≥10B |
| `logprob_chunk_size` | 4096 | standard |

All numbers are placeholders pending upstream support + a memory-proof smoke run.

## Blockers

1. **BLOCKER — transformers lacks `deepseek_v4`**. No release, no main, no PR, no remote code in the HF repo. Must write the model code (config + modeling + auto registration) before any bring-up is possible. Estimated effort: medium-to-large (MLA with per-layer compress_ratios + noaux_tc/sqrtsoftplus router + Sinkhorn + indexer + MTP + FP8 loader).
2. **BLOCKER — vLLM lacks `DeepseekV4ForCausalLM`**. Open PR #40760 exists but is not merged and has conflicts. Unblock by building vLLM from `zyongye:dsv4` and replacing the pinned pypi wheel. This PR also implicitly depends on the transformers implementation existing.
3. **CO-DEPENDENCY**: the vLLM PR cannot run end-to-end without the transformers support — the transformers blocker is the critical path.
4. **FP8 load path**: unverified, investigate during modeling code bring-up.

## Recommended next steps

1. **Pause the `/new-model` pipeline at Phase 1.5** — do NOT proceed to `/env-profile` yet.
2. **Investigate (parallel tracks)**:
   - Is there an internal DeepSeek drop of the transformers modeling code (email/Slack DeepSeek partners)? If yes, import it as a patch against transformers 5.3.0.
   - Otherwise, ask whether writing the transformers port is in scope for this team, or if we wait N weeks for an upstream PR.
   - Build a local vLLM wheel from PR #40760 in a scratch container; sanity-check it imports + constructs a DeepseekV4ForCausalLM stub (with a transformers modeling stub).
3. **If we proceed**: reshape `/env-profile` around this: transformers needs local source-install (editable), vllm needs submodule from a fork, both need matching compat. This is a much heavier env-build than a normal new-model.

## Sources

- transformers v5.3.0 models dir: https://github.com/huggingface/transformers/tree/v5.3.0/src/transformers/models
- transformers main models dir: https://github.com/huggingface/transformers/tree/main/src/transformers/models
- vllm v0.17.1 registry: https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/model_executor/models/registry.py
- vllm main registry: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py
- vLLM PR #40760 "Support DeepseekV4": https://github.com/vllm-project/vllm/pull/40760
- HF model repo: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/tree/main
- Arch-check JSON: docs/model-readiness/deepseek-v4/arch-check-report.json
