# Silent-Drop Knob Audit (v07 Tier 1/Tier 2 + adjacent perf surfaces)

Question: which `policy.megatron_cfg.*` keys reach Megatron, and which are silently dropped because NeMo-RL has no interface/wiring?

Source-of-truth files reviewed:
- `nemo_rl/models/policy/__init__.py:199` — `MegatronConfig` TypedDict (declared schema)
- `nemo_rl/models/megatron/setup.py:380` — runtime config application (`_apply_*_config` + `_build_comm_overlap_config` + `_create_megatron_config`)
- `nemo_rl/models/megatron/community_import.py:43` — HF-to-Megatron *checkpoint conversion path only* (not runtime)
- `Megatron-Bridge/.../comm_overlap.py:354` — `CommOverlapConfig` field list
- `Megatron-LM/.../model_parallel_config.py:184` — `tp_comm_*` TransformerConfig fields
- `Megatron-Bridge/.../models/gpt_provider.py` — Bridge's `GPTModelProvider` defaults

---

## Part A — Knobs explicitly used in v07 yamls

| Knob (yaml) | Tier 2 var | Wired? | Where | Verdict |
|-------------|-----------|--------|-------|---------|
| `bias_dropout_add_fusion` | v07_01/05 | yes | `setup.py:492` G6 bundle | APPLIED |
| `masked_softmax_fusion` | v07_01/05 | yes | `setup.py:493` G6 bundle | APPLIED |
| `persist_layer_norm` | v07_01/05 | yes | `setup.py:494` G6 bundle | APPLIED |
| `fused_residual_rmsnorm` | v07_11 (Phase A) | yes | `setup.py:495` G6 bundle | APPLIED (added this session) |
| `tp_comm_overlap` | v07_02/05/09 | yes | `setup.py:640` `_build_comm_overlap_config` | APPLIED |
| `delay_wgrad_compute` | v07_03/05 | yes | `setup.py:641` | APPLIED |
| `overlap_moe_expert_parallel_comm` | v07_04/05 | yes | `setup.py:642` | APPLIED |
| `gradient_accumulation_fusion` | v07_06 | partial | `community_import.py:110` only at HF→megatron import time | UNCLEAR — see note |
| `moe_shared_expert_overlap` | v07_07 (30B), v07_10 (235B) | yes | `setup.py:430` | APPLIED |
| `overlap_p2p_comm` | v07_07 (32B / 235B PP=4) | **NO** | not enumerated anywhere | **SILENT DROP** |
| `defer_embedding_wgrad_compute` | v07_08 (32B / 235B) | **NO** | not enumerated anywhere | **SILENT DROP** |
| `tp_comm_atomic_ag` | v07_09 | **NO** | TransformerConfig field, no NeMo-RL passthrough | **SILENT DROP** |
| `tp_comm_atomic_rs` | v07_09 | **NO** | TransformerConfig field, no NeMo-RL passthrough | **SILENT DROP** |

**Note on `gradient_accumulation_fusion`**: only set during the HF→Megatron checkpoint conversion (`community_import.py:110`). At runtime training the model loads from the saved Megatron checkpoint, so the value persists *only if force_reconvert_from_hf=True or the checkpoint hasn't been built yet*. If the cached checkpoint was built with `gradient_accumulation_fusion=False`, flipping the yaml does nothing on subsequent runs. Worth adding a runtime override in `_apply_performance_config`.

**Implication for v07 results**: v07_07/08/09 measurements in matrix.csv reflect the *baseline* config, not the intended override. The +1% / +1.4% deltas are within-batch noise floor (±2%), confirmed by paired baselines. Re-measure after wiring.

---

## Part B — Bridge `CommOverlapConfig` fields *not* exposed by NeMo-RL

`_build_comm_overlap_config` (setup.py:632) only enumerates 3 of 14 fields. Missing:

| Field | Purpose | Used in v07? | Priority |
|-------|---------|--------------|----------|
| `overlap_p2p_comm` | overlap pipeline P2P send/recv with compute (PP>1) | yes (v07_07) | **HIGH** |
| `defer_embedding_wgrad_compute` | defer embedding wgrad to overlap PP bubble | yes (v07_08) | **HIGH** |
| `tp_comm_overlap_cfg` | per-layer UB cfg (qkv/proj/fc1/fc2) tuning | no | medium |
| `tp_comm_bootstrap_backend` | NCCL vs symmem | no | low |
| `batch_p2p_comm` | batch-mode P2P (alternative to overlap_p2p) | no | low |
| `overlap_grad_reduce` | duplicates DDP knob, but accepted here | no | low (already in DDP) |
| `overlap_param_gather` | duplicates DDP knob | no | low (already in DDP) |
| `overlap_param_gather_with_optimizer_step` | overlap fwd-pass param gather with opt step | no | medium |
| `align_param_gather` | align param gather buckets | no | low |
| `bucket_size` | DP comm bucket | no | low |
| `wgrad_deferral_limit` | bound on `defer_embedding_wgrad_compute` | no | medium (pairs w/ defer_embed) |

---

## Part C — `tp_comm_*` TransformerConfig fields *not* exposed by NeMo-RL

These live on `TransformerConfig` (model_parallel_config.py:184), not `CommOverlapConfig`. Reach the model via `model_cfg.<attr>` direct setattr. Currently zero NeMo-RL passthrough.

| Field | Default | Used in v07? | Priority |
|-------|---------|--------------|----------|
| `tp_comm_atomic_ag` | False | yes (v07_09) | **HIGH** |
| `tp_comm_atomic_rs` | False | yes (v07_09) | **HIGH** |
| `tp_comm_split_ag` | True | no | medium (toggle off for atomic) |
| `tp_comm_split_rs` | True | no | medium |
| `tp_comm_bulk_wgrad` | True | no | low (default ok) |
| `tp_comm_bulk_dgrad` | True | no | low |
| `tp_comm_overlap_ag` | True | no | low |
| `tp_comm_overlap_rs` | True | no | low |
| `tp_comm_overlap_disable_qkv` | False | no | low |
| `tp_comm_overlap_disable_fc1` | False | no | low |

---

## Part D — DDP perf knobs *not* exposed

`_create_megatron_config` (setup.py:675) wires only 5 of DDPConfig's perf-relevant fields.

| Field | Default | Priority | Notes |
|-------|---------|----------|-------|
| `align_param_gather` | False | medium | small overlap gain |
| `num_distributed_optimizer_instances` | 1 | medium | NDO multi-instance dist optimizer (>10K GPU scaling) |
| `bucket_size` | None | medium | DP bucket size tune |
| `gradient_reduce_div_fusion` | True | low | default ok |
| `fp8_param_gather` | False | medium-high | mxfp8-only, but big perf win when applicable |
| `nccl_ub` | False | **HIGH** | NCCL UserBuffers, large multi-node gain |
| `use_custom_fsdp` | False | n/a | deprecated |

---

## Part E — TransformerConfig perf knobs *not* exposed

| Field | Bridge `GPTModelProvider` default | Priority | Notes |
|-------|----------------------------------|----------|-------|
| `moe_grouped_gemm` | False | **HIGH** | MoE perf — already user-tested separately |
| `enable_cuda_graph` | False | **HIGH** | CUDA graph capture for fwd |
| `external_cuda_graph` | False | **HIGH** | partial CUDA graphs |
| `cuda_graph_use_single_mempool` | False | medium | mem fragmentation control |
| `cross_entropy_loss_fusion` | True | low | already on by default |
| `cross_entropy_fusion_impl` | "native" | medium | "te" gives speedup on Qwen3 |
| `attention_softmax_in_fp32` | False | low | numerical knob |
| `moe_apply_probs_on_input` | False | medium | MoE accuracy/perf trade-off |
| `moe_router_force_load_balancing` | False | low | training-stability knob |
| `cp_comm_type` | None | medium | "a2a" preferred for CP>1 |
| `hierarchical_context_parallel_sizes` | None | low | only if CP>1 with hierarchy |

Key gap: **no CUDA-graph plumbing in NeMo-RL**. Adding `enable_cuda_graph`/`external_cuda_graph` requires schema + setattr; potentially the largest single perf delta available.

---

## Bottom line: how many measurements in matrix.csv are invalid

Of the 11 v07 Tier 1+2 variants tested:

- **Valid** (knob actually applied): v07_01, v07_02, v07_03, v07_04, v07_05, v07_06 (with caveats), v07_10, v07_11 (Phase A)
- **Invalid** (silent drop, measured baseline): v07_07 (overlap_p2p), v07_08 (defer_embed), v07_09 (tp_comm_atomic) — **all three Tier 2 measurements**

The earlier "all knobs ±2%" finding holds for the 8 valid variants. The 3 invalid Tier 2 measurements need re-running after wiring.

---

## Proposed patches

**P1 — Wire 4 silent-drop knobs already used in v07 yamls (smallest delta to fix existing experiments):**

1. `setup.py:_build_comm_overlap_config` — add `overlap_p2p_comm`, `defer_embedding_wgrad_compute`, `wgrad_deferral_limit`. Add to `MegatronConfig` TypedDict.
2. `setup.py:_apply_performance_config` — add G7-style allowlist for `tp_comm_atomic_ag`, `tp_comm_atomic_rs` (and `tp_comm_split_ag`/`rs` since atomic disables split). Add to `MegatronConfig` TypedDict.
3. `setup.py:_apply_performance_config` — add runtime override for `gradient_accumulation_fusion` to fix the import-time-only wiring.

**P2 — Expose CUDA graphs (largest expected perf delta):**

4. `setup.py:_apply_performance_config` — add `enable_cuda_graph`, `external_cuda_graph`, `cuda_graph_use_single_mempool` allowlist. Add to TypedDict.

**P3 — Expose remaining DDP/CommOverlap perf knobs as a generic passthrough:**

5. Refactor `_apply_performance_config` to drive all model-level passthroughs from a single allowlist constant; refactor `_build_comm_overlap_config` similarly. Reduces future silent-drop risk.
6. Add `nccl_ub`, `num_distributed_optimizer_instances`, `align_param_gather`, `bucket_size` to DDP wiring (setup.py:675).

**P4 — Cross-entropy / MoE knobs not measured yet:**

7. `cross_entropy_fusion_impl` ("te" vs "native") — Qwen3 recipes set "te" but NeMo-RL doesn't preserve.
8. `moe_grouped_gemm` — wire and re-measure.

---

## Why this matters for the v07 → Phase B narrative

The v07 conclusion ("all in-pin knobs ±2%, no real signal") becomes more defensible after fixing P1: if even *correctly-wired* overlap_p2p / defer_embed / tp_comm_atomic are within noise on this workload+pin, that's a real null result. Right now we can't claim that — we measured baselines under three different yaml labels.
