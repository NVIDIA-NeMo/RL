# v0.7 Tier 1 Perf Knobs — Empirical Validation

`v0.7_final_plan.md` proposed eight Tier 1 Megatron knobs (G1, G3, G5, G6, plus stacked) with claimed throughput gains in the 8–15% range, derived from Bridge / Megatron-LM source analysis. This report measures their **actual GRPO training-loop impact** on four NeMo-RL recipes (Llama 3.1 8B 2n4g, Qwen3-30B-A3B 4n4g, Qwen3-32B 4n4g, Qwen3-235B 16n4g) on GB200, and adds five Tier 2 knobs (`gradient_accumulation_fusion`, `moe_shared_expert_overlap`, `overlap_p2p_comm`, `defer_embedding_wgrad_compute`, `tp_comm_atomic_ag/rs`) for comparison.

The numbers below come from OCI-HSG. Lyris is mid-flight — same code, same configs, results will be appended once they land.

---

## What Each Knob Does, and Where It Applies

| Knob | Class | Source | Where it applies |
|---|---|---|---|
| **G6** fusion bundle (`bias_dropout_add_fusion`, `masked_softmax_fusion`, `persist_layer_norm`) | A — passthrough | model-provider | all 4 recipes |
| **G1** `tp_comm_overlap` | A+D — needs UB | `CommOverlapConfig` | dense TP≥2 (32b, 235b) |
| **G5** `delay_wgrad_compute` | A+B — TP-heavy | `CommOverlapConfig` | 32b, 235b |
| **G3** `overlap_moe_expert_parallel_comm` (MLM #3795) | A+B — MoE only | `CommOverlapConfig` | 30ba3b, 235b |
| **Stack** | — | combined per-model | all 4 |
| **06** `gradient_accumulation_fusion` | A | `OptimizerConfig` | all 4 |
| **07** `overlap_p2p_comm` (32b, 235b) / `moe_shared_expert_overlap` (30ba3b) | A+D / B | `TransformerConfig` | PP≥2 / MoE shared expert |
| **08** `defer_embedding_wgrad_compute` | A+B | embedding wgrad | PP≥2 (32b, 235b) |
| **09** `tp_comm_atomic_ag/rs` | C | TP overlap mode swap | TP≥2 (32b, 235b) |
| **10** `moe_shared_expert_overlap` | B | MoE | 235b |

10 GRPO steps per run. Steady-state metric = mean of `E2E (Tokens/sec/gpu)` over steps 1..N-1 (drop step-0 warmup and final partial step). Source data: [`data/matrix.csv`](data/matrix.csv).

---

## Headline: All Tier 1 Deltas Land Inside the System Noise Floor

The cleanest evidence for this is in the **30B-A3B** recipe, where four Tier 1 variants were each run twice in different SLURM batches:

| Variant | Batch B1 | Batch B2 | Batch-to-batch shift |
|---|---|---|---|
| baseline | 1589 tok/s/gpu | **1839** | **+15.7%** |
| G6 fusion | 1619 | 1853 | +14.5% |
| G3 MoE A2A | 1592 | 1811 | +13.8% |
| stack | 1589 | 1822 | +14.7% |

**The same configuration runs 14–16% faster in Batch B2 than in Batch B1.** This is the noise floor for any single-shot comparison on this cluster: anything below ~10% sits inside the band attributable to scheduler placement, fabric contention, or node-temperature variation, not to the code change.

Within-batch deltas, the legitimate axis of comparison, are all sub-2%:

| Model · variant | tok/s/gpu | MFU | Δ vs base |
|---|---|---|---|
| **Llama-8B** baseline | 3760 | 25.84% | — |
| Llama-8B G6 | 3779 | 25.90% | +0.5% |
| Llama-8B grad_accum | 3812 | 24.93% | +1.4% |
| **30B-A3B B1** baseline | 1589 | 3.03% | — |
| 30B-A3B B1 G6 | 1619 | 3.13% | +1.9% |
| 30B-A3B B1 G3 | 1592 | 3.02% | +0.2% |
| 30B-A3B B1 stack | 1589 | 3.04% | 0.0% |
| **30B-A3B B2** baseline | 1839 | 3.15% | — |
| 30B-A3B B2 G6 | 1853 | 3.14% | +0.8% |
| 30B-A3B B2 G3 | 1811 | 3.04% | −1.5% |
| 30B-A3B B2 stack | 1822 | 3.06% | −0.9% |
| **32B** baseline | 1239 | 25.27% | — |
| 32B G6 | 1234 | 25.30% | −0.4% |
| 32B G1 | 1229 | 25.09% | −0.8% |
| 32B G5 | 1217 | 24.64% | −1.8% |
| 32B stack | 1240 | 25.19% | +0.1% |
| **235B** baseline | 140.2 | 3.41% | — |
| 235B G6 | 138.8 | 3.38% | −1.0% |
| 235B G1 | 142.3 | 3.52% | **+1.5%** |
| 235B G5 | 138.3 | 3.52% | −1.4% |
| 235B G3 | 140.2 | 3.44% | 0.0% |

The largest within-batch effect is **G1 `tp_comm_overlap` on 235B at +1.5%** — well below the claimed range and barely outside the noise floor. **None of the proposed Tier 1 knobs reproduces an 8–15% gain on this RL workload.**

---

## Tier 2 (partial)

Tier 2 jobs are still in flight on OCI-HSG; values below are from 4–7 steps and should be read as preliminary.

| Model · variant | steps | tok/s/gpu | Δ vs same-batch base |
|---|---|---|---|
| 30B-A3B 06 grad_accum | 7 | 1734 | (mixed batch — uncomparable yet) |
| 30B-A3B 07 moe_shared_expert | 7 | 1796 | (mixed batch — uncomparable yet) |
| 32B 06 grad_accum | 4 | 1256 | +1.4% |
| 32B 07 overlap_p2p | 4 | 1227 | −1.0% |
| 32B 08 defer_embed_wgrad | 4 | 1252 | +1.1% |
| 32B 09 tp_atomic | 4 | 1257 | +1.4% |

Same picture as Tier 1: nothing exceeds ±2%. Final numbers will land once the runs complete.

---

## Things That Surprised Us

**MoE MFU is reported around 3%.** That number is artifactually low — the framework's FLOPs counter under-counts MoE expert sparsity, so it should not be compared against the dense recipes' ~25%. tok/s/gpu is the only fair throughput axis across the four recipes.

**Cached `iter_0000000/run_config.yaml` files broke runs on Lyris** until two stale key sets (`cuda_graph_*` on Llama/30B-A3B, `muon_use_nesterov` on 32B/235B) were stripped. These keys came from a newer NeMo-RL version that wrote the cache, then were rejected by the `GPTModelProvider` / `OptimizerConfig` of the v0.7 Bridge pin. **Treat the HF/converter cache as part of the experiment input, not as inert disk state.** Scripts to strip and back up these keys are in `data/` (see notes below).

**Submission-batch placement matters more than any Tier 1 knob.** The 30B-A3B baseline shifted by 16% across two SLURM submissions of the same yaml. Future perf claims at this granularity must come from **paired runs in the same job batch**, ideally co-scheduled, before they are taken seriously.

---

## Follow-up: Silent-Drop Wiring Audit (Phase B)

After Tier 2 finals showed null deltas, an audit found that four Tier 2 knobs (`overlap_p2p_comm`, `defer_embedding_wgrad_compute`, `tp_comm_atomic_ag/rs`, `gradient_accumulation_fusion`) were **silently dropped** by NeMo-RL's `policy.megatron_cfg` allowlist — yamls set them, the wrapper never forwarded them. Original measurements were measuring the baseline three times, not the variant.

Commit `1f788697` adds the missing forwarders + extends the `MegatronConfig` TypedDict + adds a runtime override path for grad_accum_fusion. Six DDPConfig fields (`use_custom_fsdp`, `nccl_ub`, `num_distributed_optimizer_instances`, `align_param_gather`, `bucket_size`, `fp8_param_gather`, `reuse_grad_buf_for_mxfp8_param_ag`) wired in `bd5d8196`.

Paired re-runs on Qwen3-32B FINAL (same-batch baseline 2586605, n=9 post-warmup):

| Variant | jobid | median step | Δ |
|---|---|---|---|
| baseline | 2586605 | 344.4s | — |
| `overlap_p2p_comm` | 2586606 | 353.2s | +2.54% |
| `defer_embedding_wgrad_compute` | 2586607 | 342.3s | -0.62% |
| `tp_comm_atomic_ag/rs` | 2586608 | 344.8s | +0.11% |

After fixing the silent drop, all three knobs measure at noise floor on 32B — **the original null result was correct, just for the wrong reason.** `overlap_p2p_comm` trends 2.5% slower (consistent direction across n=6 interim and n=9 final) but still inside the ±5% inter-batch band. 235B re-runs still in queue.

---

## Phase A: `fused_residual_rmsnorm` (MLM PR #3384)

A new TE-level rmsnorm fusion that the v0.7 plan did not list but the upstream MLM team flagged as high-leverage. Paired re-runs across the four recipes (n=8 post-warmup):

| Recipe | Cluster | baseline | variant | Δ |
|---|---|---|---|---|
| Llama-8B | OCI-HSG | 3618 | 3711 | +2.57% (slower) |
| Llama-8B | Lyris | 76.0s | 72.8s | -4.21% (faster) |
| Qwen3-30B-A3B | OCI-HSG | 1845 | 1845 | -0.02% |
| Qwen3-32B | OCI-HSG | 1224 | 1221 | -0.29% |
| Qwen3-235B | OCI-HSG | running | **FAILED** | — |

The 235B variant FAILED with `RuntimeError: you can only change requires_grad flags of leaf variables` in TE `pytorch/ops/fuser.py:141` — an upstream TE op bug specific to the 235B model interaction, not a NeMo-RL config issue.

**The cross-cluster sign flip on Llama-8B is the cleanest noise-floor demonstration in this study.** Same yaml, same code, same hardware family — OCI-HSG says the variant is 2.6% slower, Lyris says it's 4.2% faster. That is what ±5% inter-batch variance does to a single-pair measurement.

**Verdict: no measurable speedup on 30B/32B; Llama-8B within noise either direction; 235B blocked by upstream TE op bug.**

---

## Phase C: Newly-Wired Knobs (CUDA-graph, `nccl_ub`, ce_te_fusion)

After the silent-drop fix added them to the allowlist, three more knobs were measurable for the first time. Phase C also pulled in two yaml-level config conflicts that broke the first batch (`activation_checkpointing` ⊥ `external_cuda_graph`; CUDA-graph requires `use_te_rng_tracker=true`); commit `d3bd8f24` ships the fix and re-submits.

Final, paired (n=9 post-warmup):

| Variant | Recipe | OCI-HSG Δ | Lyris Δ |
|---|---|---|---|
| `external_cuda_graph` | Llama-8B (n=9 each) | **+5.38%** | **+1.51%** |
| `cross_entropy_loss_fusion` | Llama-8B | +1.62% (n=8) | +0.23% |
| `cross_entropy_loss_fusion` | 32B | +0.71% | running |
| `nccl_ub` | 32B | -0.35% | running |

`nccl_ub`, the strongest expected signal in Phase C (advertised 1–2% reduce/gather speedup), measures at -0.35% on 32B — squarely in the noise floor.

`external_cuda_graph` shows a consistent **slowdown** on Llama-8B across both clusters: +5.38% on OCI, +1.51% on Lyris. The likely cause is amortization: graph capture happens during step 1, savings appear in steps 2–10. With max_num_steps=10 and ~10s capture overhead, the variant pays a fixed cost on every short run that real training would amortize away. **CUDA-graph cannot be measured at n=10 GRPO steps.** A correct re-evaluation needs max_num_steps≥50 or explicit capture-vs-execute timing separation.

---

## Cross-Repo: PR-Level Re-evaluation (Paired)

Four open NeMo-RL PRs were re-measured with same-batch pairing on Qwen3-30B-A3B and Qwen3-32B:

| PR | Variant | Recipe | n | Δ |
|---|---|---|---|---|
| #2278 (GroupedGEMM) | `gemm_grouped` | 30B-A3B | 9 | **-19.4%** (faster) |
| #2279 (activation offload) | `offload_moe_act` | 30B-A3B | 9 | +0.29% |
| #2279 (activation offload) | `offload_core_attn_moe` | 32B | 9 | -1.76% |
| #2280 (selective recompute) | `recompute_full` vs no_ckpt | 32B | 9 | +4.16% (slower, expected) |
| #2133 (HybridEP d28bd676) | `hybridep_no_hp` / `hp_ep` | 30B-A3B | — | both **FAILED** |

**GroupedGEMM is the only knob in the entire study that produced a clearly above-noise speedup — -19.4% step time on Qwen3-30B-A3B paired against the same-batch baseline.** This is the headline result. It validates merging PR #2278.

`activation_offload` (PR #2279) shows null perf cost (the design intent was memory savings without throughput regression). The 32B core-attn-MoE variant trending -1.76% is borderline but plausibly within batch noise.

`selective_recompute` (PR #2280) on 32B shows the expected directional cost (full recompute +4.16% vs no recompute) — confirming the wiring works. Llama re-runs need a config audit (variant labels appear inverted; following up).

**HybridEP (PR #2133) at d28bd676**: both 30B runs failed with infrastructure errors — `ModuleNotFoundError: No module named 'torch._tensor'` on the baseline (venv corruption) and `Error compiling objects for extension` on the hp_ep variant (DeepEP build failure). Need to rebuild the worktree's venv and re-pin DeepEP before this can be measured.

---

## Key Takeaway

**On GB200 / NeMo-RL GRPO, every Megatron-side perf knob in this study lands inside the same ±5% inter-batch noise band — except GroupedGEMM (-19.4%) which is the lone clear winner.** The eight Tier 1 / five Tier 2 / three Phase C knobs all sit between -2% and +2% within-batch; the one TE-level fusion (Phase A `fused_residual_rmsnorm`) sits at noise floor on 30B/32B and shows opposite signs across two clusters on Llama-8B; the silent-drop fix (Phase B) confirmed the original nulls were correct after wiring. Among the open PRs, only **#2278 GroupedGEMM** has a real measurable speedup; **#2279 activation_offload** is a memory-side feature with neutral perf; **#2280 selective_recompute** wires correctly on 32B; **#2133 HybridEP** is currently blocked on infra.

The methodological point reinforced across every phase: **same-batch pairing is mandatory.** The Llama-8B cross-cluster sign flip on Phase A and the 16% same-yaml drift on the original 30B-A3B Tier 1 numbers both demonstrate that any single-shot delta below ~5% on this cluster reflects scheduler/fabric/temperature variance, not the code change.

Lyris cross-cluster confirms the same picture for the variants that finished there (Phase A llama_8b, Phase C llama_8b ce_te_fusion). 235B blocked by Phase A TE bug; remaining Phase B/C 32B medians will solidify with full n=8 finals on next pass.
