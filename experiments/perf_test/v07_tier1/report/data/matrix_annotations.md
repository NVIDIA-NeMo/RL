# matrix.csv Row Validity

Joins to `matrix.csv` by `(cluster, jobid)`. Marks which measurements reflect the *intended* yaml configuration vs. silent-drop baselines.

| jobid | cluster | variant | validity | reason |
|-------|---------|---------|----------|--------|
| 2584834 | oci-hsg | qwen3_32b/v07_07_overlap_p2p | **INVALID** | `overlap_p2p_comm` silently dropped (pre commit 1f788697); measurement is baseline |
| 2584836 | oci-hsg | qwen3_32b/v07_08_defer_embed_wgrad | **INVALID** | `defer_embedding_wgrad_compute` silently dropped; baseline |
| 2584838 | oci-hsg | qwen3_32b/v07_09_tp_atomic | **INVALID** | `tp_comm_atomic_ag/rs` silently dropped + split=true overrides anyway; baseline |
| 1692105 | lyris | qwen3_32b/v07_07_overlap_p2p | **INVALID** | same as 2584834 (silent drop) |
| 2584828 | oci-hsg | llama_8b/v07_06_grad_accum_fusion | UNCLEAR | `gradient_accumulation_fusion` only set at HF→megatron import — depends on whether re-import happened |
| 2584829 | oci-hsg | qwen3_30ba3b/v07_06_grad_accum_fusion | UNCLEAR | same |
| 2584830 | oci-hsg | qwen3_32b/v07_06_grad_accum_fusion | UNCLEAR | same |
| 1692104 | lyris | qwen3_32b/v07_06_grad_accum_fusion | UNCLEAR | same |

All other rows are VALID — the knob in their yaml is correctly enumerated by `setup.py` (G6 fusion bundle, G1 tp_comm_overlap, G5 delay_wgrad_compute, G3 overlap_moe_expert_parallel_comm, moe_shared_expert_overlap, fused_residual_rmsnorm).

## Phase B re-runs

After commit 1f788697 (Tier 2 wiring), submitted paired re-runs:

- OCI-HSG: jobs 2586605–2586608 (32B base, p2p, deferembed, tpatomic) and 2586609–2586612 (235B same set)
- Lyris: jobs 1692862–1692867 + the two earlier in batch (32B suite, 235B suite)

Replace the INVALID rows with these once they steady-state.

## gradient_accumulation_fusion runtime fix

Commit 1f788697 also adds runtime override in `_apply_performance_config`. Prior runs may or may not reflect the yaml depending on whether the cached megatron checkpoint was built with the toggle. Phase B does not re-run grad_accum, but next pass should re-measure 06 to lock the verdict.

## Phase A finals: fused_residual_rmsnorm

OCI-HSG (n=8 post-warmup each) — E2E TPS:

| pair | baseline | variant | delta |
|------|----------|---------|-------|
| llama_8b 2586264/2586265 | 3618 | 3711 | +2.57% |
| qwen3_30b 2586266/2586267 | 1845 | 1845 | -0.02% |
| qwen3_32b 2586268/2586269 | 1224 | 1221 | -0.29% |
| qwen3_235b 2586270/2586271 | running | **FAILED** | — |

Lyris (n=9 post-warmup each) — Total step time medians:

| pair | baseline | variant | delta |
|------|----------|---------|-------|
| llama_8b 1692688/1692689 | 76.01s | 72.81s | **-4.21%** |

**235B variant FAILED** (job 2586271): `RuntimeError: you can only change requires_grad flags of leaf variables` in TE `pytorch/ops/fuser.py:141`. Upstream TE op bug specific to the 235B model interaction with fused_residual_rmsnorm. Cannot pair-test 235B until TE patch lands.

**Llama_8b cross-cluster discrepancy**: OCI-HSG shows +2.57% (slower with variant), Lyris shows -4.21% (faster). Both within ±5% inter-batch noise band but signs flip — variance dominates the signal. Conclusion: no measurable speedup on 30B/32B; llama_8b within-noise either direction; 235B blocked by upstream bug.

## Phase B 32B FINAL (OCI-HSG, n=9 post-warmup)

Paired against same-batch baseline 2586605:

| variant | jobid | median step | delta |
|---------|-------|-------------|-------|
| baseline | 2586605 | 344.42s | — |
| v07_07 overlap_p2p | 2586606 | 353.16s | +2.54% |
| v07_08 defer_embed_wgrad | 2586607 | 342.27s | -0.62% |
| v07_09 tp_atomic | 2586608 | 344.79s | +0.11% |

After silent-drop fix (commit 1f788697), all three previously-silent knobs measure at noise floor on 32B. `overlap_p2p_comm` trends 2.5% slower (consistent across n=6 interim and n=9 final) but still inside ±5% inter-batch band; defer_embed/tp_atomic both clean nulls.

## Phase C interim

**OCI-HSG llama_8b ce_te_fusion** (n=8 post-warmup):
- baseline 2586671 / variant 2586673: 3749 vs 3810 TPS → **+1.62%** (within noise). Baseline differs from Phase A's same-yaml baseline by 3.6% (3618 vs 3749) — confirms inter-batch variance is real and pair-isolation matters.

**OCI-HSG 32B FINAL (n=9)**, paired against 2586676:

| variant | jobid | median | delta |
|---------|-------|--------|-------|
| baseline | 2586676 | 344.64s | — |
| v07_13 ce_te_fusion | 2586678 | 347.08s | +0.71% |
| v07_14 nccl_ub | 2586679 | 343.43s | -0.35% |

nccl_ub was the highest-expected-delta knob in Phase C (advertised 1-2% reduce/gather speedup). Final n=9 -0.35% confirms it sits at noise floor on 32B.

**Lyris llama_8b ce_te_fusion** (n=9):
- baseline 1692930 / variant 1692932: 74.86s vs 75.03s → **+0.23%** (noise). Confirms OCI's +1.62% was upper-edge of the noise band.

## Phase C retry: external_cuda_graph llama_8b FINAL (n=9)

After commit d3bd8f24 fix (use_te_rng_tracker + activation_checkpointing=false), retry batch ran successfully:

| Cluster | baseline | variant | delta |
|---------|----------|---------|-------|
| OCI-HSG | 80.67s (2587286) | 85.01s (2587287) | **+5.38%** |
| Lyris   | 78.18s (1693100) | 79.36s (1693101) | **+1.51%** |

Both clusters show external_cuda_graph SLOWING DOWN llama_8b. Likely cause: graph capture in step 1 amortizes over remaining steps; with max_num_steps=10 and ~10s capture overhead, the variant pays a fixed cost on short runs that real training would amortize. **CUDA-graph requires max_num_steps≥50 to fairly measure.** Current Phase C verdict: cannot conclude — need longer-run re-eval.

## Phase C jobs

- OCI-HSG initial: 2586671-2586679 (9 jobs at 5b5a3473)
- Lyris initial: 1692930-1692938 (all still PENDING)
- **CUDA-graph variants FAILED** (2586672, 2586675, 2586677): missing `use_te_rng_tracker=true` + llama recompute conflict. Fixed in d3bd8f24.
- OCI-HSG retry: 2587286-2587292 (paired base+v07_12 for all 3 models)
- Lyris retry: 1693100-1693105

Behavioral equivalence note: original Phase C jobs ran at 5b5a3473; subsequent commit bd5d8196 added `use_custom_fsdp` wiring + `reuse_grad_buf_for_mxfp8_param_ag` allowlist. Neither changed runtime behavior since all yamls use mcore defaults for these.
