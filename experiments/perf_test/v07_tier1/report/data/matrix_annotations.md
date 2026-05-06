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

## Phase A finals: fused_residual_rmsnorm (OCI-HSG, n=8 post-warmup each)

| pair | baseline E2E | variant E2E | delta |
|------|--------------|-------------|-------|
| llama_8b 2586264/2586265 | 3618 | 3711 | +2.57% |
| qwen3_30b 2586266/2586267 | 1845 | 1845 | -0.02% |
| qwen3_32b 2586268/2586269 | 1224 | 1221 | -0.29% |
| qwen3_235b 2586270/2586271 | pending | pending | — |

3/3 completed pairs show no measurable speedup at GB200 scale. Llama_8b +2.57% is borderline (above ±2% but within inter-batch noise observed across same-yaml runs at 3.6%). Conclusion subject to 235B confirm.

## Phase C interim: llama_8b ce_te_fusion (OCI-HSG)

- Baseline 2586671: median 3749 (n=8)
- Variant ce_te_fusion 2586673: median 3810 (n=8)
- **Delta: +1.62%** — within noise. Note: baseline differs from Phase A's same-yaml baseline by 3.6% (3618 vs 3749) — confirms inter-batch variance is real and pair-isolation matters.

## Phase C jobs

- OCI-HSG initial: 2586671-2586679 (9 jobs at 5b5a3473)
- Lyris initial: 1692930-1692938 (all still PENDING)
- **CUDA-graph variants FAILED** (2586672, 2586675, 2586677): missing `use_te_rng_tracker=true` + llama recompute conflict. Fixed in d3bd8f24.
- OCI-HSG retry: 2587286-2587292 (paired base+v07_12 for all 3 models)
- Lyris retry: 1693100-1693105

Behavioral equivalence note: original Phase C jobs ran at 5b5a3473; subsequent commit bd5d8196 added `use_custom_fsdp` wiring + `reuse_grad_buf_for_mxfp8_param_ag` allowlist. Neither changed runtime behavior since all yamls use mcore defaults for these.
