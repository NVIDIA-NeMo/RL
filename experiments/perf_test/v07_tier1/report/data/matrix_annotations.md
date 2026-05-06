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

## Phase A interim: llama_8b fused_residual_rmsnorm (OCI-HSG)

- Baseline 2586264: E2E TPS/GPU median 3618 (post-warmup, n=8)
- Variant  2586265: E2E TPS/GPU median 3711 (post-warmup, n=8)
- **Delta: +2.57%** — borderline (within-batch noise band ±2%). Need 30B/32B/235B pairs to draw conclusion.

## Phase C jobs

- OCI-HSG: 2586671–2586679 (9 jobs, llama_8b + 30B + 32B variants for cuda_graph + ce_te_fusion + nccl_ub)
- Lyris: 1692930–1692938 (mirror set), all PENDING due to 235B-priority queue head
- Submitted at commit 5b5a3473; running jobs equivalent to bd5d8196 since `use_custom_fsdp=false` matches mcore default and `reuse_grad_buf_for_mxfp8_param_ag` not set.
