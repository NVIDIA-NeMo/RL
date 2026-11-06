# Same-checkpoint, real-rollout streaming investigation

Distinguish metric aggregation, normalization placement, production DP placement,
policy drift and actual overlap. This diagnostic does not relax or replace the
failed original BF16 qualification contract. Product worker and existing parity
harness stay frozen; replay reuses that harness's model, public/raw optimizer
and preclip-gradient helpers. All computation uses allocated nodes and image1030.

## Large-model protocol

Use a read-only hardlink snapshot of the existing run's step210 checkpoint,
including optimizer/scheduler. Disable W&B and checkpoint writes; diagnostic
outputs are separate. Fresh capture resumes the checkpoint's data-loader cursor
without the rollout recovery ledger and leaves the convergence job untouched.

1. Resume step210 for two streaming LA0 updates, threshold4. Save all32 actual
   response tokens, generation/policy LPs, masks and computed advantages. Capture
   happens after all backward work, before TQ cleanup, so the extra tensor I/O
   cannot delay the first training dispatch. Its duration is recorded separately.
2. Independently resume the same state/seed for two full-batch updates, threshold8.
   Compare valid token-sequence hash multisets before calling native timing a
   paired workload result. One cold update with CPU capture is not steady throughput.
3. Repeat streaming with rank0 CUDA profiling on the second update, after the
   first has warmed up the pipeline, and no artificial delay or admission gate.
   Profiler/synchronization changes timing: use it for execution ordering/overlap.
   Missing traces fail analysis; zero overlap is a valid measured result.
   Cross-node sub-ms intersections need an independent clock-error bound.
4. Restore the exact source model/public+raw optimizer/scheduler/RNG before each
   numerical case below; retain the same captured generation LPs and masks.

| Case | Purpose |
| --- | --- |
| full | One full-batch SC dispatch. |
| full_repeat | Identical repeat to expose nondeterministic variation. |
| numerator_full | Independent ordinary full-input worker call; move only normalization to clip entry. |
| split_fixed_shards | Shard once, then split rank rows; preserve microbatch order and padding. |
| split_production_shards | Split global groups, then apply actual DP sharding to each chunk. |
| full_recompute | Full-input control with newly recomputed policy LPs. |
| split_production_recompute | Production placement plus newly recomputed policy LPs. |

Compare valid-token LPs, independent token-weighted K3 versus logged metrics,
with outer padding held fixed in these worker controls. The production-sharding
case changes DP row placement, not every native execution detail; native captures
include the real per-dispatch padding.
normalized preclip gradients, model/public+raw optimizer and scheduler. Per-rank
reports are retained and global exact flags require all ranks. New LPs in the
last case are injected into the materialized microbatch: the local adapter
permits only one put per partition; real TQ delta writes occur in native capture.

Evaluate updated models on a common full-input LP layout to isolate weight
changes. This still uses the old rollout and is labelled one-update KL rather
than live LA0 KL. Restore public FSDP shards before snapshot restoration after
inference, which may leave the root unsharded.

After numerical warmup, alternate full/split order for three fixed-input timing
repeats. State restoration, gradient/model comparisons and exports stay outside
the timer. The timed region includes LP/preparation/local fetch/backward/commit;
it excludes rollout GPU work and network TQ/Ray latency. This measures backend
overhead, not end-to-end streaming speedup. Native timings and ungated traces
address that separately.

Expected distinctions: fixed stored LPs/masks should give the same K3 apart from
scalar reduction roundoff; a systematic scaling discrepancy indicates a metric
contract defect. Fixed-shard split should reproduce the independent numerator
reference. Retain ordinary full-vs-stream BF16 differences and original limits.
If all generation finishes before the first training RPC, no generation remains
to overlap on that update. Check actual kernels as well as controller receipts.

## Runner smoke

Debug smoke uses Qwen2.5-1.5B,2train+2rollout GPUs,32responses, shortened768total/
256generation. It validates capture/replay/profiler plumbing. Its first shortened
rollout has uniform rewards and zero GRPO gradients, so it does not qualify
nonzero-gradient or large-MoE updates. EP16 uses the unchanged Qwen3.5-35B-A3B
recipe,16train+8rollout GPUs and the step210 policy; inspect actual advantages.

Entrypoint: run.sh inside one three-node ray.sub allocation. Preserve raw
failures/source hashes; diagnostic completion does not mean numerical acceptance.
Large-model results are pending until that allocation actually executes.
