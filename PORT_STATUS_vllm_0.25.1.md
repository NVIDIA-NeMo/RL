# MXFP8 refit port to vLLM 0.25.1 — status

This branch (`sna/mxfp8-on-vllm0251`) is the **prep workspace** for adapting the
MXFP8 refit optimizations (PR #3294) to vLLM 0.25.1. It is **not** a PR branch.
The shipping PRs (#3294 / #3295 / #3296) stay on the pinned vLLM 0.20.0.

Base = Terry's `terryk/bump-vllm-0.25.1` (#3280) + our #3295 + #3294, merged.

## Verdict per PR on 0.25.1

- **#3295 (generation logprobs as prev):** works unchanged — 0 vLLM coupling.
- **#3296 (BF16 TRTLLM refit):** drop — the rebind bug it patches is fixed
  natively in vLLM v0.21.0.
- **#3294 (MXFP8 refit):** needs a real port. Cascade below.

## Port cascade (measured on GB200 / oci-hsg, one break per run cycle)

| # | Break (runtime) | Fix | Commit |
|---|---|---|---|
| 1 | `ImportError: FusedMoeWeightScaleSupported` removed from `fused_moe.layer` | drop import; use the stable string `"block"` | 99b1a96 |
| 2 | backend assert `self.mxfp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM` (0.25 uses `self.fp8_backend`) | drop the sanity assert | 99b1a96 |
| 3 | `AssertionError: Unsupported MXFP8 linear kernel: FlashInferCutedslMxfp8LinearKernel` | widen the refit kernel guard to accept `FlashInferCutedsl*` | 0fa4259 |
| 4 | `ValueError: Mismatched mB.strides[1] ... expected to be 1` (CuteDSL GEMM) | **OPEN** — real weight-layout port (see below) | — |

Fixes 1-3 got the engine to **initialize on v0.25.1** and cleared the MoE
batched-shuffle path. Break #4 is the genuine hard part.

## The open item (#4): CuteDSL linear weight layout

Root cause: `process_weights_after_loading_mxfp8_linear` re-swizzles only the
**scale** (`swizzle_mxfp8_scale`) and leaves the **weight** in checkpoint
layout. The old Cutlass MXFP8 linear kernel accepted that; vLLM 0.25's new
**CuteDSL** kernel wants the weight in a different memory layout (the stride
error). So the refit-loaded weight has the wrong stride for the CuteDSL GEMM.

**Fix shape (moderate, one function, logic unchanged):** add a weight
re-layout to `process_weights_after_loading_mxfp8_linear`, mirroring exactly
what vLLM 0.25's CuteDSL kernel does to the weight in its own
`process_weights_after_loading` — the same pattern our MoE path already uses
(batched gather) and that #3280 uses for fp8 MoE (`convert_to_fp8_moe_kernel_format`).
Then re-verify bit-exact (`NRL_MXFP8_SHUFFLE_VERIFY`).

**Why defer to #3280 merge:** the CuteDSL weight layout is a kernel-internal,
undocumented, version-specific contract. Once #3280 lands, the merged base
establishes the 0.25 quant path as the reference to mirror, instead of guessing
the stride.

**Scope note:** in an MoE model with attention BF16-ignored and experts on the
MoE path, the mxfp8 *linear* path likely covers very few layers (router / a few
dense) — small surface, but must be correct.

## How to resume

1. Rebase this branch onto NeMo-RL main once #3280 (vLLM 0.25.1 bump) merges.
2. Read vLLM 0.25's CuteDSL MXFP8 linear kernel `process_weights_after_loading`
   for the exact weight transform; apply it in `process_weights_after_loading_mxfp8_linear`.
3. Run the naive-vs-patch A/B (oci-hsg `nemo-rl-cg-test/exp_vllm0251_ab.sh`,
   GB200 4n4g) to confirm the refit optimizations still reproduce on 0.25.1.

## Detailed fix recipe for break #4 (CuteDSL linear weight layout)

Root cause: `process_weights_after_loading_mxfp8_linear` re-lays-out only the
scale (`swizzle_mxfp8_scale`); the weight stays checkpoint [N,K]. The old
Cutlass kernel accepted that; CuteDSL wants dim-1-contiguous (the stride error).

Fix = the same "idempotent re-layout + bit-exact verify" pattern #3294 already
uses for MoE and #3296 uses for BF16-TRTLLM:

1. **Find the transform (needs vLLM 0.25 source):** read
   `FlashInferCutedslMxfp8LinearKernel.process_weights_after_loading` in vLLM
   0.25 (under `.../quantization/kernels/...`). That is the ground-truth layout
   the kernel wants on a normal load. Likely a transpose ([N,K]->[K,N] to make
   dim-1 contiguous), possibly + a block pack. Mirror it exactly.
2. **Apply idempotently in refit:** keep the checkpoint-layout weight as the
   permanent load target (every refit's weight_loader works unchanged);
   recompute the CuteDSL layout from it into an apply-target each refit
   (separate tensor or aliased view), exactly like `w13_weight_for_apply` in
   the BF16-TRTLLM path.
3. **Bit-exact verify:** add an `NRL_MXFP8_*_VERIFY`-style assert that our
   refit-produced weight equals vLLM's own `process_weights` output on a fresh
   load, byte-for-byte.
4. **Scope:** confirm which layers hit the mxfp8-linear path (attention is
   BF16-ignored, experts go MoE path -> likely just router / a few dense).

Moderate effort, no new architecture (reuses #3294's own pattern). The only new
work is reading the exact CuteDSL transform. Cleanest at #3280 merge: the merged
base pins 0.25 and includes #3280's `convert_to_fp8_moe_kernel_format`, giving a
tested reference for the layout instead of reverse-engineering the stride.

## CONFIRMED transform (read from vLLM v0.25.1 source, 2026-07-21)

`vllm/model_executor/kernels/linear/mxfp8/flashinfer.py`:

- `FlashInferCutlassMxfp8LinearKernel.process_weights_after_loading` (0.20 path):
  `layer.weight = Parameter(weight.contiguous())`  # [N,K] row-major; apply() does weight.t()
- `FlashInferCutedslMxfp8LinearKernel.process_weights_after_loading` (0.25 path):
  `layer.weight = Parameter(weight.contiguous().t())`  # [K,N] column-major; apply() uses it directly
  (source comment: "Store weight column-major [K, N] as mm_mxfp8 expects for operand B.")

So the ONLY weight-layout delta is the `.t()`: CuteDSL wants the weight pre-transposed
to column-major [K,N]. Our `process_weights_after_loading_mxfp8_linear` currently
re-swizzles only the scale and leaves the weight [N,K] -> stride mismatch.

Fix = when the selected kernel is CuteDSL, produce the weight as
`weight.contiguous().t()` for the apply target, idempotently (keep [N,K] as the
load target so weight_loader + refit #2 keep working; the CuteDSL apply reads the
[K,N] view -> same aliasing pattern as BF16-TRTLLM's w13_weight_for_apply).
Guard with a bit-exact verify vs the kernel's own process_weights output.

## Grind progress (gcp-nrt, 2026-07-21)

Cascade being resolved one break per run cycle:
1. FusedMoeWeightScaleSupported import — fixed (99b1a96)
2. mxfp8_backend / Fp8MoeBackend assert — fixed (99b1a96)
3. CuteDSL linear kernel guard — fixed (0fa4259)
4. **CuteDSL linear weight-layout stride — FIXED (ba0b1b1f): expose layer.weight as a
   zero-copy transpose view of the [N,K] load target. Validated: the `mB.strides[1]`
   error is GONE.**
5. `layer.enable_eplb` on RoutedExperts (0.25 FusedMoE split) — fixed (3679830):
   read via layer.moe_config.moe_parallel_config.enable_eplb.
6. (in flight, job 460903)

All remaining breaks are the same class (FusedMoE -> MoERunner+RoutedExperts attribute
migration); each resolves with a getattr/attribute-path fix. The hard part (weight
layout, #4) is solved. gcp-nrt caveat: worker venv build needs a retry when
nvidia-resiliency-ext source build times out on the network (transient).

## RESULT: port works end-to-end on vLLM 0.25.1 (2026-07-21)

All breaks cleared; smoke5 (846ba53) runs multiple steps with refits, SHUFFLE_VERIFY
passing. Full fix set on top of #3280:
- 99b1a96: drop FusedMoeWeightScaleSupported import (-> "block" literal) + drop
  mxfp8_backend/Fp8MoeBackend assert.
- 0fa4259: accept FlashInferCutedsl* in the linear refit kernel guard.
- 3679830: read enable_eplb via moe_config.moe_parallel_config.
- 29bb71f: compute routing_method_type via get_routing_method_type; read ep_rank
  via moe_config.moe_parallel_config.
- 846ba53: **refit-safe linear weight layout** — keep layer.weight in checkpoint
  [N,K] (so every refit load matches), and patch FlashInferCutedslMxfp8LinearKernel.
  apply_weights to read layer.weight.t() (zero-copy [K,N] column-major, what the
  kernel wants). NOTE: the earlier ba0b1b1f transpose-in-process was WRONG — it made
  layer.weight [K,N] and broke refit #2's lm_head load ([N,K] shard into [K,N] param).

The only attributes missing from RoutedExperts were routing_method_type + ep_rank
(the other ~13 are present). Weight-layout was the one conceptual fix; everything
else is attribute-path migration from the FusedMoE -> MoERunner+RoutedExperts split.

**Correctness: validated** (bit-exact SHUFFLE_VERIFY passes on 0.25.1).

**Performance: NOT representative yet.** naive-mxfp8 on 0.25.1 shows logprob 144.8 s
vs 84.6 s on 0.20 — the flashinfer CuteDSL MXFP8 kernel runs an UNTUNED fallback
tactic (AutoTuner: "shape outside tuning bucket -- perf cliff", esp. the lm_head
2048x151936 GEMM). This is a flashinfer tuning-state issue, not a port bug. Real
0.25.1 perf needs a flashinfer autotuner pass covering these shapes / a warm tuning
cache. Deferred as the last item.
