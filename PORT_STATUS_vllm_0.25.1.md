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
