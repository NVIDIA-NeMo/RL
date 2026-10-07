# Qwen3-30B-A3B MXFP8 training comparison

Async-1off, 4 GB200 nodes x 4 GPUs, GBS 2048, 20 steps. All arms use the
same Qwen3 performance recipe, FlashInfer TRTLLM MoE backend, NCCL reshard
refit, policy and reference logprobs, and checkpoint/validation-disabled
measurement settings. Report means over completed steps 2-20 and valid counts
per metric; use W&B throughput rather than deriving it from mean time.

| Arm | Training | Rollout | Grouped GEMM |
| --- | --- | --- | --- |
| `bf16-bf16` | BF16 | BF16 | Default |
| `bf16-mxfp8` | BF16 | MXFP8 routed experts | Default |
| `mxfp8-default` | MXFP8 routed experts, `fp8_param=true` | MXFP8 routed experts | Default |
| `mxfp8-param-false` | MXFP8 routed experts, `fp8_param=false` | MXFP8 routed experts | Default |
| `mxfp8-option-b` | MXFP8 routed experts, `fp8_param=true` | MXFP8 routed experts | TE op-fuser + CuTeDSL/cuDNN flags |
| `mxfp8-option-b-param-false` | MXFP8 routed experts, `fp8_param=false` | MXFP8 routed experts | TE op-fuser + CuTeDSL/cuDNN flags |

The training TE recipe matches `*mlp.experts.linear_fc*` and forces all other
modules to BF16. Its MXFP8 matcher does not override a disabled outer FP8
context, so the first two and last six training layers remain BF16. The
rollout applies the same BF16 boundary and excludes attention and router
weights from MXFP8. `mxfp8-param-false` differs from `mxfp8-default` only in
parameter storage; the same frozen source archive and config file are used.
Compare `mxfp8-option-b-param-false` with `mxfp8-param-false` to isolate
Option B with BF16 parameter storage. Option B is a diagnostic for Qwen's
SwiGLU expert layout: its configuration does not by itself prove which GEMM
kernel ran, and its refit/accuracy must pass before quoting a speedup.

Pull the pushed source commit once before preparing its immutable archive.
Wait for archive preparation to finish before submitting the GPU arms, so
concurrent `git pull` calls do not race on the shared checkout.

Set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` to select Triton attention while
keeping the configured FlashInfer TRTLLM MoE backend. The launcher gives
this variant a separate run name and node-local cache. `SOURCE_COMMIT` may
point to an existing immutable source archive when only the submission
script changes; set `LAUNCH_COMMIT` to the pushed launcher commit in that case.
Compare the BF16/BF16 attention A/B to measure backend overhead. The
FlashInfer-attention MXFP8 runs have nonfinite generation logprobs, so their
timing is diagnostic rather than an accuracy-qualified speed comparison.
