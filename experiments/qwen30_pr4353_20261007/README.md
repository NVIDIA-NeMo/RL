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
| `mxfp8-option-b` | Same as above | Same as above | TE op-fuser + CuTeDSL/cuDNN flags |

The MXFP8 arms keep the first two and last six layers BF16. Option B is a
diagnostic for Qwen's SwiGLU expert layout: its configuration does not by
itself prove which GEMM kernel ran, and its refit/accuracy must pass before
quoting a speedup.

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
