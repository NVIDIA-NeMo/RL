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

## Reproduce the reported comparison

The shared integration branch is `sna/mxfp8-perf` in NVIDIA-NeMo/RL. It
contains the Qwen3-30B-A3B recipes, launcher, TE routed-expert scope, and
MXFP8 refit code on top of current `main`. The published [W&B report](<https://wandb.ai/nvidia/nemo-rl-mxfp8-training/reports/Qwen3-30B-A3B:-BF16-vs-MXFP8-Training-and-Rollout-(Option-B,-GB200)--VmlldzoxODA3Mzg2NA==>)
covers six successful 20-step jobs. Those measurements used frozen runtime
source `15fc687fb7aaf342128bc4f10c91e622cc1f4079`, with launcher changes
through `27346dfd600e9e27e3db419c585ae10e2ee5aaaf`. The later merge from
`main` has not yet been remeasured, so do not attribute the published numbers
to the new branch head.

On OCI-HSG, check out the branch under `/home` and initialize submodules:

```bash
git fetch origin sna/mxfp8-perf
git switch --track origin/sna/mxfp8-perf
git submodule update --init --recursive
```

Prepare an immutable source archive containing the checked-out commit **and
its initialized submodule contents**; a plain `git archive` omits submodule
files. Use a validated vLLM 0.29 image with the matching NeMo-RL actor
environments. The original runs used
`nemo_rl_main_aligned_20261007_7776525.sqsh`. Stage the source archive and
image on cluster storage, keep the checkout in `/home`, and let the launcher
place caches under node-local `/raid/scratch`. Then set `CONTAINER`,
`SOURCE_ARCHIVE`, `SOURCE_COMMIT` (the archive's commit), `RESULT_ROOT`, and
`WANDB_API_KEY`. `LAUNCH_COMMIT` defaults to `SOURCE_COMMIT`; set it separately
only when replaying the original frozen source with a newer launcher.

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
bash experiments/qwen30_pr4353_20261007/submit.sh bf16-bf16 test-only
bash experiments/qwen30_pr4353_20261007/submit.sh bf16-mxfp8 test-only
bash experiments/qwen30_pr4353_20261007/submit.sh mxfp8-option-b test-only
bash experiments/qwen30_pr4353_20261007/submit.sh mxfp8-option-b-param-false test-only
```

After the Slurm dry runs pass, submit the same arms without `test-only`.
The launcher defaults to 20 steps and `coreai_dlalgo_nemorl`; override
`SLURM_ACCOUNT` when needed. The two `mxfp8-default`/`mxfp8-param-false`
arms provide the Option B ablation. Compare only matched 20-step runs using
steps 2-20, both logprobs, finite generation KL, and logged throughput
metrics. `exposed_generation` is an Async wait, not full generation latency.
In Async, `generation_tokens_per_sec_per_gpu` uses policy + logprob + exposed
wait in its denominator, so it is a training-coupled worker-group proxy,
not standalone vLLM decode throughput. The earlier MXFP8 run with default
FlashInfer attention had NaN generation KL and a large reward shift; its raw
time must not be quoted as a valid MXFP8 speedup.
