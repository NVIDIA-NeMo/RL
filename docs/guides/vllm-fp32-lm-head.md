# vLLM FP32 LM-head comparison

On Qwen3-8B, native FP32 LM-head output removes the final BF16 rounding of
logits while leaving the checkpoint weights and transformer activations in BF16.
The measured rollout latency is unchanged within run-to-run variation.

NeMo RL maps `generation.vllm_cfg.fp32_lm_head: true` to vLLM 0.29's native
`hf_overrides.head_dtype="float32"`. Its CUDA path uses
`torch.mm(..., out_dtype=torch.float32)`, the same operation used by
[TensorRT-LLM #19850](https://github.com/NVIDIA/TensorRT-LLM/pull/19850).
This replaces NeMo RL's Nemotron-H source patch and applies to the native
`LogitsProcessor` used by models such as Qwen and Llama.

See [configuration and reproduction commands](../design-docs/generation.md#fp32-lm-head-logits)
and [machine-readable results with all timing samples](../assets/vllm_fp32_lm_head_results.json).
The benchmark source is
[`benchmark_vllm_fp32_lm_head.py`](../../tools/benchmarks/benchmark_vllm_fp32_lm_head.py).

## Setup

Measured on October 29, 2026:

| Item | Value |
|---|---|
| NeMo RL main base | `1849f2256725b783b34c9f91ae9444f93bb26587` |
| Hardware | One NVIDIA GB200, driver 580.126.20 |
| Runtime | Python 3.13.14, PyTorch 2.13.0+cu130, vLLM 0.29.0, Transformers 5.12.1 |
| Model | Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Data | First 64 GSM8K test questions, dataset revision `740312add88f781978c0658806c59bc2815b9866` |
| Prompt | Qwen chat template, `enable_thinking=False` |
| Generation | Greedy, seed 42, exactly 256 new tokens/request, `ignore_eos=True`, raw logprobs |
| Engine | NeMo RL synchronous worker, TP=1, FlashAttention 4, eager mode, prefix cache disabled |
| Scheduling | 64 max sequences, 4096 max batched tokens, 2048 max model length, 0.65 memory utilization |

The experiment uses a standalone wheel environment with the pinned inference
components, not the complete NeMo RL training container. No Megatron trainer was
launched. The matrix-projection, generated-token and timing measurements below
must not be interpreted as GRPO reward, convergence, or trainer/rollout mismatch
measurements. CUDA graphs and the default FlashInfer attention backend were not
benchmarked.

## Projection accuracy

After timed generation, the BF16 arm captured 2,048 actual rollout hidden states.
Both head paths were evaluated on those **identical BF16 inputs and weights**;
the reference is their FP64 matrix product. There are 311,164,928 compared logit
entries. The transformer itself is unchanged.

| Metric versus FP64 reference | BF16 output, then upcast | Direct FP32 output |
|---|---:|---:|
| Maximum absolute logit error | 0.125019 | 0.000289416 |
| Reference-greedy token logprob absolute error, mean | 0.00639957 | 0.00000140896 |
| Reference-greedy token logprob absolute error, p99 | 0.0737542 | 0.0000146984 |
| Sampled-token logprob absolute error, mean | 0.00890977 | 0.00000204711 |
| Sampled-token logprob absolute error, maximum | 0.208985 | 0.0000689185 |
| KL(reference distribution ‖ head distribution), mean | 0.000271022 | 1.47908e-11 |
| Argmax differences | 17 / 2048 | 0 / 2048 |

For sampled-token errors, eight tokens per position were drawn from the FP64
reference distribution at temperature 1 using seed 42 (16,384 samples). Logprob
errors include FP32 log-softmax rounding. KL uses FP64 normalization for both
paths to isolate projection error; it is **not** NeMo RL's `gen_kl_error` metric.

All BF16 logits were bit-identical to direct-FP32 logits rounded back to BF16.
This verifies the intended mechanism: the native path removes the final output
rounding. It does not make the entire model FP32 or eliminate earlier rounding.

## LM-head latency

Native projection plus conversion to FP32, measured with CUDA events on the live
checkpoint weight `[151936, 4096]`. Each size has five warmups and twenty timed
samples per path, with the path order alternated. Values are medians in ms.

| Tokens per projection | BF16 GEMM + FP32 upcast | FP32-output GEMM |
|---:|---:|---:|
| 1 | 0.2504 | 0.2452 |
| 8 | 0.2545 | 0.2491 |
| 64 | 0.2812 | 0.2640 |
| 512 | 0.6505 | 0.4659 |
| 4096 | 4.9520 | 3.1476 |

These measurements exclude softmax, transformer layers and tensor-parallel
communication. The FP32 path avoids the separate output-upcast kernel.

## Rollout latency

Four fresh processes ran in BF16 → FP32 → FP32 → BF16 order on the same GPU.
Each process used one untimed full-workload warmup and five timed repetitions.
Every repetition generated 16,384 tokens. Numerical capture and FP64 analysis
happened after the timed repetitions.

| Metric, ten repetitions per mode | BF16 | FP32 |
|---|---:|---:|
| Median seconds | 7.47147 | 7.47302 |
| Min–max seconds | 7.40109–7.53123 | 7.32746–7.58299 |
| Generated tokens/s at median latency | 2192.88 | 2192.42 |

The median FP32 latency is 0.0208% higher, well inside observed timing variation;
there is no measurable end-to-end speed difference in this setup. Process-level
medians, in execution order, were 7.52153, 7.50861, 7.45632 and 7.41876 seconds.
Generated token sequences were identical across the two launches of each mode.
No claim of identical outputs between BF16 and FP32 is made.

## Validation

- Configuration, policy validation and source-compatibility regressions: 86
  distinct CPU test cases, including seven dtype-alias cases added after the
  initial 79-case run.
- Native CUDA numerical tests: seven cases for BF16/FP16, strided and batched
  states, FP64 comparison, live weight updates, and quantized-head rejection.
- Two-GPU native LM-head test: 16 combinations covering BF16/FP16 output,
  FP32 output, divisible/padded vocabulary sizes and in-place weight reloads.
- NeMo RL worker smoke test: Qwen3-0.6B generated successfully with an FP32
  head and BF16 weights. All four Qwen3-8B runs verified those live dtypes.

The TP test is [`vllm_fp32_lm_head_tp.py`](../../tests/functional/vllm_fp32_lm_head_tp.py).
The precision unit tests are
[`test_vllm_fp32_lm_head.py`](../../tests/unit/models/generation/test_vllm_fp32_lm_head.py).
