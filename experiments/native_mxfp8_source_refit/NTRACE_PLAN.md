# Native MXFP8 Policy ntrace

## Question

Why does Nemotron3 Nano policy training with native MXFP8 parameters improve
less than expected over BF16 training?

## Matched arms

Both arms use the same model, dataset, rollout precision, NCCL Reshard refit,
parallelism, and generated-token workload.

| Arm | Policy training | Rollout | Parameter storage |
| --- | --- | --- | --- |
| Control | BF16 | MXFP8 | BF16 |
| Candidate | MXFP8 | MXFP8 | native MXFP8 (`fp8_param=true`) |

The Nano policy topology is TP2, PP2, CP2, EP8 on 16 policy GPUs. Another 16
GPUs run disaggregated rollout. The final eight policy layers stay in BF16 in
the candidate arm.

## Capture

- Warm up policy updates 0 and 1.
- Capture updates 2 through 4 with ntrace.
- Start with global policy rank 0. Expand to all 16 policy ranks after the
  rank-0 artifact gate passes.
- Record unlimited Python stack depth, NVTX ranges, CUDA graph provenance, and
  three complete policy updates.

## Required gates

1. The resolved config and worker log both show `fp8_param=true` for the
   candidate arm and disabled training FP8 for the control arm.
2. Each traced rank writes non-empty records and stack Parquet files.
3. ntrace closes the requested three-update window without a lifecycle error.
4. `llm-analyzer` parses each capture and reports policy GPU time by GEMM,
   communication, quantization/layout, routing, optimizer, and other kernels.
5. Comparisons use active GPU time per generated/training token when the two
   runs process different token counts.
