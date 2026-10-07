# Lightning MXFP8 training and grouped GEMM A/B

This experiment inherits the matched GBS512, 8xGB200 Lightning recipes in
`../lightning_main_20261006/`. Every arm uses MXFP8 routed-expert rollout with
FlashInfer TRTLLM, first 2 and last 6 layers in BF16, and both policy and
reference logprobs. Async uses NCCL reshard refit; Sync uses colocated refit.

| Arm | Training storage/compute | Grouped expert path |
| --- | --- | --- |
| `bf16-storage` | BF16/BF16 | Existing recipe |
| `default` | MXFP8 params/MXFP8 routed experts | Existing recipe |
| `option-a` | Same as default | TE GroupedTensor, cuBLASLt candidate |
| `option-b` | Same as default | TE op-fuser with grouped tensor, CuTeDSL/cuDNN candidate |

The MXFP8 arms use #4353 to send logical BF16 refit weights; vLLM converts
them to MXFP8. The TE precision recipe keeps non-routed modules in BF16 and
inherits the outer model-init storage policy for first/last BF16 layers.

Option A must not run until the cuBLASLt fix for NVBUG 6815125 is verified in
the immutable container and recorded in `CUBLAS_GROUPED_GEMM_PATCH_RECORD`.
Option B additionally enables fused weighted squared ReLU because the model
uses `relu2`; without it, this MCore version rejects the op-fuser path.

Compare steps 2-20 of successful 20-step runs. Record source/image/submodule
SHAs, resolved config, W&B URL, generation KL/reward, policy/logprob/refit/
generation/E2E times and tok/s/GPU. Do not claim the requested cuDNN kernel is
active until a trace or TE kernel log confirms it.
