# Lightning MXFP8 training and grouped GEMM A/B

This experiment inherits the matched GBS512, 8xGB200 Lightning recipes in
`../lightning_main_20261006/`. Every arm uses MXFP8 routed-expert rollout with
FlashInfer TRTLLM, first 2 and last 6 layers in BF16, and both policy and
reference logprobs. Async uses NCCL reshard refit; Sync uses colocated refit.

| Arm | Training storage/compute | Grouped expert path |
| --- | --- | --- |
| `bf16-storage` | BF16/BF16 | Existing recipe |
| `default` | MXFP8 params/MXFP8 routed experts | Existing recipe |
| `option-a` | Same as default | TE 2.18 grouped-tensor/cuBLASLt path via the legacy environment switch |
| `option-b` | Same as default | TE op-fuser with grouped tensor, CuTeDSL/cuDNN candidate |

The MXFP8 arms use #4353 to send logical BF16 refit weights; vLLM converts
them to MXFP8. The TE precision recipe keeps non-routed modules in BF16 and
inherits the outer model-init storage policy for first/last BF16 layers.

Option A is explicitly an unpatched diagnostic at the user's request. NVBUG
6815125 describes possible silent output corruption, so its timing is not a
usable recommendation without a numerical comparison against the control.
The pinned TE 2.18 `GroupedLinear` has no `use_grouped_tensor` constructor
argument: the literal MCore flag fails during model initialization. The TE
2.18 equivalent is `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1` with
`moe_use_grouped_tensor=false`; this selects the grouped-tensor GEMM for
supported MXFP8 inputs on Blackwell. It does not use single grouped-parameter
storage. Option B inherits the base recipe directly, so it cannot inherit the
Option A environment switch.
Option B additionally enables fused weighted squared ReLU because the model
uses `relu2`; without it, this MCore version rejects the op-fuser path.

Compare steps 2-20 of successful 20-step runs. Record source/image/submodule
SHAs, resolved config, W&B URL, generation KL/reward, policy/logprob/refit/
generation/E2E times and tok/s/GPU. Do not claim the requested cuDNN kernel is
active until a trace or TE kernel log confirms it.
