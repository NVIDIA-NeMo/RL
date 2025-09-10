# FP8 for NeMo-RL

This module provides a suite of tools to enable FP8 quantization for large language models. This module is still in developement. Currently we support 

* FP8 generation, using Deepseek style FP8 (sub channel scaling)
* FP8 training, using TransformerEngine as linear layer implementation, supporting Deepseek style FP8 (sub channel scaling) and per-tensor scaling

NeMo-RL monkey patches several vLLM functions to enable FP8 generations for reinforcement learning. The `init_fp8` function patches key `vLLM` components when initialized:
1.  **`RayDistributedExecutor`**: For multi-GPU inference, the executor is patched to ensure that every worker process applies the same FP8 patches before model initialization.
2.  **Quantization Utilities**: Functions within `vllm.model_executor.layers.quantization` are replaced with versions that support power-of-2 scaling and other custom features.
3.  **Weight Loading**: A custom `load_weights` function handles the on-the-fly quantization of model weights from a higher-precision format to FP8 with the correct scaling factors.

---

## Usage

FP8 generations are recommended to be configured with the following settings:

   ```
    loss_fn:
        # importance sampling helps improve stability
        use_importance_sampling_correction: true

    policy:
        generation:
            vllm_cfg:
                precision: 'fp8'
                # DeepGemm is much more performant than vLLM's default cutlass fp8 subchannel scaling kernels
                use_deep_gemm: true
                # Keeping the first and last three layers in bf16 reduces the multi-token error without
                # a signficant effect to performance
                num_last_layers_in_bf16: 3
                num_first_layers_in_bf16: 1
                # Use FP32 scaling factors. Rounding scaling factors to the nearest pow2 may improve quantization 
                # fidelity however this feature is still under research.
                use_weight_pow2_scale: False
                use_activation_pow2_scale: False
```

FP8 training requires megatron path, and is recommented to be configured with the following settings:

```
    policy:
        megatron_cfg:
            fp8_cfg:
                fp8: "hybrid"               # choices: [hybrid, e4m3]
                fp8_recipe: "tensorwise"    # choicse: [tensorwise, blockwise]
                fp8_param: false            # boolean value
```

### Special note with using FP8 training with Deepseek-style FP8 (sub channel scaling)*

The TransformerEngine implementation of this recipe requires cublas version >= 12.9; however, nemo-rl currently depends on torch 2.7.1 which depends on cuda 12.8; therefore, using the default way will cause the following error 
```
File "/opt/ray_venvs/nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker/lib/python3.12/site-packages/transformer_engine/pytorch/fp8.py", line 646, in fp8_autocast
FP8GlobalStateManager.fp8_autocast_enter(
File "/opt/ray_venvs/nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker/lib/python3.12/site-packages/transformer_engine/pytorch/fp8.py", line 465, in fp8_autocast_enter
assert fp8_block_available, reason_for_no_fp8_block
           ^^^^^^^^^^^^^^^^^^^
AssertionError: FP8 block scaled GEMM requires Hopper and CUDA >= 12.9.
```
The issue will be resolved when we bump torch version to >=2.8.0 in the future. For now, the following temporal solutions can be used to try Deepseek style FP8 training:
* Build the NGC pytorch based container from `docker/Dockerfile.ngc_pytorch`. In this way you will use the torch in system python environment, which has cuda version 12.9 or higher.


## Accuracy

We observe on the Llama 8b recipe a ~5% accuracy loss is incurred with FP8 generations. Convergence is still under active research and FP8 generations should be used with caution. We are investigating ways to close the accuracy gap and further improve performance. 
