# FSDP2 Parallel Plan

This guide outlines the parallelization strategy for Fully Sharded Data Parallel version 2 (FSDP2) training in NeMo RL.

## Fallback Priority

NeMo RL supports three parallelization strategies, applied in the following order of fallback priority:

### 1. Custom Parallel Plan

Your user-defined custom parallel plans always take precedence when available. For detailed implementation and usage, refer to the [Custom Parallel Plan Example](#custom-parallel-plan-example).

### 2. Optimized Parallel Plan

Optimized parallel plans are available for specific model architectures. They may offer superior performance compared to Hugging Face's tensor parallel implementation. This approach is used if no custom parallel plan is specified and the model class supports optimized parallelization.

### 3. Hugging Face Tensor Parallel Plan

The Hugging Face tensor parallel plan is the default. It's available for most models via `._tp_plan` and is used when neither a custom nor an optimized parallel plan is available.

## Custom Parallel Plan Example

A custom parallel plan should be defined in a separate file, such as the example provided in `examples/custom_parallel.py`.

To implement the custom parallel plan, either update the value of `custom_parallel_plan` in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_grpo_math.py \
    policy.dtensor_cfg.custom_parallel_plan=examples.custom_parallel.custom_parallel_plan
```
