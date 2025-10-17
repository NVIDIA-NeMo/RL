# Evaluation

We provide evaluation tools to assess model capabilities.

## Convert Model Format (Optional)

If you have trained a model and saved the checkpoint in the PyTorch DCP format, you first need to convert it to the Hugging Face format before running evaluation:

```sh
# Example for a GRPO checkpoint at step 170
uv run python examples/converters/convert_dcp_to_hf.py \
    --config results/grpo/step_170/config.yaml \
    --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
    --hf-ckpt-path results/grpo/hf
```

If you have a model saved in Megatron format, you can use the following command to convert it to Hugging Face format prior to running evaluation. This script requires Megatron Core, so make sure you launch with the mcore extra:

```sh
# Example for a GRPO checkpoint at step 170
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
    --config results/grpo/step_170/config.yaml \
    --megatron-ckpt-path results/grpo/step_170/policy/weights/iter_0000000 \
    --hf-ckpt-path results/grpo/hf
```

```{note}
Adjust the paths according to your training output directory structure.
```

For an in-depth explanation of checkpointing, refer to the [Checkpointing documentation](../design-docs/checkpointing.md).

## Run Evaluation

Run the evaluation script with the converted model:

```sh
uv run python examples/run_eval.py generation.model_name=$PWD/results/grpo/hf
```

Run the evaluation script with custom settings:

```sh
# Example: Evaluation of DeepScaleR-1.5B-Preview on MATH-500 using 8 GPUs
#          Pass@1 accuracy averaged over 16 samples for each problem
uv run python examples/run_eval.py \
    --config examples/configs/evals/math_eval.yaml \
    generation.model_name=agentica-org/DeepScaleR-1.5B-Preview \
    generation.temperature=0.6 \
    generation.top_p=0.95 \
    generation.vllm_cfg.max_model_len=32768 \
    data.dataset_name=math500 \
    eval.num_tests_per_prompt=16 \
    cluster.gpus_per_node=8
```

```{note}
Evaluation results may vary slightly due to various factors, such as sampling parameters, random seed, inference engine version, and inference engine settings.
```

Refer to `examples/configs/evals/eval.yaml` for a full list of parameters that can be overridden. For an in-depth explanation of evaluation, refer to the [Evaluation documentation](../guides/eval.md).

