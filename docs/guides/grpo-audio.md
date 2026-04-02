# Audio GRPO on AVQA

This guide explains how to use NeMo RL to train [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) with GRPO on the [AVQA](https://huggingface.co/datasets/avqa) audio question-answering dataset, following the approach described in the [R1-AQA paper](https://arxiv.org/abs/2503.11197), and then evaluate the trained model on the [MMAU benchmark](https://huggingface.co/datasets/TwinkStart/MMAU).

## 1. Train the Model

Run GRPO training with the provided config:

```
uv run examples/run_vlm_grpo.py --config examples/configs/audio_grpo_3B_megatron.yaml
```

Config: `examples/configs/audio_grpo_3B_megatron.yaml`

Key hyperparameters:

| Parameter | Value |
| --- | --- |
| Model | Qwen2.5-Omni-3B |
| Dataset | AVQA (train split) |
| GPUs | 8 x 1 node, Megatron backend |
| Learning rate | 1e-6 |
| KL penalty | 0.01 |
| Generations per prompt | 8 |
| Prompts per step | 8 |
| Max steps | 200 |
| Save period | 100 |
| Reward | format (0.2) + exact_alnum (0.8) |

## 2. Convert Checkpoint (Megatron to HF)

Throughout training, checkpoints are saved to the `results/audio_grpo_3B_megatron` directory (specified by `checkpointing.checkpoint_dir`). To evaluate a checkpoint, first convert it from Megatron format to Hugging Face format:

```
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
    --config results/audio_grpo_3B_megatron/step_200/config.yaml \
    --megatron-ckpt-path results/audio_grpo_3B_megatron/step_200/policy/weights/iter_0000000 \
    --hf-ckpt-path results/audio_grpo_3B_megatron/step_200/hf --no-strict
```

Replace the step number with the checkpoint you want to evaluate. Note the `--extra mcore` flag is required for the Megatron converter.

## 3. Evaluate on MMAU

Evaluate the converted checkpoint on the [MMAU benchmark](https://huggingface.co/datasets/TwinkStart/MMAU):

```
uv run examples/run_eval.py \
    --config=examples/configs/evals/mmau.yaml \
    generation.model_name=results/audio_grpo_3B_megatron/step_200/hf \
    data.dataset_name=TwinkStart/MMAU
```

Config: `examples/configs/evals/mmau.yaml`

Use `generation.model_name` to specify the path to the converted Hugging Face checkpoint.

## 4. Results

Evaluating the step-200 checkpoint on MMAU, we get the following result:

```
============================================================
model_name='hf_iter_0000000' dataset_name='MMAU'
max_new_tokens=8000 temperature=0.0 top_p=1.0 top_k=-1 seed=42

metric=pass@1 num_tests_per_prompt=1

score=0.7210 (721.0/1000)
============================================================
```

As a reference, here are results comparing the baseline, the [R1-AQA](https://arxiv.org/abs/2503.11197) HuggingFace vanilla implementation, and NeMo-RL:

| Model | MMAU Score |
| --- | --- |
| Qwen2.5-Omni-3B (baseline) | 69.8 |
| Qwen2.5-Omni-3B + GRPO (HF vanilla) | 71.6 |
| Qwen2.5-Omni-3B + GRPO (NeMo-RL) | 72.1 |

The NeMo-RL result (72.1) is comparable to and slightly higher than the Huggingface Transformers reference implementation (71.6), confirming that the training pipeline reproduces expected improvements over the baseline.
