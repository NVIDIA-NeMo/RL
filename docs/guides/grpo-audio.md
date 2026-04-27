# Audio GRPO on AVQA

This guide explains how to use NeMo RL to train [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) with GRPO on the [AVQA](https://mn.cs.tsinghua.edu.cn/avqa) audio question-answering dataset, following the approach described in the [R1-AQA paper](https://arxiv.org/abs/2503.11197), and then evaluate the trained model on the [MMAU benchmark](https://huggingface.co/datasets/TwinkStart/MMAU).

> **Alternative dataset**: see the [AudioMCQ-StrongAC-GeminiCoT](#audio-grpo-on-audiomcq-strongac-geminicot) section below for an alternative audio dataset wrapped under `dataset_name: audiomcq` plus a Qwen2.5-Omni-7B Megatron recipe.

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

## Audio GRPO on AudioMCQ-StrongAC-GeminiCoT

NeMo-RL also ships a wrapper for the [Harland/AudioMCQ-StrongAC-GeminiCoT](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT) dataset under the registry key `audiomcq`. The dataset is a curated subset of [inclusionAI/AudioMCQ](https://huggingface.co/datasets/inclusionAI/AudioMCQ) that is already filtered to the StrongAC partition (samples whose questions cannot be answered without analyzing the audio) and additionally restricted to rows whose Gemini chain-of-thought annotations passed quality review. It contains roughly 19,480 multiple-choice rows totalling ~9.72 GB across seven source folders (AudioCaps, SpeechCraft, CompA-R, Tacos, LP-MusicCaps-MTT, Clotho, MusicCaps).

### Asset prerequisites

The dataset snapshot ships every audio file inline next to the `data.jsonl` manifest, so a one-time `snapshot_download` is sufficient. The `AudioMCQDataset` wrapper performs an eager head-row asset probe at construction time, so a missing or partial snapshot fails fast with a clear error before any Ray actor or model spins up.

If the `Harland/AudioMCQ-StrongAC-GeminiCoT` snapshot is not yet cached, the loader will trigger the download automatically the first time it runs. To pre-stage:

```
huggingface-cli download Harland/AudioMCQ-StrongAC-GeminiCoT --repo-type=dataset
```

### Switching the existing 3B recipe to AudioMCQ

The existing `examples/configs/audio_grpo_3B_megatron.yaml` can be retargeted to AudioMCQ purely with command-line overrides — no new YAML file is required:

```
uv run examples/run_vlm_grpo.py \
    --config examples/configs/audio_grpo_3B_megatron.yaml \
    data.train.dataset_name=audiomcq \
    data.validation=null
```

### Training Qwen2.5-Omni-7B on AudioMCQ

NeMo-RL also ships a 7B Megatron recipe sized for a single node with 8 × H100/H200 80 GB:

```
uv run examples/run_vlm_grpo.py --config examples/configs/audio_grpo_7B_megatron.yaml
```

Config: `examples/configs/audio_grpo_7B_megatron.yaml`

Key hyperparameters:

| Parameter | Value |
| --- | --- |
| Model | Qwen2.5-Omni-7B |
| Dataset | Harland/AudioMCQ-StrongAC-GeminiCoT (`audiomcq`) |
| GPUs | 8 x 1 node, Megatron backend |
| Megatron `tensor_model_parallel_size` | 4 |
| vLLM `tensor_parallel_size` | 2 |
| `train_global_batch_size` | 16 |
| `train_micro_batch_size` | 1 |
| `max_total_sequence_length` | 2048 |
| Learning rate | 1e-6 |
| Reward | format (0.2) + exact_alnum (0.8) |

Use `grpo.max_num_steps`, `checkpointing.enabled`, and `logger.wandb_enabled` overrides to gate or shorten a run; for a quick smoke that exercises the dataset and processor plumbing:

```
uv run --no-sync examples/run_vlm_grpo.py \
    --config examples/configs/audio_grpo_7B_megatron.yaml \
    grpo.max_num_steps=2 \
    checkpointing.enabled=false \
    logger.wandb_enabled=false
```

The 7B recipe inherits its non-audio defaults from `grpo_math_1B_megatron.yaml` (rather than chaining through the 3B audio recipe) and explicitly redeclares the audio-specific vLLM, Megatron, processor, env, and reward settings inline so it stands alone.
