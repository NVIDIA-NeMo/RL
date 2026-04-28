# Audio Post-training with Nemo-RL

This guide explains how to use NeMo-RL to train [Qwen2.5-Omni](https://huggingface.co/Qwen) (3B or 7B) with GRPO on audio question-answering data, convert the resulting Megatron checkpoint to Hugging Face format, and evaluate it on the [MMAU benchmark](https://huggingface.co/datasets/TwinkStart/MMAU).

NeMo-RL ships two recipes out of the box, but the pieces are independent and can be mixed:

| Recipe | Model | Default dataset | Config |
| --- | --- | --- | --- |
| 3B (R1-AQA reproduction) | Qwen2.5-Omni-3B | [AVQA](https://mn.cs.tsinghua.edu.cn/avqa) | `examples/configs/audio_grpo_3B_megatron.yaml` |
| 7B (StrongAC + Gemini CoT) | Qwen2.5-Omni-7B | [Harland/AudioMCQ-StrongAC-GeminiCoT](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT) | `examples/configs/audio_grpo_7B_megatron.yaml` |

The 3B recipe accepts the AudioMCQ dataset via a CLI override (no new YAML needed), and the 7B recipe is standalone (it inherits non-audio defaults from `grpo_math_1B_megatron.yaml` rather than chaining through the 3B audio recipe), so you can swap models and datasets independently.

## 1. Datasets

### AVQA

The [Audio-Visual Question Answering (AVQA)](https://mn.cs.tsinghua.edu.cn/avqa) dataset is the original training corpus from the [R1-AQA paper](https://arxiv.org/abs/2503.11197). NeMo-RL exposes it under the registry key `audiomcq`'s sibling, `dataset_name: avqa`, and pulls the pre-processed split from [`gijs/avqa-processed`](https://huggingface.co/datasets/gijs/avqa-processed) on the Hub. AVQA has native `train` and `validation` splits, so the 3B recipe references both directly.

### AudioMCQ-StrongAC-GeminiCoT

[Harland/AudioMCQ-StrongAC-GeminiCoT](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT) is a curated subset of [inclusionAI/AudioMCQ](https://huggingface.co/datasets/inclusionAI/AudioMCQ) released alongside the [AudioMCQ work](https://arxiv.org/abs/2509.21060). Two filters are applied upstream:

- **StrongAC partition** — only samples where ≥ 2 LALMs answered incorrectly when the audio was silenced, i.e. questions that genuinely require listening to the audio.
- **Gemini chain-of-thought review** — only rows whose Gemini-generated CoT annotations passed quality review (no hallucinations / invalid reasoning).

It contains ~19,480 multiple-choice rows totalling ~9.72 GB across seven source folders (AudioCaps, SpeechCraft, CompA-R, Tacos, LP-MusicCaps-MTT, Clotho, MusicCaps). The snapshot ships every audio file inline next to a `data.jsonl` manifest, so a one-time `snapshot_download` is sufficient — no per-source corpus fetch is needed.

NeMo-RL exposes the dataset under `dataset_name: audiomcq`. The wrapper performs an eager head-row asset probe at construction time, so a missing or partial snapshot fails fast with a clear error before any Ray actor or model spins up. To pre-stage:

```
huggingface-cli download Harland/AudioMCQ-StrongAC-GeminiCoT --repo-type=dataset
```

Because the upstream manifest only ships a native `train` split, the wrapper synthesizes a deterministic validation slice from `split_validation_size` + `seed`. The 7B yaml uses an explicit `data.validation` block plus `populate_val_dataset: false` on `data.train` so the train slice excludes the held-out rows (no leakage) and the validation pool is not double-counted (no duplication).

## 2. Train

### 3B (Qwen2.5-Omni-3B) — `audio_grpo_3B_megatron.yaml`

```
uv run examples/run_vlm_grpo.py --config examples/configs/audio_grpo_3B_megatron.yaml
```

Key hyperparameters:

| Parameter | Value |
| --- | --- |
| Model | Qwen2.5-Omni-3B |
| Dataset | AVQA (`dataset_name: avqa`) |
| GPUs | 8 × 1 node, Megatron backend |
| Learning rate | 1e-6 |
| KL penalty | 0.01 |
| Generations per prompt | 8 |
| Prompts per step | 8 |
| Reward | format (0.2) + exact_alnum (0.8) |

To retarget the 3B recipe to AudioMCQ purely via CLI overrides:

```
uv run examples/run_vlm_grpo.py \
    --config examples/configs/audio_grpo_3B_megatron.yaml \
    data.train.dataset_name=audiomcq \
    data.validation=null
```

(`data.validation=null` is required because AVQA's native `split=validation` doesn't exist on the AudioMCQ manifest. The train wrapper's auto-populated `val_dataset` provides the held-out rows instead.)

### 7B (Qwen2.5-Omni-7B) — `audio_grpo_7B_megatron.yaml`

```
uv run examples/run_vlm_grpo.py --config examples/configs/audio_grpo_7B_megatron.yaml
```

Key hyperparameters (sized for 1 × 8 × H100/H200 80 GB):

| Parameter | Value |
| --- | --- |
| Model | Qwen2.5-Omni-7B |
| Dataset | Harland/AudioMCQ-StrongAC-GeminiCoT (`dataset_name: audiomcq`) |
| Megatron `tensor_model_parallel_size` | 4 |
| vLLM `tensor_parallel_size` | 2 |
| `train_global_batch_size` | 32 |
| `train_micro_batch_size` | 1 |
| `logprob_batch_size` | 1 (TP ≥ 4 requires `train_micro_batch_size == logprob_batch_size`) |
| `max_total_sequence_length` | 2048 |
| Learning rate | 1e-6 |
| Reward | format (0.2) + exact_alnum (0.8) |

For a quick smoke run that exercises the dataset and processor plumbing without committing to a long run:

```
uv run --no-sync examples/run_vlm_grpo.py \
    --config examples/configs/audio_grpo_7B_megatron.yaml \
    grpo.max_num_steps=2 \
    checkpointing.enabled=false \
    logger.wandb_enabled=false
```

## 3. Convert checkpoint (Megatron → HF)

Throughout training, checkpoints are saved under `${checkpointing.checkpoint_dir}` (`results/audio_grpo_3B_megatron/` for 3B, `results/audio_grpo_7B_megatron/` for 7B). To evaluate a checkpoint, first convert it from Megatron format to Hugging Face format:

```
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
    --config <ckpt_dir>/step_<N>/config.yaml \
    --megatron-ckpt-path <ckpt_dir>/step_<N>/policy/weights/iter_0000000 \
    --hf-ckpt-path <ckpt_dir>/step_<N>/hf --no-strict
```

Notes:

- Replace `<ckpt_dir>` and `<N>` with your run's checkpoint directory and step.
- `--extra mcore` is required for the Megatron converter.
- If the converter hits a Hugging Face Hub `429 Too Many Requests` while fetching tokenizer metadata, prepend `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` so it uses the cached snapshot only.

## 4. Evaluate on MMAU

Evaluate the converted HF checkpoint on the [MMAU benchmark](https://huggingface.co/datasets/TwinkStart/MMAU):

```
uv run examples/run_eval.py \
    --config=examples/configs/evals/mmau.yaml \
    generation.model_name=<ckpt_dir>/step_<N>/hf \
    data.dataset_name=TwinkStart/MMAU
```

Config: `examples/configs/evals/mmau.yaml` (vLLM colocated, bf16, 8k context). For 7B add `generation.vllm_cfg.tensor_parallel_size=2 cluster.gpus_per_node=2` so the weights fit.

## 5. Results

### 3B + AVQA (R1-AQA reproduction)

Evaluating `audio_grpo_3B_megatron / step_200`:

```
============================================================
model_name='hf_iter_0000000' dataset_name='MMAU'
max_new_tokens=8000 temperature=0.0 top_p=1.0 top_k=-1 seed=42
metric=pass@1 num_tests_per_prompt=1
score=0.7210 (721.0/1000)
============================================================
```

| Model | MMAU pass@1 |
| --- | --- |
| Qwen2.5-Omni-3B (baseline) | 69.8 |
| Qwen2.5-Omni-3B + GRPO (HF vanilla, R1-AQA) | 71.6 |
| Qwen2.5-Omni-3B + GRPO (NeMo-RL) | **72.1** |

The NeMo-RL number is comparable to and slightly higher than the Hugging Face Transformers reference implementation, confirming that the pipeline reproduces the expected improvement over baseline.

### 7B + AudioMCQ

Early sanity check at `audio_grpo_7B_megatron / step_50`:

```
============================================================
model_name='hf' dataset_name='MMAU'
max_new_tokens=8000 temperature=0.0 top_p=1.0 top_k=-1 seed=42
metric=pass@1 num_tests_per_prompt=1
score=0.7250 (725.0/1000)
============================================================
```

This is a short-run snapshot meant to validate the dataset + recipe path.
