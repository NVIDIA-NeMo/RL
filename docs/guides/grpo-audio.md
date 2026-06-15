# Audio Post-training with Nemo-RL

This guide explains how to use NeMo-RL to train [Qwen2.5-Omni](https://huggingface.co/Qwen) (3B or 7B) and [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) with GRPO on audio question-answering data, convert the resulting Megatron checkpoint to Hugging Face format, and evaluate it on the [MMAU benchmark](https://huggingface.co/datasets/TwinkStart/MMAU).

NeMo-RL ships three recipes out of the box, but the pieces are independent and can be mixed:

| Recipe | Model | Default dataset | Config |
| --- | --- | --- | --- |
| 3B (R1-AQA reproduction) | Qwen2.5-Omni-3B | [AVQA](https://mn.cs.tsinghua.edu.cn/avqa) | `examples/configs/audio_grpo_3B_megatron.yaml` |
| 7B (StrongAC + Gemini CoT) | Qwen2.5-Omni-7B | [Harland/AudioMCQ-StrongAC-GeminiCoT](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT) | `examples/configs/recipes/vlm/vlm_grpo-qwen2.5-omni-7b-audiomcq-1n8g-megatron.v1.yaml` |
| 30B MoE (Qwen3-Omni) | Qwen3-Omni-30B-A3B-Instruct | Harland/AudioMCQ-StrongAC-GeminiCoT | `examples/configs/recipes/vlm/vlm_grpo-qwen3-omni-30ba3b-audiomcq-4n8g-megatron.v1.yaml` |

The 3B recipe accepts the AudioMCQ dataset via a CLI override (no new YAML needed); the 7B and 30B recipes live under `examples/configs/recipes/vlm/` and inherit non-audio defaults from `grpo_math_1B_megatron.yaml`, redeclaring the audio-specific blocks inline. The 30B recipe targets the Qwen3-Omni MoE thinker on 4 × 8 H100/H200. You can swap models and datasets independently.

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

Because the upstream manifest only ships a native `train` split, the wrapper synthesizes a deterministic held-out validation slice from `split_validation_size` + `seed` — the same train-and-validate-from-train convention used by AVQA. The 7B and 30B yamls set `data.train.split_validation_size` to an absolute count (256, matching `grpo.max_val_samples`); the held-out rows are exposed as the validation set and excluded from training (no leakage), and no separate `data.validation` entry is needed.

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
    data.train.split_validation_size=256 \
    data.validation=null
```

(`data.validation=null` drops AVQA's native `split=validation` entry, which doesn't exist on the AudioMCQ manifest; `data.train.split_validation_size=256` makes the train wrapper auto-populate `val_dataset` with a held-out slice instead.)

### 7B (Qwen2.5-Omni-7B) — `vlm_grpo-qwen2.5-omni-7b-audiomcq-1n8g-megatron.v1.yaml`

```
uv run examples/run_vlm_grpo.py --config examples/configs/recipes/vlm/vlm_grpo-qwen2.5-omni-7b-audiomcq-1n8g-megatron.v1.yaml
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
    --config examples/configs/recipes/vlm/vlm_grpo-qwen2.5-omni-7b-audiomcq-1n8g-megatron.v1.yaml \
    grpo.max_num_steps=2 \
    checkpointing.enabled=false \
    logger.wandb_enabled=false
```

### 30B MoE (Qwen3-Omni-30B-A3B-Instruct) — `vlm_grpo-qwen3-omni-30ba3b-audiomcq-4n8g-megatron.v1.yaml`

```
uv run examples/run_vlm_grpo.py --config examples/configs/recipes/vlm/vlm_grpo-qwen3-omni-30ba3b-audiomcq-4n8g-megatron.v1.yaml
```

Key hyperparameters (sized for 4 × 8 × H100/H200 80 GB):

| Parameter | Value |
| --- | --- |
| Model | Qwen3-Omni-30B-A3B-Instruct (MoE thinker) |
| Dataset | Harland/AudioMCQ-StrongAC-GeminiCoT (`dataset_name: audiomcq`) |
| `cluster.num_nodes` × `gpus_per_node` | 4 × 8 |
| Megatron `tensor_model_parallel_size` / `expert_model_parallel_size` / `pipeline_model_parallel_size` | 1 / 8 / 1 |
| Megatron `moe_token_dispatcher_type` | allgather (matches EP≥16 large-MoE peers) |
| vLLM `tensor_parallel_size` / `expert_parallel_size` / `async_engine` | 4 / 8 / false |
| `train_global_batch_size` | 32 |
| `max_total_sequence_length` | 2048 |
| Learning rate | 1e-6 |
| Reward | format (0.2) + exact_alnum (0.8) |

The Qwen3-Omni recipe has two model-specific gotchas baked into the yaml:

- **Thinker-only training.** The Megatron `Qwen3OmniBridge` only converts the thinker (LLM + audio + vision encoders); talker / code2wav modules emit a one-line `talker/code2wav audio-output is not supported yet` warning at convert time and stay frozen at the original HF weights, so checkpoint conversion in §3 needs `--no-strict`.
- **vLLM `tensor_parallel_size: 4`, not 1.** With TP=1 + EP > 1, NeMo-RL's `VllmGenerationWorker` enters the `else` branch in `vllm_worker.py:431` (no `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`, no `VLLM_RAY_PER_WORKER_GPUS`); vLLM then auto-picks `RayDistributedExecutor` (because the worker actor itself runs inside a Ray actor) and `_init_workers_ray` blocks forever in `ray.get` waiting for a Ray sub-worker that has no GPU bundle to land on. TP=4 enters the `if model_parallel_size > 1` branch, which sets the per-worker GPU fraction so the sub-workers can co-tenant the parent actor's GPU bundle. TP must also divide the audio tower's 20 attention heads, which rules out TP=8.

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
- For **Qwen3-Omni-30B-A3B-Instruct**, `--no-strict` is mandatory (not optional) — talker / code2wav tensors live only in the original HF snapshot, so the bridge writes thinker-side shards and copies the rest verbatim. The converter prints `Warning: model-000{14,15}-of-00015.safetensors: missing N tensors ... still saved because strict=False`, which is expected.

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
