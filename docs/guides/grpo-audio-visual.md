# Audio+Video Intent GRPO on IntentTrain / IntentBench

This guide explains how to use NeMo RL to train [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) with GRPO on the [PhilipC/IntentTrain](https://huggingface.co/datasets/PhilipC/IntentTrain) audio-visual intent-recognition dataset and validate on [PhilipC/IntentBench](https://huggingface.co/datasets/PhilipC/IntentBench), following the dataset structure used in the [HumanOmniV2 reference](https://github.com/HumanMLLM/HumanOmniV2).

Each training sample feeds the Qwen2.5-Omni processor both the video stream (8 frames) and the audio track decoded from the same file at 16 kHz mono. Audio and video flow as two **independent multimodal items** per prompt: the dataset emits `{type: video}` + `{type: audio}` content items, the Qwen2.5-Omni chat template renders both `<|VIDEO|>` and `<|AUDIO|>` placeholders, and vLLM rollouts populate `multi_modal_data["video"]` and `multi_modal_data["audio"]` from the same sample. The explicit time-alignment hint `use_audio_in_video=True` is **not** used because the installed transformers + vLLM Qwen2.5-Omni stack rejected that path; both modalities still reach the model, just without that alignment hint.

## 1. Train the Model

Run GRPO training with the provided config:

```
uv run examples/run_vlm_grpo.py --config examples/configs/intent_grpo_7B_megatron.yaml
```

Config: `examples/configs/intent_grpo_7B_megatron.yaml`

Key hyperparameters:

| Parameter | Value |
| --- | --- |
| Model | Qwen2.5-Omni-7B |
| Train dataset | PhilipC/IntentTrain (problem_type = "multiple choice") |
| Validation dataset | PhilipC/IntentBench (problem_type = "multiple choice") |
| Modalities per prompt | video (8 frames, `<\|VIDEO\|>` placeholder) + audio (16 kHz mono, `<\|AUDIO\|>` placeholder) — independent multimodal items, no `use_audio_in_video` alignment |
| GPUs | 8 x 1 node, Megatron backend, `tensor_model_parallel_size=2` (data parallel = 4) |
| Learning rate | 1e-6 |
| KL penalty | 0.01 |
| Generations per prompt | 8 |
| Prompts per step | 32 |
| Train global / micro batch | 32 / 1 |
| Max steps | 1000 |
| Save period | 20 |
| Reward | format (0.2) + exact_alnum (0.8) |

The dataset class downloads `PhilipC/IntentTrain` and `PhilipC/IntentBench` via `huggingface_hub.snapshot_download` and extracts each `videos.zip` once into the corresponding HuggingFace cache directory. Re-instantiating the dataset on a machine that already has the archives extracted is a no-op.

Only `problem_type == "multiple choice"` samples are used. The allow-list is configurable through `data.train.allowed_problem_types` and `data.validation.allowed_problem_types` if you want to extend scope (for example, to `emer_ov_mc`); doing so requires picking an answer-correctness reward that handles those answer formats.

### 7B training notes

- **Per-forward batch must be exactly 1 sample/rank** (`train_micro_batch_size=1`, `logprob_batch_size=1`). Otherwise the Qwen2.5-Omni `get_rope_index` path crashes with `IndexError: index 1 is out of bounds for dimension 0 with size 1`. `train_global_batch_size=32` only sets gradient accumulation and must stay divisible by `micro × data_parallel_size` (32 % (1 × 4) == 0).
- **8 video frames** keep the prompt around ~4.5k tokens (8×360 video + ~1.5k audio + text), under `max_total_sequence_length=8192`, and roughly halve the training-forward activation memory versus 16 frames. Do **not** switch to fps-based sampling — at fps=2 the clips expand to ~43k video tokens, blow past the token budget, and `vlm_hf_data_processor` then empties the multimodal items and sets `loss_multiplier=0`.
- **`activation_checkpointing: true` + `gpu_memory_utilization: 0.4`** keep the Megatron forward inside the memory vLLM leaves resident after sleep mode. If `tensor_model_parallel_size=2` OOMs, fall back to `tensor_model_parallel_size=4` (proven to run at 8 frames).
- If `loss_multiplier` is logged at 0 for many samples, the multimodal prompt is exceeding `max_total_sequence_length`; bump it until validation samples consistently produce non-zero loss.
- Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` once `Qwen/Qwen2.5-Omni-7B`, `PhilipC/IntentTrain`, and `PhilipC/IntentBench` are pre-fetched, so Megatron's tokenizer worker doesn't hit the network.

## 2. Convert Checkpoint (Megatron to HF)

Checkpoints are saved under `results/intent_grpo_7B_megatron` (`checkpointing.checkpoint_dir`), one every `save_period=20` steps. Convert a checkpoint from Megatron to Hugging Face format before evaluating:

```
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
    --config results/intent_grpo_7B_megatron/step_43/config.yaml \
    --megatron-ckpt-path results/intent_grpo_7B_megatron/step_43/policy/weights/iter_0000000 \
    --hf-ckpt-path results/intent_grpo_7B_megatron/step_43/hf --no-strict
```

Replace the step number with the checkpoint you want to evaluate. `--no-strict` is expected here: only the Qwen2.5-Omni *thinker* is trained, so the talker tensors are reported as "not written". The `--extra mcore` flag is required for the Megatron converter.

## 3. Evaluate

In-training validation uses IntentBench as the validation set, so `val_period`, `val_batch_size`, and `max_val_samples` from the config drive evaluation cadence.

For a standalone benchmark, decode the converted HF checkpoint on [Daily-Omni](https://huggingface.co/datasets/liarliar/Daily-Omni) (1197 audio-visual multiple-choice questions) with `examples/run_eval.py`:

```
uv run examples/run_eval.py --config examples/configs/evals/daily_omni.yaml \
    generation.model_name=results/intent_grpo_7B_megatron/step_43/hf
```

The eval config (`examples/configs/evals/daily_omni.yaml`) feeds audio + video (32 frames — eval has no training-forward memory pressure, so it samples more densely than training), uses the same think+answer prompt as training, and scores with `exact_alnum` (case-insensitive exact match on the `<answer>` content).

## 4. Results

Daily-Omni accuracy (1197 questions, greedy decoding) for the base Qwen2.5-Omni-7B versus the GRPO-trained checkpoint:

| Question type | Base | After GRPO |
| --- | --- | --- |
| **Overall** | **0.498** | **0.590** |
| AV Event Alignment | 0.353 | 0.450 |
| Comparative | 0.618 | 0.725 |
| Context understanding | 0.446 | 0.534 |
| Event Sequence | 0.395 | 0.490 |
| Inference | 0.714 | 0.760 |
| Reasoning | 0.651 | 0.766 |

GRPO lifts overall Daily-Omni accuracy by ~9 points, with gains across every question category. The largest relative gains are on the reasoning-style questions; AV Event Alignment (which most depends on precise audio↔video synchronization) improves but remains the weakest category, consistent with the recipe not using the `use_audio_in_video` time-alignment path.
