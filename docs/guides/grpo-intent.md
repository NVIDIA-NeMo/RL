# Audio+Video Intent GRPO on IntentTrain / IntentBench

This guide explains how to use NeMo RL to train [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) with GRPO on the [PhilipC/IntentTrain](https://huggingface.co/datasets/PhilipC/IntentTrain) audio-visual intent-recognition dataset and validate on [PhilipC/IntentBench](https://huggingface.co/datasets/PhilipC/IntentBench), following the joint audio+video setup used in the [HumanOmniV2 reference](https://github.com/HumanMLLM/HumanOmniV2).

Each training sample feeds the Qwen2.5-Omni processor both the video stream (16 frames) and the audio track decoded from the same file at 16 kHz mono. The recipe sets `use_audio_in_video=True` on the HuggingFace processor and on every vLLM rollout request so audio and video tokens are aligned.

## 1. Train the Model

Run GRPO training with the provided config:

```
uv run examples/run_vlm_grpo.py --config examples/configs/intent_grpo_3B_megatron.yaml
```

Config: `examples/configs/intent_grpo_3B_megatron.yaml`

Key hyperparameters:

| Parameter | Value |
| --- | --- |
| Model | Qwen2.5-Omni-3B |
| Train dataset | PhilipC/IntentTrain (problem_type = "multiple choice") |
| Validation dataset | PhilipC/IntentBench (problem_type = "multiple choice") |
| Modalities per prompt | video (16 frames) + audio (16 kHz mono, joint via `use_audio_in_video=True`) |
| GPUs | 8 x 1 node, Megatron backend |
| Learning rate | 1e-6 |
| KL penalty | 0.01 |
| Generations per prompt | 8 |
| Prompts per step | 8 |
| Max steps | 1000 |
| Save period | 400 |
| Reward | format (0.2) + exact_alnum (0.8) |

The dataset class downloads `PhilipC/IntentTrain` and `PhilipC/IntentBench` via `huggingface_hub.snapshot_download` and extracts each `videos.zip` once into the corresponding HuggingFace cache directory. Re-instantiating the dataset on a machine that already has the archives extracted is a no-op.

Only `problem_type == "multiple choice"` samples are used in v1. The allow-list is configurable through `data.train.allowed_problem_types` and `data.validation.allowed_problem_types` if you want to extend scope (for example, to `emer_ov_mc`); doing so requires picking an answer-correctness reward that handles those answer formats.

## 2. Convert Checkpoint (Megatron to HF)

Throughout training, checkpoints are saved to the `results/intent_grpo_3B_megatron` directory (specified by `checkpointing.checkpoint_dir`). To evaluate a checkpoint, first convert it from Megatron format to Hugging Face format:

```
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
    --config results/intent_grpo_3B_megatron/step_400/config.yaml \
    --megatron-ckpt-path results/intent_grpo_3B_megatron/step_400/policy/weights/iter_0000000 \
    --hf-ckpt-path results/intent_grpo_3B_megatron/step_400/hf --no-strict
```

Replace the step number with the checkpoint you want to evaluate. Note the `--extra mcore` flag is required for the Megatron converter.

## 3. Evaluate

In-training validation uses IntentBench as the validation set, so `val_period`, `val_batch_size`, and `max_val_samples` from the config drive evaluation cadence. A standalone `examples/run_eval.py` flow for IntentBench is intentionally out of scope for this recipe in v1 — extend `nemo_rl/data/datasets/eval_datasets/` and add an eval YAML if you want one.

## 4. Results

This guide ships as a starting point for audio+video GRPO on IntentTrain/IntentBench. The recipe is exercised end to end (load → rollout → reward → checkpoint → validation) but does not commit to a particular IntentBench accuracy target — IntentBench's evaluation methodology and any published numerical comparison are out of scope for this recipe. Use the validation reward and answer-correctness reward signal in the wandb / tensorboard logs to track training progress.

If `loss_multiplier` is logged at 0 for many samples, the multimodal prompt is exceeding `policy.max_total_sequence_length` (default 8192 in this recipe) and the truncation branch in `vlm_hf_data_processor` is masking those samples out. Bump `max_total_sequence_length` until validation samples consistently produce non-zero loss.
