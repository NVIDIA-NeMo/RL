# SpeechLM Post-training with GSPO

This guide describes how to post-train a Nemotron SpeechLM model for automatic
speech recognition with Group Sequence Policy Optimization (GSPO), resume a run,
and export the trained checkpoint for inference. The recipe uses the
DTensor/Automodel training backend, vLLM rollouts, and transcript error-rate rewards.

| Component | Configuration |
| --- | --- |
| Model | Compatible Nemotron SpeechLM 30B-A3B Hugging Face checkpoint |
| Training resources | 4 nodes, 8 GPUs per node |
| Dataset | User-provided Parquet manifest and shared audio files |
| Reward | Word error rate; character error rate for Mandarin ASR |
| Recipe | [SpeechLM GSPO YAML](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml) |

## 1. Prepare the code and environment

Use `nvcr.io/nvidia/nemo-rl:v0.7.0` with the compatible SpeechLM revisions of
[NeMo RL](https://github.com/NVIDIA-NeMo/RL),
[NeMo Speech](https://github.com/NVIDIA-NeMo/Speech),
[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel), and
[NeMo Skills](https://github.com/NVIDIA-NeMo/Skills).

> [!NOTE]
> This guide accompanies the SpeechLM recipe changes under review. Those changes
> are not yet published in the upstream repositories. The revision variables
> below must identify releases or review branches containing the coordinated
> changes; the base container or an arbitrary upstream checkout is insufficient.

From a shared source directory, check out those revisions:

```bash
: "${NEMO_RL_REF:?Set the SpeechLM-enabled NeMo RL revision}"
: "${SPEECH_REF:?Set the matching Speech revision}"
: "${AUTOMODEL_REF:?Set the matching Automodel revision}"
: "${SKILLS_REF:?Set the matching Skills revision}"

git clone https://github.com/NVIDIA-NeMo/RL.git nemo-rl
git -C nemo-rl checkout "$NEMO_RL_REF"
git -C nemo-rl submodule update --init --recursive
git clone https://github.com/NVIDIA-NeMo/Speech.git speech
git -C speech checkout "$SPEECH_REF"
git clone https://github.com/NVIDIA-NeMo/Skills.git skills
git -C skills checkout "$SKILLS_REF"

export RL_REPO="$(realpath nemo-rl)"
export SPEECH_REPO="$(realpath speech)"
export SKILLS_REPO="$(realpath skills)"
export AUTOMODEL_REPO="$RL_REPO/3rdparty/Automodel-workspace/Automodel"
git -C "$AUTOMODEL_REPO" fetch origin "$AUTOMODEL_REF"
git -C "$AUTOMODEL_REPO" checkout "$AUTOMODEL_REF"
```

Install these packages in the container's **separate policy and generation
environments** using [SpeechLM environment setup](../recipes/speechlm-environment.md).
Run the setup on every node before starting Ray, or bake the same installation
into a reusable image. Installing packages only in the driver environment does
not configure the Ray actors. Keep the image's compiled Torch, CUDA,
Transformer Engine, DeepEP, and vLLM packages.

The [cluster guide](../cluster.md) describes the standard multi-node launcher.
Make source checkouts, the model, and audio available at the same absolute paths
on all nodes. Mount model and dataset inputs read-only and give the run its own
writable checkpoint, log, and dataset-cache directories. Audio stays on shared
storage; the manifest contains paths to it.

## 2. Prepare the model and dataset

### Model

Start from a complete SpeechLM Hugging Face checkpoint containing the model
configuration, tokenizer, and weights. The recipe's parameter names and expected
tensor counts target this model architecture. A native distributed checkpoint
is used for **resuming RL**, rather than as the initial HF model path.

```bash
export SPEECHLM_GSPO_MODEL=/shared/models/speechlm-seed
export SPEECHLM_GSPO_DATA=/shared/data/asr/train.parquet
export SPEECHLM_GSPO_OUTPUT_DIR=/shared/results/speechlm-gspo
export HF_DATASETS_CACHE="$SPEECHLM_GSPO_OUTPUT_DIR/dataset-cache"
mkdir -p "$SPEECHLM_GSPO_OUTPUT_DIR" "$HF_DATASETS_CACHE"
```

### Manifest format

Each Parquet row describes one utterance. The audio must be nonempty, sampled at
16 kHz, and readable by SoundFile. Multichannel audio is averaged to mono. The
processor does not resample or crop the recording.

| Field | Requirement | Purpose |
| --- | --- | --- |
| `id` or `cut_id` | Recommended | Stable utterance identifier |
| `audio_filepath` or `audio_path` | Required | Audio file path visible to every worker |
| `prompt` or `instruction` | Required, nonempty string | User instruction |
| `expected_answer` or `reference` | Required string | Reference transcript; may be empty for nonspeech |
| `messages` | Required column | Preserves metadata through `ResponseDataset`; use a user-message list |
| `language` | Recommended; required for Mandarin CER | Language identifier, such as `en`, `hi`, `nl`, or `zh` |
| `task` | Required for Mandarin CER | Use `ordinary_asr` for ordinary transcription |
| `reward_metric` | Optional, defaults to `wer` | `wer`, or `cer` for Mandarin ordinary ASR |
| `cer_whitespace_mode` | Required for Mandarin CER | `han_adjacent` |
| `responses_create_params` | Optional | Per-row `max_output_tokens`; null uses the global cap |

For example, the following creates a manifest illustrating the schema. Replace
the examples with a training corpus before launching a run:

```python
import pandas as pd

prompt = "Transcribe the audio into text."
rows = [
    {
        "id": "en-0001",
        "audio_filepath": "/shared/audio/en-0001.wav",
        "prompt": prompt,
        "expected_answer": "the next train arrives at nine",
        "messages": [{"role": "user", "content": prompt}],
        "language": "en",
        "task": "ordinary_asr",
        "reward_metric": "wer",
        "cer_whitespace_mode": None,
        "responses_create_params": {"max_output_tokens": None},
    },
    {
        "id": "zh-0001",
        "audio_filepath": "/shared/audio/zh-0001.wav",
        "prompt": prompt,
        "expected_answer": "下一班火车九点到达",
        "messages": [{"role": "user", "content": prompt}],
        "language": "zh",
        "task": "ordinary_asr",
        "reward_metric": "cer",
        "cer_whitespace_mode": "han_adjacent",
        "responses_create_params": {"max_output_tokens": None},
    },
]
pd.DataFrame(rows).to_parquet("train.parquet", index=False)
```

Run manifest preparation in a CPU environment with pandas and a Parquet engine.
Keep `messages` in the saved table: without it, `ResponseDataset` reformats the
rows and removes the audio/reward columns. The ASR processor builds the actual
model request from `prompt` and the waveform, independently of stored messages:
one user message, **instruction before audio**, no system message. References
are used only for reward calculation. A supplied `system_prompt` must be empty;
`input_order`, if supplied, must be `text_before_audio`.

Prompt variation is configured per row. For serialized multi-speaker
transcription, use the matching speaker-tag instruction and reference format;
its training reward is WER on the serialized text. Nonspeech uses an empty
reference and WER metadata, not the Mandarin ordinary-ASR CER path. Keep genuine
speech examples in the dataset so suppressing all output is not rewarded
throughout the corpus.

The recipe uses the manifest's row order, with shuffling disabled and seed 42.
It does not mine data, create a validation split, or exclude evaluation audio
automatically. Prepare the training/evaluation separation before training.

### Reward and group selection

For nonempty references, the reward is `1 - error_rate`, with error rates
expressed as fractions. A WER of 0.1 gives reward 0.9; insertions can produce
negative rewards. WER normalization casefolds and separates punctuation while
preserving combining marks, apostrophes, and underscores. It does not expand
numbers or apply benchmark-specific normalization. Mandarin ordinary ASR uses
NeMo Skills CER with standard normalization and Han-adjacent whitespace handling.
For an empty WER reference, reward is `1 - number_of_hypothesis_words`.

Each update initially offers **40 utterances and generates 64 responses per
utterance**. Groups with identical rewards are discarded. The first 20 remaining
groups, in input order, form the training batch: 1,280 responses per update.
Groups are not ranked by reward spread. The standard dynamic-sampling loop
draws additional batches when needed and fails if its refill allowance cannot
produce a complete batch. Its configuration is
`grpo.dynamic_sampling_max_gen_batches: 2`.

Advantages use a leave-one-out mean and the full group's sample standard
deviation. GSPO uses sequence-level importance ratios and sequence-level loss
reduction. A group needs varying rewards to contribute learning signal; a
perfect transcript is not required in every group.

## 3. Train

The principal settings are defined in the recipe YAML:

| Setting | Value |
| --- | --- |
| Optimizer updates | 400 |
| Selected utterances / responses per utterance | 20 / 64 |
| AdamW peak LR / weight decay | `1e-6` / `0.01` |
| LR schedule | 5-update linear warmup from 0.1×, then constant |
| GSPO clipping / reference KL penalty | 0.2 / 0 |
| Microbatch / policy expert parallelism | 1 / 8 |
| vLLM tensor / expert parallelism | 8 / 8 |
| Sampling temperature | 0.8 |
| Total sequence limit / maximum generated tokens | 62,464 / 17,278 |
| Checkpoint cadence / retained checkpoints | Every 20 updates / 20 |

The output cap is a ceiling, not a target. A positive row cap can lower it;
generation is also bounded by the available context. Audio-expanded sequence
length must fit the model's total sequence limit.

### Trainable parameters

`policy.native_model.trainable_names` selects the update surface **before
optimizer construction**. The supplied list contains 269 tensors
(1,074,318,400 parameters), whose master weights are FP32. All other tensors
remain frozen. The configured tensor/scalar counts detect unintended changes
to that selection. Perception dropout and augmentation are disabled for
consistent policy/rollout conditioning. MTP weights remain frozen; MTP auxiliary
loss and speculative decoding are disabled in this recipe.

Run the following **once on the head of an initialized four-node Ray cluster**,
using the generation environment configured in step 1:

```bash
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_DISABLE_QWENVL_VISION_PATCH=1
export NRL_NSYS_WORKER_PATTERNS=
export VLLM_USE_DEEP_GEMM=0
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 NO_VCS_VERSION=1
unset PYTHONPATH

cd "$RL_REPO"
uv run --no-project --python "$GENERATION_PYTHON" \
  examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml
```

The image actor environments are reused intentionally; the environment setup
pins their packages independently of the source-checkout fingerprint. A generic
`uv run` that resolves a new project environment is a different setup.

For Slurm, save the driver block above as `/shared/scripts/train-speechlm.sh`
and the environment page's shell blocks as
`/shared/scripts/setup-speechlm-env.sh`, both starting with `set -euo pipefail`.
Export the model/data/output and repository variables from the preceding steps
and submit through the [standard `ray.sub` launcher](../cluster.md):

```bash
cd "$RL_REPO"
export CONTAINER=/shared/containers/nemo-rl-v0.7.0.sqsh
export MOUNTS=/shared:/shared
export SETUP_COMMAND='bash /shared/scripts/setup-speechlm-env.sh'
export COMMAND='bash /shared/scripts/train-speechlm.sh'
sbatch --nodes=4 --gres=gpu:8 --account=YOUR_ACCOUNT \
  --partition=YOUR_PARTITION --time=4:00:00 \
  --job-name=speechlm-gspo ray.sub
```

This example assumes the checkouts and all inputs/outputs are under `/shared`;
adjust mounts for your filesystem. The setup script runs separately on every
node. Define `GENERATION_PYTHON` in the submission environment as on the
environment page; variables exported by a child setup script are not inherited
by the later driver script. The recipe's checkpoint time budget is 3 hours
20 minutes and may stop a segment before all 400 updates complete.

### Resume and monitor

Run the same command with the same output directory to resume the latest valid
native checkpoint, including optimizer, scheduler, and dataloader state.
`grpo.max_num_steps` is the **global** update target, not the number of additional
updates. Use a fresh output directory for an independent experiment.

TensorBoard logs are under `$SPEECHLM_GSPO_OUTPUT_DIR/logs`; checkpoints are under
`checkpoints/step_<N>`. Monitor error-rate rewards, valid groups, generated-token
lengths, gradient norm, LR, and rollout/update timing. Compare throughput together
with token workload; long rollouts can dominate step time.

An optional [20-update plus two-update resume test](../../tests/test_suites/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.sh)
checks continuation mechanics. Set `SPEECHLM_GSPO_PYTHON` to
`$GENERATION_PYTHON` when using that test. It is a separate test, with its own
output directory; the training command above runs the 400-update recipe directly.

## 4. Export a checkpoint

Use the canonical exporter in NeMo Speech with an eight-GPU allocation and the
policy environment. `SPEECH_EXPORT_CONFIG` must be the compatible Speech
experiment YAML/JSON containing `model` and `trainer.strategy`; it is not the
HF `config.json`. Any model asset paths referenced by that configuration must
be available during conversion.

```bash
export SPEECH_EXPORT_CONFIG=/shared/configs/speechlm-export.yaml
export SPEECH_HF_OUTPUT=/shared/results/speechlm-gspo-hf-step400

cd "$AUTOMODEL_REPO"
uv run --no-project --python "$POLICY_PYTHON" \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  "$SPEECH_REPO/examples/speechlm2/to_hf.py" \
  class_path=nemo.collections.speechlm2.models.SALMAutomodel \
  checkpoint_format=nemo_rl \
  ckpt_path="$SPEECHLM_GSPO_OUTPUT_DIR/checkpoints/step_400/policy/weights/model" \
  ckpt_config="$SPEECH_EXPORT_CONFIG" output_dir="$SPEECH_HF_OUTPUT"
```

Native checkpoints preserve training and optimizer precision. HF export and
vLLM serving apply their supported dtype conversions; FP32 training masters do
not imply every serving tensor remains FP32. Keep the native checkpoint for
further RL and the HF export for inference.

## 5. Evaluate

Evaluate the seed and exported checkpoints with the same inference and scoring
configuration, using [NeMo Skills](https://github.com/NVIDIA-NeMo/Skills) and the
SpeechLM vLLM plugin. Pin the evaluator revision, prompts, text/audio ordering,
decoding settings, length limits, and text normalization for both models.

Use held-out English and multilingual ASR, multi-speaker transcription, and
nonspeech audio. Report WER/CER for speech, a speaker-aware metric for
multi-speaker transcription, and nonempty-output rates for nonspeech. Inspect
empty speech outputs and truncation as well as aggregate error rates. Training
reward WER and benchmark WER can use different normalization; report which
scorer produced each result.

The recipe disables in-training validation (`grpo.val_period: 0`), so run these
evaluations on saved checkpoints. Compare the seed, intermediate checkpoints,
and final checkpoint to detect regressions as well as improvements.
