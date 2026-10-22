# SpeechLM post-training with GSPO

This guide describes the full-scale ASR post-training recipe exercised for 400
updates on an internal 20,000-recording dataset. It accepts a user-supplied
Parquet manifest with the same public schema; the internal dataset is not
distributed. The code and installation instructions here target current public
source checkouts and independently prepared policy and generation environments.

This recipe trains a SpeechLM from its full Hugging Face seed directory using
fractional WER/CER rewards and sequence-level GSPO. The native policy preserves
its audio encoder, frozen MTP head and checkpoint names. An explicit parameter
surface keeps selected optimizer masters in FP32 while frozen weights stay in
their declared storage dtype. Generation loads the seed before applying selected
weight refits, so frozen weights are never replaced by dummy initialization.

## Requirements

Use compatible NeMo RL, NeMo Speech and NeMo Automodel checkouts. Speech must
provide component `automodel_freeze_config` and `perception_freeze_config`
settings, strict native checkpoint loading and validated partial perception
refits. Automodel must provide exact `FreezeConfig.unfreeze_parameters`
selection, pre-shard frozen-storage preparation, `compute_logits=False` and
`compute_mtp=False`.
These come from companion pull requests
([Speech #16387](https://github.com/NVIDIA-NeMo/Speech/pull/16387),
[Automodel #4377](https://github.com/NVIDIA-NeMo/Automodel/pull/4377) and
[Skills #1539](https://github.com/NVIDIA-NeMo/Skills/pull/1539)); earlier
revisions and current releases do not provide them.

Use a CUDA environment matching the current RL installation guide (Python
3.13.14 and Torch 2.13). Set `RL`, `SPEECH`, `AUTOMODEL` and `SKILLS` to `main`
checkouts that include the companion pull requests. Obtain the four packages
from their public repositories:

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git
git clone https://github.com/NVIDIA-NeMo/Speech.git
git clone https://github.com/NVIDIA-NeMo/Automodel.git
git clone https://github.com/NVIDIA-NeMo/Skills.git
export RL="$PWD/RL" SPEECH="$PWD/Speech" AUTOMODEL="$PWD/Automodel" SKILLS="$PWD/Skills"
```

Record each resolved commit in the run artifacts. Image-compatible branches are
maintained separately; their package versions are not interchangeable with this
current-source installation.

Native policy workers use the driver's prepared interpreter. Existing Automodel
workers keep their pinned extra and submodule; vLLM keeps its own environment.
Prepare packages once at paths visible on every Ray node:

```bash
export SPEECHLM_GSPO_VENV=/path/to/venvs/speechlm-policy
export SPEECHLM_GSPO_PYTHON="$SPEECHLM_GSPO_VENV/bin/python"
export NEMO_RL_VENV_DIR=/path/to/venvs/rl-actors

# Install Automodel's declared native CUDA/MoE dependencies from its checkout.
UV_PROJECT_ENVIRONMENT="$SPEECHLM_GSPO_VENV" uv sync \
  --project "$AUTOMODEL" --python 3.13.14 --extra moe --extra fa
uv pip install --python "$SPEECHLM_GSPO_PYTHON" \
  -e "$RL" -e "$SPEECH[speechlm2]" -e "$SKILLS[core]" \
  -r "$SKILLS/requirements/audio.txt"

# Keep generation on RL's independently locked vLLM stack.
GENERATION_ENV="$NEMO_RL_VENV_DIR/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
UV_PROJECT_ENVIRONMENT="$GENERATION_ENV" uv sync --locked \
  --project "$RL" --extra vllm --extra nemo_gym
uv pip install --python "$GENERATION_ENV/bin/python" \
  -e "$SPEECH[asr]" 'transformers==5.12.1' 'torch==2.13.0' 'torchvision==0.28.0'
```

Use the CUDA/compiler prerequisites in the [installation guide](../about/installation.md)
and each dependency's build instructions. The policy and generation processes
must have matching Torch/CUDA ABIs for colocated refits. Check the resulting
package set with `uv pip check --python <interpreter>` for each interpreter.
Do not enable the global `NEMO_RL_PY_EXECUTABLES_SYSTEM` override: it would also
move generation into the policy environment. The ASR reward worker runs CPU
text alignment in the driver environment; it needs no judging model or API.

These are source-install instructions for the main contribution. Fresh installation
and the full distributed recipe need the GPU qualification below; the focused
local tests do not establish a complete CUDA dependency installation.

## Data and seed

Use the **full Hugging Face snapshot** of
`nvidia/Nemotron-3.5-Transcribe-30B-A4B`, including config, tokenizer, backbone
configuration and safetensors. Pin the model revision when downloading:

```bash
export SPEECHLM_GSPO_MODEL=/shared/models/Nemotron-3.5-Transcribe-30B-A4B
HF_HUB_DISABLE_XET=0 hf download nvidia/Nemotron-3.5-Transcribe-30B-A4B \
  --revision 471d3b578578edef8c1ed919190140ef7c1f6d8d \
  --local-dir "$SPEECHLM_GSPO_MODEL"
```

Authenticate using your approved Hugging Face credentials if access is required.
The same recipe also accepts compatible full HF seeds without a CTC head, as in
the original internal-data campaign. The optional bundled CTC timestamp decoder
remains frozen. Its original floating
storage, serialized tokenizer buffer and safetensors metadata survive native
resume and canonical HF export; checkpoints without this artifact retain their
ordinary strict core-loading path.
Freezing the head does not guarantee unchanged timestamp accuracy: the selected
audio-encoder layers still change. Timestamp quality needs its own evaluation.
Native DCP resume outputs are a separate format.
Use one recording per row in a local Parquet file. Keep the selected dataset's
row order, task balance, reference text, and user-prompt variation. The processor
uses each row's instruction, including task-specific SOT instructions; it does
not substitute one global ASR prompt.

A minimal row is:

```python
{
    "messages": [{"role": "user", "content": "Transcribe the audio."}],
    "audio_filepath": "/data/audio/example.wav",
    "instruction": "Transcribe the audio.",
    "expected_answer": "The reference transcript.",
    "language": "en",
    "task": "ordinary_asr",
    "reward_metric": "wer",
    "responses_create_params": {"max_output_tokens": None},
}
```

Preserve the `messages` column for `ResponseDataset` passthrough. The ASR
processor builds the actual request with text before audio, no system prompt
and no reference leakage. Audio must be nonempty, sampled at 16 kHz, and readable by SoundFile; channels are averaged.
For Mandarin ordinary ASR, set `language="zh"`, `task="ordinary_asr"`,
`reward_metric="cer"` and `cer_whitespace_mode="han_adjacent"`. This route requires
a nonempty normalized reference. Other tasks use `reward_metric="wer"`; empty
references are valid for nonspeech. Rewards are fractional `1 - error_rate`, not percentages and not clipped to
zero. WER rewards use case/punctuation normalization that preserves Indic
combining marks; they are not interchangeable with downstream benchmark scores.
For an empty WER reference, the error is the number of normalized hypothesis
words, so an empty hypothesis earns 1 and extra words reduce the reward.

A positive per-row output cap limits generation; a null cap retains the global
limit. Dataset preparation must establish any benchmark exclusion policy;
this recipe does not verify audio-level disjointness.

## Train and resume

Set the three paths below, then launch from the RL checkout inside a four-node,
32-GPU Ray allocation (eight GPUs per node), with shared model/data/output paths.
Start the Ray cluster through your scheduler using the normal
[cluster setup](../cluster.md) instructions. The recipe's complete selected-name list describes the intended
Nemotron SpeechLM topology; setup rejects missing, extra or mismatched tensors.

```bash
export SPEECHLM_GSPO_MODEL=/shared/models/Nemotron-3.5-Transcribe-30B-A4B
export SPEECHLM_GSPO_DATA=/path/to/consumer.parquet
export SPEECHLM_GSPO_OUTPUT_DIR=/path/to/fresh/output
cd "$RL"
uv run --no-project "$SPEECHLM_GSPO_PYTHON" examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml
```

The full configuration is
[`vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml).
Its inherited defaults and overrides resolve the following settings:

| Setting | Value |
| --- | --- |
| Updates | 400, with at most 2 passes through the ordered dataset |
| Offered recordings / retained groups | 40 / 20; dynamic-sampling refill threshold 2 |
| Responses per recording | 64; temperature 0.8 |
| Learning rate | 1e-6 after a 5-update linear warmup from 0.1 of the target LR |
| Update surface | 269 named tensors, 1,074,318,400 scalars; FP32 storage |
| Frozen components | Remaining backbone/perception weights, MTP, and optional CTC head |
| Training / generation parallelism | Policy TP=CP=1, EP=8; generation TP=8, EP=8 |
| Context / global output limits | 62,464 / 17,278 tokens, further bounded by per-row caps and remaining context |
| Checkpoint cadence / retention | Every 20 updates / 20 native checkpoints |

The update surface is configured by `policy.native_model.trainable_names`,
`expected_trainable_tensors`, and `expected_trainable_scalars`; the complete list
is in the recipe. No CTC parameter is selected. Keep the tested list when changing
only the dataset or substituting a checkpoint of the same architecture.
The native policy translates this list into Automodel's existing `FreezeConfig`:
freeze all component modules, then unfreeze the exact selected parameter names.
The loader uses FP32 storage; the BF16 FSDP2 compute policy reduces ordinary
frozen storage before allocation while retaining model-required FP32 tensors.
Exact-name, count and storage validation runs before the optimizer is created.

Each offered batch has 40 recordings and 64 responses per recording. Dynamic
sampling retains the first 20 nonconstant groups in order. It does not rank
valid groups by reward spread. At 400 completed updates, 8,000 groups and 512,000
responses contribute to training, with possible repetition of dataset rows;
additional offered groups can be discarded or trigger a refill. With the current
loop's threshold of 2, an insufficient initial batch can trigger two refills
(three offered batches total); training fails if those cannot supply 20 valid
groups. This threshold is unchanged from the tested recipe. Advantages use a
leave-one-out mean and **full valid-group sample standard deviation**, with
`1e-6` stabilization. The new estimator option defaults off for existing recipes.
Native SpeechLM requires unpacked sequences, policy TP=CP=1 and synchronous
MTP-off generation. Router projection caching remains enabled during activation
checkpoint recomputation; auxiliary MTP computation is disabled without deleting
its frozen checkpoint state.

When auditing retained rows in `train_data_step*.jsonl`, use `filtered_rewards`.
With dynamic sampling enabled, raw `rewards` contains the unfiltered batch and
is not aligned with retained `source_idx` or content.

The recipe saves every 20 updates. Reusing its output directory restores the
latest finalized native optimizer, scheduler and dataloader state. Use a fresh
output directory for an independent run. For the separate 20-plus-fresh-2
qualification, run the [nightly shell driver](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/test_suites/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.sh); it verifies saved state
before and after restarting the training process.

## Export and evaluate

First generate the export configuration from the original local HF seed. This
includes the CTC construction metadata when present; no private experiment YAML
is needed. Keep this seed available for its tokenizer and backbone config.

```bash
export MODEL_CONFIG=/shared/results/speechlm-export-config.json
export EXPORT_OUTPUT=/shared/results/speechlm-hf
"$SPEECHLM_GSPO_PYTHON" "$SPEECH/examples/speechlm2/prepare_rl_export_config.py" \
  --seed "$SPEECHLM_GSPO_MODEL" --output "$MODEL_CONFIG"
```

In an eight-GPU allocation with the prepared policy environment, run Speech's
canonical converter:

```bash
uv run --no-project "$SPEECHLM_GSPO_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "$SPEECH/examples/speechlm2/to_hf.py" \
  class_path=nemo.collections.speechlm2.models.SALMAutomodel \
  checkpoint_format=nemo_rl \
  ckpt_path="$SPEECHLM_GSPO_OUTPUT_DIR/checkpoints/step_400/policy/weights/model" \
  ckpt_config="$MODEL_CONFIG" output_dir="$EXPORT_OUTPUT"
```

`MODEL_CONFIG` is the compatible Speech configuration file containing `model` and
`trainer.strategy`; the YAML loader also accepts the helper's JSON output. It is
not the HF `config.json`. The exporter preserves intrinsic
FP32 storage through the owning adapter, the CTC artifact's original storage and
metadata, and applies canonical casts elsewhere;
FP32 optimizer masters do not imply lossless FP32 serving.

Evaluate the exported HF directory using the public NeMo Skills SpeechLM
serving path and unchanged benchmark manifests, prompts, decoder limits and
scorers. FullHF combines its SPGI/non-SPGI profiles; SOT uses cpWER, and MUSAN
requires all four RNG seeds. Preserve raw finish reasons and speech empty-output
checks alongside headline scores. Nonempty/character-rate MUSAN proxies do not
establish semantic hallucination quality.

## Validation and measured scope

The completed internal-data campaign reproduced the original recipe closely on
its matched evaluations:

| Metric (lower is better) | Original 400-update model | Reproduced 400-update model |
| --- | ---: | ---: |
| FullHF macro WER | 4.040% | 4.042% |
| ML7 equal-language mean WER | 4.902% | 4.910% |
| SOT cpWER | 12.828% | 12.828% |
| Monsoon WER | 3.542% | 3.548% |

These measurements use the internal seed/data and separately pinned runtime,
not the public HF seed with arbitrary replacement data. Noise suppression was
weaker in the reproduction (MUSAN noise nonempty rate 40.134% to 43.952%); matched
regression evaluation remains necessary. The run used a supported router
activation-checkpoint caching correction for updates 281–400; the final recipe
sets it explicitly from the start. It was not independently rerun for 400 updates
with that setting throughout.

Optional CTC persistence was additionally exercised through 20 real updates,
fresh-process resume, and HF export: all 38 CTC state entries and metadata were
preserved, with the same 269-tensor update surface. That checks preservation,
not post-RL timestamp accuracy.


The main-based contribution has focused CPU tests for rewards, group statistics,
audio transport, native input expansion/logprobs, precision selection, checkpoint
loading, refits and export. Earlier distributed campaigns used separate immutable source/package pins.
They do not qualify a fresh current-main distributed CUDA/EP installation. GPU CI and the normal nightly
20-plus-fresh-2 test remain required for that environment.
