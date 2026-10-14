# SpeechLM GSPO from current source

For the v0.7.0 image workflow, use [SpeechLM Post-training with GSPO](gspo-speechlm.md). This page describes the separate current-source installation.

This recipe trains a SpeechLM from its full Hugging Face seed directory using
fractional WER/CER rewards and sequence-level GSPO. The native policy preserves
its audio encoder, frozen MTP head and checkpoint names. An explicit parameter
surface keeps selected optimizer masters in FP32 while frozen weights stay in
their declared storage dtype. Generation loads the seed before applying selected
weight refits, so frozen weights are never replaced by dummy initialization.

## Requirements

Use compatible NeMo RL, NeMo Speech and NeMo Automodel checkouts. Speech must
provide `SALMAutomodel(parameter_config=...)`, strict native checkpoint loading
and validated partial perception refits. Automodel must provide `ParameterConfig`,
`preserve_parameter_dtypes`, `compute_logits=False` and `compute_mtp=False`.
These companion contributions must land before this recipe can use an upstream
release; installing unrelated public-main revisions is not sufficient.

Use a CUDA environment matching the current RL installation guide (Python
3.13.14 and Torch 2.13). Set `RL`, `SPEECH`, `AUTOMODEL` and `SKILLS` to public
source checkouts containing the companion contributions. Before they merge,
use the companion PR heads. The supplied v0.7.0 image recipe has separate pins;
it is not a current-main environment.

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

The model path must be the **full Speech HF seed directory**, including config,
tokenizer and safetensors. Native DCP resume outputs are a separate format.
Use a local Parquet file with one recording per row:

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
and no reference leakage. Audio must be nonempty 16-kHz WAV; channels are averaged.
For Mandarin ordinary ASR, set `language="zh"`, `task="ordinary_asr"`,
`reward_metric="cer"` and `cer_whitespace_mode="han_adjacent"`. This route requires
a nonempty normalized reference. Other tasks use `reward_metric="wer"`; empty
references are valid for nonspeech.
A positive per-row output cap limits generation; a null cap retains the global
limit. Dataset preparation must establish any benchmark exclusion policy;
this recipe does not verify audio-level disjointness.

## Train and resume

Set the three paths below, then launch inside the normal four-node, 32-GPU Ray
allocation. The recipe's complete selected-name list describes the intended
Nemotron SpeechLM topology; setup rejects missing, extra or mismatched tensors.

```bash
export SPEECHLM_GSPO_MODEL=/path/to/full/speech-seed-hf
export SPEECHLM_GSPO_DATA=/path/to/consumer.parquet
export SPEECHLM_GSPO_OUTPUT_DIR=/path/to/fresh/output
uv run --no-project "$SPEECHLM_GSPO_PYTHON" examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml
```

Each offered batch has 40 recordings and 64 responses per recording. Dynamic
sampling retains the first 20 nonconstant groups in order. Advantages use a
leave-one-out mean and **full valid-group sample standard deviation**, with
`1e-6` stabilization. The new estimator option defaults off for existing recipes.
Native SpeechLM requires unpacked sequences, policy TP=CP=1 and synchronous
MTP-off generation. Router projection caching remains enabled during activation
checkpoint recomputation; auxiliary MTP computation is disabled without deleting
its frozen checkpoint state.

The recipe saves every 20 updates. Reusing its output directory restores the
latest finalized native optimizer, scheduler and dataloader state. Use a fresh
output directory for an independent run. For the separate 20-plus-fresh-2
qualification, run the matching nightly shell driver; it verifies saved state
before and after restarting the training process.

## Export and evaluate

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

`MODEL_CONFIG` is the compatible Speech experiment YAML containing `model` and
`trainer.strategy`, not the HF `config.json`. The exporter preserves intrinsic
FP32 storage through the owning adapter and applies canonical casts elsewhere;
FP32 optimizer masters do not imply lossless FP32 serving.

Evaluate the exported HF directory using the public NeMo Skills SpeechLM
serving path and unchanged benchmark manifests, prompts, decoder limits and
scorers. FullHF combines its SPGI/non-SPGI profiles; SOT uses cpWER, and MUSAN
requires all four RNG seeds. Preserve raw finish reasons and speech empty-output
checks alongside headline scores. Nonempty/character-rate MUSAN proxies do not
establish semantic hallucination quality.

## Validation scope

The main-based contribution has focused CPU tests for rewards, group statistics,
audio transport, native input expansion/logprobs, precision selection, checkpoint
loading, refits and export. The prior supplied-image 400-update campaign used
separate immutable source/package pins, including a router-cache repair from
update 281. It is evidence for those pins and does not qualify a fresh
current-main distributed CUDA/EP installation. GPU CI and the normal nightly
20-plus-fresh-2 test remain required for that environment.
