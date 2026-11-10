# SpeechLM GSPO

GSPO ASR training of native NeMo Speech SALM checkpoints, such as
[Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B),
with NeMo RL's Automodel policy and vLLM generation. The project uses only
NeMo RL extension points:

- `speechlm_gspo.model.SpeechLMPolicy` wraps Speech's `SALMAutomodel`. It is
  registered in `AUTOMODEL_FACTORY` for `model_type: nemo_speechlm` and
  configured through `policy.automodel_cfg.automodel_kwargs.native_model`
  (exact trainable tensor names and the expanded audio context limit). Its
  `forward` expands audio placeholders and returns logits aligned to the text
  tokens, so NeMo RL's logprob and loss code runs unchanged.
- The checkpoint's processor class is registered with `AutoProcessor`; audio
  features are computed inside the model.
- The ASR data processor and the WER/CER reward environment are registered as
  `asr` with `register_processor` and `register_env`. Mandarin `ordinary_asr`
  rows use NeMo Skills' audio CER.
- Policy and generation workers are selected with `worker_extension_cls_fqn`
  and run in the separately locked environments below.

## Environments

| Actor | Project | Key pins |
| --- | --- | --- |
| Driver | `research/speechlm_gspo` (root workspace member) | root `uv.lock` |
| Policy, ASR reward | `envs/policy` | Speech `732041d29`, Automodel `752a71eb`, Skills `5ea1990a`, transformers 5.18.0, TransformerEngine 2.19.0, Torch 2.13.0 |
| vLLM generation | `envs/generation` | NeMo RL `vllm` extra (vLLM 0.29.0, transformers 5.12.1) and Speech `732041d29` |

Speech and Automodel main require transformers and TransformerEngine versions
outside the root lock, so `envs/` carry their own locks. Their Torch, CUDA
libraries and indexes follow the root `pyproject.toml`. Actors build their
virtual environments on first use under `NEMO_RL_VENV_DIR`. The first policy
build compiles TransformerEngine's PyTorch extension, mamba-ssm,
causal-conv1d and DeepEP, so point `NEMO_RL_VENV_DIR` at a persistent
directory visible on every node. An existing venv is reused without syncing and
installs this checkout in editable mode, so reuse it only from the same code
path and set `NRL_FORCE_REBUILD_VENVS=true` after moving the code, changing a
lock, or a failed build.

Relock an environment with the uv version of the NeMo RL image:

```bash
cd research/speechlm_gspo/envs/policy && uv lock
```

## Run

Training rows are Parquet records with `audio_filepath` (16 kHz),
`prompt` (the instruction), `expected_answer` (the reference), and a nonempty
`messages` column holding the user prompt without the reference. Without
`messages`, `ResponseDataset` rewrites each row and drops the other columns.
Optional `language`, `task`, `reward_metric` and `cer_whitespace_mode` fields
select the reward metric (see `speechlm_gspo/reward.py`).

```bash
export SPEECHLM_GSPO_MODEL=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
export SPEECHLM_GSPO_DATA=/path/to/train.parquet
export SPEECHLM_GSPO_OUTPUT_DIR=/path/to/output
export NEMO_RL_VENV_DIR=/shared/path/to/venvs

cd research/speechlm_gspo
uv run --locked run_speechlm_gspo.py \
  --config configs/recipes/gspo-nemotron-3.5-transcribe-30b-a4b-4n8g-automodel-ep8.yaml
```

The recipe uses 4 nodes with 8 GPUs. Add Hydra overrides such as
`grpo.max_num_steps=2 checkpointing.enabled=false` for a short smoke run.
`tests/test_suites/gspo-nemotron-3.5-transcribe-30b-a4b-4n8g-automodel-ep8.sh`
trains 20 steps, verifies the checkpoint with `tools/verify_resume.py`, and
resumes for two more steps.

## Tests

```bash
cd research/speechlm_gspo
# CPU tests in the driver environment; Speech-dependent tests are skipped.
uv run --locked --group test pytest tests/unit
# All tests, including the model and Mandarin CER tests, in the policy environment.
uv run --locked --project envs/policy --with pytest pytest tests/unit
```
