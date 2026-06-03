---
name: gsm8k-checkpoint-eval
description: Evaluate NeMo-RL/Nemotron diffusion or Ministral3 RL checkpoints on GSM8K or MATH500 by converting Megatron checkpoints to Hugging Face format, running eval_3b_checkpoint.sh, or running the NemoSkills eval pipeline from the nemo-rl clone.
---

# GSM8K / MATH500 Checkpoint Evaluation

Use this workflow when evaluating a saved NeMo-RL Megatron checkpoint on the SGLang GSM8K or MATH500 benchmark. The checkpoint must be converted to Hugging Face format before evaluation. The same scripts handle both benchmarks; choose with `BENCHMARK=gsm8k` (default) or `BENCHMARK=math`.

## Workflow

1. Work from the JustGRPO worktree:

```bash
cd /home/snorouzi/diffusion_RL/RL
```

2. Convert the checkpoint to HF format with `convert_checkpoint.sh`.

For native Nemotron diffusion checkpoints:

```bash
STEP_DIR=/path/to/run/checkpoints/step_N \
OUT=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
./convert_checkpoint.sh
```

For `Ministral3ForCausalLM` / `ministral3_ar` checkpoints, pass `--ministral3` so the Megatron-Bridge runtime patch is added:

```bash
STEP_DIR=/path/to/run/checkpoints/step_N \
OUT=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
./convert_checkpoint.sh --ministral3
```

The `--ministral3` flag is needed because the causal `Ministral3ForCausalLM` bridge is registered through the runtime patch directory.

3. Evaluate the converted HF checkpoint with `eval_3b_checkpoint.sh`.

### GSM8K (default)

```bash
MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
OUTDIR=/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/<name>_gsm8k \
./eval_3b_checkpoint.sh
```

### MATH500

```bash
BENCHMARK=math \
MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
OUTDIR=/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/<name>_math500 \
./eval_3b_checkpoint.sh
```

To submit either as a Slurm batch job, add `--sbatch`:

```bash
BENCHMARK=math \
MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_hf \
OUTDIR=/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/<name>_math500 \
./eval_3b_checkpoint.sh --sbatch
```

## NemoSkills eval path from nemo-rl

Use this path when you want to evaluate with the NemoSkills wrapper in the `nemo-rl` clone instead of the SGLang benchmark script. This starts the LLaDA/Nemotron API server, runs the NemoSkills GSM8K client against `localhost:8000/v1`, and writes a generated `.gpu_only_cmd_*.sh` script into the output directory for exact reproducibility.

If the input is a saved NeMo-RL/Megatron checkpoint such as `/path/to/run/checkpoints/step_N`, convert it to Hugging Face format before using this NemoSkills path. Follow step 2 in this same skill file (`/home/snorouzi/diffusion_RL/RL/skills/gsm8k-checkpoint-eval/SKILL.md`) and run `./convert_checkpoint.sh`; then set `SERVER_MODEL_PATH` to the converted HF output directory. Do not point `SERVER_MODEL_PATH` directly at the raw Megatron step directory unless intentionally using the DCP conversion flow below.

Work from the `nemo-rl` clone:

```bash
cd /home/snorouzi/code/yonggan-rl
```

### Diffusion / Nemotron decoding

For the Nemotron-Labs-Diffusion-3B base checkpoint on GSM8K with full evaluation:

```bash
ACCOUNT=coreai_dlalgo_llm \
SERVER_PARTITION=batch \
SERVER_TIME=04:00:00 \
SERVER_GPUS=8 \
SERVER_MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B \
SERVER_TOKENIZER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B \
SERVER_ENGINE=nemotron \
SEQ_EVAL_BENCHMARK=gsm8k:1 \
SEQ_EVAL_EXPNAME=baseline_3b_nemoskills_diffusion_steps1024_full \
SEQ_EVAL_OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_results/nemorskills_baseline_3b_diffusion_steps1024_full_$(date +%Y%m%d_%H%M%S) \
SEQ_EVAL_GENERATION_ALGORITHM=nemotron \
SEQ_EVAL_TOKENS_TO_GENERATE=1024 \
SEQ_EVAL_STEPS=1024 \
SEQ_EVAL_BLOCK_LENGTH=32 \
SEQ_EVAL_TEMPERATURE=0 \
SEQ_EVAL_EXTRA_ARGS="--exclude-unfinished-nfe false" \
bash xp/examples/run_llada_eval_pipeline_gpu_only.sh
```

For a smoke test, keep the same command but add `--max-samples 16`:

```bash
SEQ_EVAL_EXTRA_ARGS="--max-samples 16 --exclude-unfinished-nfe false"
```

### AR-native decoding

For AR-native evaluation, switch both the server engine and NemoSkills generation algorithm:

```bash
SERVER_ENGINE=ar_native \
SEQ_EVAL_GENERATION_ALGORITHM=ar_native \
SEQ_EVAL_STEPS=64 \
bash xp/examples/run_llada_eval_pipeline_gpu_only.sh
```

Keep the same `SERVER_MODEL_PATH`, `SERVER_TOKENIZER`, benchmark, output directory, and token settings as above unless intentionally changing the model or benchmark.

### DCP / trained checkpoint eval

For a NeMo-RL DCP checkpoint, pass the DCP path and base model. The pipeline converts the DCP checkpoint to a temporary HF directory before starting the server:

```bash
SERVER_DCP_PATH=/path/to/run/checkpoints/step_N \
SERVER_BASE_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B \
SERVER_MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B \
SERVER_TOKENIZER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B \
bash xp/examples/run_llada_eval_pipeline_gpu_only.sh
```

If the trained checkpoint was already converted to HF, use that HF directory directly as `SERVER_MODEL_PATH` and omit `SERVER_DCP_PATH`.

For NemoSkills evals, prefer the `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/...` checkpoint path when it exists. We have seen the eval container fail to load an otherwise valid checkpoint from the shorter `/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/...` path with an error like "Can't load the configuration". The `/lustre/fs1/.../projects/...` path is the path family used by the successful NemoSkills runs.

Also make sure custom HF remote-code files are real files visible inside the container, not symlinks to an unmounted checkpoint directory. For Ministral/Nemotron diffusion checkpoints, verify at least these files exist in `SERVER_MODEL_PATH` before submitting:

```bash
ls -l "${SERVER_MODEL_PATH}"/config.json \
      "${SERVER_MODEL_PATH}"/configuration_ministral_dlm.py \
      "${SERVER_MODEL_PATH}"/modeling_ministral_dlm.py
```

If `configuration_ministral_dlm.py` or `modeling_ministral_dlm.py` is a symlink, either copy the real files into the HF checkpoint directory or use a `basefiles`/non-symlink checkpoint copy. Broken container-visible symlinks show up as missing remote-code files during model load.

NemoSkills eval notes:

- `SERVER_ENGINE=nemotron` corresponds to the diffusion/Nemotron server path; `SERVER_ENGINE=ar_native` corresponds to native AR decoding.
- `SEQ_EVAL_GENERATION_ALGORITHM` must match the intended server-side mode (`nemotron` or `ar_native`) so the OpenAI-compatible request body carries the right extra fields.
- `SEQ_EVAL_TOKENS_TO_GENERATE` is the maximum generated length. For GSM8K, `1024` is the usual setting used in these runs.
- `SEQ_EVAL_STEPS=1024` was used for full diffusion-style Nemotron decoding; `SEQ_EVAL_STEPS=64` was used for AR-native NemoSkills evals.
- The generated command script is saved as `${SEQ_EVAL_OUTPUT_DIR}/.gpu_only_cmd_*.sh`; use that file as the source of truth for rerunning an exact completed eval.
- The server worker logs are saved under `${SEQ_EVAL_OUTPUT_DIR}/worker_logs`.

## Notes

- `STEP_DIR` must point at a checkpoint step directory containing `config.yaml` and `policy/weights/iter_0000000`.
- `OUT` is the converted HF model directory passed later as `MODEL`.
- `eval_3b_checkpoint.sh` accepts `MODEL`, `OUTDIR`, and `BENCHMARK` as environment variables.
- `BENCHMARK` values: `gsm8k` (default) or `math` (= the HuggingFaceH4/MATH-500 test set). Anything else fails fast.
- `MAX_TOKENS` defaults: 1024 for GSM8K, 2048 for MATH500. MATH500 solutions are longer; override with `MAX_TOKENS=...` if needed.
- The Slurm job name defaults to `eval_3b_instruct_${BENCHMARK}`; override with `SBATCH_JOB_NAME=...`.
- The underlying evaluator is `benchmark/gsm8k/eval_sglang.py` in the SGLang fork, which dispatches on `--benchmark gsm8k|math`.
- For diffusion-style evaluation, the converted HF checkpoint should use a diffusion-compatible HF template/base model and the eval script should run with its diffusion settings. The default decoding mode is FastDiffuser regardless of how the checkpoint was trained — see [[feedback-eval-decoding-mode]].

## MATH500 dataset prefetch (one-time)

The container runs with `HF_HUB_OFFLINE=1`, so `HuggingFaceH4/MATH-500` must be in the shared HF cache before the first MATH500 eval. From the login node:

```bash
HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home \
HF_TOKEN=$(cat ~/hf_token.txt) \
/lustre/fsw/portfolios/coreai/users/snorouzi/sglang_nemotron_torch291_cu129_uvpy312_venv/bin/python \
  -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/MATH-500', split='test')"
```

Cache persists across runs; only needed once per dataset. `openai/gsm8k` is already cached.
