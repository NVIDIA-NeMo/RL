---
name: gsm8k-checkpoint-eval
description: Evaluate native Nemotron Diffusion NeMo-RL Megatron checkpoints on math benchmarks such as GSM8K and AIME by converting them to Hugging Face format, then running the yonggon-rl NemoSkills pipeline or the standalone No-Ray NeMoRL-validation replication evaluator.
---

# Math Checkpoint Evaluation

Use this workflow for native Nemotron Diffusion NeMo-RL Megatron checkpoints. The evaluation has two major steps:

1. Convert the raw Megatron checkpoint step directory to Hugging Face format.
2. Evaluate the converted Hugging Face checkpoint on GSM8K, AIME, or related math benchmarks with the standalone evaluator or the `yonggon-rl` NemoSkills pipeline.

Do not run conversion directly on the login node. Run conversion inside the container specified below so CUDA, Transformer Engine, cuDNN, and C++ runtime dependencies are consistent.

## Step 1: Convert Checkpoint

Required inputs:

- `STEP_DIR`: raw checkpoint step directory containing `config.yaml` and `policy/weights/iter_0000000`.
- `OUT`: destination Hugging Face checkpoint directory.

Use this container for conversion:

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh
```

Use the repo wrapper for conversion. Do not inline or recreate the converter command from `tools/nemotron_diffusion/convert_nemotron_diffusion_checkpoint_to_hf.sh`; `convert_checkpoint.sh` owns the native Nemotron Diffusion conversion settings.

For diffuGRPO/native NeMo-RL checkpoints, set the same uv tag/env/cache family used by training. `convert_checkpoint.sh` does not derive these paths from `ENV_TAG` by itself, so pass both `ENV_TAG` and the `NEMOTRON_UV_*` overrides explicitly:

```bash
cd /home/snorouzi/diffusion_RL/RL

ENV_TAG=mb_3rdparty_sglagn_local_fork \
UV_PROJECT_ENVIRONMENT=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_diffu_grpo \
UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache_diffu_grpo \
NEMOTRON_UV_PROJECT_ENVIRONMENT=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_mb_3rdparty_sglagn_local_fork \
NEMOTRON_UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache_diffu_grpo \
NEMOTRON_HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home \
STEP_DIR=/path/to/run/checkpoints/step_N \
OUT=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_step_N_hf \
./convert_checkpoint.sh
```

When running under Slurm, load the container above and run the same wrapper command inside `/home/snorouzi/diffusion_RL/RL`. Example:

```bash
srun --container-image=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh \
  --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
  --container-workdir=/home/snorouzi/diffusion_RL/RL \
  bash -lc 'export ENV_TAG=diffu_grpo
export UV_PROJECT_ENVIRONMENT=/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_diffu_grpo
export UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache_diffu_grpo
export NEMOTRON_UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT
export NEMOTRON_UV_CACHE_DIR=$UV_CACHE_DIR
export NEMOTRON_HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home
STEP_DIR=/path/to/run/checkpoints/step_N OUT=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_step_N_hf ./convert_checkpoint.sh'
```

Agents should not generate a full custom conversion script unless the user explicitly asks for one.

After conversion succeeds, verify the Hugging Face directory contains at least:

```bash
ls -l "${OUT}"/config.json \
      "${OUT}"/model.safetensors \
      "${OUT}"/tokenizer.json \
      "${OUT}"/configuration_ministral_dlm.py \
      "${OUT}"/modeling_ministral_dlm.py
```

## Step 2: Evaluate Converted Checkpoint

Default evaluation must be standlone No-Ray NemoRL validation.

### 1. Standalone No-Ray NeMoRL Validation Replication

Use this path when the user asks to reproduce the NeMoRL validation schedule, compare against training-time validation, or avoid Ray while keeping the NeMoRL prompt/generation/grading logic. Prefer the checked-in submit wrapper; do not reconstruct the full Slurm command by hand.

Set benchmark-specific budgets explicitly. The wrapper defaults may be tuned for recent AIME runs, so do not rely on defaults for GSM8K.

Recommended standalone budgets:

- `BENCHMARK=gsm8k`: `MAX_NEW_TOKENS=750`, `MAX_STEPS=32`, `CONTEXT_LENGTH=1024`.
- `BENCHMARK=aime2024`/`aime24` or `BENCHMARK=aime2025`/`aime25`: `MAX_NEW_TOKENS=8192`, `MAX_STEPS=8192`, `CONTEXT_LENGTH=20480`.

```bash
cd /home/snorouzi/diffusion_RL/RL

CKPT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/<name>_step_N_hf \
TOKENIZER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B \
BENCHMARK=gsm8k \
ALG=FastDiffuser \
BS=32 \
TEMP=0.0 \
MAX_NEW_TOKENS=750 \
MAX_STEPS=32 \
CONTEXT_LENGTH=1024 \
TAG=<name>_step_N_nemorl_rep_fastdiffuser_b32_t1 \
./submit_standalone_gsm8k_eval.sh
```

Key parameters:

- `BENCHMARK=gsm8k` by default. Use `BENCHMARK=aime2024`/`aime24` or `BENCHMARK=aime2025`/`aime25` to run the same standalone NeMoRL-style evaluator on AIME.
- `ALG=FastDiffuser` for diffusion/FastDiffuser eval.
- `ALG=LinearSpec` for linear speculation eval.
- `BS=16` or `BS=32` for block size.
- `TEMP=0.0` always use greedy eval to remove the noise.
- `MAX_NEW_TOKENS`, `MAX_STEPS`, and `CONTEXT_LENGTH` must be set per benchmark: use `750/32/1024` for GSM8K and `8192/8192/20480` for AIME.
- AIME standalone eval loads the Nemo Skills AIME JSONLs from `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_data/nemo_skills_aime/{aime24,aime25}/test.jsonl` by default. Override with `NEMO_SKILLS_AIME_DATA_DIR` if needed; do not commit benchmark JSONLs into the repo.
- For AIME chat-completions parity with Yonggon/NemoSkills, use `GENERATION_API=chat_completions`, `TOP_P=0.95`, and `PROMPT_FILE=/home/snorouzi/diffusion_RL/RL/examples/prompts/generic_math.txt`.
- Observed baseline prompt sensitivity for the 3B checkpoint with SGLang/FastDiffuser block size 16, temp 0, high AIME budget: `generic_math.txt` gives AIME24 `4/30 = 13.33%` and AIME25 `3/30 = 10.00%`; `aime_no_cot.txt` gives AIME24 `5/30 = 16.67%` and AIME25 `2/30 = 6.67%`. Use `aime_no_cot.txt` when trying to match the higher observed AIME24 number.
- `NUM_SAMPLES=-1` by default for the full benchmark; set `NUM_SAMPLES=16` for a smoke test.

Examples:

```bash
# GSM8K, FastDiffuser, NeMoRL validation-style budget
BENCHMARK=gsm8k ALG=FastDiffuser BS=32 TEMP=0 MAX_NEW_TOKENS=750 MAX_STEPS=32 CONTEXT_LENGTH=1024 TAG=<name>_gsm8k_fd_b32_t1 ./submit_standalone_gsm8k_eval.sh

# AIME 2024, FastDiffuser, Yonggon/NemoSkills-style high budget
BENCHMARK=aime2024 ALG=FastDiffuser BS=16 TEMP=0 MAX_NEW_TOKENS=8192 MAX_STEPS=8192 CONTEXT_LENGTH=20480 GENERATION_API=chat_completions TOP_P=0.95 PROMPT_FILE=/home/snorouzi/diffusion_RL/RL/examples/prompts/generic_math.txt TAG=<name>_aime24_fd_b16_chat ./submit_standalone_gsm8k_eval.sh

# AIME 2025, FastDiffuser, Yonggon/NemoSkills-style high budget
BENCHMARK=aime2025 ALG=FastDiffuser BS=16 TEMP=0 MAX_NEW_TOKENS=8192 MAX_STEPS=8192 CONTEXT_LENGTH=20480 GENERATION_API=chat_completions TOP_P=0.95 PROMPT_FILE=/home/snorouzi/diffusion_RL/RL/examples/prompts/generic_math.txt TAG=<name>_aime25_fd_b16_chat ./submit_standalone_gsm8k_eval.sh

# Linear speculation, block size 32
BENCHMARK=gsm8k ALG=LinearSpec BS=32 TEMP=0 MAX_NEW_TOKENS=750 MAX_STEPS=32 CONTEXT_LENGTH=1024 TAG=<name>_gsm8k_linearspec_b32_t1 ./submit_standalone_gsm8k_eval.sh
```

The wrapper uses `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh`, mounts `/home/snorouzi` and `/lustre`, and writes `metrics.json`, `records.jsonl`, `server.log`, `server_command.txt`, `dllm_config.yaml`, and `slurm-<job>.out` under the output directory.

If a converted HF checkpoint is a symlink-heavy directory and fails inside the container with a missing custom-code file such as `configuration_ministral_dlm.py`, create an eval-only materialized copy with real files and the intended `config.json`, then point `CKPT` at that materialized directory. Do not patch the original converted checkpoint in place unless the user explicitly asks.

### 2. GSM8K NemoSkills Diffusion Eval

Use `xp/examples/run_llada_eval_pipeline_gpu_only.sh`. This starts the Nemotron/LLaDA API server, runs the NemoSkills client against `localhost:8000/v1`, and writes a generated `.gpu_only_cmd_*.sh` script into the output directory for exact reruns.

Set `SERVER_MODEL_PATH` to the converted Hugging Face checkpoint directory from Step 1. Set `SERVER_TOKENIZER` to the known 3B base checkpoint/tokenizer path, not to the raw `step_N` Megatron checkpoint directory.

Important path rule for NemoSkills: use the canonical `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/...` path family for `SERVER_MODEL_PATH`, `SERVER_TOKENIZER`, and `SEQ_EVAL_OUTPUT_DIR`. The shorter `/lustre/fsw/portfolios/coreai/users/snorouzi/...` path may resolve correctly on the login node, but the NemoSkills container can fail to load an otherwise valid HF checkpoint from that path with an error like "Can't load the configuration".

```bash
HF_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/<name>_step_N_hf
BASE_3B_TOKENIZER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B
OUTDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_results/<name>_step_N_gsm8k_nemoskills

ACCOUNT=coreai_dlalgo_llm \
SERVER_PARTITION=batch \
SERVER_TIME=04:00:00 \
SERVER_GPUS=8 \
SERVER_MODEL_PATH="${HF_MODEL}" \
SERVER_TOKENIZER="${BASE_3B_TOKENIZER}" \
SERVER_ENGINE=nemotron \
SEQ_EVAL_BENCHMARK=gsm8k:1 \
SEQ_EVAL_EXPNAME=<name>_step_N_gsm8k_nemoskills \
SEQ_EVAL_OUTPUT_DIR="${OUTDIR}" \
SEQ_EVAL_GENERATION_ALGORITHM=nemotron \
SEQ_EVAL_TOKENS_TO_GENERATE=750 \
SEQ_EVAL_STEPS=750 \
SEQ_EVAL_BLOCK_LENGTH=32 \
SEQ_EVAL_TEMPERATURE=0 \
SEQ_EVAL_EXTRA_ARGS="--exclude-unfinished-nfe false" \
bash xp/examples/run_llada_eval_pipeline_gpu_only.sh
```

For a smoke test, keep the same command but add a small sample cap:

```bash
SEQ_EVAL_EXTRA_ARGS="--max-samples 16 --exclude-unfinished-nfe false"
```

#### GSM8K NemoSkills AR-Native Eval

When the user asks for AR mode with the same `yonggon-rl` / NemoSkills setup, use the native AR path rather than the Hugging Face `ar` path. The required flags are:

```bash
SERVER_ENGINE=ar_native
SEQ_EVAL_GENERATION_ALGORITHM=ar_native
```

Use a separate output directory and keep `SERVER_MODEL_PATH` pointed at the original converted Hugging Face checkpoint. Do not create an AR-specific checkpoint copy for `ar_native`; the `ar_native` engine loads the model with `AutoModel` and calls `model.ar_generate()`.

```bash
HF_MODEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/<name>_step_N_hf
BASE_3B_TOKENIZER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B
OUTDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_results/<name>_step_N_gsm8k_nemoskills_ar_native

ACCOUNT=coreai_dlalgo_llm \
SERVER_PARTITION=batch \
SERVER_TIME=04:00:00 \
SERVER_GPUS=8 \
SERVER_MODEL_PATH="${HF_MODEL}" \
SERVER_TOKENIZER="${BASE_3B_TOKENIZER}" \
SERVER_ENGINE=ar_native \
SEQ_EVAL_BENCHMARK=gsm8k:1 \
SEQ_EVAL_EXPNAME=<name>_step_N_gsm8k_nemoskills_ar_native \
SEQ_EVAL_OUTPUT_DIR="${OUTDIR}" \
SEQ_EVAL_GENERATION_ALGORITHM=ar_native \
SEQ_EVAL_TOKENS_TO_GENERATE=750 \
SEQ_EVAL_STEPS=750 \
SEQ_EVAL_BLOCK_LENGTH=1 \
SEQ_EVAL_TEMPERATURE=0 \
bash xp/examples/run_llada_eval_pipeline_gpu_only.sh
```

Avoid `SERVER_ENGINE=hf` with `SEQ_EVAL_GENERATION_ALGORITHM=ar` unless the user explicitly asks for the Hugging Face AR path. That path patches `config.json` in place for AR loading, so it must never be run against a converted checkpoint directory that will also be reused for diffusion eval.

#### Alternate Block Length

Use the same NemoSkills command as above with a separate output directory and the block-length override:

```bash
SEQ_EVAL_OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_results/<name>_step_N_gsm8k_nemoskills_block16 \
SEQ_EVAL_BLOCK_LENGTH=16 \
bash xp/examples/run_llada_eval_pipeline_gpu_only.sh
```

### 3. Optional SGLang Fallback

Use the SGLang wrapper only when the user explicitly asks for the SGLang benchmark path. For native Nemotron Diffusion, pass FastDiffuser explicitly and clear `JSON_MODEL_OVERRIDE_ARGS` so the wrapper does not inject AR mode.

```bash
cd /home/snorouzi/diffusion_RL/RL

HF_MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/<name>_step_N_hf
OUTDIR=/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/<name>_step_N_gsm8k_sglang_fastdiffuser
SGLANG_COMMIT=$(git -C /home/snorouzi/code/sglang-nemotron-dllm-a652eb48 rev-parse HEAD)

MODEL="${HF_MODEL}" \
OUTDIR="${OUTDIR}" \
BENCHMARK=gsm8k \
SGLANG_COMMIT="${SGLANG_COMMIT}" \
DLLM_ALGORITHM=FastDiffuser \
JSON_MODEL_OVERRIDE_ARGS= \
BLOCK_SIZE=32 \
MAX_STEPS=32 \
SBATCH_JOB_NAME=<name>_step_N_sglang_fd_b32 \
./eval_3b_checkpoint.sh --sbatch
```

## Notes

- `STEP_DIR` is the raw NeMo-RL Megatron checkpoint step directory.
- `OUT` is the converted Hugging Face checkpoint directory created by `convert_checkpoint.sh`.
- `SERVER_MODEL_PATH` for NemoSkills evaluation must point to `OUT`, not to the raw `step_N` directory.
- For NemoSkills, prefer the `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/...` canonical path over the shorter `/lustre/fsw/portfolios/coreai/users/snorouzi/...` path for model, tokenizer, and output directories.
- If a converted checkpoint exists under `/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/...`, resolve its canonical path with `readlink -f <checkpoint_dir>` and use that `/lustre/fs1/...` result for `SERVER_MODEL_PATH`.
- `SERVER_TOKENIZER` should point to the 3B base checkpoint/tokenizer path: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B`.
- Keep conversion and evaluation output directories separate.
- Submit most eval jobs to the `batch_short` partition. It is usually more available, and GSM8K eval jobs usually finish under 2 hours, which matches the `batch_short` time limit.
- Keep output directories separate across block lengths, decoding modes, and benchmark variants.
- For AR-native NemoSkills eval, use `SERVER_ENGINE=ar_native`, `SEQ_EVAL_GENERATION_ALGORITHM=ar_native`, and `SEQ_EVAL_BLOCK_LENGTH=1`.
- Do not use `SERVER_ENGINE=hf` / `SEQ_EVAL_GENERATION_ALGORITHM=ar` by default; that Hugging Face AR path patches `config.json` in place.
- For standalone No-Ray NeMoRL validation replication, use `/home/snorouzi/diffusion_RL/RL/submit_standalone_gsm8k_eval.sh` and set `BENCHMARK`, `ALG`, `BS`, `TEMP`, `CKPT`, `TAG`, `MAX_NEW_TOKENS`, `MAX_STEPS`, and `CONTEXT_LENGTH`.
- In standalone eval, `TEMP=1.0` matches the NeMoRL validation-style schedule; `TEMP=0.0` is greedy. For offlie evaluations always use `TEMP=0` unless otherwise specified.
- NemoSkills writes a generated `.gpu_only_cmd_*.sh` script into `SEQ_EVAL_OUTPUT_DIR`; use it as the source of truth for rerunning an exact completed eval.
- The NemoSkills server worker logs are saved under `${SEQ_EVAL_OUTPUT_DIR}/worker_logs`.
- GSM8K uses `SEQ_EVAL_TOKENS_TO_GENERATE=750` in the NemoSkills command above and `MAX_NEW_TOKENS=750` in the standalone command.
- AIME uses the high-budget standalone settings `MAX_NEW_TOKENS=8192`, `MAX_STEPS=8192`, and `CONTEXT_LENGTH=20480`.
- If using the optional SGLang fallback, set `SGLANG_COMMIT` to the actual checked-out SGLang commit and use `JSON_MODEL_OVERRIDE_ARGS=` with FastDiffuser so the wrapper does not inject AR mode.
