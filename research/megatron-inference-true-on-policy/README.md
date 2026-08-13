# Megatron-Inference True On-Policy Study

Colocated Megatron inference GRPO with `zero_train_gen_mismatch` (`gen_kl_error` → 0).

## Setup

```bash
cp research/megatron-inference-true-on-policy/.env.template research/megatron-inference-true-on-policy/.env
# Edit .env: RL_DIR, CONTAINER_IMAGE, HF_TOKEN, HF_HOME, WANDB_API_KEY, WANDB_PROJECT
```

## Run

Recipe knobs live in yaml; the launcher only sets `MODEL`, `PRECISION`, and cluster overrides.

```bash
cd research/megatron-inference-true-on-policy

sbatch --export=MODEL=qwen1.5b,PRECISION=bf16  run_zero_kl_precision.sh
sbatch --export=MODEL=qwen30ba3b,PRECISION=bf16  run_zero_kl_precision.sh
sbatch --export=MODEL=qwen30ba3b,PRECISION=mxfp8 run_zero_kl_precision.sh
sbatch --export=MODEL=nanov3,PRECISION=bf16  run_zero_kl_precision.sh
sbatch --export=MODEL=nanov3,PRECISION=mxfp8 run_zero_kl_precision.sh
sbatch --export=MODEL=qwen30ba3b,ZERO_TRAIN_GEN_MISMATCH=false run_zero_kl_precision.sh
```

Recipes:
- `examples/configs/recipes/llm/grpo-qwen1.5b-megatron-zero-train-gen-kl.yaml`
- `examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl.yaml`
- `examples/configs/recipes/llm/grpo-nanov3-30ba3b-megatron-zero-train-gen-kl.yaml`

Per-model wrappers (`run_qwen1.5b_zero_kl_precision.sh`, etc.) forward to `run_zero_kl_precision.sh`.
