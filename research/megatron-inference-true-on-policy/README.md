# Megatron-Inference True On-Policy Study

True on-policy RL training using Megatron-Inference to minimize training-generation mismatch (`gen_kl_error` -> 0).

## Setup

```bash
# 1. Clone the RL repo
git clone -b yigongq/minf-onpolicy git@github.com:YigongQin/RL.git
cd RL && git submodule update --init --recursive

# 2. Download the container (one-time)
enroot import --output nemo_rl_v0.7.0.sqsh 'docker://nvcr.io#nvidia/nemo-rl:v0.7.0'

# 3. Create .env with your config
cp .env.template .env
# Edit .env: set RL_DIR, CONTAINER_IMAGE, HF_TOKEN, HF_HOME, WANDB_API_KEY, WANDB_PROJECT
```

## Run

Single launcher for all three study models. Zero-KL knobs live in the recipe yaml; the script only handles runtime sweeps (`MODEL`, `PRECISION`, ablations).

```bash
# Qwen2.5-1.5B (default 1×8)
sbatch --export=MODEL=qwen1.5b,PRECISION=bf16  run_zero_kl_precision.sh

# Qwen3-30B-A3B DAPO (default 1×8; scale with SLURM --nodes)
sbatch --export=MODEL=qwen30ba3b,PRECISION=bf16  run_zero_kl_precision.sh
sbatch --export=MODEL=qwen30ba3b,PRECISION=mxfp8 run_zero_kl_precision.sh

# Nemotron-3-Nano-30B-A3B DAPO (default 2×8)
sbatch --export=MODEL=nanov3,PRECISION=bf16  run_zero_kl_precision.sh
sbatch --export=MODEL=nanov3,PRECISION=mxfp8 run_zero_kl_precision.sh

# Ablations
sbatch --export=MODEL=qwen30ba3b,ZERO_TRAIN_GEN_MISMATCH=false run_zero_kl_precision.sh
```

Recipes:
- `examples/configs/recipes/llm/grpo-qwen1.5b-megatron-zero-train-gen-kl.yaml`
- `examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl.yaml`
- `examples/configs/recipes/llm/grpo-nanov3-30ba3b-megatron-zero-train-gen-kl.yaml`

