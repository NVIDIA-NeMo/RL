# Megatron-Inference True On-Policy Study

True on-policy RL training using Megatron-Inference to minimize training-generation mismatch (`gen_kl_error` -> 0).


## Setup

```bash
# 1. Clone the RL repo
git clone -b yigongq/minf-onpolicy git@github.com:YigongQin/RL.git
cd RL && git submodule update --init --recursive

# 2. Download the container (one-time)
enroot import --output nemo_rl_v0.7.0.sqsh 'docker://nvcr.io#nvidia/nemo-rl:v0.7.0'
# This creates nemo_rl_v0.7.0.sqsh in the current directory.
# Move it to your preferred location and update CONTAINER_IMAGE in the scripts.

# 3. Create .env with your config
cp .env.template .env
# Edit .env: set RL_DIR, CONTAINER_IMAGE, HF_TOKEN, HF_HOME, WANDB_API_KEY, WANDB_PROJECT
```

## Run

```bash
# Qwen2.5-1.5B (1 node, 8GPU, TP=1)
# ZERO_TRAIN_GEN_MISMATCH is set to true by default
sbatch --export=PRECISION=bf16  run_qwen1.5b_zero_kl_precision.sh
sbatch --export=PRECISION=bf16,ZERO_TRAIN_GEN_MISMATCH=false run_qwen1.5b_zero_kl_precision.sh
sbatch --export=PRECISION=mxfp8 run_qwen1.5b_zero_kl_precision.sh
sbatch --export=PRECISION=mxfp8,ZERO_TRAIN_GEN_MISMATCH=false run_qwen1.5b_zero_kl_precision.sh


# Qwen3-30B-A3B (1 node, 8GPU, TP=1)
sbatch --export=PRECISION=bf16  run_qwen30ba3b_zero_kl_precision.sh
sbatch --export=PRECISION=mxfp8 run_qwen30ba3b_zero_kl_precision.sh
sbatch --export=PRECISION=bf16,ZERO_TRAIN_GEN_MISMATCH=false run_qwen30ba3b_zero_kl_precision.sh
sbatch --export=PRECISION=mxfp8,ZERO_TRAIN_GEN_MISMATCH=false run_qwen30ba3b_zero_kl_precision.sh
```

