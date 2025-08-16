#!/bin/bash

export CHECKPOINT_DIR=/home/zhaochengz/lustre/reinforcer/results/Qwen2.5-3B-sft-xxx
export MAX_MODEL_LEN=32768
export JOB_ID=1

output=$(
COMMAND="./scripts/convert_ckpt.sh" \
    sbatch --account=llmservice_modelalignment_ppo --job-name=convert_ckpt${JOB_ID} \
    --nodes=1 --partition=interactive --time=4:0:0 --gres=gpu:1 \
    --output=${BASE_LOG_DIR}/slurm-%j.out \
    /home/zhaochengz/lustre/reinforcer/ray.sub
)
echo "$output"
job_id=$(echo "$output" | awk '{print $4}')

COMMAND="./scripts/run_gpqa.sh" TEMPERATURE=0.0 TOP_P=1.0 TOP_K=-1 NUM_GENERATION=1 TAG=greedy \
    sbatch --account=llmservice_modelalignment_ppo --job-name=gpqa_greedy${JOB_ID} \
    --dependency=afterok:${job_id},singleton \
    --nodes=1 --partition=batch --time=4:0:0 --gres=gpu:8 --exclusive \
    --output=${BASE_LOG_DIR}/slurm-%j.out \
    /home/zhaochengz/lustre/reinforcer/ray.sub

COMMAND="./scripts/run_gpqa.sh" TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 NUM_GENERATION=4 TAG=recommended \
    sbatch --account=llmservice_modelalignment_ppo --job-name=gpqa_recommended${JOB_ID} \
    --dependency=afterok:${job_id},singleton \
    --nodes=1 --partition=batch --time=4:0:0 --gres=gpu:8 --exclusive \
    --output=${BASE_LOG_DIR}/slurm-%j.out \
    /home/zhaochengz/lustre/reinforcer/ray.sub

COMMAND="./scripts/run_gpqa.sh" TEMPERATURE=1.0 TOP_P=1.0 TOP_K=-1 NUM_GENERATION=4 TAG=high \
    sbatch --account=llmservice_modelalignment_ppo --job-name=gpqa_high${JOB_ID} \
    --dependency=afterok:${job_id},singleton \
    --nodes=1 --partition=batch --time=4:0:0 --gres=gpu:8 --exclusive \
    --output=${BASE_LOG_DIR}/slurm-%j.out \
    /home/zhaochengz/lustre/reinforcer/ray.sub