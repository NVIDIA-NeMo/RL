export CHECKPOINT_DIR=/home/zhaochengz/lustre/reinforcer/results/Qwen2.5-3B-sft-xxx
export MAX_MODEL_LEN=32768
export JOB_ID=1
export TEMPERATURE=0.6
export TOP_P=1.0
export TOP_K=-1
export NUM_GENERATION=4
export TAG=temp0.6

output=$(
COMMAND="./scripts/convert_ckpt.sh" \
    sbatch --account=llmservice_modelalignment_ppo --job-name=convert_ckpt${JOB_ID} \
    --nodes=1 --partition=interactive --time=4:0:0 --gres=gpu:1 \
    --output=${BASE_LOG_DIR}/slurm-%j.out \
    /home/zhaochengz/lustre/reinforcer/ray.sub
)
echo "$output"
job_id=$(echo "$output" | awk '{print $4}')

models=$(ls -d ${CHECKPOINT_DIR}/hf_step_* | sort -V)
last_step=0
worker_ids=()
for model in $models; do
    step=$(basename $model)
    step=${step/hf_step_/}
    output=$(
        COMMAND="./scripts/run_gpqa.sh" START_STEP=$last_step END_STEP=$step \
            sbatch --account=llmservice_modelalignment_ppo --job-name=gpqa_${TAG}${JOB_ID} \
            --dependency=afterok:${job_id},singleton \
            --nodes=1 --partition=batch --time=4:0:0 --gres=gpu:8 --exclusive \
            --output=${BASE_LOG_DIR}/slurm-%j.out \
            /home/zhaochengz/lustre/reinforcer/ray.sub
    )
    echo "$output"
    worker_id=$(echo "$output" | awk '{print $4}')
    worker_ids+=($worker_id)
    last_step=$step
done

sbatch --account=llmservice_modelalignment_ppo --job-name=collect_gpqa_steps${JOB_ID} \
    --dependency=afterok:${worker_ids[@]// /:}
    --nodes=1 --partition=cpu --time=4:0:0 \
    ./scripts/collect_gpqa_steps.sh