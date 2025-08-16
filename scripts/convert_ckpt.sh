CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/home/zhaochengz/lustre/reinforcer/results/Qwen2.5XXX"}

model_family=$(basename "$CHECKPOINT_DIR")
summary_files=(
    "logs/${model_family}_greedy_summary.txt"
    "logs/${model_family}_recommended_summary.txt"
    "logs/${model_family}_high_summary.txt"
    "logs/${model_family}_temp0.6_summary.txt"
)

set -e
step_dirs=$(ls -d ${CHECKPOINT_DIR}/step_* | sort -V)

for step_dir in $step_dirs; do
    new_dir=$(dirname $step_dir)/hf_$(basename $step_dir)
    if [ -d "${new_dir}" ]; then
        continue
    fi
    record="model_name='hf_$(basename $step_dir)'"
    for summary_file in "${summary_files[@]}"; do
        if [ ! -f "$summary_file" ] || ! grep -q "$record" "$summary_file"; then
            uv run python examples/convert_dcp_to_hf.py --config ${step_dir}/config.yaml \
            --dcp-ckpt-path ${step_dir}/policy/weights --hf-ckpt-path ${new_dir}
            continue 2
        fi
    done
    echo "$step_dir: Found evaluation records in all summary files. Skipping."
done
