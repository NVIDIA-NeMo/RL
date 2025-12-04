source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
export HF_HOME=/lustre/fsw/portfolios/coreai/users/zhiyul/hf

# Ensure NRL_FORCE_REBUILD_VENVS is set to true only on first run
FLAG_FILE="/tmp/nrl_venv_built_$(whoami).flag"
if [ ! -f "$FLAG_FILE" ]; then
    export NRL_FORCE_REBUILD_VENVS=true
    touch "$FLAG_FILE"
    echo "First run detected: NRL_FORCE_REBUILD_VENVS=true"
else
    export NRL_FORCE_REBUILD_VENVS=false
    echo "Subsequent run: NRL_FORCE_REBUILD_VENVS=false (flag file exists)"
fi

rm -f standalone_router.log quick_debug.log

# REAL DATA THRASHING CONFIGURATION (CORRECTED)
# - 128 unique prompts (~1.2M tokens total)
# - Memory 0.05 -> Cluster Capacity ~1.1M tokens (Contention!)
# - 20 iterations -> We reuse the 128 prompts 20 times.
uv run --extra dynamo examples/run_router_benchmark_ray.py \
    --num-nodes 2 --gpus-per-node 8 \
    --dataset OpenMathInstruct-2 \
    --dataset-samples 128 \
    --batch-size 128 \
    --num-generations-per-prompt 1 \
    --use-manual-routing \
    --sequential \
    --seq-len 16384 \
    --max-seq-length 16384 \
    --generation-max-tokens 16 \
    --gpu-memory-utilization 0.6 \
    --warmup-iterations 1 --num-iterations 3 2>&1 