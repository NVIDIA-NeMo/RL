#!/bin/bash
set -e

REPO=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/src/NeMo-RL/nemo-rl-deepseek-v4
cd "$REPO"

export NRL_FORCE_REBUILD_VENVS=true
export HF_TOKEN=""
export WANDB_MODE=disabled
export UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/shuangy/uv_cache
export UV_HTTP_TIMEOUT=600
mkdir -p "$UV_CACHE_DIR"

# Driver venv lacks nemo_automodel (workspace dep doesn't materialize there).
# Make it importable via the workspace tree so run_grpo.py's registry import
# can run even before Ray worker venvs are rebuilt.
export PYTHONPATH="${REPO}/3rdparty/Automodel-workspace/Automodel:${PYTHONPATH:-}"

echo "=== Starting env refresh: rebuilding Ray worker venvs ==="
echo "=== Using default grpo_math_1B config (Qwen2.5-1.5B) for venv rebuild ==="
echo "=== REPO=$REPO ==="
echo "=== UV_CACHE_DIR=$UV_CACHE_DIR ==="
echo "=== Start time: $(date) ==="

LOG=/tmp/env_refresh.log
rm -f "$LOG"
uv run python examples/run_grpo.py 2>&1 | tee "$LOG" &
TRAIN_PID=$!

echo "=== Waiting for first training step to complete (up to 90 min for first-time venv build) ==="
START_WAIT=$(date +%s)
while true; do
    if grep -q "Total step time" "$LOG" 2>/dev/null; then
        echo "=== First training step completed after $(( $(date +%s) - START_WAIT ))s ==="
        break
    fi
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "=== Training process exited unexpectedly; dumping log ==="
        tail -n 200 "$LOG"
        exit 1
    fi
    sleep 15
done

echo "=== Killing training process ==="
kill $TRAIN_PID 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
pkill -f "run_grpo" 2>/dev/null || true
sleep 5

echo ""
echo "=== Verifying package versions ==="

POLICY_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
VLLM_VENV=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker

if [ -d "$POLICY_VENV" ]; then
    echo "--- Policy worker venv ---"
    "$POLICY_VENV/bin/python" -c "import torch; print(f'torch: {torch.__version__}')"
    "$POLICY_VENV/bin/python" -c "import transformers; print(f'transformers: {transformers.__version__}')"
    "$POLICY_VENV/bin/python" -c "import nemo_automodel; print('automodel: OK')" 2>/dev/null || echo "automodel: not installed in this venv"
else
    echo "WARNING: Policy venv not found at $POLICY_VENV"
fi

if [ -d "$VLLM_VENV" ]; then
    echo "--- vLLM worker venv ---"
    "$VLLM_VENV/bin/python" -c "import torch; print(f'torch: {torch.__version__}')"
    "$VLLM_VENV/bin/python" -c "import transformers; print(f'transformers: {transformers.__version__}')"
    "$VLLM_VENV/bin/python" -c "import vllm; print(f'vllm: {vllm.__version__}')"
    "$VLLM_VENV/bin/python" -c "import vllm._C; print('vllm._C ABI: OK')"
    "$VLLM_VENV/bin/python" -c "from vllm.model_executor.models.registry import ModelRegistry; print('DSV4 registered:', 'DeepseekV4ForCausalLM' in ModelRegistry.get_supported_archs())"
else
    echo "WARNING: vLLM venv not found at $VLLM_VENV"
fi

echo ""
echo "=== Environment refresh complete ==="
echo "=== End time: $(date) ==="
echo "=== Container will be saved on exit via --container-save ==="
