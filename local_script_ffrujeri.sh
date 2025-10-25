source /lustre/fsw/coreai_dlalgo_genai/zhiyul/secrets.sh
set -eoux pipefail
CONFIG_PATH="examples/configs/grpo_acemath_rl.yaml"

FIRST=${FIRST:-0}
if [ $FIRST -eq 1 ]; then
    uv pip install -e .
    export NRL_FORCE_REBUILD_VENVS=true
fi

uv run examples/run.py \
    --config=${CONFIG_PATH} \
    loss_fn.use_importance_sampling_correction=False \
    cluster.num_nodes=2 \
    cluster.gpus_per_node=8 \
    logger.wandb_enabled=True \
    logger.wandb.name='grpo-acemath_rl-dtensor_tp1-vllm_tp2-tk-big-version-bump-zhiyul' \
    ++logger.wandb.id='grpo-acemath_rl-dtensor_tp1-vllm_tp2-tk-big-version-bump-zhiyul' \
    logger.wandb.project='nemo-rl-acemath-rl-tk-big-version-bump' \
    ++policy.generation.vllm_kwargs.compilation_config.use_inductor=False