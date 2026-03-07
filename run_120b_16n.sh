#!/bin/bash
# GPT-OSS 120B 16-node: Megatron TP=8 matches vLLM TP=8 — eliminates Bug 11 OOM structurally.
# With TP=8 on both sides, prepare_refit_info() streams weights 1:1 (no all_gather reshape needed).
# 16n8g: TP=8, PP=2, EP=8 (128 GPUs total)
set -e

PIP_CUDNN_LIB=$(uv run python3 -c "import nvidia.cudnn, pathlib; print(pathlib.Path(list(nvidia.cudnn.__path__)[0]) / 'lib')")
export LD_LIBRARY_PATH="${PIP_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
ln -sf libcudnn.so.9 "${PIP_CUDNN_LIB}/libcudnn.so" 2>/dev/null || true
echo "[INFO] PIP_CUDNN_LIB=${PIP_CUDNN_LIB}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

VENV_DIR=$(uv run --no-sync python3 -c "import sys; print(sys.prefix)" 2>/dev/null || echo "/opt/nemo_rl_venv")
"${VENV_DIR}/bin/pip" install "torchvision==0.24.0+cu129" --index-url https://download.pytorch.org/whl/cu129 --no-deps -q
/opt/nemo_rl_venv/bin/pip install "nvidia-cudnn-cu12==9.19.0.56" --no-deps -q || true

unset RAY_ADDRESS
export RAY_IGNORE_VERSION_MISMATCH=1
export VLLM_USE_DEEP_GEMM=0

SESSION_DIR=$(ls -d /tmp/ray/session_[0-9]*/ 2>/dev/null | head -1 | sed 's|/$||')
if [[ -n "$SESSION_DIR" && ! -f "$SESSION_DIR/node_ip_address.json" ]]; then
  HEAD_IP=$(hostname -i | awk '{print $1}')
  echo "{\"node_ip_address\": \"${HEAD_IP}\"}" > "$SESSION_DIR/node_ip_address.json"
  echo "[INFO] Created node_ip_address.json in $SESSION_DIR (ip=${HEAD_IP})"
fi

uv run --no-sync python3 -c "
import ctypes
cudnn = ctypes.CDLL('libcudnn.so.9')
cudnn.cudnnGetVersion.restype = ctypes.c_size_t
v = cudnn.cudnnGetVersion()
print(f'[cuDNN] libcudnn.so.9 version: {v // 10000}.{(v % 10000) // 100}.{v % 100}')
"

LOGFILE="gpt-oss-120b_16n_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Starting GPT-OSS 120B 16n run (Bug11 structural fix: Megatron TP=8 == vLLM TP=8), logging to ${LOGFILE}"

# 16n8g: TP=8, PP=2, EP=8 (128 GPUs)
# Key: tensor_model_parallel_size=8 matches vllm tensor_parallel_size=8
#      => prepare_refit_info() does 1:1 weight copy, no reshape all_gather => no OOM
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 CUDNN_INSTALL=0 NRL_FORCE_REBUILD_VENVS=false NRL_IGNORE_VERSION_MISMATCH=1 VLLM_USE_DEEP_GEMM=0 \
uv run --no-sync ./examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-gptoss-120b-8n8g-megatron.yaml \
  cluster.num_nodes=16 \
  cluster.gpus_per_node=8 \
  policy.megatron_cfg.tensor_model_parallel_size=8 \
  policy.megatron_cfg.pipeline_model_parallel_size=2 \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
  policy.megatron_cfg.moe_permute_fusion=true \
  policy.generation.vllm_cfg.tensor_parallel_size=8 \
  policy.train_global_batch_size=128 \
  grpo.num_prompts_per_step=16 \
  grpo.num_generations_per_prompt=8 \
  grpo.max_num_steps=5 \
  grpo.val_period=1000 \
  checkpointing.enabled=false \
  logger.wandb_enabled=False \
  2>&1 | tee "${LOGFILE}"

echo "[INFO] 120B 16n run complete. Log: ${LOGFILE}"
