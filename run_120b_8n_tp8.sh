#!/bin/bash
# GPT-OSS 120B 8-node TP=8 run: pip cuDNN 9.19 + alltoall + moe_permute_fusion + sequence_packing
# 8n8g: TP=8, PP=2, EP=4 (64 GPUs total)
#
# Memory calc (H100 80GB per GPU):
#   vLLM  (TP=8, util=0.5): 40GB alloc = ~30GB model + ~10GB KV cache → 64 seq OK
#   Megatron training (TP=8, PP=2, EP=4):
#     - PP=1/TP=8 needed 54.73GB → PP=2 halves to ~27GB model base
#     - EP=4 (vs 16n EP=8) doubles expert params per GPU → ~35GB estimate
#     - Activations (activation_checkpointing=true): ~5-8GB → total ~42GB → fits H100 80GB ✓
#   Note: EP=4 is the only valid EP with TP=8 PP=2 on 8 nodes (8x2x4=64; 8x2x8=128>64)
#
# Key fix: Megatron TP=8 == vLLM TP=8 → 1:1 weight copy at prepare_refit_info → no Bug 11 OOM
# Batch reduced: 8 prompts x 8 generations = 64 (vs 128) to halve KV cache during generation
set -e

PIP_CUDNN_LIB=$(uv run python3 -c "import nvidia.cudnn, pathlib; print(pathlib.Path(list(nvidia.cudnn.__path__)[0]) / 'lib')")
export LD_LIBRARY_PATH="${PIP_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
ln -sf libcudnn.so.9 "${PIP_CUDNN_LIB}/libcudnn.so" 2>/dev/null || true

echo "[INFO] PIP_CUDNN_LIB=${PIP_CUDNN_LIB}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

VENV_DIR=$(uv run --no-sync python3 -c "import sys; print(sys.prefix)" 2>/dev/null || echo "/opt/nemo_rl_venv")
echo "[INFO] Pinning torchvision==0.24.0+cu129 in ${VENV_DIR} (fixes torch 2.9.0 ABI compat)"
"${VENV_DIR}/bin/pip" install "torchvision==0.24.0+cu129" \
  --index-url https://download.pytorch.org/whl/cu129 --no-deps -q

echo "[INFO] Installing nvidia-cudnn-cu12==9.19.0.56 into /opt/nemo_rl_venv for worker cuDNN propagation"
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

LOGFILE="gpt-oss-120b_8n_tp8_$(date +%Y%m%d_%H%M%S).log"
RUN_NAME="gptoss-120b-8n-tp8-ep4-$(date +%Y%m%d_%H%M%S)"
echo "[INFO] Starting GPT-OSS 120B 8n TP=8 PP=2 EP=4 run, logging to ${LOGFILE}"

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 CUDNN_INSTALL=0 NRL_FORCE_REBUILD_VENVS=false NRL_IGNORE_VERSION_MISMATCH=1 VLLM_USE_DEEP_GEMM=0 \
uv run --no-sync ./examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-gptoss-120b-8n8g-megatron.yaml \
  cluster.num_nodes=8 \
  cluster.gpus_per_node=8 \
  policy.generation.vllm_cfg.tensor_parallel_size=8 \
  policy.megatron_cfg.tensor_model_parallel_size=8 \
  policy.megatron_cfg.pipeline_model_parallel_size=2 \
  policy.megatron_cfg.expert_model_parallel_size=4 \
  policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
  policy.megatron_cfg.moe_permute_fusion=true \
  policy.train_global_batch_size=64 \
  grpo.num_prompts_per_step=8 \
  grpo.num_generations_per_prompt=8 \
  grpo.max_num_steps=20 \
  grpo.val_period=1000 \
  checkpointing.enabled=false \
  logger.wandb_enabled=True \
  logger.wandb.project=sync-grpo-h100-gptoss-exp \
  +logger.wandb.entity=nvidia \
  "logger.wandb.name=${RUN_NAME}" \
  2>&1 | tee "${LOGFILE}"

echo "[INFO] 120B 8n TP=8 run complete. Log: ${LOGFILE}"
