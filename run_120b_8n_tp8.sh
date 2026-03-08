#!/bin/bash
# GPT-OSS 120B 8-node TP=8 PP=1 run: pip cuDNN 9.19 + alltoall + moe_permute_fusion + sequence_packing
# 8n8g: TP=8, PP=1, EP=8 (64 GPUs total) — PP=1 for better performance (no PP bubble overhead)
#
# Memory calc (H100 80GB per GPU):
#   vLLM  (TP=8, util=0.55): 44GB alloc = ~30GB model + ~14GB KV cache → OK
#   Megatron training (TP=8, PP=1, EP=8):
#     - PP=1/TP=8 estimated ~54.73GB (model + optimizer states + activations w/ act_ckpt)
#     - EP=8 (vs EP=4): halves expert params per GPU → lower memory than EP=4
#     - Total ~55-59GB → fits H100 80GB ✓
#   Note: TP=8×PP=1×EP=8 = 64 GPUs ✓ (8 nodes × 8 GPUs)
#
# Key fix: Megatron TP=8 == vLLM TP=8 → 1:1 weight copy at prepare_refit_info → no Bug 11 OOM
# Previous attempt (9869349) used gpu_util=0.40 → KV cache -3.14 GiB → FIXED: use 0.55
# Batch: 16 prompts x 8 generations = 128
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

LOGFILE="gpt-oss-120b_8n_tp8pp1_$(date +%Y%m%d_%H%M%S).log"
RUN_NAME="gptoss-120b-8n-tp8pp1-ep8-$(date +%Y%m%d_%H%M%S)"
echo "[INFO] Starting GPT-OSS 120B 8n TP=8 PP=1 EP=8 run, logging to ${LOGFILE}"

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 CUDNN_INSTALL=0 NRL_FORCE_REBUILD_VENVS=false NRL_IGNORE_VERSION_MISMATCH=1 VLLM_USE_DEEP_GEMM=0 \
uv run --no-sync ./examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-gptoss-120b-8n8g-megatron.yaml \
  cluster.num_nodes=8 \
  cluster.gpus_per_node=8 \
  policy.generation.vllm_cfg.tensor_parallel_size=8 \
  policy.generation.vllm_cfg.pipeline_parallel_size=1 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.55 \
  policy.megatron_cfg.tensor_model_parallel_size=8 \
  policy.megatron_cfg.pipeline_model_parallel_size=1 \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
  policy.megatron_cfg.moe_permute_fusion=true \
  policy.train_global_batch_size=128 \
  grpo.num_prompts_per_step=16 \
  grpo.num_generations_per_prompt=8 \
  grpo.max_num_steps=3 \
  grpo.val_period=1000 \
  checkpointing.enabled=false \
  logger.wandb_enabled=True \
  logger.wandb.project=sync-grpo-h100-gptoss120b-exp \
  +logger.wandb.entity=nvidia \
  "logger.wandb.name=${RUN_NAME}" \
  2>&1 | tee "${LOGFILE}"

echo "[INFO] 120B 8n TP=8 PP=1 EP=8 run complete. Log: ${LOGFILE}"
