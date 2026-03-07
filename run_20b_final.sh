#!/bin/bash
# GPT-OSS 20B final run: pip cuDNN 9.19 + moe_permute_fusion + sequence_packing
set -e

# Override LD_LIBRARY_PATH with pip cuDNN 9.19 (prepend over tarball 9.18.1 set by attach script)
# Note: this uv run also triggers env sync; it may install torchvision 0.25.0 (wrong for torch 2.9.0)
PIP_CUDNN_LIB=$(uv run python3 -c "import nvidia.cudnn, pathlib; print(pathlib.Path(list(nvidia.cudnn.__path__)[0]) / 'lib')")
export LD_LIBRARY_PATH="${PIP_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
ln -sf libcudnn.so.9 "${PIP_CUDNN_LIB}/libcudnn.so" 2>/dev/null || true

echo "[INFO] PIP_CUDNN_LIB=${PIP_CUDNN_LIB}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Fix torchvision: uv base env (no --extra vllm) resolves torchvision 0.25.0+cu129
# which requires torch 2.10.0, but we use 2.9.0 → C++ ext fails → RuntimeError.
# Force 0.24.0+cu129 (compatible with torch 2.9.0) and then skip re-sync on the training run.
VENV_DIR=$(uv run --no-sync python3 -c "import sys; print(sys.prefix)" 2>/dev/null || echo "/opt/nemo_rl_venv")
echo "[INFO] Pinning torchvision==0.24.0+cu129 in ${VENV_DIR} (fixes torch 2.9.0 ABI compat)"
"${VENV_DIR}/bin/pip" install "torchvision==0.24.0+cu129" \
  --index-url https://download.pytorch.org/whl/cu129 --no-deps -q

# Install pip cuDNN 9.19 into system venv (/opt/nemo_rl_venv) so Ray workers see it.
# With NEMO_RL_PY_EXECUTABLES_SYSTEM=1, workers use /opt/nemo_rl_venv/bin/python3.
# worker_groups.py detects nvidia/cudnn/lib in the worker venv and sets LD_LIBRARY_PATH.
# Without this, workers fall back to container's cuDNN 9.10.1 → FusedAttention disabled.
echo "[INFO] Installing nvidia-cudnn-cu12==9.19.0.56 into /opt/nemo_rl_venv for worker cuDNN propagation"
/opt/nemo_rl_venv/bin/pip install "nvidia-cudnn-cu12==9.19.0.56" --no-deps -q || true

# The attach script sets RAY_ADDRESS=ray://IP:10001 which triggers Ray client mode
# and fails with "resources must not be provided". Unset it to use address="auto"
# which connects to the local Ray process via socket (works with version mismatch).
unset RAY_ADDRESS
export RAY_IGNORE_VERSION_MISMATCH=1

# Ray 2.54.0 no longer creates node_ip_address.json; create it for the 2.49.2 driver.
SESSION_DIR=$(ls -d /tmp/ray/session_[0-9]*/ 2>/dev/null | head -1 | sed 's|/$||')
if [[ -n "$SESSION_DIR" && ! -f "$SESSION_DIR/node_ip_address.json" ]]; then
  HEAD_IP=$(hostname -i | awk '{print $1}')
  echo "{\"node_ip_address\": \"${HEAD_IP}\"}" > "$SESSION_DIR/node_ip_address.json"
  echo "[INFO] Created node_ip_address.json in $SESSION_DIR (ip=${HEAD_IP})"
fi

# Verify cuDNN version (--no-sync to avoid reinstalling torchvision 0.25.0)
uv run --no-sync python3 -c "
import ctypes
cudnn = ctypes.CDLL('libcudnn.so.9')
cudnn.cudnnGetVersion.restype = ctypes.c_size_t
v = cudnn.cudnnGetVersion()
print(f'[cuDNN] libcudnn.so.9 version: {v // 10000}.{(v % 10000) // 100}.{v % 100}')
"

LOGFILE="gpt-oss-20b_final_$(date +%Y%m%d_%H%M%S).log"
RUN_NAME="gptoss-20b-2n-tp2-ep8-$(date +%Y%m%d_%H%M%S)"
echo "[INFO] Starting GPT-OSS 20B run, logging to ${LOGFILE}"

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 CUDNN_INSTALL=0 NRL_FORCE_REBUILD_VENVS=false NRL_IGNORE_VERSION_MISMATCH=1 VLLM_USE_DEEP_GEMM=0 \
uv run --no-sync ./examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-gptoss-20b-8n8g-megatron.yaml \
  cluster.num_nodes=2 \
  cluster.gpus_per_node=8 \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.megatron_cfg.moe_permute_fusion=true \
  policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
  policy.train_global_batch_size=128 \
  policy.sequence_packing.enabled=true \
  grpo.num_prompts_per_step=16 \
  grpo.num_generations_per_prompt=8 \
  grpo.max_num_steps=20 \
  grpo.val_period=1000 \
  checkpointing.enabled=false \
  logger.wandb_enabled=True \
  logger.wandb.project=sync-grpo-h100-gptoss-exp \
  +logger.wandb.entity=nvidia \
  "logger.wandb.name=${RUN_NAME}" \
  2>&1 | tee "${LOGFILE}"

echo "[INFO] 20B run complete. Log: ${LOGFILE}"
