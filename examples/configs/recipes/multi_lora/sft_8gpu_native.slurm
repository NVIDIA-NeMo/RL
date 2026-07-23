#!/bin/bash
# =============================================================================
# 2026-07-23 — Multi-LoRA SFT, NeMo-RL NATIVE launcher (no nousnet anywhere)
# 1 node x 8 GPUs, DTensor v2. The multi-adapter implementation is the package
# vendored at nemo_rl/models/multi_lora in this worktree (RL_multilora_native,
# branch phuc/multilora-native-nemorl); entry = examples/run_sft_multi_lora.py.
# Usage: sbatch --export=ALL,CFG=<abs-or-rel-config> [-J name] this.slurm
# =============================================================================
#SBATCH --partition=batch
#SBATCH --chdir=/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --exclude=d2dfac12-009,d2dfac12-017,d2dfac12-014,d2dfac12-013,d2dfac12-022,d2dfac12-023,d2dfac12-025

set -euo pipefail

if [[ -z "${CFG:-}" ]]; then
    echo "ERROR: CFG must be set" >&2
    exit 1
fi

NATIVE_DIR="${NATIVE_DIR:-/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native}"
HOST_AUTOMODEL_DIR="${HOST_AUTOMODEL_DIR:-/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/Automodel_container_base}"
cd "${NATIVE_DIR}"
case "${CFG}" in
  /*) CFG_ABS="${CFG}" ;;
  *)  CFG_ABS="${NATIVE_DIR}/${CFG}" ;;
esac

NEMO_RL_SOURCE="${NEMO_RL_SOURCE:-/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/nousnet_pre_multilora_8178f9c/slurm_helpers}"
CONTAINER="${NOUSNET_CONTAINER:-/home/shared/containers/nemo-rl-super-v3.sqsh}"
RUN_TAG="${SLURM_JOB_NAME:-mlnative}-${SLURM_JOB_ID:-local}"
EXP_DIR="${NATIVE_DIR}/results/${RUN_TAG}"
mkdir -p "${EXP_DIR}"

cp -f "${CFG_ABS}" "${EXP_DIR}/config.yaml"

INNER_CMD="
set -eoux pipefail
cd /opt/nemo-rl
echo '>>> Overlaying NATIVE multi-lora branch onto container /opt/nemo-rl (no nousnet)'
cp -f ${NATIVE_DIR}/nemo_rl/algorithms/sft.py /opt/nemo-rl/nemo_rl/algorithms/sft.py
cp -f ${NATIVE_DIR}/nemo_rl/algorithms/loss_functions.py /opt/nemo-rl/nemo_rl/algorithms/loss_functions.py
cp -f ${NATIVE_DIR}/nemo_rl/data/processors.py /opt/nemo-rl/nemo_rl/data/processors.py
cp -f ${NATIVE_DIR}/nemo_rl/models/policy/lm_policy.py /opt/nemo-rl/nemo_rl/models/policy/lm_policy.py
cp -f ${NATIVE_DIR}/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py /opt/nemo-rl/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py
cp -f ${NATIVE_DIR}/nemo_rl/models/automodel/train.py /opt/nemo-rl/nemo_rl/models/automodel/train.py
cp -f ${NATIVE_DIR}/nemo_rl/models/automodel/setup.py /opt/nemo-rl/nemo_rl/models/automodel/setup.py
rm -rf /opt/nemo-rl/nemo_rl/models/multi_lora
cp -r ${NATIVE_DIR}/nemo_rl/models/multi_lora /opt/nemo-rl/nemo_rl/models/multi_lora
cp -f ${NATIVE_DIR}/examples/run_sft_multi_lora.py /opt/nemo-rl/examples/run_sft_multi_lora.py
cp -f ${NATIVE_DIR}/patches/automodel/nemo_automodel_components__peft_lora.py /opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/_peft/lora.py
cp -f ${NATIVE_DIR}/patches/automodel/nemo_automodel_components_distributed_parallelizer.py /opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/distributed/parallelizer.py
cp -f ${NATIVE_DIR}/patches/automodel/nemo_automodel_components_distributed_parallelizer_utils.py /opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/distributed/parallelizer_utils.py
cp -f ${NATIVE_DIR}/patches/automodel/nemo_automodel_components_checkpoint_checkpointing.py /opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/checkpoint/checkpointing.py
md5sum /opt/nemo-rl/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py /opt/nemo-rl/nemo_rl/models/multi_lora/adapter.py /opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/_peft/lora.py
python - <<'EOF'
import importlib.util, sys
assert importlib.util.find_spec('nousnet') is None, 'nousnet is importable inside the container — NOT a clean native run'
print('CLEAN: nousnet not importable')
EOF
pip install 'numpy<2' -q 2>&1 | tail -2 || true
export HF_HOME=/home/id_ed25519_nous_work/.cache/huggingface
export NRL_IGNORE_VERSION_MISMATCH=1
unset PYTORCH_CUDA_ALLOC_CONF || true
# --- runtime env kept verbatim from the campaign launcher so native runs stay
# comparable. NOUSNET_* names are read by the vendored nemo_rl.models.multi_lora
# modules (env-var names intentionally unchanged during vendoring). ---
export NVTE_DISABLE_TRITON_KERNELS=1
export TRITON_CACHE_AUTOTUNING=1
export NOUSNET_DETERMINISTIC=1
export NOUSNET_DETERMINISTIC_SEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NOUSNET_DIAG_ENABLED=\${NOUSNET_DIAG_ENABLED:-0}
export NOUSNET_DIAG_DUMP=\${NOUSNET_DIAG_DUMP:-0}
export NOUSNET_DIAG_LOSS_TRACE=\${NOUSNET_DIAG_LOSS_TRACE:-0}
export NOUSNET_DIAG_LORA_STEP=\${NOUSNET_DIAG_LORA_STEP:-1}
export NOUSNET_DIAG_TRACE_ONLY=\${NOUSNET_DIAG_TRACE_ONLY:-0}
export NOUSNET_DIAG_WHO=\${NOUSNET_DIAG_WHO:-single_unknown}
export NOUSNET_DIAG_ADAPTER_NAMES=\${NOUSNET_DIAG_ADAPTER_NAMES:-}
export NOUSNET_INIT_EXPORT_DIR=\${NOUSNET_INIT_EXPORT_DIR:-}
export NOUSNET_INIT_IMPORT_DIR=\${NOUSNET_INIT_IMPORT_DIR:-}
export NOUSNET_INIT_IMPORT_SLOT=\${NOUSNET_INIT_IMPORT_SLOT:-}
export PYTHONHASHSEED=\${PYTHONHASHSEED:-0}
export NCCL_ALGO=Ring
export NCCL_DETERMINISTIC=1
export NOUSNET_FUSED_GRAD_CLIP=\${NOUSNET_FUSED_GRAD_CLIP:-0}
export NOUSNET_PER_ADAPTER_GRAD_CLIP=\${NOUSNET_PER_ADAPTER_GRAD_CLIP:-0}
export NOUSNET_TRITON_GRAD_CLIP=\${NOUSNET_TRITON_GRAD_CLIP:-0}
export NOUSNET_MULTI_LORA_LOSS_TRITON=\${NOUSNET_MULTI_LORA_LOSS_TRITON:-0}
export NOUSNET_FORCE_PAD_TO=\${NOUSNET_FORCE_PAD_TO:-1024}
export NOUSNET_FORCE_UNIFORM_ROUTER=\"\${NOUSNET_FORCE_UNIFORM_ROUTER:-1}\"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_NCCL_BLOCKING_WAIT=0
export NOUSNET_PAD_TOKEN_ID=0
export RAY_DEDUP_LOGS=0
export NOUSNET_RUN_DIR=${EXP_DIR}
# Driver-side imports resolve from the native worktree (the NeMo-RL fork under
# test) — same role the campaign's HOST_RL_DIR PYTHONPATH entry played. Worker
# processes inherit the ray daemons' env, so the cp overlay above covers them.
export PYTHONPATH=${NATIVE_DIR}:\${PYTHONPATH:-}
cd /opt/nemo-rl
python examples/run_sft_multi_lora.py --config ${EXP_DIR}/config.yaml
echo 'SFT_DONE_OK'
"

export COMMAND="${INNER_CMD}"
export CONTAINER
export MOUNTS="${NEMO_RL_SOURCE}:${NEMO_RL_SOURCE},${NATIVE_DIR}:${NATIVE_DIR},/home/phuc:/home/phuc,/home/id_ed25519_nous_work:/home/id_ed25519_nous_work,/home/mormio/.cache:/home/mormio/.cache,/home/mormio/RL/data:/home/mormio/RL/data,/home/shared:/home/shared"
export NOUSNET_RAY_DYNAMIC_PORTS="${NOUSNET_RAY_DYNAMIC_PORTS:-1}"
export GPUS_PER_NODE=8
export CPUS_PER_WORKER=64
export BASE_LOG_DIR="${EXP_DIR}"

bash "${NEMO_RL_SOURCE}/ray.sub"
