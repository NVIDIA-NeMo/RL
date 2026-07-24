#!/bin/bash
# =============================================================================
# Standalone Multi-LoRA SFT launcher for NeMo-RL (1 node x 8 GPUs, DTensor v2).
# All source and launch infrastructure come from this checkout. NeMo Automodel
# remains NeMo-RL's normal package/submodule dependency; the four required
# Multi-LoRA integration files are carried under patches/automodel in this repo.
# Usage: sbatch --export=ALL,CFG=<abs-or-rel-config> [-J name] this.slurm
# =============================================================================
#SBATCH --partition=batch
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

# Resolve the checkout from the submitted script path. SLURM copies scripts to
# its spool, so BASH_SOURCE alone is not stable after submission; NATIVE_DIR is
# an explicit override for nonstandard launchers.
if [[ -n "${NATIVE_DIR:-}" ]]; then
    REPO_ROOT="$(cd -- "${NATIVE_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/nemo_rl/__init__.py" ]]; then
    REPO_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
fi
NATIVE_DIR="${REPO_ROOT}"
cd "${NATIVE_DIR}"
case "${CFG}" in
  /*) CFG_ABS="${CFG}" ;;
  *)  CFG_ABS="${NATIVE_DIR}/${CFG}" ;;
esac

# The stock repo-root ray.sub assumes an --mpi=pmix cluster; this recipe ships
# a cluster-adapted copy (pmi2, dynamic Ray ports, NCCL IB env) next to the
# launcher so the whole launch path is tracked in this repository.
RAY_SUB="${RAY_SUB:-${NATIVE_DIR}/examples/configs/recipes/multi_lora/ray.sub}"
CONTAINER="${CONTAINER:-/home/shared/containers/nemo-rl-super-v3.sqsh}"
HF_HOME_HOST="${HF_HOME_HOST:-${HOME}/.cache/huggingface}"
RUN_TAG="${SLURM_JOB_NAME:-mlnative}-${SLURM_JOB_ID:-local}"
EXP_DIR="${NATIVE_DIR}/results/${RUN_TAG}"
mkdir -p "${EXP_DIR}"

for required in "${RAY_SUB}" "${CONTAINER}" "${CFG_ABS}"; do
    [[ -e "${required}" ]] || { echo "ERROR: required path missing: ${required}" >&2; exit 1; }
done
mkdir -p "${HF_HOME_HOST}"

cp -f "${CFG_ABS}" "${EXP_DIR}/config.yaml"

INNER_CMD="
set -eoux pipefail
cd /opt/nemo-rl
echo '>>> Overlaying this NeMo-RL Multi-LoRA checkout onto container /opt/nemo-rl'
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
print('STANDALONE_OK: nousnet not importable')
EOF
python - <<'EOF'
from pathlib import Path
root = Path('/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel')
checks = {
    'multi_dispatch': (root / 'components/_peft/lora.py', 'from nemo_rl.models.multi_lora.adapter import MultiLinearLoRA'),
    'shard_hint': (root / 'components/distributed/parallelizer_utils.py', 'def attr_shard_placement_fn'),
    'checkpoint_restore': (root / 'components/checkpoint/checkpointing.py', 'missing'),
}
for name, (path, marker) in checks.items():
    text = path.read_text(encoding='utf-8')
    assert marker in text, f'Automodel integration check failed: {name} ({path})'
print('AUTOMODEL_INTEGRATION_OK: repo-contained patches active')
EOF
pip install 'numpy<2' -q 2>&1 | tail -2 || true
export HF_HOME=${HF_HOME_HOST}
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
MOUNT_ITEMS=("${NATIVE_DIR}:${NATIVE_DIR}" "${HF_HOME_HOST}:${HF_HOME_HOST}")
for optional_mount in /home/shared /home/phuc /home/mormio/.cache /home/mormio/RL/data; do
    [[ -e "${optional_mount}" ]] && MOUNT_ITEMS+=("${optional_mount}:${optional_mount}")
done
MOUNTS="$(IFS=,; echo "${MOUNT_ITEMS[*]}")"
export MOUNTS
export NOUSNET_RAY_DYNAMIC_PORTS="${NOUSNET_RAY_DYNAMIC_PORTS:-1}"
export GPUS_PER_NODE=8
export CPUS_PER_WORKER=64
export BASE_LOG_DIR="${EXP_DIR}"

bash "${RAY_SUB}"
