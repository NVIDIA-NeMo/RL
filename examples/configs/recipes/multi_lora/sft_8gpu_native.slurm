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
cp -f ${HOST_AUTOMODEL_DIR}/nemo_automodel/components/distributed/parallelizer.py /opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/distributed/parallelizer.py
cp -f ${HOST_AUTOMODEL_DIR}/nemo_automodel/components/checkpoint/checkpointing.py /opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/checkpoint/checkpointing.py
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
cd /opt/nemo-rl
python examples/run_sft_multi_lora.py --config ${EXP_DIR}/config.yaml
echo 'SFT_DONE_OK'
"

COMMAND="${INNER_CMD}" \
CONTAINER="${CONTAINER}" \
MOUNTS="/home/phuc:/home/phuc,/home/id_ed25519_nous_work:/home/id_ed25519_nous_work,/home/shared:/home/shared" \
BASE_LOG_DIR="${EXP_DIR}" \
bash "${NEMO_RL_SOURCE}/ray.sub"
