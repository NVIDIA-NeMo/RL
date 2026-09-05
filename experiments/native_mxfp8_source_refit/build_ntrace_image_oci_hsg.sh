#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-01:00:00}

case "${ACTION}" in
  render|test-only|submit) ;;
  *)
    echo "ACTION must be render, test-only, or submit" >&2
    exit 2
    ;;
esac

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"
: "${BASE_IMAGE:?Set BASE_IMAGE to an immutable image under /lustre}"
: "${OUTPUT_IMAGE:?Set OUTPUT_IMAGE to a new immutable image under /lustre}"
: "${NTRACE_REPO:?Set NTRACE_REPO to a pinned checkout under /home}"

case "${BASE_IMAGE}" in
  /lustre/*) ;;
  *) echo "BASE_IMAGE must be under /lustre" >&2; exit 2 ;;
esac
case "${OUTPUT_IMAGE}" in
  /lustre/*) ;;
  *) echo "OUTPUT_IMAGE must be under /lustre" >&2; exit 2 ;;
esac
case "${NTRACE_REPO}" in
  /home/*) ;;
  *) echo "NTRACE_REPO must be under /home" >&2; exit 2 ;;
esac

test -s "${BASE_IMAGE}"
test -d "${NTRACE_REPO}/src/ntrace"
test -f "${NTRACE_REPO}/pyproject.toml"

SRUN_CANDIDATE=$(command -v srun 2>/dev/null || true)
SRUN=${SRUN:-$(readlink -f "${SRUN_CANDIDATE}" 2>/dev/null || true)}
if [[ -z "${SRUN}" || "${SRUN}" != /* || ! -x "${SRUN}" ]]; then
  echo "Set SRUN to the absolute path of the Slurm srun executable" >&2
  exit 2
fi

NTRACE_SHA=$(git -C "${NTRACE_REPO}" rev-parse HEAD)
OUTPUT_LOG=${OUTPUT_IMAGE%.sqsh}.build-%j.out

if [[ "${ACTION}" == render ]]; then
  printf 'base_image=%s\noutput_image=%s\nntrace_repo=%s\nntrace_sha=%s\n' \
    "${BASE_IMAGE}" "${OUTPUT_IMAGE}" "${NTRACE_REPO}" "${NTRACE_SHA}"
  exit 0
fi

if [[ -e "${OUTPUT_IMAGE}" ]]; then
  echo "OUTPUT_IMAGE already exists: ${OUTPUT_IMAGE}" >&2
  exit 2
fi

BUILD_COMMAND=$(cat <<EOF
set -euo pipefail
rm -rf /opt/ntrace-runtime
NTRACE_INSTALL_SOURCE=${NTRACE_REPO} \
NTRACE_INSTALL_TARGET=/opt/ntrace-runtime \
NTRACE_INSTALL_PYTHON=/opt/nemo_rl_venv/bin/python \
  bash ${NTRACE_REPO}/scripts/ntrace_nemo_rl_install_target.sh
printf '%s\n' '${NTRACE_SHA}' > /opt/ntrace-runtime/.source-revision
PYTHONPATH=/opt/ntrace-runtime /opt/nemo_rl_venv/bin/python -c \
  'import ntrace, pyarrow; from ntrace.backends import get_backend, selected_backend_name; assert selected_backend_name() == "cpp"; get_backend()'
EOF
)
BUILD_COMMAND_B64=$(printf '%s' "${BUILD_COMMAND}" | base64 | tr -d '\n')

SBATCH_ACTION=()
if [[ "${ACTION}" == test-only ]]; then
  SBATCH_ACTION=(--test-only)
fi

exec sbatch \
  "${SBATCH_ACTION[@]}" \
  --nodes=1 \
  --gres=gpu:4 \
  --exclusive \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --job-name="${SLURM_ACCOUNT}.build-ntrace-${NTRACE_SHA:0:8}" \
  --output="${OUTPUT_LOG}" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"container_build","description":"build ntrace runtime image"}}' \
  --wrap="${SRUN} --ntasks=1 --ntasks-per-node=1 \
    --no-container-mount-home \
    --container-remap-root \
    --container-image=${BASE_IMAGE} \
    --container-mounts=/home:/home,/lustre:/lustre \
    --container-workdir=/tmp \
    --container-writable \
    --container-save=${OUTPUT_IMAGE} \
    bash -lc 'printf %s ${BUILD_COMMAND_B64} | base64 -d | bash' \
  && sha256sum ${OUTPUT_IMAGE} > ${OUTPUT_IMAGE}.sha256"
