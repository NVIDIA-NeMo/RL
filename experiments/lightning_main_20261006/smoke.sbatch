#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --mem=0
set -euo pipefail
: "${CONTAINER:?Set immutable nightly image}"
: "${SOURCE_ARCHIVE:?Set prepared source archive}"
export LOCAL_ROOT="/raid/scratch/${USER}/nr-smoke-${SLURM_JOB_ID}"
export EXPECTED_FINGERPRINT="${SOURCE_ARCHIVE}.fingerprint.json"
export PYTHONPATH="${LOCAL_ROOT}/source:${LOCAL_ROOT}/source/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${LOCAL_ROOT}/source/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export PYTHONPYCACHEPREFIX="${LOCAL_ROOT}/pycache"
export UV_CACHE_DIR="${LOCAL_ROOT}/uv"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_ROOT}/inductor"
export TRITON_CACHE_DIR="${LOCAL_ROOT}/triton"
export VLLM_CACHE_ROOT="${LOCAL_ROOT}/vllm"
export FLA_TILELANG=0
export RAY_TMPDIR="/tmp/nr${SLURM_JOB_ID}"
unset PYTHONOPTIMIZE NRL_IGNORE_VERSION_MISMATCH
srun --gres=gpu:4 --mem=0 --no-container-mount-home \
  --container-image="$CONTAINER" --container-mounts=/lustre:/lustre,/raid/scratch:/raid/scratch \
  bash -c '
    set -euo pipefail
    mkdir -p "$LOCAL_ROOT/source" "$UV_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
    tar -xf "$SOURCE_ARCHIVE" -C "$LOCAL_ROOT/source"
    cd "$LOCAL_ROOT/source"
    for role in driver policy vllm-sync vllm-async; do
      case "$role" in
        driver) executable=/opt/nemo_rl_venv/bin/python ;;
        policy) executable=/usr/local/bin/python-MegatronPolicyWorker ;;
        vllm-sync) executable=/usr/local/bin/python-VllmGenerationWorker ;;
        vllm-async) executable=/usr/local/bin/python-VllmAsyncGenerationWorker ;;
      esac
      "$executable" experiments/lightning_main_20261006/smoke.py "$role"
    done
    /opt/nemo_rl_venv/bin/python tools/config_cli.py expand experiments/lightning_main_20261006/sync-bf16.yaml
    /opt/nemo_rl_venv/bin/python tools/config_cli.py expand experiments/lightning_main_20261006/async-bf16.yaml
    /usr/local/bin/python-MegatronPolicyWorker experiments/lightning_main_20261006/ray_socket_probe.py
    /usr/local/bin/python-MegatronPolicyWorker -m pytest -q --mcore-only \
      tests/unit/models/megatron/test_hybridep_data.py \
      tests/unit/models/megatron/test_group_experts.py::test_build_hf_to_local_param_map_train_side \
      tests/unit/models/policy/test_megatron_worker.py::test_refit_metadata_map_precedes_transport_init
  '
