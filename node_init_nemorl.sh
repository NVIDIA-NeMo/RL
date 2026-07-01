for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done

# v13 image: tensorrt_llm is editable-installed in the VENV (/opt/nemo_rl_venv,
# from /workspace/TensorRT-LLM), NOT system python. tensorrt_llm's getExecPath()
# in cpp/.../fmhaKernels.h calls popen("pip show tensorrt_llm") to derive the
# NVRTC -I include dirs for the fmha JIT, so `pip` MUST resolve to the venv — if
# it resolves to system python (no tensorrt_llm) it returns empty, NVRTC gets
# zero include dirs, and the kernel compile dies with 'could not open source
# file "cuda.h" (no directories in search list)'.
# This matters cluster-wide: nemo-rl passes the driver's os.environ through to
# every Ray actor as runtime_env env_vars (virtual_cluster.py init_ray), incl.
# TRT-LLM's nested RayWorkerWrapper TP ranks. So the driver PATH must be venv-
# first HERE. (Shiki's original `/usr/bin`-first line was for the v12 image where
# tensorrt_llm lived in system python; it is inverted for this v13 image.)
export PATH=/opt/nemo_rl_venv/bin:/usr/bin:$PATH

# Note: Dockerfile sets CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl, which
# alone breaks #include_next <math.h>. We fix this further below by prepending
# /usr/include while keeping the original cccl path for deep_gemm builds.

export RAY_DEDUP_LOGS=0
export TRTLLM_UCX_INTERFACE=eth0
# Megatron needs this for its CUDA kernel compilation. gb200 = Blackwell (sm_100).
# Some Megatron internals also touch sm_90 (Hopper) so include it for safety.
export TORCH_CUDA_ARCH_LIST='9.0 10.0'

# OpenMPI in v13 can't bootstrap singleton mode standalone, but works fine under
# mpirun -n 1. Each Ray-spawned trtllm worker is wrapped via trtllm_py_wrapper.sh
# (see NEMO_RL_PY_EXECUTABLES_TRTLLM above) so MPI_Init has a proper launcher context.
# Only need --allow-run-as-root semantics here for cases where Ray daemons/clusters
# need MPI without the wrapper.
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# Surface logger.info from py_executor _handle_control_request + others;
# trtllm Logger defaults to "error", which silently drops INFO output.
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-info}
export LLM_MODELS_ROOT=/lustre/fsw/coreai_comparch_trtllm/common
# TrtllmGenerationWorker actors are launched via PY_EXECUTABLES.CONTAINER_SYSTEM,
# which reads this env var on the DRIVER at import time. /usr/bin/python3 in this
# image is cp312 with tensorrt_llm pre-installed; the driver's uv venv is cp312
# but lacks tensorrt_llm. Must be set on the driver, not just the actors.
# Pin TRT-LLM Ray actors to the base image's system Python (where the wheel
# is installed below). Without this, PY_EXECUTABLES.TRTLLM falls back to
# sys.executable, which under `uv run` is the uv-managed venv and lacks
# tensorrt_llm. See nemo_rl/distributed/virtual_cluster.py.
# For v13 (cp313): tensorrt_llm is installed in /opt/nemo_rl_venv as an editable
# install (from /workspace/TensorRT-LLM). /usr/bin/python3.13 doesn't have it.
# We can't point Ray directly at the venv python because v13's OMPI 4.1.9a1
# can't bootstrap singleton MPI (opal_init fails when launched outside mpirun).
# Wrap the venv python with `mpirun -n 1` via a tiny script so each Ray actor's
# MPI initializes as a 1-rank world. Since orchestrator_type="ray", inter-rank
# comm goes through Ray, not MPI — singleton MPI per actor is sufficient.
export NEMO_RL_PY_EXECUTABLES_TRTLLM=${NEMO_RL_PY_EXECUTABLES_TRTLLM:-/opt/nemo-rl/trtllm_py_wrapper.sh}

# Install ray into the base image's cp312 system Python so that actors using
# PY_EXECUTABLES.SYSTEM=/usr/bin/python3 (e.g. TrtllmGenerationWorker, when
# tensorrt_llm is only available in the cp312 dist-packages) can be launched
# by Ray. Idempotent: pip skips if already up-to-date.
if [[ -x /usr/bin/python3 ]]; then
    /usr/bin/python3 -m pip install --quiet --no-input "decord2" || true
    # Install uv into system python so `uv run` from the driver works (v13 doesn't ship uv).
    /usr/bin/python3 -m pip install --quiet --no-input uv || true
fi


# Force inference-side (system py) nccl4py to dlopen the SAME libnccl that
# torch.distributed loads on the train side. Even with nccl4py versions
# matched, libnccl bootstrap protocol differs between 2.28.x (venv via
# nvidia-nccl-cu13) and 2.29.x (apt-installed). Override the system symlink
# to point at the venv's libnccl so both interpreters dlopen the same lib.
NCCL_PIN_SRC=$(ls /opt/nemo_rl_venv/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null | head -1)
NCCL_SYS_LINK=/usr/lib/aarch64-linux-gnu/libnccl.so.2
if [[ -f "$NCCL_PIN_SRC" ]] && [[ -L "$NCCL_SYS_LINK" ]]; then
    target=$(readlink "$NCCL_SYS_LINK")
    if [[ "$target" != "$NCCL_PIN_SRC" ]]; then
        if [[ ! -e "${NCCL_SYS_LINK}.apt_orig" ]]; then
            cp -P "$NCCL_SYS_LINK" "${NCCL_SYS_LINK}.apt_orig"
        fi
        ln -sf "$NCCL_PIN_SRC" "$NCCL_SYS_LINK"
    fi
fi

export CPATH=/usr/local/cuda/targets/sbsa-linux/include:/usr/include:/usr/include/aarch64-linux-gnu:/usr/lib/gcc/aarch64-linux-gnu/13/include
export C_INCLUDE_PATH=/usr/local/cuda/targets/sbsa-linux/include:/usr/include:/usr/include/aarch64-linux-gnu:/usr/lib/gcc/aarch64-linux-gnu/13/include
# Fully unset CPLUS_INCLUDE_PATH so g++ falls back to its built-in C++ stdlib
# include order. Adding /usr/include via CPLUS_INCLUDE_PATH (instead of
# unsetting) does NOT fix #include_next <math.h>: the var is consumed with
# `-isystem` semantics and disrupts libstdc++'s internal header chain. If
# deep_gemm / cuda::std builds need cccl headers, pass `-I/usr/local/cuda/include/cccl`
# to those build commands directly rather than via this env var.
unset CPLUS_INCLUDE_PATH
export CUDA_INCLUDE_PATH=/usr/local/cuda/targets/sbsa-linux/include
export CUDA_PATH=/usr/local/cuda
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}

# Install into /usr/bin/python3 (system python where tensorrt_llm lives), not
# the uv venv. Bare `pip` resolves to /opt/nemo_rl_venv/bin/pip which puts
# ray under the wrong interpreter and was also corrupting the venv install.
if [[ -x /usr/bin/python3 ]]; then
    /usr/bin/python3 -m pip install --quiet --no-input \
        mako \
        "ray[default]==2.54.0" \
        oyaml \
        pytest-unused-fixtures || true
fi

# trtllm_backend.NcclExtension.init_collective uses nemo-rl's own
# StatelessProcessGroup (backed by nccl4py from core deps). Pin to the same
# version resolved in uv.lock for train-side venvs (megatron policy worker)
# — nccl4py bumps the bootstrap wire format between minor releases, so train
# and inference must run identical versions or ncclCommInitRankScalable will
# return RemoteError(6).
if [[ -x /usr/bin/python3 ]]; then
    /usr/bin/python3 -m pip install --quiet --no-input "nccl4py==0.2.0" || true
fi

# Secrets + HF cache (tokens, HF_HOME, HF_DATASETS_CACHE, WANDB_*) live in one
# place so they aren't duplicated across launchers. Edit env.sh to change them.
# (Replaces Shiki's hardcoded shikiw/hf_cache + ${HF_TOKEN}/${WANDB_API_KEY} block.)
source /lustre/fsw/coreai_comparch_trtllm/erinh/launch_scripts/env.sh
