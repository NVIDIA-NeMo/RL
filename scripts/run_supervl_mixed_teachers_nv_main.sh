#!/usr/bin/env bash
# In-container version of submit_super_vl_35_mm_trainer_vllm_v2_32n4g.sh.
# Source: cspades/RL cye/nemotron-omni-super-base @ 52a7096c.
# rem handles allocation, container entry, and Ray startup.
set -euo pipefail
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/uv
export CHECKPOINTING_ENABLED=true
export CHECKPOINT_KEEP_TOP_K=1
export CHECKPOINT_SAVE_PERIOD=10
export CONFIG=examples/configs/recipes/vlm/super_vl_35_mixed_teachers_nv_main.yaml
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DATA_ROOT=/opt/nemo-rl/workspace/datasets/mm-trainer-unified
export EXP_AVG_DTYPE=float32
export EXP_AVG_SQ_DTYPE=float32
export FLASHINFER_DISABLE_VERSION_CHECK=1
export GENERATION_BACKEND=vllm
export GENERATION_ROUTER_BACKEND_TIMEOUT_S=1800
export GPUS_PER_NODE=4
export HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/huggingface/datasets
export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/huggingface
export HF_HUB_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/huggingface/hub
export HF_MODULES_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/huggingface/modules
export HUGGINGFACE_HUB_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/huggingface/hub
export INFER_EP=4
export INFER_TP=4
export LOGPROB_MB_TOKENS=65536
export MAX_BUFFERED_ROLLOUTS=256
export MAX_INFLIGHT_PROMPTS=128
export MAX_LOOKAHEAD_VERSIONS=1
export MAX_NEW_TOKENS=32768
export MAX_SEQUENCE_LENGTH=65536
export MAX_STEPS=125
export MIN_GENERATION_TOKENS=32768
export MM_TRAINER_DATA_PATH=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/datasets/mm-trainer-unified/training.jsonl
export MM_TRAINER_GYM_VENV_DIR=/opt/gym_venvs
export MM_TRAINER_MEDIA_ROOT=/lustre
export MM_TRAINER_MODEL_PATH=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/models/super-vl-35-rlvr-v43-falcon-r3-20260905/hf
export MM_TRAINER_RESULTS_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/experiments/super-vl-35-mm-trainer-vllm-v2-32n4g
export MM_TRAINER_WANDB_ENTITY=nvidia
export MM_TRAINER_WANDB_ID=super-vl-35-mm-trainer-vllm-v2-32n4g
export MM_TRAINER_WANDB_NAME=super-vl-35-mm-trainer-vllm-v2-32n4g
export MM_TRAINER_WANDB_PROJECT=rohit-unified-teacher-supervl3p5
export MODEL_NAME=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/models/super-vl-35-rlvr-v43-falcon-r3-20260905/hf
export MOE_BACKEND=flashinfer_cutlass
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NEMO_GYM_EXTRA_ROOTS=/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/runtime/gym-extra
export NEMO_GYM_ROLLOUT_TIMEOUT_S=2100
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export NEMO_LENS_RUNTIME_REV=b0f977d414b2f89938604a0b7eaa78ee08bc8700
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_RL_VIDEO_MEDIA_ROOT=/lustre
export NEMO_RL_VIDEO_TRAIN_JSONL=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/datasets/mm-trainer-unified/training.jsonl
export NEMO_RL_VIDEO_VAL_JSONL=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/datasets/mm-trainer-unified/training.jsonl
export NRL_FORCE_REBUILD_VENVS=false
export NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/megatron-checkpoints-super-vl-35-unified-final-ln-v2
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.006
export NRL_VENVS_TRUST_EXISTING=1
export NRL_VIDEO_BACKEND=torchcodec
export NRL_VIDEO_SAMPLING_STYLE=nemotron_vl
export NRL_VIDEO_TEMPORAL_PATCH_SIZE=2
export NUM_FRAMES=64
export NUM_GENERATIONS_PER_PROMPT=16
export NUM_GEN_NODES=16
export NUM_NODES=32
export NUM_PROMPTS_PER_STEP=128
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export OFFLOAD_OPTIMIZER_FOR_LOGPROB=false
export OPTIMIZER_CPU_OFFLOAD=false
export OPTIMIZER_OFFLOAD_FRACTION=0.0
export OVERLAP_GRAD_REDUCE=false
export OVERLAP_PARAM_GATHER=false
export POLICY_CP=2
export POLICY_EP=16
export POLICY_TP=2
export PREPARE_VSTAT=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RESULTS_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/experiments/super-vl-35-mm-trainer-vllm-v2-32n4g
export SEGMENT_SIZE=8
export STALL_WATCHDOG_TIMEOUT_S=12600
export STORE_PARAM_REMAINDERS=false
export TASK=vstat
export TEMPORAL_PATCH_SIZE=2
export TORCH_CUDA_ARCH_LIST=10.0
export TORCH_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/torch
export TRAIN_GBS=2048
export TRAIN_MB_TOKENS=65536
export TRANSFORMERS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/huggingface/transformers
export TRITON_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/triton
export USE_PRECISION_AWARE_OPTIMIZER=true
export VIDEO_TARGET_PATCHES=1024
export VLLM_CAP_MAX_TOKENS_TO_CONTEXT=true
export VLLM_ENABLE_PREFIX_CACHING=false
export VLLM_ENFORCE_EAGER=false
export VLLM_GPU_MEMORY_UTILIZATION=0.8
export VLLM_LIMIT_MM_IMAGES=64
export VLLM_MAX_NUM_BATCHED_TOKENS=65536
export VLLM_MAX_NUM_SEQS=128
export VLLM_REFIT_TIMEOUT_S=300
export VLLM_RUNTIME_PATCH_SCRIPT=/opt/nemo-rl/scripts/patch_vllm_super_omni_radio_layernorm_0_29.py
export VLLM_TRITON_FORCE_FIRST_CONFIG=1
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
export WANDB_INIT_TIMEOUT=300
export WANDB_NAME=super-vl-35-mm-trainer-vllm-v2-32n4g
export WANDB_PROJ=rohit-unified-teacher-supervl3p5
export XDG_CACHE_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/xdg

# The project wrapper sources only these exports before starting Ray.
if [[ "${1:-}" == "--env" ]]; then
  return 0
fi

if [[ "${1:-}" == "--setup" ]]; then
set -euo pipefail
cd /opt/nemo-rl
ensure_nemo_lens_runtime() {
  local python=$1
  if "${python}" -c 'from nemo.lens.groups import SpanRegistry; from nemo.lens.instruments import MetricSpec, register_metric_group' >/dev/null 2>&1; then
    return
  fi
  echo "[nemo-lens] Updating ${python} to b0f977d414b2f89938604a0b7eaa78ee08bc8700"
  uv pip install --python "${python}" "nemo-lens[sdk,aiohttp] @ git+https://github.com/NVIDIA-NeMo/Lens.git@b0f977d414b2f89938604a0b7eaa78ee08bc8700"
  "${python}" -c 'from nemo.lens.groups import SpanRegistry; from nemo.lens.instruments import MetricSpec, register_metric_group'
}
ensure_nemo_lens_runtime /opt/nemo_rl_venv/bin/python
for python in /opt/ray_venvs/*/bin/python; do
  if [[ -x "${python}" ]]; then
    ensure_nemo_lens_runtime "${python}"
  fi
done
MEGATRON_WORKER_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
if [[ ! -x ${MEGATRON_WORKER_PYTHON} ]]; then
  FORCE_REBUILD_VENV=false uv run --no-sync python -c 'import os; from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES; from nemo_rl.utils.venvs import create_local_venv; create_local_venv(PY_EXECUTABLES.MCORE, "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker", force_rebuild=os.environ["FORCE_REBUILD_VENV"].lower() == "true")'
fi
ensure_nemo_lens_runtime ${MEGATRON_WORKER_PYTHON}
uv run --no-project --no-sync --python "${MEGATRON_WORKER_PYTHON}" python /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/runtime/build_mcore_helpers_nv_main.py
AUDIO_DEPS_STAGGER_MAX_S=30 RAY_MEGATRON_PYTHON=${MEGATRON_WORKER_PYTHON} bash tools/install_audio_deps.sh
if [[ ! -x /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python ]]; then
  FORCE_REBUILD_VENV=false VLLM_WORKER_CLASS=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker uv run --no-sync python -c 'import os; from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES; from nemo_rl.utils.venvs import create_local_venv; create_local_venv(PY_EXECUTABLES.VLLM_GYM, os.environ["VLLM_WORKER_CLASS"], force_rebuild=os.environ["FORCE_REBUILD_VENV"].lower() == "true")'
fi
ensure_nemo_lens_runtime /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python /opt/nemo-rl/scripts/patch_vllm_super_omni_radio_layernorm_0_29.py
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python -c 'import torchcodec'
  exit 0
fi
if [[ ! -s "${MM_TRAINER_DATA_PATH}" || ! -f "${MM_TRAINER_MODEL_PATH}/config.json" || ! -f "${MM_TRAINER_MODEL_PATH}/chat_template.jinja" ]]; then
  echo "Missing mixed-teacher data or model files" >&2
  exit 1
fi
cd /opt/nemo-rl
BRIDGE=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
export PYTHONPATH=/opt/nemo-rl:${NEMO_GYM_EXTRA_ROOTS}:${BRIDGE}/src:${BRIDGE}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}
exec uv run --no-sync --python /opt/nemo_rl_venv/bin/python python examples/run_grpo_single_controller.py --config "${CONFIG}" "$@"
