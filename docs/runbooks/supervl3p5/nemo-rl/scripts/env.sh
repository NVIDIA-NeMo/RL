#!/usr/bin/env bash
# Source inside a provisioned container. Keep the image's pinned Python envs.
export RL_DIR=${RL_DIR:-/opt/nemo-rl}
: "${MM_TRAINER_MODEL_PATH:?Set the local HF checkpoint directory}"
: "${MM_TRAINER_DATA_PATH:?Set the mixed-teacher training JSONL}"
: "${MM_TRAINER_RESULTS_DIR:?Set a writable experiment directory}"
: "${MM_TRAINER_WANDB_ID:?Set the experiment W&B ID}"
: "${GYM_EXTRA_DIR:?Set the custom Gym resource/agent directory}"
export MM_TRAINER_WANDB_NAME=${MM_TRAINER_WANDB_NAME:-$MM_TRAINER_WANDB_ID}
export MM_TRAINER_WANDB_ENTITY=${MM_TRAINER_WANDB_ENTITY:-nvidia}
export MM_TRAINER_WANDB_PROJECT=${MM_TRAINER_WANDB_PROJECT:-rohit-unified-teacher-supervl3p5}
export MM_TRAINER_GYM_VENV_DIR=${MM_TRAINER_GYM_VENV_DIR:-/opt/gym_venvs}
export NEMO_GYM_VENV_DIR=$MM_TRAINER_GYM_VENV_DIR
export MM_TRAINER_MEDIA_ROOT=${MM_TRAINER_MEDIA_ROOT:-/lustre}
export NEMO_GYM_EXTRA_ROOTS=$RL_DIR/3rdparty/Gym-workspace/Gym:$GYM_EXTRA_DIR
export SUPER_CACHE_DIR=${SUPER_CACHE_DIR:-$MM_TRAINER_RESULTS_DIR/cache}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-$SUPER_CACHE_DIR/megatron-checkpoints-supervl3p5}
export UV_CACHE_DIR=$SUPER_CACHE_DIR/uv
export HF_HOME=$SUPER_CACHE_DIR/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$SUPER_CACHE_DIR/torch
export TRITON_CACHE_DIR=$SUPER_CACHE_DIR/triton
export XDG_CACHE_HOME=$SUPER_CACHE_DIR/xdg
export DRIVER_PYTHON=${DRIVER_PYTHON:-/opt/nemo_rl_venv/bin/python}
export MEGATRON_WORKER_PYTHON=${MEGATRON_WORKER_PYTHON:-/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}
export VLLM_WORKER_PYTHON=${VLLM_WORKER_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}
export BRIDGE_DIR=$RL_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
export PYTHONPATH=$RL_DIR:$NEMO_GYM_EXTRA_ROOTS:$BRIDGE_DIR/src:$BRIDGE_DIR/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_VENVS_TRUST_EXISTING=1
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_LENS_RUNTIME_REV=b0f977d414b2f89938604a0b7eaa78ee08bc8700
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.006
export NEMO_GYM_ROLLOUT_TIMEOUT_S=2100
export NRL_VIDEO_BACKEND=torchcodec
export NRL_VIDEO_SAMPLING_STYLE=nemotron_vl
export NRL_VIDEO_TEMPORAL_PATCH_SIZE=2
export NUM_FRAMES=64
export TEMPORAL_PATCH_SIZE=2
export VIDEO_TARGET_PATCHES=1024
export NEMO_RL_VIDEO_MEDIA_ROOT=$MM_TRAINER_MEDIA_ROOT
export NEMO_RL_VIDEO_TRAIN_JSONL=$MM_TRAINER_DATA_PATH
export NEMO_RL_VIDEO_VAL_JSONL=$MM_TRAINER_DATA_PATH
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
export VLLM_RUNTIME_PATCH_SCRIPT=$RL_DIR/scripts/patch_vllm_super_omni_radio_layernorm_0_29.py
export VLLM_TRITON_FORCE_FIRST_CONFIG=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCH_CUDA_ARCH_LIST=10.0
export WANDB_INIT_TIMEOUT=300
# Production disables the optional mismatch dump/stop mechanism.
unset NRL_TMPE_DIAGNOSTIC_DIR
