#!/bin/bash
export PR3_REPO=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
export PR3_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python
export PYTHONPATH="$PR3_REPO:$PR3_REPO/3rdparty/Automodel-workspace/Automodel:$PR3_REPO/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${PYTHONPATH:-}"
export NEMO_RL_VENV_DIR=/opt/ray_venvs UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_NO_SYNC=1 UV_NO_PROGRESS=1
export HF_HOME=/lustre/fsw/portfolios/coreai/users/yuekaiz/.cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NRL_IGNORE_VERSION_MISMATCH=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 NCCL_DEBUG=WARN
cd "$PR3_REPO"
