#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ----- PARAMETERS -----

EXP_NAME=grpo-nanov3-12B-8n8g-async-gym
NUM_ACTOR_NODES=8
NUM_SLURM_NODES=8
REPO_LOCATION=/home/haitianj/repos/nemo-RL
CONTAINER_IMAGE_PATH=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/nvcr.io/nvidian/nemo-rl:29fc948-55351550.squashfs
SLURM_ACCOUNT=coreai_dlalgo_nemorl
SLURM_PARTITION=batch

# Lustre locations (everything the job writes at runtime).
LUSTRE_USER_DIR=/lustre/fsw/portfolios/coreai/users/haitianj
CHECKPOINT_DIR=$LUSTRE_USER_DIR/results/$EXP_NAME
UV_CACHE_DIR_OVERRIDE=$LUSTRE_USER_DIR/uv_cache
HF_HOME=$LUSTRE_USER_DIR/hf_cache
# Shared gym server venvs: must be on a filesystem visible from every node,
# because code_gen's unit-test Ray tasks (SPREAD + py_executable) start
# workers with this venv's python on arbitrary nodes. Pre-built serially —
# see repair_gym_venvs.sh on lustre (parallel setup on lustre corrupts venvs).
NEMO_GYM_VENV_DIR=$LUSTRE_USER_DIR/gym_venvs
# Training data dumps (train_data_step*.jsonl, hundreds of MB per step),
# W&B local files (wandb.init(dir=log_dir)), and GPU monitoring logs.
RESULTS_DIR=$LUSTRE_USER_DIR/results/$EXP_NAME
# ray-driver/head/worker logs (ray.sub BASE_LOG_DIR) and sbatch stdout.
RUN_LOGS_DIR=$LUSTRE_USER_DIR/run-logs
# policy.model_name / policy.tokenizer.name in the config live here, outside
# the default mounts.
MODEL_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/data/nano-v3-12b-hf

mkdir -p "$RUN_LOGS_DIR"

# ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION

# Construct the command
read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=$HF_HOME \
HF_TOKEN=$HF_TOKEN \
WANDB_API_KEY=$WANDB_API_KEY \
NEMO_GYM_VENV_DIR=$NEMO_GYM_VENV_DIR \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/nemo_gym/rohit_nanov3.yaml \
    ++cluster.num_nodes=$NUM_ACTOR_NODES \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=$RESULTS_DIR/logs \
    ++checkpointing.checkpoint_dir=$RESULTS_DIR \
    ++logger.wandb_enabled=True \
    ++logger.wandb.project=areal \
    ++logger.wandb.entity=joc \
    $@
EOF

echo -e "Running command:\n$COMMAND"

FINAL_NUM_SLURM_NODES="${NUM_SLURM_NODES:-$NUM_ACTOR_NODES}"

# NRL_IGNORE_VERSION_MISMATCH: mounted dev repo vs baked container pyproject/uv.lock
# drift is a fatal RuntimeError otherwise.
# TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD: torch>=2.6 DCP load of our own trusted megatron
# shards fails under weights_only=True (belt and suspenders with the monkeypatch in
# nemo_rl/__init__.py).
# BASE_LOG_DIR + --output/--error: ray and slurm logs go to lustre; ray.sub uses
# set -e, so an unwritable stdout kills the job within seconds.
COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS="$REPO_LOCATION:$REPO_LOCATION,$LUSTRE_USER_DIR:$LUSTRE_USER_DIR,$MODEL_DIR:$MODEL_DIR" \
BASE_LOG_DIR=$RUN_LOGS_DIR \
NRL_IGNORE_VERSION_MISMATCH=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
HF_HOME=$HF_HOME \
UV_CACHE_DIR_OVERRIDE=$UV_CACHE_DIR_OVERRIDE \
sbatch \
    --nodes=$FINAL_NUM_SLURM_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    --output=$RUN_LOGS_DIR/slurm-%j.out \
    --error=$RUN_LOGS_DIR/slurm-%j.out \
    ray.sub
