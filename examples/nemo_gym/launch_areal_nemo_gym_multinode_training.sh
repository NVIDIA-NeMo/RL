#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# AReaL variant of launch_nemo_gym_multinode_training.sh: same Slurm/ray.sub
# flow, but the entrypoint is examples/run_grpo_areal.py
# (SingleControllerArealActor: two-phase decoupled-PPO + interruptible refit)
# with the SC/AReaL config examples/nemo_gym/grpo_nanov3_8n8g.yaml
# (8 nodes: 4 vLLM async generation + 4 megatron training).

# ----- PARAMETERS -----
# WANDB_API_KEY, HF_TOKEN, EXP_NAME, NUM_ACTOR_NODES, NUM_SLURM_NODES (optional), REPO_LOCATION, CONTAINER_IMAGE_PATH, SLURM_ACCOUNT, SLURM_PARTITION

REPO_LOCATION=/home/haitianj/repos/nemo-RL

HF_HOME=/lustre/fsw/portfolios/coreai/users/haitianj/hf_cache
EXP_NAME=${EXP_NAME:-grpo-nanov3-12B-8n8g-gym-areal}
NUM_ACTOR_NODES=8
NUM_SLURM_NODES=8
CONTAINER_IMAGE_PATH=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/nvcr.io/nvidian/nemo-rl:29fc948-55351550.squashfs
SLURM_ACCOUNT=coreai_dlalgo_nemorl
SLURM_PARTITION=batch
# Checkpoints are large (12B Megatron + optimizer state); keep them on lustre, not $HOME.
# (SC-path checkpointing is not wired yet; the override is inert until it is.)
CHECKPOINT_DIR=/lustre/fsw/portfolios/coreai/users/haitianj/results/$EXP_NAME
# policy.model_name / policy.tokenizer.name in the config live under this path,
# which is outside the default mounts.
MODEL_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/data/nano-v3-12b-hf


# ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION

# Construct the command
read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=$HF_HOME \
HF_TOKEN=$HF_TOKEN \
WANDB_API_KEY=$WANDB_API_KEY \
NEMO_GYM_VENV_DIR=/lustre/fsw/portfolios/coreai/users/haitianj/gym_venvs \
NRL_TQ_SKIP_ACTOR_RUNTIME_ENV=1 \
uv run python examples/run_grpo_areal.py \
    --config examples/nemo_gym/grpo_nanov3_8n8g.yaml \
    ++cluster.num_nodes=$NUM_ACTOR_NODES \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=results/$EXP_NAME \
    ++checkpointing.checkpoint_dir=$CHECKPOINT_DIR \
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
COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS="$REPO_LOCATION:$REPO_LOCATION,/lustre/fsw/portfolios/coreai/users/haitianj:/lustre/fsw/portfolios/coreai/users/haitianj,$MODEL_DIR:$MODEL_DIR" \
NRL_IGNORE_VERSION_MISMATCH=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
HF_HOME=$HF_HOME \
sbatch \
    --nodes=$FINAL_NUM_SLURM_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
