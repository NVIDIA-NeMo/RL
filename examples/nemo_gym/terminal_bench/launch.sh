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

# Terminal Bench GRPO on Slurm through ray.sub. Inputs are environment variables.
# Required: CONTAINER SLURM_ACCOUNT SLURM_PARTITION NUM_TRAIN_NODES NUM_GEN_NODES
#           MODEL (policy checkpoint dir) DATA (TB JSONL)
#           TASK_ROOT (host dir mounted at /terminal-bench-tasks, matching task_folder in DATA)
#           RUN_DIR (checkpoints, logs, Gym output) OPENSANDBOX_DOMAIN OPENSANDBOX_API_KEY WANDB_API_KEY
# Optional: WALLTIME (4:00:00) EXP_NAME WANDB_PROJECT HF_HOME MOUNTS (extra host:container,...) SBATCH_ARGS
# Additional recipe overrides may be passed as arguments.
set -euo pipefail

for v in CONTAINER SLURM_ACCOUNT SLURM_PARTITION NUM_TRAIN_NODES NUM_GEN_NODES MODEL DATA TASK_ROOT RUN_DIR OPENSANDBOX_DOMAIN OPENSANDBOX_API_KEY WANDB_API_KEY; do
  : "${!v:?$v is required}"
done

RL_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$RUN_DIR"
export RUN_DIR=$(realpath "$RUN_DIR")  # Slurm uses the physical path as the container workdir
EXP_NAME=${EXP_NAME:-tb-opencode-$(date +%Y%m%d-%H%M%S)}
NUM_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES))
export GPUS_PER_NODE=4

# Runs in every node's container before Ray starts. The image has no unversioned
# python3-config, which Megatron's dataset-helper Makefile calls, and the prebuilt
# helper is hidden by the checkout mount; build it once under a lock so nodes do not race.
read -r -d '' SETUP_COMMAND <<'EOF' || true
ln -sfn "$(readlink -f /opt/nemo_rl_venv/bin/python)-config" /usr/local/bin/python3-config
flock "$RUN_DIR/.helpers.lock" make -C /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets
EOF

read -r -d '' COMMAND <<EOF || true
cd /opt/nemo-rl && uv run --locked --extra nemo_gym examples/run_grpo_single_controller.py \
  --config examples/nemo_gym/terminal_bench/opencode_sc.yaml \
  cluster.num_nodes=$NUM_NODES \
  policy.generation.colocated.resources.num_nodes=$NUM_GEN_NODES \
  policy.generation.vllm_cfg.reasoning_parser_plugin=/opt/nemo-rl/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py \
  policy.model_name=$MODEL \
  data.train.data_path=$DATA \
  checkpointing.checkpoint_dir=$RUN_DIR/checkpoints \
  logger.log_dir=$RUN_DIR/logs \
  logger.wandb_enabled=true logger.wandb.project=${WANDB_PROJECT:-tb-opencode} logger.wandb.name=$EXP_NAME \
  ++env.nemo_gym.nemo_gym_log_dir=$RUN_DIR/gym ++env.nemo_gym.results_dir=$RUN_DIR/gym/results ++env.nemo_gym.cache_dir=$RUN_DIR/gym/cache \
  $@
EOF

cd "$RUN_DIR"
COMMAND=$COMMAND SETUP_COMMAND=$SETUP_COMMAND CONTAINER=$CONTAINER \
MOUNTS="$RL_ROOT:/opt/nemo-rl,$MODEL:$MODEL,$DATA:$DATA,$TASK_ROOT:/terminal-bench-tasks,$RUN_DIR:$RUN_DIR${HF_HOME:+,$HF_HOME:$HF_HOME}${MOUNTS:+,$MOUNTS}" \
sbatch ${SBATCH_ARGS:-} --nodes="$NUM_NODES" --gres=gpu:4 --exclusive --mem=0 \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --time="${WALLTIME:-4:00:00}" \
  --job-name="$EXP_NAME" --output="$RUN_DIR/slurm-%j.out" "$RL_ROOT/ray.sub"
