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

# Launch Inkling 2 Small GRPO (Automodel policy + SGLang BF16 or vLLM MXFP8 rollouts)
# on a Slurm cluster of 4-GPU GB200 nodes through ray.sub. Submits one self-ending job.
#
#   ACCOUNT=<acct> CONTAINER=<image.sqsh> WORKDIR=<dir> bash tools/inkling2/launch_slurm.sh proxy
#   ... launch_slurm.sh full        # SGLang, 16 nodes (does not fit; see the 16n recipe)
#   ... launch_slurm.sh full32      # SGLang, 32 nodes, EP64 + HybridEP in 16-node rack segments
#   ... launch_slurm.sh vllm-tiny   # vLLM MXFP8, 1 node, tiny checkpoint smoke test
#   ... launch_slurm.sh vllm-full   # vLLM MXFP8, 16 nodes, colocated
#   ... launch_slurm.sh vllm-async  # vLLM MXFP8, 24 nodes: 16 training + 8 generation
#
# The vllm-* modes need a CONTAINER whose vLLM serves Inkling2 (vllm-project/vllm#67325)
# and views built with make_vllm_mxfp8_views.py (make_tiny_mxfp8_view.py for vllm-tiny).
#
# Required: ACCOUNT (Slurm account), CONTAINER (NeMo RL image with the sglang and
# automodel extras), WORKDIR (shared dir for logs, checkpoints and the HF cache).
# Set policy.model_name and policy.generation.sglang_cfg.model_path (SGLang) or
# policy.generation.model_name (vLLM) through
# EXTRA_OVERRIDES or edit the recipe; DATA_MOUNTS adds container mounts for them.
#
# Optional: PARTITION (batch), QOS_OVERRIDE / TIME_OVERRIDE (per-mode QOS/time limit),
# EXTRA_OVERRIDES='key=value ...' (config overrides), DATA_MOUNTS='src:dst,...',
# AM_OVERRIDE=/path/nemo_automodel (mount another Automodel source tree),
# SEGMENT_OVERRIDE=16 (Slurm segment size), SECRETS_FILE (shell file sourced in a
# subshell for WANDB_API_KEY and HF_TOKEN only; never echoed).
#
# full/full32/vllm-full/vllm-async keep checkpoints in a fixed dir ($WORKDIR/checkpoints/<mode>;
# CKPT_DIR_OVERRIDE changes it) and resume from its latest checkpoint. The job saves
# one checkpoint SAVE_MARGIN_MIN (default 10) minutes before its time limit (or at
# SAVE_BY_OVERRIDE, DD:HH:MM:SS from job start), then exits.
set -euo pipefail
MODE=${1:?proxy|full|full32|vllm-tiny|vllm-full|vllm-async}
: "${ACCOUNT:?set ACCOUNT}" "${CONTAINER:?set CONTAINER}" "${WORKDIR:?set WORKDIR}"
PARTITION=${PARTITION:-batch}
RL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)  # NeMo RL tree: nemo_rl mount, recipes, ray.sub
RECIPES=$RL/examples/configs/recipes/llm
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$WORKDIR/runs/$MODE-$STAMP
mkdir -p "$OUT"
ENVS="export PYTHONPATH=/opt/nemo-rl:\${PYTHONPATH:-} HF_HOME=$WORKDIR/hf_home HF_DATASETS_CACHE=$WORKDIR/hf_home/datasets"

AM=$RL/3rdparty/Automodel-workspace/Automodel/nemo_automodel
SEGMENT=""
case $MODE in
  proxy)
    RECIPE=$RECIPES/grpo-inkling2-proxy-4n4g-gb200-automodel-sglang.yaml
    NODES=4; QOS=interactive; TIME=01:00:00; EXTRA=""
    ;;
  full)
    RECIPE=$RECIPES/grpo-inkling2-16n4g-gb200-automodel-sglang.yaml
    NODES=16; QOS=normal; TIME=04:00:00
    EXTRA="logger.wandb_enabled=true logger.wandb.name=grpo-inkling2-16n4g-gb200-automodel-sglang-$STAMP"
    ;;
  full32)
    RECIPE=$RECIPES/grpo-inkling2-32n4g-gb200-automodel-sglang.yaml
    NODES=32; QOS=normal; TIME=04:00:00
    # The recipe uses EP64 + HybridEP inside 16-node rack segments (cluster.segment_size=16).
    SEGMENT=16
    EXTRA="logger.wandb_enabled=true logger.wandb.name=grpo-inkling2-32n4g-gb200-automodel-sglang-$STAMP"
    ;;
  vllm-tiny)
    RECIPE=$RECIPES/grpo-inkling2-tiny-1n4g-automodel-vllm-mxfp8.yaml
    NODES=1; QOS=interactive; TIME=01:00:00; EXTRA=""
    ;;
  vllm-full)
    RECIPE=$RECIPES/grpo-inkling2-16n4g-automodel-vllm-mxfp8.yaml
    NODES=16; QOS=normal; TIME=04:00:00
    EXTRA="logger.wandb_enabled=true logger.wandb.name=grpo-inkling2-16n4g-automodel-vllm-mxfp8-$STAMP"
    ;;
  vllm-async)
    RECIPE=$RECIPES/grpo-inkling2-24n4g-automodel-vllm-mxfp8-async.yaml
    NODES=24; QOS=normal; TIME=04:00:00
    EXTRA="logger.wandb_enabled=true logger.wandb.name=grpo-inkling2-24n4g-automodel-vllm-mxfp8-async-$STAMP"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
QOS=${QOS_OVERRIDE:-$QOS}; TIME=${TIME_OVERRIDE:-$TIME}
AM=${AM_OVERRIDE:-$AM}; SEGMENT=${SEGMENT_OVERRIDE:-$SEGMENT}
case $MODE in proxy|vllm-tiny) LONG_RUN=false ;; *) LONG_RUN=true ;; esac
if $LONG_RUN; then CKPT_DIR=$WORKDIR/checkpoints/$MODE; else CKPT_DIR=$OUT/ckpt; fi
CKPT_DIR=${CKPT_DIR_OVERRIDE:-$CKPT_DIR}
# checkpoint_must_save_by (DD:HH:MM:SS from job start) = time limit - SAVE_MARGIN_MIN.
# Slurm time is [D-]HH:MM:SS. The margin covers one step plus the ~70 s synchronous
# save of the ~6.5 TB checkpoint and Ray teardown.
SAVE_MARGIN_MIN=${SAVE_MARGIN_MIN:-10}
t=$TIME; days=0
if [[ $t == *-* ]]; then days=${t%%-*}; t=${t#*-}; fi
IFS=: read -r hh mm ss <<<"$t"
save_by=$(( ((10#$days*24 + 10#$hh)*60 + 10#$mm)*60 + 10#${ss:-0} - SAVE_MARGIN_MIN*60 ))
(( save_by > 0 )) || { echo "TIME=$TIME is shorter than SAVE_MARGIN_MIN=$SAVE_MARGIN_MIN" >&2; exit 2; }
SAVE_BY=$(printf '%02d:%02d:%02d:%02d' $((save_by/86400)) $((save_by%86400/3600)) $((save_by%3600/60)) $((save_by%60)))
SAVE_BY=${SAVE_BY_OVERRIDE:-$SAVE_BY}  # e.g. SAVE_BY_OVERRIDE=00:01:30:00 (DD:HH:MM:SS)
EXTRA="$EXTRA checkpointing.checkpoint_must_save_by=$SAVE_BY"
MOUNTS="$WORKDIR:$WORKDIR,$RL/nemo_rl:/opt/nemo-rl/nemo_rl,$AM:/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel"
[[ -n "${DATA_MOUNTS:-}" ]] && MOUNTS="$MOUNTS,$DATA_MOUNTS"
EXTRA="$EXTRA ${EXTRA_OVERRIDES:-}"  # e.g. EXTRA_OVERRIDES='grpo.val_period=2 grpo.max_num_steps=4'

# Only WANDB_API_KEY and HF_TOKEN are taken from SECRETS_FILE (sourced in a subshell). Never echoed.
SECRETS=${SECRETS_FILE:-}
if [[ -n "$SECRETS" && -r "$SECRETS" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] || WANDB_API_KEY=$(bash -c "source '$SECRETS' >/dev/null 2>&1; printf '%s' \"\${WANDB_API_KEY:-}\"")
  [[ -n "${HF_TOKEN:-}" ]] || HF_TOKEN=$(bash -c "source '$SECRETS' >/dev/null 2>&1; printf '%s' \"\${HF_TOKEN:-}\"")
fi
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN
if $LONG_RUN; then
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is not set (env or SECRETS_FILE)" >&2; exit 1; }
  export WANDB_API_KEY
fi

# SEGMENT=16 asks Slurm for 16-node segments (each inside one 18-node NVL72 rack) and
# tells NeMo RL to place workers by segment, so a 64-GPU HybridEP group never crosses
# racks (HybridEP's inter-rack RDMA path segfaulted).
SEGMENT_ARG=""
if [[ -n "$SEGMENT" ]]; then
  SEGMENT_ARG="--segment=$SEGMENT"
  EXTRA="$EXTRA cluster.segment_size=$SEGMENT"
fi
mkdir -p "$CKPT_DIR"
echo "checkpoint_dir=$CKPT_DIR  must_save_by=$SAVE_BY (limit $TIME)"
# SGLang jobs end in -sglang so they are easy to tell apart in squeue.
if [[ $MODE == vllm-* ]]; then JOB_NAME=inkling2-$MODE-gb200; else JOB_NAME=inkling2-$MODE-gb200-sglang; fi
cd "$RL"
CONTAINER=$CONTAINER MOUNTS=$MOUNTS GPUS_PER_NODE=4 BASE_LOG_DIR=$OUT \
COMMAND="$ENVS; cd /opt/nemo-rl && uv run --no-sync python examples/run_grpo.py --config $RECIPE logger.log_dir=$OUT/logs checkpointing.checkpoint_dir=$CKPT_DIR $EXTRA" \
sbatch -A "$ACCOUNT" -p "$PARTITION" -q $QOS -N$NODES --gres=gpu:4 --exclusive -t $TIME $SEGMENT_ARG \
  -J "$JOB_NAME" -o "$OUT/slurm-%j.out" ray.sub
echo "logs: $OUT"
