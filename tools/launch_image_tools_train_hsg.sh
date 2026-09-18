#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${PROJECT_ROOT:?Completed immutable HSG snapshot required}"
: "${RUN_DIR:?New user-owned HSG results directory required}"
: "${OVERLAY_DIR:?Qualified additive overlay required}"
: "${MODEL_CHECKPOINT:?Use the user-selected step_120/hf checkpoint}"
: "${TRAIN_MANIFEST:?Content-grouped HSG training manifest required}"
: "${EVAL_MANIFEST:?Content-grouped HSG validation manifest required}"
: "${WANDB_RUN_ID:?Stable unique W&B run ID required}"
DRY_RUN=${DRY_RUN:-1}
IMAGE_TOOLS_SUITE=${IMAGE_TOOLS_SUITE:-standalone}
case "$IMAGE_TOOLS_SUITE" in standalone|visual-games) ;; *) exit 2;; esac
HSG_ROOT=${HSG_ROOT:?Set your shared HSG user root}
case "$PROJECT_ROOT" in "$HSG_ROOT"/*/snapshots/*) ;; *) exit 2;; esac
case "$RUN_DIR" in "$HSG_ROOT"/*/results/*) ;; *) exit 2;; esac
case "$OVERLAY_DIR" in "$HSG_ROOT"/*/cache/*) ;; *) exit 2;; esac
case "$DRY_RUN" in 0|1) ;; *) exit 2;; esac
if test -e "$RUN_DIR"; then echo 'Refusing to reuse RUN_DIR' >&2; exit 2; fi
test -r "$PROJECT_ROOT/.env"
test -r "$OVERLAY_DIR/READY.json"
test -r "$MODEL_CHECKPOINT/model.safetensors.index.json"
test -r "$EVAL_MANIFEST"
if [[ "$IMAGE_TOOLS_SUITE" == standalone ]]; then
  : "${QUALIFICATION_LOG:?Successful runtime qualification stdout required}"
  test -r "$TRAIN_MANIFEST"
  python3 "$PROJECT_ROOT/tools/image_tools_launch_checks.py" --log "$QUALIFICATION_LOG" --overlay "$OVERLAY_DIR" --lock "$PROJECT_ROOT/uv.lock" --project-root "$PROJECT_ROOT" --model-checkpoint "$MODEL_CHECKPOINT"
  training_nodes=8
else
  : "${IMAGE_QUALIFIED_SOURCE:?}" "${IMAGE_QUALIFIED_RUN:?}" "${COMPONENT_SOURCE:?}" "${COMPONENT_RUN:?}"
  : "${GAMES_VENV_ROOT:?}" "${GAMES_TRAIN_MANIFEST:?}" "${IMAGE_TRAIN_MANIFEST:?}" "${VISGYM_ASSET_ARCHIVE:?}"
  test "$TRAIN_MANIFEST" = "$RUN_DIR/train.jsonl"
  test -r "$VISGYM_ASSET_ARCHIVE"
  # Fail before requesting nodes if the offline test dependency closure is absent.
  python3 "$PROJECT_ROOT/tools/callback_test_runtime.py" >/dev/null
  export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  python3 -m tools.check_visual_image_tools_launch --project-root "$PROJECT_ROOT" \
    --image-source "$IMAGE_QUALIFIED_SOURCE" --image-run "$IMAGE_QUALIFIED_RUN" \
    --component-source "$COMPONENT_SOURCE" --component-run "$COMPONENT_RUN" \
    --games-venv-root "$GAMES_VENV_ROOT" --overlay "$OVERLAY_DIR" --model-checkpoint "$MODEL_CHECKPOINT"
  python3 -m tools.check_visual_image_tools_mix --train "$GAMES_TRAIN_MANIFEST" \
    --train "$IMAGE_TRAIN_MANIFEST" --validation "$EVAL_MANIFEST" --max-output-tokens 512
  training_nodes=40
fi
OPERATION_ROOT=${PROJECT_ROOT%/snapshots/*}
if [[ -n "${RESUME_CHECKPOINT_DIR:-}" ]]; then
  case "$RESUME_CHECKPOINT_DIR" in "$OPERATION_ROOT"/results/*/checkpoints) ;; *) echo 'Resume checkpoints must belong to this operation' >&2; exit 2;; esac
  : "${RESUME_STEP:?Exact completed step required for resume}"
  python3 - "$RESUME_CHECKPOINT_DIR" "$RESUME_STEP" <<'PY'
import json
import sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
steps = [int(p.name[5:]) for p in root.glob('step_*') if p.name[5:].isdigit()]
assert steps and max(steps) == expected, (steps, expected)
checkpoint = root / f'step_{expected}'
assert json.loads((checkpoint / 'training_info.json').read_text())['current_step'] == expected
assert json.loads((root / 'latest_checkpoint_status.json').read_text())['last_checkpoint_step'] == expected
for name in ['config.yaml', 'train_dataloader.pt', 'rollouts.pt', 'replay_buffer.pt',
             'policy/weights/iter_0000000/.metadata']:
    assert (checkpoint / name).stat().st_size > 0, name
print(f'IMAGE_TOOLS_RESUME_CHECKPOINT_OK path={checkpoint}', flush=True)
PY
  export CHECKPOINT_DIR="$RESUME_CHECKPOINT_DIR" IMAGE_TOOLS_WANDB_RESUME=must RESUME_STEP
else
  export CHECKPOINT_DIR="$RUN_DIR/checkpoints" IMAGE_TOOLS_WANDB_RESUME=allow
fi
export CONTAINER="${CONTAINER:?Set the qualified container image path}"
test -r "$CONTAINER"
export IMAGE_TOOLS_RUN_NAME="super-image-tools-step120-$(basename "$RUN_DIR")"
export IMAGE_TOOLS_WANDB_ID="$WANDB_RUN_ID"
export IMAGE_TOOLS_CONFIG=/opt/nemo-rl/examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-image-tools-8n4g-megatron-tp8ep16cp2-async.v1.yaml
if [[ "$IMAGE_TOOLS_SUITE" == visual-games ]]; then
  export IMAGE_TOOLS_CONFIG=/opt/nemo-rl/examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-visual-games-image-tools-40n4g-megatron-tp8ep16cp2-async.v1.yaml
  export IMAGE_TOOLS_RUN_NAME="super-visual-games-image-tools-step120-$(basename "$RUN_DIR")"
fi
export IMAGE_TOOLS_RUN_NAME="${RESUME_WANDB_RUN_NAME:-$IMAGE_TOOLS_RUN_NAME}"
export IMAGE_TOOLS_SUITE
export MODEL_CHECKPOINT TRAIN_MANIFEST EVAL_MANIFEST OVERLAY_DIR RUN_DIR
export RUN_LOG_DIR="$RUN_DIR/logs"
export VLLM_TOKENIZER="$RUN_DIR/tokenizer"
export WANDB_MODE=online WANDB_DISABLED=false WANDB_RESUME="$IMAGE_TOOLS_WANDB_RESUME"
export WANDB_PROJECT=games-rlvr-nemotron-super WANDB_ENTITY=nvidia
export WANDB_RUN_NAME="$IMAGE_TOOLS_RUN_NAME" WANDB_RUN_ID
export NEMO_RL_VENV_DIR=/opt/ray_venvs NEMO_GYM_VENV_DIR=/opt/gym_venvs
export NRL_FORCE_REBUILD_VENVS=false NRL_PATCH_NSIGHT=0
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_RAY_CLI=/opt/nemo_rl_venv/bin/ray NRL_RAY_PYTHON=/opt/nemo_rl_venv/bin/python
export IMAGE_TOOLS_BUILD_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
export PATH="/opt/nemo-rl/tools/image_tools_build_bin:$PATH"
export VIRTUAL_ENV=/opt/nemo_rl_venv UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_SOURCE_OVERRIDE_ROOT=/opt/nemo-rl
export NEMO_GYM_EXTRA_ROOTS=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym
export HF_HOME="$RUN_DIR/cache/huggingface" HF_MODULES_CACHE="$RUN_DIR/cache/huggingface/modules"
export NRL_MEGATRON_CHECKPOINT_DIR="$OPERATION_ROOT/cache/megatron-checkpoints"
export MEGATRON_CONFIG_LOCK_DIR="$RUN_DIR/cache/hf-locks"
# ray.sub bind-mounts UV_CACHE_DIR_OVERRIDE over /root/.cache/uv. The bundled
# venvs symlink into that directory, so an empty replacement breaks imports.
unset UV_CACHE_DIR_OVERRIDE
export IMAGE_TOOLS_OUTPUT_DIR="$RUN_DIR/image-tool-crops"
export NEMO_GYM_COMPACT_ROLLOUT_DIR="$RUN_DIR/rollout-diagnostics"
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 GPUS_PER_NODE=4
export BASE_LOG_DIR="$RUN_LOG_DIR/ray" RAY_LOG_SYNC_FREQUENCY=60
export PYTHONPATH="$OVERLAY_DIR/packages:/opt/nemo-rl/tools/runtime_source_override:$HF_MODULES_CACHE:/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
export MOUNTS="$OPERATION_ROOT:$OPERATION_ROOT,$PROJECT_ROOT:/opt/nemo-rl:ro,$MODEL_CHECKPOINT:$MODEL_CHECKPOINT:ro,$RUN_DIR/gym-cache:/opt/nemo-rl/3rdparty/Gym-workspace/Gym/cache"
# The content-addressed dataset belongs to the earlier operation, not this new source mirror.
DATA_ROOT="$HSG_ROOT/visual-games-image-tools-21647a62f8/data"
export MOUNTS="$MOUNTS,$DATA_ROOT:$DATA_ROOT:ro"
if [[ "$IMAGE_TOOLS_SUITE" == visual-games ]]; then
  # Add only the four independent game components to the baked image services.
  # Never mount an empty replacement over /opt/gym_venvs or its uv archive.
  for component in resources_servers/gym_v resources_servers/visgym responses_api_agents/gymv_agent responses_api_agents/visgym_agent; do
    test -d "$GAMES_VENV_ROOT/$component/.venv"
    export MOUNTS="$MOUNTS,$GAMES_VENV_ROOT/$component:/opt/gym_venvs/$component:ro"
  done
  export MOUNTS="$MOUNTS,$EVAL_MANIFEST:$EVAL_MANIFEST:ro,$VISGYM_ASSET_ARCHIVE:$VISGYM_ASSET_ARCHIVE:ro"
  export NEMO_GYM_EXTRA_ROOTS="$NEMO_GYM_EXTRA_ROOTS:$RUN_DIR/assets"
  export VISGYM_ASSET_ARCHIVE
  export MPLBACKEND=Agg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
fi
MCORE_DATASETS_REL=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets
export IMAGE_TOOLS_DATASET_DIR="/opt/nemo-rl/$MCORE_DATASETS_REL"
export MOUNTS="$MOUNTS,$RUN_DIR/runtime/mcore-datasets:$IMAGE_TOOLS_DATASET_DIR"
export COMMAND='exec bash /opt/nemo-rl/tools/image_tools_train_hsg.sh'
export SETUP_COMMAND='set -euo pipefail
bash /opt/nemo-rl/tools/image_tools_dataset_helpers.sh build "$IMAGE_TOOLS_DATASET_DIR"
for fqn in nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker nemo_rl.algorithms.async_utils.ReplayBuffer nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector nemo_rl.environments.nemo_gym.NemoGym; do
  test -x "/opt/ray_venvs/$fqn/bin/python"
done
test -r /opt/nemo_rl_venv/lib/python3.13/site-packages/transformers/__init__.py
/opt/nemo_rl_venv/bin/python -c "from transformers import AutoConfig, AutoProcessor, AutoTokenizer; print(\"IMAGE_TOOLS_NODE_IMPORTS_OK\", flush=True)"'
submit=(sbatch --parsable --account="${SLURM_ACCOUNT:?Set your Slurm account}" --partition="${SLURM_PARTITION:-batch}" --qos="${SLURM_QOS:-normal}"
  --nodes="$training_nodes" --ntasks-per-node=1 --gpus-per-node=4 --exclusive --time=04:00:00
  --job-name="$IMAGE_TOOLS_RUN_NAME" --chdir="$PROJECT_ROOT"
  --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" "$PROJECT_ROOT/ray.sub")
printf 'W&B: https://wandb.ai/nvidia/games-rlvr-nemotron-super/runs/%s (requested run ID; not yet connected)\n' "$WANDB_RUN_ID"
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${submit[@]}"; printf '\n'; exit 0; fi
mkdir "$RUN_DIR"
if [[ "$IMAGE_TOOLS_SUITE" == visual-games ]]; then
  PYTHONPATH="$PROJECT_ROOT" python3 -m tools.prepare_visual_image_tools_mix \
    --games "$GAMES_TRAIN_MANIFEST" --images "$IMAGE_TRAIN_MANIFEST" \
    --validation "$EVAL_MANIFEST" --output "$TRAIN_MANIFEST" > "$RUN_DIR/manifest-audit.json"
fi
mkdir "$RUN_DIR/runtime"
bash "$PROJECT_ROOT/tools/image_tools_dataset_helpers.sh" stage "$PROJECT_ROOT/$MCORE_DATASETS_REL" "$RUN_DIR/runtime/mcore-datasets"
mkdir -p "$RUN_LOG_DIR" "$RUN_DIR/gym-cache" "$HF_MODULES_CACHE" "$NRL_MEGATRON_CHECKPOINT_DIR" "$MEGATRON_CONFIG_LOCK_DIR" "$IMAGE_TOOLS_OUTPUT_DIR" "$NEMO_GYM_COMPACT_ROLLOUT_DIR"
cd "$PROJECT_ROOT"
job_id=$("${submit[@]}")
printf '%s\n' "$job_id" > "$RUN_DIR/gpu-job-id.txt"
printf 'Submitted image-tool training job %s\n' "$job_id"
