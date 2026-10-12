#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CPU-only validation of the built container against this checkout: worker venv
# imports, and the SingleController master config for the AgenticVBench recipe
# with context compaction on and off (validate_single_controller_config, the
# same check setup_single_controller runs before allocating workers).
#   bash scripts/agentic_vbench/container/validate_container.sh [CONTAINER]
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../../.." && pwd)
container=${1:-${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh}}
[[ -f $container ]] || { echo "container not found: $container" >&2; exit 1; }
recipe=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc.v1.yaml}
: "${MODEL_PATH:?MODEL_PATH must point to a complete Hugging Face checkpoint visible in the container}"
model_path=${MODEL_PATH}
mounts="/lustre:/lustre,/scratch:/scratch"
for sub in nemo_rl examples tools scripts 3rdparty/Gym-workspace/Gym 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge; do
  mounts+=",$code_dir/$sub:/opt/nemo-rl/$sub"
done
export MODEL_PATH=$model_path TRAIN_PATH=${TRAIN_PATH:-/dev/null} VAL_PATH=${VAL_PATH:-/dev/null} AVB_IMAGE_CACHE=${AVB_IMAGE_CACHE:-/tmp/avb-images}
srun --account="${SLURM_ACCOUNT:-nemotron_n4_omni}" --partition="${VALIDATE_PARTITION:-cpu}" --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=00:40:00 --job-name=nrlcc-validate \
  --container-image="$container" --container-mounts="$mounts" --container-workdir=/opt/nemo-rl \
  --export=ALL,NRL_CONTAINER=1,NRL_IGNORE_VERSION_MISMATCH=1,UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv,RAY_USAGE_STATS_ENABLED=0 \
  bash -c '
set -euo pipefail
cd /opt/nemo-rl
# As in launch_sc.sh: the driver venv imports Gym config helpers from the source tree.
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:${PYTHONPATH:-}
echo "== fingerprint =="; cat /opt/nemo_rl_container_fingerprint; echo
echo "== worker venvs =="; ls /opt/ray_venvs
uv run --no-sync python -c "import ray, transformers, nemo_rl.algorithms.single_controller, nemo_rl.environments.nemo_gym; print(\"driver venv ok\", ray.__version__, transformers.__version__)"
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python -c "import vllm, torch; from nemo_rl.models.generation.vllm.vllm_worker_async import VllmAsyncGenerationWorker; from nemo_rl.models.generation.capture_context import decide_capture_input; print(\"vllm venv ok\", vllm.__version__, torch.__version__)"
# Transformer Engine dlopens libcuda at import; on a CPU node only check the installs exist.
/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python -c "import importlib.util as u, importlib.metadata as m; assert u.find_spec(\"megatron.core\") and u.find_spec(\"transformer_engine\") and u.find_spec(\"megatron.bridge\"); print(\"mcore venv ok TE\", m.version(\"transformer-engine\"))"
/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python -c "import nemo_rl.environments.gym_selection, nemo_gym.token_id_capture.completion, responses_api_agents.agentic_vbench_agent.app, responses_api_agents.agentic_vbench_agent.capture_proxy; print(\"gym venv ok\")"
/opt/ray_venvs/nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor/bin/python -c "import nemo_rl.experience.rollout_reassembler; print(\"reassembler venv ok\")"
echo "== Ray version consistency (the head runs from /opt/nemo_rl_venv; every worker venv must match) =="
head_ray=$(/opt/nemo_rl_venv/bin/python -c "import ray; print(ray.__version__)")
for v in /opt/ray_venvs/*; do
  worker_ray=$("$v/bin/python" -c "import ray; print(ray.__version__)")
  printf "%-95s ray=%s\n" "$v" "$worker_ray"
  [[ "$worker_ray" == "$head_ray" ]] || { echo "Ray version skew: $v has $worker_ray, head venv has $head_ray" >&2; exit 1; }
done
echo "ray=$head_ray everywhere"
echo "== SingleController config validation =="
uv run --no-sync python - "'"$recipe"'" <<PY
import sys
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.algorithms.single_controller_utils.config import validate_single_controller_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
register_omegaconf_resolvers()
recipe = sys.argv[1]
common = [
    "grpo.num_prompts_per_step=8", "grpo.num_generations_per_prompt=4", "policy.train_global_batch_size=32",
    "token_capture.enabled=true", "token_capture.capture_dir=/tmp/token_capture",
    "cluster.num_nodes=8", "++cluster.segment_size=4", "data_plane.simple.num_storage_units=16",
    "policy.generation.colocated.resources.num_nodes=4", "logger.wandb_enabled=false",
]
for cc in ("false", "true"):
    cfg = parse_hydra_overrides(load_config(recipe), common + [f"token_capture.context_compaction={cc}"])
    master = MasterConfig(**OmegaConf.to_container(cfg, resolve=True))
    assert master.grpo.async_grpo is None and master.data_plane.get("enabled")
    validate_single_controller_config(master)
    print(f"validate_single_controller_config ok: context_compaction={cc}")
PY
echo CONTAINER_VALIDATION_OK'
