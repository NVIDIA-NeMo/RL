#!/usr/bin/env bash
# Smoke-test launcher for the gym-v multimodal recipes on a single interactive
# allocation (2 nodes × 8 GPUs). Run this from INSIDE the NeMo-RL container,
# with the repo checked out at the working directory (typically /opt/nemo-rl).
#
# Usage:
#   examples/nemo_gym/run_gymv_smoke.sh qwen                # Qwen2.5-VL-3B (1n, 4vllm+4mcore)
#   examples/nemo_gym/run_gymv_smoke.sh nemotron            # Nemotron-Omni-30B (2n, 1vllm+1mcore)
#   examples/nemo_gym/run_gymv_smoke.sh <recipe.yaml>       # any recipe under examples/nemo_gym/
#
# Extra CLI args after the recipe pick are forwarded to the training script,
# so you can layer Hydra overrides on top, e.g.:
#   examples/nemo_gym/run_gymv_smoke.sh qwen grpo.max_num_steps=1

set -euo pipefail

RECIPE_KEY="${1:-qwen}"; shift || true

case "${RECIPE_KEY}" in
    qwen)
        RECIPE="examples/nemo_gym/grpo_qwen25vl_gymv_smoke.yaml"
        ;;
    nemotron|omni)
        RECIPE="examples/nemo_gym/grpo_nemotron_omni_30ba3b_gymv_smoke.yaml"
        ;;
    *)
        RECIPE="${RECIPE_KEY}"
        ;;
esac

if [[ ! -f "${RECIPE}" ]]; then
    echo "error: recipe not found at ${RECIPE}" >&2
    exit 1
fi

# HF token — needed to download Qwen2.5-VL-3B / Nemotron-Omni checkpoints.
# Sourced from the environment; if you keep it in a dotenv, `source` it first.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "warn: HF_TOKEN unset — HF hub downloads may 401" >&2
fi

# Shared caches keep model weights on the mounted Lustre workspace rather than
# in the container's ephemeral rootfs. Override via env if you already have
# these pointed elsewhere.
export HF_HOME="${HF_HOME:-${PWD}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

# Ray's AF_UNIX socket path is capped at 107 bytes on Linux. On Lustre-rooted
# working directories (long paths) the default under $PWD/tmp overruns it and
# Ray fails to spin up. Force it under /tmp.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
mkdir -p "${RAY_TMPDIR}"

# Multimodal recipes use the multimodal entry point; the text-only path uses
# run_grpo_nemo_gym.py (kept here as a fallback branch even though both
# checked-in recipes currently need the multimodal script).
if grep -q '^\s*is_vlm:\s*true' "${RECIPE}"; then
    ENTRY="examples/nemo_gym/run_multimodal_grpo_nemo_gym.py"
else
    ENTRY="examples/nemo_gym/run_grpo_nemo_gym.py"
fi

echo "==> recipe: ${RECIPE}"
echo "==> entry:  ${ENTRY}"
echo "==> HF_HOME=${HF_HOME}"
echo "==> RAY_TMPDIR=${RAY_TMPDIR}"

	#--locked --no-sync \

exec uv run \
    "${ENTRY}" \
    --config "${RECIPE}" \
    "$@"
