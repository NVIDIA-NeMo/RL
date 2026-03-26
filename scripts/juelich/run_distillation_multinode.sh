#!/bin/bash
# Submits a 2-node distillation job using the Jülich 1.7B self-distill config.
# All parameters (topk, batch size, naming, etc.) are defined in the YAML config —
# no overrides needed.
#
# Usage: ./run_distillation_multinode.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="examples/configs/opsd/juelich/distill-1.7b-to-1.7b-2n4g-gen8k.yaml"

export COMMAND="uv run python examples/run_distillation.py --config ${CONFIG} \
    data.default.prompt_file=examples/prompts/prefix.txt \
    data.default.teacher_prompt_file=examples/prompts/teacher-concise.txt"

sbatch --nodes=2 --job-name="distill-1.7b-self" "$SCRIPT_DIR/ray.sub"
