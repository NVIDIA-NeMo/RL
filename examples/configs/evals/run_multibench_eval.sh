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
#
# Run a NeMo-RL eval config across a fixed benchmark suite, saving each
# benchmark's results to its own subdirectory. run_eval.py evaluates a single
# dataset per invocation, so this issues one call per benchmark.
#
# Usage:
#   examples/configs/evals/run_multibench_eval.sh <eval_config.yaml> <run_name> [extra hydra overrides...]
#
# Examples:
#   examples/configs/evals/run_multibench_eval.sh \
#     examples/configs/evals/nemotron3_nano_30b_eval.yaml nemotron3_nano_30b
#
#   examples/configs/evals/run_multibench_eval.sh \
#     examples/configs/evals/qwen3vl_4b_eval.yaml qwen3vl_4b_instruct
#
# Any extra args are forwarded to every run_eval.py call (e.g. to change a
# model path or lower max_model_len for a quick smoke test).
set -euo pipefail

CONFIG="${1:?path to eval config yaml}"
RUN_NAME="${2:?short run name used in the save path}"
shift 2
EXTRA_OVERRIDES=("$@")

# Where per-benchmark results land. Defaults to a repo-local path for local
# runs; the SkyPilot job overrides EVAL_RESULTS_ROOT to a persistent FSx path.
RESULTS_ROOT="${EVAL_RESULTS_ROOT:-results/evals/${RUN_NAME}}"

# Each row: name | dataset_name | prompt_file | verifier_type | num_tests_per_prompt
# num_tests_per_prompt tuned per benchmark: AIME has only 30 problems so it
# needs more samples for a stable avg@k; MMLU-Pro is ~12k questions so 1 sample
# keeps cost sane. Tune these for your compute budget.
BENCHMARKS=(
  "math500|math500|examples/prompts/cot.txt|math|2"
  "aime2024|aime2024|examples/prompts/cot.txt|math|16"
  "aime2025|aime2025|examples/prompts/cot.txt|math|16"
  "gpqa_diamond|gpqa_diamond|examples/prompts/gpqa.txt|multilingual_multichoice|2"
  "mmlu_pro|mmlu_pro|examples/prompts/mmlu_pro.txt|english_multichoice|1"
)

echo "Config:       ${CONFIG}"
echo "Run name:     ${RUN_NAME}"
echo "Results root: ${RESULTS_ROOT}"
echo

FAILED_BENCHMARKS=()
for entry in "${BENCHMARKS[@]}"; do
  IFS='|' read -r name dataset prompt verifier ntests <<< "${entry}"
  echo "=================================================================="
  echo "  Evaluating ${RUN_NAME} on ${name} (num_tests_per_prompt=${ntests})"
  echo "=================================================================="
  # Don't let one benchmark's failure (e.g. a gated dataset, an OOM) abort the
  # whole suite: log it, record it, and keep going. The aggregation below writes
  # null for any benchmark without a results.json.
  if ! uv run python examples/run_eval.py \
    --config "${CONFIG}" \
    data.dataset_name="${dataset}" \
    data.prompt_file="${prompt}" \
    env.math.verifier_type="${verifier}" \
    eval.num_tests_per_prompt="${ntests}" \
    eval.save_path="${RESULTS_ROOT}/${name}" \
    logger.wandb_enabled=true \
    logger.log_dir="${RESULTS_ROOT}/${name}/wandb" \
    logger.wandb.project="${WANDB_PROJECT:-nemo-rl-evals}" \
    logger.wandb.name="${RUN_NAME}_${name}_eval" \
    logger.wandb.group="${RUN_NAME}_eval" \
    ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}; then
    echo "!! Benchmark '${name}' FAILED; continuing with the rest of the suite."
    FAILED_BENCHMARKS+=("${name}")
  fi
done

if [ "${#FAILED_BENCHMARKS[@]}" -gt 0 ]; then
  echo
  echo "WARNING: these benchmarks failed and will be null in summary.json: ${FAILED_BENCHMARKS[*]}"
fi

echo
echo "All benchmarks complete. Results saved under ${RESULTS_ROOT}/<benchmark>/"

# Aggregate every benchmark's results.json (written by run_eval.py) into one
# summary.json at the results root, so the whole suite's scores live in a single
# file. Missing per-benchmark files (e.g. a benchmark that crashed) are recorded
# as null rather than failing the aggregation.
echo
echo "Aggregating benchmark scores -> ${RESULTS_ROOT}/summary.json"
BENCH_NAMES="$(printf '%s\n' "${BENCHMARKS[@]}" | cut -d'|' -f1 | paste -sd, -)"
RUN_NAME="${RUN_NAME}" RESULTS_ROOT="${RESULTS_ROOT}" BENCH_NAMES="${BENCH_NAMES}" \
  python3 - <<'PY'
import json
import os

results_root = os.environ["RESULTS_ROOT"]
run_name = os.environ["RUN_NAME"]
names = [n for n in os.environ["BENCH_NAMES"].split(",") if n]

results = {}
scores = []
for name in names:
    path = os.path.join(results_root, name, "results.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        results[name] = None
        print(f"  ! {name}: results.json missing or unreadable")
        continue
    results[name] = data
    scores.append(data["score"])
    print(f"  - {name}: {data['metric']} = {data['score']:.4f}")

summary = {
    "run": run_name,
    "results": results,
    "average_over_benchmarks": (sum(scores) / len(scores)) if scores else None,
    "num_benchmarks_reported": len(scores),
    "num_benchmarks_expected": len(names),
}
out_path = os.path.join(results_root, "summary.json")
os.makedirs(results_root, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  -> wrote {out_path}")
PY
