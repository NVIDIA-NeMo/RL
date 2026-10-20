#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Exercise the real launcher in a minimal synthetic checkout. This keeps the
# contract test independent of whether optional git submodules were initialized
# in the caller's worktree; neither DRY_RUN nor the submission stub executes
# these placeholder training sources.
fixture=$tmp/repo
mkdir -p \
  "$fixture/scripts/agentic_vbench" \
  "$fixture/examples" \
  "$fixture/3rdparty/Gym-workspace/Gym/responses_api_agents/agentic_vbench_agent" \
  "$fixture/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
cp "$repo_dir/scripts/agentic_vbench/launch_sc.sh" "$fixture/scripts/agentic_vbench/launch_sc.sh"
printf '# placeholder\n' >"$fixture/recipe.yaml"
printf '# placeholder\n' >"$fixture/examples/run_grpo_single_controller.py"
printf '# placeholder\n' >"$fixture/3rdparty/Gym-workspace/Gym/responses_api_agents/agentic_vbench_agent/app.py"
printf '# placeholder\n' >"$fixture/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/.fixture"
printf '# placeholder\n' >"$fixture/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/.fixture"
printf '#!/usr/bin/env bash\n' >"$fixture/ray.sub"
git init --quiet --initial-branch=fixture "$fixture"
git -C "$fixture" add .
git -C "$fixture" -c user.name='Launcher fixture' -c user.email='fixture@example.invalid' \
  commit --quiet -m 'Create committed launcher fixture'

model=$tmp/model
mkdir -p "$model"
printf '{}\n' >"$model/config.json"
printf '{"weight_map":{"dummy":"model-00001-of-00001.safetensors"}}\n' >"$model/model.safetensors.index.json"
printf 'x\n' >"$model/model-00001-of-00001.safetensors"
printf '{}\n' >"$model/tokenizer.json"
printf '{}\n' >"$model/processor_config.json"
printf '{}\n' >"$tmp/train.jsonl"
touch "$tmp/container.sqsh"

run_dry() {
  local results=$1
  shift
  env \
    DRY_RUN=1 \
    ENV_FILE=/dev/null \
    WANDB_API_KEY=submission-fixture-does-not-call-wandb \
    WANDB_CLUSTER_NAME=dfw \
    LAUNCH_BENCHMARK=agentic_vbench \
    PERSISTENT_CACHE="$tmp/cache" \
    CONTAINER="$tmp/container.sqsh" \
    MODEL_PATH="$model" \
    TRAIN_PATH="$tmp/train.jsonl" \
    VAL_PATH="$tmp/train.jsonl" \
    RESULTS_DIR="$results" \
    BASE_NAME=agentic_vbench_long_cycle_contract \
    CONFIG_IN_CONTAINER=recipe.yaml \
    MAX_STEPS=1000000 \
    SLURM_QOS=normal \
    SLURM_TIME_LIMIT=24:00:00 \
    MIN_SLURM_TIME_SECONDS=86400 \
    MIN_CHECKPOINT_BUDGET_SECONDS=82800 \
    REQUIRE_RESUMABLE_JOB_CYCLES=true \
    JOB_CYCLES=6 \
    "$@" \
    bash "$fixture/scripts/agentic_vbench/launch_sc.sh"
}

run_dry "$tmp/results" >"$tmp/output"

grep -Fqx 'time_limit=24:00:00 checkpoint_must_save_by=00:23:40:00' "$tmp/output"
grep -Fqx 'allocation_boundary_seconds=86400 controller_boundary_seconds=85200 grpo_horizon_steps=1000000 continuation_chain=afterany cycles=6' "$tmp/output"
grep -F -- '--time=24:00:00' "$tmp/output" >/dev/null
[[ $(grep -c '^sbatch ' "$tmp/output") == 6 ]]
if grep -F -- '--array=' "$tmp/output" >/dev/null; then
  echo 'launcher used array throttling instead of explicit cycle dependencies' >&2
  exit 1
fi
grep -F -- '--qos=normal' "$tmp/output" >/dev/null
grep -F 'grpo.max_num_steps="1000000"' "$tmp/results/training_command.sh" >/dev/null
grep -F 'grpo.max_num_epochs=null' "$tmp/results/training_command.sh" >/dev/null
grep -F 'checkpointing.checkpoint_must_save_by="00:23:40:00"' "$tmp/results/training_command.sh" >/dev/null

if run_dry "$tmp/too-short" SLURM_TIME_LIMIT=02:00:00 >/dev/null 2>&1; then
  echo 'launcher accepted a cycle below MIN_SLURM_TIME_SECONDS' >&2
  exit 1
fi

if run_dry "$tmp/short-controller" CHECKPOINT_MUST_SAVE_BY=00:01:40:00 >/dev/null 2>&1; then
  echo 'launcher accepted a controller boundary below MIN_CHECKPOINT_BUDGET_SECONDS' >&2
  exit 1
fi

if run_dry "$tmp/no-continuation" JOB_CYCLES=1 >/dev/null 2>&1; then
  echo 'launcher accepted a non-resumable one-cycle production launch' >&2
  exit 1
fi

# Exercise actual submission control flow without scheduling model compute.
mkdir -p "$tmp/bin"
cat >"$tmp/bin/sbatch" <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
count=0
[[ ! -f $SBATCH_TEST_COUNT ]] || count=$(cat "$SBATCH_TEST_COUNT")
count=$((count + 1))
printf '%s\n' "$count" >"$SBATCH_TEST_COUNT"
printf '%s\0' "$@" >"${SBATCH_TEST_COUNT}.args-${count}"
[[ ${SBATCH_TEST_FAIL_AT:-0} != "$count" ]] || exit 42
if [[ ${SBATCH_TEST_BAD_AT:-0} == "$count" ]]; then
  printf 'not-a-job-id\n'
else
  printf '%s;fixture-cluster\n' "$((70000 + count))"
fi
SBATCH
chmod +x "$tmp/bin/sbatch"
run_dry "$tmp/chained" DRY_RUN=0 JOB_CYCLES=20 \
  NUM_NODES=16 NUM_GEN_NODES=8 SEGMENT_SIZE=8 \
  CONTAINER="$tmp/container.sqsh" PATH="$tmp/bin:$PATH" \
  SBATCH_TEST_COUNT="$tmp/submitted" >"$tmp/chained-output"
python3 - "$tmp" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
assert (root / "submitted").read_text().strip() == "20"
for index in range(1, 21):
    args = (root / f"submitted.args-{index}").read_bytes().decode().split("\0")[:-1]
    dependency = "singleton" if index == 1 else f"afterany:{70000 + index - 1}"
    assert f"--dependency={dependency}" in args, (index, args)
    assert "--nodes=16" in args and not any(arg.startswith("--array=") for arg in args)
    assert any(arg.startswith("--output=") and "%j" in arg for arg in args)
receipts = list((root / "chained").glob("job_cycles.*"))
assert len(receipts) == 1
rows = [line.split("\t") for line in receipts[0].read_text().splitlines()]
assert len(rows) == 20
assert rows[0] == ["1", "70001", "singleton"]
assert rows[-1] == ["20", "70020", "afterany:70019"]
PY

for failure in FAIL BAD; do
  if run_dry "$tmp/failure-$failure" DRY_RUN=0 JOB_CYCLES=20 \
    CONTAINER="$tmp/container.sqsh" PATH="$tmp/bin:$PATH" \
    SBATCH_TEST_COUNT="$tmp/count-$failure" "SBATCH_TEST_${failure}_AT=2" \
    >"$tmp/failure-$failure.out" 2>&1; then
    echo "launcher accepted failed or ambiguous submission: $failure" >&2
    exit 1
  fi
  [[ $(cat "$tmp/count-$failure") == 2 ]]
done

run_dry "$tmp/custom-dependency" DRY_RUN=0 JOB_CYCLES=2 \
  SLURM_DEPENDENCY=afterok:12345 CONTAINER="$tmp/container.sqsh" \
  PATH="$tmp/bin:$PATH" SBATCH_TEST_COUNT="$tmp/custom-count" \
  >"$tmp/custom-output"
grep -F 'dependency=afterok:12345' "$tmp/custom-output" >/dev/null
grep -F 'dependency=afterany:70001' "$tmp/custom-output" >/dev/null

echo 'agentic-vbench long-cycle contracts: PASS'
