#!/usr/bin/env bash
set -Eeuo pipefail

physical_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "${physical_root}" == /lustre/fs?/* ]]; then
  ROOT="/lustre/fsw/${physical_root#/*/*/}"
else
  ROOT="${physical_root}"
fi
RUNTIME_ROOT="${OSWORLD_RUNTIME_ROOT:-$(dirname "${ROOT}")/osworld-cc-runtime}"

: "${OSWORLD_GRPO_VAL_DATA:?Set OSWORLD_GRPO_VAL_DATA to the full 361-task JSONL}"
: "${EVAL_NAME:?Set EVAL_NAME}"

PARITY_SHARD_SOURCE_DIR="${PARITY_SHARD_SOURCE_DIR:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/jianh/data/osworld-eval}"
PARITY_SHARD_DIR="${PARITY_SHARD_DIR:-${RUNTIME_ROOT}/data/eval-parity-shards}"
SUBMIT="${EVAL_SUBMIT_SCRIPT:-${ROOT}/examples/nemo_gym/submit_osworld_cc_eval.sh}"
mkdir -p "${PARITY_SHARD_DIR}"

python - "${OSWORLD_GRPO_VAL_DATA}" "${PARITY_SHARD_SOURCE_DIR}" "${PARITY_SHARD_DIR}" <<'PY'
import json
import os
import sys
from pathlib import Path

full_path = Path(sys.argv[1])
reference_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

def task_id(row):
    metadata = row.get("verifier_metadata") or {}
    return (
        metadata.get("id")
        or metadata.get("example_id")
        or row.get("example_id")
        or str(row.get("id", "")).split("/")[-1]
    )

full_rows = [json.loads(line) for line in full_path.read_text().splitlines() if line.strip()]
full_by_id = {task_id(row): row for row in full_rows}
if len(full_rows) != 361 or len(full_by_id) != 361:
    raise SystemExit(
        f"Expected 361 unique NeMo-Gym validation tasks, got rows={len(full_rows)} unique={len(full_by_id)}"
    )

used = set()
expected_counts = [91, 90, 90, 90]
for index, expected_count in enumerate(expected_counts, start=1):
    shard = f"{index:02d}"
    reference_path = reference_dir / f"validation_nogdrive_361_q{shard}.jsonl"
    reference_rows = [
        json.loads(line)
        for line in reference_path.read_text().splitlines()
        if line.strip()
    ]
    ids = [task_id(row) for row in reference_rows]
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise SystemExit(
            f"Reference shard {shard} expected {expected_count} unique tasks, got {len(ids)}"
        )
    missing = [value for value in ids if value not in full_by_id]
    if missing:
        raise SystemExit(f"NeMo-Gym validation data is missing {shard} task IDs: {missing[:5]}")
    overlap = used.intersection(ids)
    if overlap:
        raise SystemExit(f"Reference shard {shard} overlaps prior shards: {sorted(overlap)[:5]}")
    used.update(ids)

    output_path = output_dir / f"validation_nogdrive_361_q{shard}.jsonl"
    tmp_path = output_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as output:
        for value in ids:
            output.write(json.dumps(full_by_id[value], separators=(",", ":")) + "\n")
    os.replace(tmp_path, output_path)

if used != set(full_by_id):
    raise SystemExit(
        f"Jianh parity shards do not cover the full NeMo-Gym set: covered={len(used)} full={len(full_by_id)}"
    )
print("Prepared Jianh parity shards: 91 + 3x90 = 361 tasks", file=sys.stderr)
PY

job_ids=()
PARITY_SHARDS="${PARITY_SHARDS:-01 02 03 04}"
for shard in ${PARITY_SHARDS}; do
  [[ "${shard}" =~ ^0[1-4]$ ]] || {
    echo "Invalid parity shard: ${shard}" >&2
    exit 2
  }
  shard_data="${PARITY_SHARD_DIR}/validation_nogdrive_361_q${shard}.jsonl"
  shard_count="$(wc -l < "${shard_data}")"
  submit_output="$(
    OSWORLD_GRPO_VAL_DATA="${shard_data}" \
    EVAL_NAME="${EVAL_NAME}-q${shard}" \
    EVAL_NUM_GENERATIONS=4 \
    EVAL_TEMPERATURE=0.6 \
    EVAL_TOP_P=1.0 \
    EVAL_MAX_STEPS=100 \
    EVAL_VAL_BATCH_SIZE="${shard_count}" \
    EVAL_NUM_WORKERS=32 \
    NUM_NODES=1 \
    "${SUBMIT}"
  )"
  job_id="${submit_output##* }"
  job_ids+=("${job_id}")
  echo "Submitted Jianh-parity eval shard=${shard} tasks=${shard_count} job=${job_id}" >&2
done

printf '%s\n' "${job_ids[*]}"
