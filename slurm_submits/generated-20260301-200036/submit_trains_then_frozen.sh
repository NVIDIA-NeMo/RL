#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Dependency type for frozen phase: "afterany" (default) or "afterok"
DEP_TYPE="${DEP_TYPE:-afterany}"
if [[ "$DEP_TYPE" != "afterany" && "$DEP_TYPE" != "afterok" ]]; then
  echo "DEP_TYPE must be afterany or afterok" >&2
  exit 2
fi

extra=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${SBATCH_EXTRA_ARGS})
fi

export_vars="${SBATCH_EXPORT_VARS:-ALL,NEMO_RL_PY_EXECUTABLES_SYSTEM=1,NRL_FORCE_REBUILD_VENVS=false,RAY_ENABLE_UV_RUN_RUNTIME_ENV=0}"

echo "Submitting TRAIN phase for all seeds..."
declare -a train_ids=()
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed123.sbatch" train)
train_ids+=("$jid")
echo "train submit_eagle_tardrift_seed123.sbatch -> $jid"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed124.sbatch" train)
train_ids+=("$jid")
echo "train submit_eagle_tardrift_seed124.sbatch -> $jid"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed125.sbatch" train)
train_ids+=("$jid")
echo "train submit_eagle_tardrift_seed125.sbatch -> $jid"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed126.sbatch" train)
train_ids+=("$jid")
echo "train submit_eagle_tardrift_seed126.sbatch -> $jid"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed127.sbatch" train)
train_ids+=("$jid")
echo "train submit_eagle_tardrift_seed127.sbatch -> $jid"
dep=$(IFS=:; echo "${train_ids[*]}")
echo "Submitting FROZEN phase for all seeds with dependency ${DEP_TYPE}:${dep}"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" --dependency="${DEP_TYPE}:${dep}" "submit_eagle_tardrift_seed123.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed123.sbatch -> $jid (dep ${DEP_TYPE}:${dep})"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" --dependency="${DEP_TYPE}:${dep}" "submit_eagle_tardrift_seed124.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed124.sbatch -> $jid (dep ${DEP_TYPE}:${dep})"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" --dependency="${DEP_TYPE}:${dep}" "submit_eagle_tardrift_seed125.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed125.sbatch -> $jid (dep ${DEP_TYPE}:${dep})"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" --dependency="${DEP_TYPE}:${dep}" "submit_eagle_tardrift_seed126.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed126.sbatch -> $jid (dep ${DEP_TYPE}:${dep})"
jid=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" --dependency="${DEP_TYPE}:${dep}" "submit_eagle_tardrift_seed127.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed127.sbatch -> $jid (dep ${DEP_TYPE}:${dep})"
echo "Done."
