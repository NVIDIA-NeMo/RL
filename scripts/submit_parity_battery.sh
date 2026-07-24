#!/bin/bash
# Submit the multi-LoRA parity battery: {noclip, clip1} x {multi, single_a..d}.
#
# Ten 100-step runs from identical exact-init LoRA shards; per-adapter loss
# traces ON. Multi-LoRA trains 4 adapters concurrently on splits a-d; each
# single run trains one plain-LoRA adapter on its own split. Equivalence holds
# when per-adapter multi curves match the corresponding single curves.
#
# Prereq: canonical exact-init shards (rank_RR_slot_II.pt). Generate once with
# a native export run (NOUSNET_INIT_EXPORT_DIR=<path>) or point CANON at an
# existing set.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CANON="${CANON:-${REPO_ROOT}/results/code7x_exactinit_canonical}"
LAUNCHER=examples/configs/recipes/multi_lora/sft_8gpu_native.slurm
CFGDIR=examples/configs/recipes/multi_lora
BASE_ENV="NOUSNET_DIAG_ENABLED=1,NOUSNET_DIAG_LOSS_TRACE=1,NOUSNET_DIAG_TRACE_ONLY=1,NOUSNET_DIAG_LORA_STEP=0,NOUSNET_FORCE_PAD_TO=1024,NOUSNET_INIT_IMPORT_DIR=${CANON}"

if [[ ! -d "${CANON}" ]]; then
  cat >&2 <<EOF
ERROR: canonical exact-init shards not found: ${CANON}
Generate them with a native export run (NOUSNET_INIT_EXPORT_DIR=<path>) or set
CANON=/path/to/rank_RR_slot_II.pt shards.
EOF
  exit 1
fi

for CKPT in results/parity100_{noclip,clip1}_{multi,single_a,single_b,single_c,single_d}_ckpt; do
  if [[ -e "${CKPT}" ]]; then
    echo "ERROR: fresh-run checkpoint path already exists: ${CKPT}" >&2
    exit 1
  fi
done

# Slot = the exact-init shard each single run imports (adapter_a..d -> 0..3).
declare -A SLOT=( [a]=0 [b]=1 [c]=2 [d]=3 )
SUBMITTED=""
for MODE in noclip clip1; do
  # Multi runs clip per-adapter iff the mode says so; singles always use the
  # stock trainer clip path driven by policy.max_grad_norm in the overlay.
  PAC=0; [[ "$MODE" == "clip1" ]] && PAC=1
  J=$(sbatch --parsable \
      --export=ALL,CFG=${CFGDIR}/parity100_${MODE}_multi.yaml,${BASE_ENV},NOUSNET_INIT_IMPORT_SLOT=,NOUSNET_PER_ADAPTER_GRAD_CLIP=${PAC} \
      -J parity-${MODE}-multi ${LAUNCHER})
  SUBMITTED+="${MODE}/multi=${J} "
  for AD in a b c d; do
    J=$(sbatch --parsable \
        --export=ALL,CFG=${CFGDIR}/parity100_${MODE}_single_${AD}.yaml,${BASE_ENV},NOUSNET_INIT_IMPORT_SLOT=${SLOT[$AD]},NOUSNET_DIAG_WHO=single_${AD},NOUSNET_PER_ADAPTER_GRAD_CLIP=0 \
        -J parity-${MODE}-s${AD} ${LAUNCHER})
    SUBMITTED+="${MODE}/single_${AD}=${J} "
  done
done
printf '%s\n' "$SUBMITTED" | tee /tmp/parity_battery_jobs.txt
