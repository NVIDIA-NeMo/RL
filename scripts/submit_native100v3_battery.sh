#!/bin/bash
# Submit native100 V3 parity battery.
# V3 explicitly restores the previous nousnet campaign's COT prompt inherited
# from the GRPO base, while keeping tokenizer model-default and fresh checkpoints.
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
CANON=/path/to/rank_RR_slot_II.pt shards. No nousnet checkout is required.
EOF
  exit 1
fi

for CKPT in results/native100v3_{noclip,clip1}_{multi,sa,sb,sc,sd}_ckpt; do
  if [[ -e "${CKPT}" ]]; then
    echo "ERROR: fresh-run checkpoint path already exists: ${CKPT}" >&2
    exit 1
  fi
done

declare -A SLOT=( [sa]=0 [sb]=1 [sc]=2 [sd]=3 )
SUBMITTED=""
for MODE in noclip clip1; do
  PAC=0; [[ "$MODE" == "clip1" ]] && PAC=1
  J=$(sbatch --parsable \
      --export=ALL,CFG=${CFGDIR}/native100v3_${MODE}_multi.yaml,${BASE_ENV},NOUSNET_INIT_IMPORT_SLOT=,NOUSNET_PER_ADAPTER_GRAD_CLIP=${PAC} \
      -J n100v3-${MODE}-m ${LAUNCHER})
  SUBMITTED+="${MODE}/multi=${J} "
  for AD in sa sb sc sd; do
    WHO="single_${AD#s}"
    J=$(sbatch --parsable \
        --export=ALL,CFG=${CFGDIR}/native100v3_${MODE}_${AD}.yaml,${BASE_ENV},NOUSNET_INIT_IMPORT_SLOT=${SLOT[$AD]},NOUSNET_DIAG_WHO=${WHO},NOUSNET_PER_ADAPTER_GRAD_CLIP=0 \
        -J n100v3-${MODE}-${AD} ${LAUNCHER})
    SUBMITTED+="${MODE}/${AD}=${J} "
  done
done
printf '%s\n' "$SUBMITTED" | tee /tmp/native100v3_battery_jobs.txt
