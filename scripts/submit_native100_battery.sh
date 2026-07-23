#!/bin/bash
# Submit the native 100-step single-vs-multi battery (both clip modes).
# 10 jobs: {noclip,clip1} x {multi, sa, sb, sc, sd} — campaign-parity env:
# exact-init from code7x_exactinit_canonical, per-adapter loss traces ON,
# TRACE_ONLY determinism mode, clip1 multi uses per-adapter grad clip.
set -euo pipefail
cd /home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native

CANON=/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/nousnet_pre_multilora_8178f9c/results/code7x_exactinit_canonical
LAUNCHER=examples/configs/recipes/multi_lora/sft_8gpu_native.slurm
CFGDIR=examples/configs/recipes/multi_lora
BASE_ENV="NOUSNET_DIAG_ENABLED=1,NOUSNET_DIAG_LOSS_TRACE=1,NOUSNET_DIAG_TRACE_ONLY=1,NOUSNET_DIAG_LORA_STEP=0,NOUSNET_FORCE_PAD_TO=1024,NOUSNET_INIT_IMPORT_DIR=${CANON}"

declare -A SLOT=( [sa]=0 [sb]=1 [sc]=2 [sd]=3 )
SUBMITTED=""

for MODE in noclip clip1; do
  # multi: no slot (imports all 4), clip1 multi needs per-adapter clip
  PAC=0; [[ "$MODE" == "clip1" ]] && PAC=1
  J=$(sbatch --parsable \
      --export=ALL,CFG=${CFGDIR}/native100_${MODE}_multi.yaml,${BASE_ENV},NOUSNET_INIT_IMPORT_SLOT=,NOUSNET_PER_ADAPTER_GRAD_CLIP=${PAC} \
      -J n100v2-${MODE}-m ${LAUNCHER})
  SUBMITTED+="${MODE}/multi=${J} "
  for AD in sa sb sc sd; do
    WHO="single_${AD#s}"
    J=$(sbatch --parsable \
        --export=ALL,CFG=${CFGDIR}/native100_${MODE}_${AD}.yaml,${BASE_ENV},NOUSNET_INIT_IMPORT_SLOT=${SLOT[$AD]},NOUSNET_DIAG_WHO=${WHO},NOUSNET_PER_ADAPTER_GRAD_CLIP=0 \
        -J n100v2-${MODE}-${AD} ${LAUNCHER})
    SUBMITTED+="${MODE}/${AD}=${J} "
  done
done

echo "${SUBMITTED}" | tee /tmp/native100_battery_jobs.txt
