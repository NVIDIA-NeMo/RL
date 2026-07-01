#!/bin/bash
nohup srun \
  --no-container-mount-home \
  -A coreai_comparch_trtllm \
  -p gb200 \
  --overlap \
  --container-name=ray-head \
  --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL \
  --nodes=1 --ntasks=1 \
  -w lyris0236 \
  --jobid 2382560 \
  bash -c 'NUM_VLLM_REPLICAS=1 bash /lustre/fsw/coreai_comparch_trtllm/erinh/nemo-rl-for-gen/examples/swe_bench/run_grpo_nano_v3_5_swe_scale_gen_hsg_vllm.sh 2>&1' \
  > /lustre/fsw/coreai_comparch_trtllm/erinh/RL/run7_20260715_005655.log 2>&1 &
echo "Run7 PID: $!"
