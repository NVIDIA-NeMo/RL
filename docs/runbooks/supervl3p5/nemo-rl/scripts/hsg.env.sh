#!/usr/bin/env bash
# HSG paths only. Source this before env.sh or run_v2.sh.
export RL_DIR=/opt/nemo-rl
export SUPER_CONTAINER=/home/rohitkumarj/data/enroot-containers/rl.nightly.sep30.2026.sqsh
export MM_TRAINER_MODEL_PATH=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/models/super-vl-35-rlvr-v43-falcon-r3-20260905/hf
export MM_TRAINER_DATA_PATH=/lustre/fsw/portfolios/coreai/users/cye/code/RL/workspace/datasets/mm-trainer-unified/training.jsonl
export GYM_EXTRA_DIR=$RL_DIR/examples/nemo_gym/supervl3p5
export HSG_RUNTIME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/rem/unified-teacher-supervl3p5/runtime
export HSG_EXPERIMENTS=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/rem/unified-teacher-supervl3p5/experiments
export SUPER_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni
export NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/cache/nemo-rl-omni/megatron-checkpoints-super-vl-35-unified-final-ln-v2
