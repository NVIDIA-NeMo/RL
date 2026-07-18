#!/bin/bash
# Front #3 (user redirect: debug at SMALLER scale): instrument the 48-node cp16 recipe
# (TP8/EP16/CP16 = 32 train + 16 gen) instead of 80-node cp32. Same max_total_sequence_length
# =65536, same data/LR/loss — a memory-layout change only, so the divergence should still occur.
# Same capture-on-first-NaN + NaN-guard. If cp16 NaNs -> debug cheaper; if clean-for-many-steps
# -> escalate to cp32. Includes ALL validated mounts (the prior cp16 run died on missing hf_cache).
set -uo pipefail
Z=/lustre/fsw/portfolios/llmservice/users/zhiyul

# READ paths (model, data, container, SIFs) stay under $Z — they are world-readable within HSG, so
# any HSG user can repro without copying them. Only WRITE targets (caches, per-agent venvs, results)
# are scoped per-user via WRITE_ROOT, which defaults to your own users/<you> dir. For zhiyul it
# resolves back to $Z, so this launcher is byte-identical for the original author.
WRITE_ROOT="${WRITE_ROOT:-/lustre/fsw/portfolios/llmservice/users/${USER:-$(whoami)}}"
# Pin the container-workdir to the /lustre/fsw (mounted) path. SLURM_SUBMIT_DIR can resolve through
# the users/<u> symlink to /lustre/fs1 (NOT mounted on GB200 compute) -> pyxis "couldn't chdir" fail.
export NRL_CONTAINER_WORKDIR="${NRL_CONTAINER_WORKDIR:-$WRITE_ROOT/RL}"

source /lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/secrets.sh > >(grep -v HF_TOKEN) 2>&1 || true
export HF_HOME="$WRITE_ROOT/hf_cache"   # WRITE: HF cache (model itself is local at MODEL_PATH)

export EXP_NAME="ultra-swe-teacher-cp16"   # was ...-nancap (leftover from the NaN investigation, now fixed via moe_backend: triton)
export CONFIG_PATH="examples/configs/ultra/swe_teacher_cp16.yaml"
export MODEL_PATH="$Z/hf_home/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
export TRAIN_PATH="$Z/RL/ultra_data/swe.train.jsonl"
export VAL_PATH="$Z/RL/ultra_data/swe.val.jsonl"
export CONTAINER="$Z/enroot-images/nvcr.io+nvidian+nemo-rl+nightly.2026-07-13.squashfs"   # newer vLLM/NCCL; requires our overlay rebased onto current main (else ImportError log_container_init_timing)
export SANDBOX_CONTAINER="$Z/containers/nemo-skills-sandbox.sqsh"
export SIF_DIR="$Z/swe_sifs"
export PERSISTENT_CACHE="$WRITE_ROOT/persistent_cache"   # WRITE: compile/uv caches
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="nemotron_sw_post"
export SLURM_QOS="short"    # QOS 'short' = priority 200 (2x 'normal'=100), NO reservation needed.
                            # Limits: MaxWall=2:00:00, MaxNodes=64 -> our 1:59h/48N job fits. This
                            # jumped the job from mid-pack (Reason=Priority) to rank #1 (Reason=Resources).
export WALLTIME="1:59:00"   # <2h to stay strictly under 'short' MaxWall=2:00:00. Ample to load 550B
                            # + refit + reach rollouts and observe zero-500s (engine init ~20min).
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"   # cp16: 32 train
export NUM_GEN_NODES="${NUM_GEN_NODES:-16}"       # 16 gen = 48 nodes
# Topology-aware placement (NVIDIA-NeMo/RL PR #2986 / issue #2937): keep each EP group (EP32 = 8
# nodes) within one NVLink rack so the MoE all-to-all avoids cross-rack NCCL transport (which
# stalls under load -> 600s watchdog abort). segment_size = EP/gpus_per_node = 32/4 = 8. Must
# match cluster.segment_size in the recipe. Overrides ultra_launch's default of 16.
export SEGMENT_SIZE=8
export NUM_GYM_NODES="${NUM_GYM_NODES:-0}"
export USE_SNAPSHOT=0

# uv cache DISABLED: the existing lustre cache ($Z/persistent_cache/uv) is stale/incompatible —
# mounting it made uv install a broken `transformers` (ImportError: AutoProcessor) that crashed
# job 4744299 at 4min. The baked-container uv build works (~5min). To re-enable later, first WIPE
# the lustre uv cache so it repopulates fresh from THIS container, then set:
#   export UV_CACHE_DIR_OVERRIDE="$Z/persistent_cache/uv"

# ALL validated mounts. ultra_launch only mounts the nemo_rl/configs overlays + PERSISTENT_CACHE
# caches; everything else the recipe references by absolute path must be mounted here:
#   $Z/RL           - pyxis container-workdir (--container-workdir=SLURM_SUBMIT_DIR); FIX for the
#                     80-node 4269660 "pyxis: couldn't chdir to $Z/RL" failure.
#   $Z/hf_home,$Z/hf_cache - model + writable HF cache (prior cp16 died on missing hf_cache).
#   $Z/gym_venvs    - recipe env.nemo_gym.uv_venv_dir (per-agent venvs) — NOT under $Z/RL.
#   $Z/swe_sifs     - SIF_DIR; its swegym/swerebench are SYMLINKS into sdevare/images, so the
#   .../sdevare/images - symlink TARGETS must be mounted too or the .sif paths dangle.
#   $Z/RL/nemo_rl overlay - capture+guard edits live there.
export EXTRA_MOUNTS="$Z/RL:$Z/RL,$Z/hf_home:$Z/hf_home,$WRITE_ROOT/hf_cache:$WRITE_ROOT/hf_cache,$WRITE_ROOT/gym_venvs:$WRITE_ROOT/gym_venvs,$Z/swe_sifs:$Z/swe_sifs,/lustre/fsw/portfolios/llmservice/users/sdevare/images:/lustre/fsw/portfolios/llmservice/users/sdevare/images,$Z/RL/nemo_rl:/opt/nemo-rl/nemo_rl,$Z/RL/examples/nemo_gym:/opt/nemo-rl/examples/nemo_gym"
# ^ examples/nemo_gym overlay: ultra_launch overlays nemo_rl + examples/configs but NOT the driver
#   examples/nemo_gym/run_grpo_nemo_gym.py. On the 07-13 container the baked driver imports a newer
#   symbol (log_container_init_timing) our 123-behind package lacks -> ImportError. Overlaying OUR
#   driver keeps driver+package consistent (both ours) while still using 07-13's vLLM/NCCL libs.

# NaN is fixed at the source (moe_backend: triton). NRL_NAN_CAPTURE (debug hooks, adds overhead) is
# dropped; keep NRL_NAN_GUARD as cheap defensive safety (turns any residual NaN logprob into a 200,
# not a 500). Set NRL_NAN_CAPTURE=1 + NRL_NAN_CAPTURE_DIR only if you need to debug a new NaN.
export NRL_NAN_GUARD=1
export RAY_DEDUP_LOGS=0
# Opt into the NCCL flight recorder (debugging the training-step collective stall). Universal flag;
# ray.sub defaults the dump dir for USER=zhiyul, else set TORCH_NCCL_DEBUG_INFO_TEMP_FILE yourself.
export NRL_NCCL_FLIGHT_RECORDER=1
# NEW vs job 5331363 (which had the FR but a 600s watchdog that fired 71s before the walltime
# SIGKILL -> never dumped): shorten the TRAINING-phase NCCL PG timeout to 120s so a stalled
# collective trips the watchdog fast and the FR dumps with ~15min of walltime to spare. The 30-min
# 550B load + initial refit run BEFORE the first train() call, so they keep the long default.
export NRL_TRAIN_PG_TIMEOUT_S=120

# Ensure per-user WRITE dirs exist so pyxis can bind-mount them (the $Z read paths already exist).
mkdir -p "$WRITE_ROOT/hf_cache" "$WRITE_ROOT/persistent_cache" "$WRITE_ROOT/gym_venvs"

# Results/checkpoints/logs are WRITE targets: pin them under WRITE_ROOT. For zhiyul this equals
# $Z/RL/results/$EXP_NAME (unchanged); ultra_launch derives CHECKPOINT_DIR/BASE_LOG_DIR from it.
export RESULTS_DIR="${RESULTS_DIR:-$WRITE_ROOT/RL/results/$EXP_NAME}"

cd "$Z/RL"
# uv_venv_dir is the recipe's per-agent venv WRITE dir (set to ??? in the YAML); scope it per-user
# here — same Hydra-override mechanism as checkpointing.save_period.
bash ultra_launch.sh checkpointing.save_period=1 env.nemo_gym.uv_venv_dir="$WRITE_ROOT/gym_venvs"
