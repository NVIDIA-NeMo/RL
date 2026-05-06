#!/bin/bash
# 4B Megatron: BF16 train (DP8) -> FP8 vLLM (DP8)
#
# Tests BF16->FP8 refit. Train side stores BF16 weights (fp8_param=false or
# no fp8_cfg); gen side runs vLLM FP8 quantized inference.
#
# STATUS (2026-05-05): NOT END-TO-END WORKING.
#   - Phase 2 lands the train-side dtype switch (default-off; this config
#     keeps it off because fp8_param!=true).
#   - Phase 3 (gen-side scale_inv routing) and BF16->FP8 quantization on the
#     gen side are NOT implemented yet — vLLM gets BF16 bytes into an FP8 param.
#   - Expected failure mode: vLLM dtype mismatch when copying BF16 into FP8 param.
# This script is the regression target for that follow-up work.

account=coreai_dlalgo_nemorl
NUM_ACTOR_NODES=2
n_gen_nodes=1
n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

Ttp=1
Gtp=1
Tcp=1

nprompt=4
grpo_group_size=16
logprob_batch_size=2
seq_len=1024
lag=1
inflight=true

rollout_size=$((nprompt * grpo_group_size))
training_gbs=${rollout_size}

sp=False

COMMAND="uv run ./examples/run_grpo.py \
--config examples/configs/grpo_math_4B_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.val_period=1000 \
grpo.num_prompts_per_step=${nprompt} \
grpo.num_generations_per_prompt=${grpo_group_size} \
policy.logprob_batch_size=${logprob_batch_size} \
checkpointing.enabled=false \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=${lag} \
grpo.async_grpo.in_flight_weight_updates=${inflight} \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.vllm_cfg.precision=fp8 \
policy.generation.vllm_cfg.use_deep_gemm=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.max_total_sequence_length=${seq_len} \
policy.megatron_cfg.tensor_model_parallel_size=${Ttp} \
policy.megatron_cfg.sequence_parallel=${sp} \
policy.megatron_cfg.context_parallel_size=${Tcp} \
policy.generation.vllm_cfg.tensor_parallel_size=${Gtp} \
policy.train_global_batch_size=${training_gbs} \
+policy.nccl_reshard_refit=true \
grpo.max_num_steps=10 \
logger.wandb_enabled=True \
logger.wandb.project='new-refit-integration' \
logger.wandb.name='4b-megatron-BF16toFP8-DP8-DP8-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo_rl.0505.sqsh \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
HF_HUB_OFFLINE=1 \
NRL_FORCE_REBUILD_VENVS=true \
NRL_IGNORE_VERSION_MISMATCH=1 \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=00:30:00 \
    --gres=gpu:8 \
    ray.sub
