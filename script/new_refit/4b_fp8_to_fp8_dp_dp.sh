#!/bin/bash
# 4B Megatron FP8 (DP8, fp8_param=true) -> FP8 vLLM (DP8)
#
# Tests FP8->FP8 refit. Train side stores TE Float8BlockwiseQTensor weights
# (fp8_param=true, fp8_recipe=blockwise); gen side runs vLLM FP8 inference.
#
# STATUS (2026-05-05): NOT END-TO-END WORKING.
#   - Phase 2 lands the train-side dtype switch: when fp8_param=true,
#     `_is_fp8_export()` returns True and Bridge's `build_export_fp8_tasks`
#     produces (FP8 weight, scale_inv) pairs.
#   - Phase 3 (gen-side scale_inv routing) is NOT implemented yet —
#     `_build_hf_to_vllm_mapping` doesn't know what to do with `*.weight_scale_inv`
#     HF names. Expected failure mode: KeyError or unmapped param warning.
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
policy.megatron_cfg.fp8_cfg.enabled=true \
policy.megatron_cfg.fp8_cfg.fp8=e4m3 \
policy.megatron_cfg.fp8_cfg.fp8_recipe=blockwise \
policy.megatron_cfg.fp8_cfg.fp8_param=true \
policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false \
policy.generation.vllm_cfg.tensor_parallel_size=${Gtp} \
policy.train_global_batch_size=${training_gbs} \
+policy.nccl_reshard_refit=true \
grpo.max_num_steps=10 \
logger.wandb_enabled=True \
logger.wandb.project='new-refit-integration' \
logger.wandb.name='4b-megatron-FP8toFP8-DP8-DP8-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo_rl.0505.sqsh \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
HF_HUB_OFFLINE=1 \
NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1 \
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
