#!/bin/bash
# Reuse the two-GPU SC+Gym capture smoke for all supported token-level IS modes.
# One GPU trains Qwen3-0.6B with Megatron and one serves async vLLM.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
BASE_TEST=$SCRIPT_DIR/grpo_async_gym_single_controller.sh

# grpo_score_centering_filtered_gym_single_controller.sh reruns this with a
# filtered sampling profile and a subset of the IS modes.
TOP_P=${SC_TOP_P:-1.0}
TOP_K=${SC_TOP_K:-null}
EXP_PREFIX=${SC_EXP_PREFIX:-grpo_score_centering_gym_single_controller}
REPLAY=false
FORCE_ON_POLICY=true
REPLAY_CHECKS=()
if [[ "$TOP_P" != 1.0 || "$TOP_K" != null ]]; then
    # Filtered score centering renormalizes over the sampler's captured support.
    # Computing prev logprobs runs the prev-logprob replay path (captured-head
    # fetch, Megatron post-processor, packed boundaries), and token_mult_prob_error
    # then compares those replayed logprobs with vLLM's processed logprobs.
    REPLAY=true
    FORCE_ON_POLICY=false
    REPLAY_CHECKS=(
        'max(data["train/token_mult_prob_error"]) < 1.05'
        # token_mult_prob_error averages over kept tokens only, so it cannot see
        # drops. Replay drops a token only when its head misses part of the kept
        # set (5+ ties at the top_k-th logit); logprobs recomputed with the
        # filters drop every sampled token outside the trainer's own support.
        'max(data["train/policy_support_excluded_tokens"]) < 10'
    )
fi

cd "$PROJECT_ROOT"

for IS_MODE in ${SC_IS_MODES:-off is tis icepop adaptive-ppo-clip}; do
    EXP_NAME=${EXP_PREFIX}_$IS_MODE
    EXP_DIR=$SCRIPT_DIR/$EXP_NAME
    IS_OVERRIDES=(
        loss_fn.use_importance_sampling_correction=false
        ++loss_fn.truncated_importance_sampling_type=null
        ++loss_fn.truncated_importance_sampling_ratio_min=null
        loss_fn.truncated_importance_sampling_ratio=null
    )
    PACK_LOSS=false
    MODE_CHECKS=()
    if [[ "$IS_MODE" == is ]]; then
        IS_OVERRIDES[0]=loss_fn.use_importance_sampling_correction=true
    elif [[ "$IS_MODE" == tis || "$IS_MODE" == icepop || "$IS_MODE" == adaptive-ppo-clip ]]; then
        PACK_LOSS=true
        IS_OVERRIDES=(
            loss_fn.use_importance_sampling_correction=true
            ++loss_fn.truncated_importance_sampling_type=$IS_MODE
            ++loss_fn.truncated_importance_sampling_ratio_min=0.5
            loss_fn.truncated_importance_sampling_ratio=2.0
        )
    fi
    if [[ "$IS_MODE" == adaptive-ppo-clip ]]; then
        # One logged k per step: the controller ran alongside the centering.
        MODE_CHECKS=('len(data["train/cur_adaptive_ppo_clip_k"]) == 3')
    fi

    # The parent recipe uses GSPO; explicitly select the supported token-level
    # objective and ordinary logits path. This smoke is not a convergence test.
    SC_TEST_EXP_NAME="$EXP_NAME" RUN_CONVERGENCE_CHECKS=0 bash "$BASE_TEST" \
        ++token_capture.enabled=true \
        ++policy.generation.top_logprobs=16 \
        ++loss_fn.score_centering=true \
        loss_fn.sequence_level_importance_ratios=false \
        loss_fn.token_level_loss=true \
        ++loss_fn.force_on_policy_ratio=$FORCE_ON_POLICY \
        loss_fn.reference_policy_kl_penalty=0 \
        grpo.skip_reference_policy_logprobs_calculation=true \
        policy.sequence_packing.enabled=$PACK_LOSS \
        ++policy.sequence_packing.fuse_loss=$PACK_LOSS \
        ++policy.megatron_cfg.use_fused_linear_logprobs=false \
        policy.logprob_chunk_size=128 \
        policy.generation.temperature=1.0 \
        policy.generation.top_p=$TOP_P \
        policy.generation.top_k=$TOP_K \
        ++policy.generation.sampler_replay.enabled=$REPLAY \
        ++policy.generation.vllm_cfg.logprobs_mode=processed_logprobs \
        grpo.max_num_steps=3 \
        "${IS_OVERRIDES[@]}"

    uv run --no-sync tests/json_dump_tb_logs.py "$EXP_DIR/logs" \
        --output_path "$EXP_DIR/metrics.json"
    uv run --no-sync tests/check_metrics.py "$EXP_DIR/metrics.json" \
        'len(data["train/loss"]) == 3' \
        'len(data["train/score_centering_head_mass"]) == 3' \
        'all_finite(data["train/loss"])' \
        'all_finite(data["train/grad_norm"])' \
        'all_finite(data["train/score_centering_head_mass"])' \
        'min(data["train/score_centering_head_mass"]) > 0' \
        'max(data["train/score_centering_head_mass"]) <= 1.0001' \
        'min(data["train/global_valid_toks"]) > 0' \
        'max(data["train/finalize/total_ms"]) > 0' \
        'max(data["train/finalize/invalid_row_rate"]) == 0' \
        'max(data["train/finalize/capture_poisoned_rollouts"]) == 0' \
        ${REPLAY_CHECKS[@]+"${REPLAY_CHECKS[@]}"} \
        ${MODE_CHECKS[@]+"${MODE_CHECKS[@]}"}
done
