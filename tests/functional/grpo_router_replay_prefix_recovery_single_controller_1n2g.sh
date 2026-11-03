#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Router-replay (R3) smoke test for generation-prefix recovery. A Gym-v2
# checkpoint is taken while vLLM is mid-decode on a MoE model with router
# replay on; the process is hard-killed and restored. Each restored call must
# resume from the exact durable token prefix and stage, for training, the routes
# the interrupted call recorded for that prefix, not routes recomputed after the
# restart.
#
# The policy is a tiny random-weight Qwen3-MoE built on the fly, so the test
# fits one Megatron GPU and one vLLM GPU. A random model cannot follow the
# Workplace task and its near-tied logits flip tokens between runs, so this
# test does no parity against an uninterrupted run (that is the Workplace
# test's job, grpo_async_gym_single_controller_prefix_recovery.sh). It reuses
# the reference-free checks of the Omni test instead: each restored prefix must
# appear verbatim in its trained row. With --require-routes, verify-completion
# also requires the routes staged for that row to hash like the restored ones.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 2 )); then
    echo "SKIP: router-replay prefix recovery requires at least two visible GPUs"
    exit 0
fi

EXP_NAME=$(basename "$0" .sh)
TEST_DIR="${SCRIPT_DIR}/${EXP_NAME}"
TEST_CONFIG="${PROJECT_ROOT}/examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml"
MODEL_DIR="${TEST_DIR}/tiny-qwen3-moe"
source "${SCRIPT_DIR}/_gym_prefix_recovery_phases.sh"
set_prefix_recovery_paths

# One prompt group per output budget. A random-weight model never emits EOS,
# so every call decodes to its budget: long enough to be cut mid-generation,
# and the distinct budgets finish the groups in a fixed order.
OUTPUT_TOKENS=(${SC_R3_PREFIX_RECOVERY_OUTPUT_TOKENS:-1024 1536})
NUM_PROMPTS=${#OUTPUT_TOKENS[@]}
NUM_GENERATIONS=${SC_R3_PREFIX_RECOVERY_NUM_GENERATIONS:-2}
MAX_MODEL_LEN=${SC_R3_PREFIX_RECOVERY_MAX_MODEL_LEN:-4096}
MAX_STEPS=${SC_R3_PREFIX_RECOVERY_MAX_STEPS:-2}
SNAPSHOT_INTERVAL_S=${SC_R3_PREFIX_RECOVERY_INTERVAL_S:-0.05}
PHASE2_SNAPSHOT_INTERVAL_S=${SC_R3_PREFIX_RECOVERY_PHASE2_INTERVAL_S:-600}
SNAPSHOT_TIMEOUT_S=${SC_R3_PREFIX_RECOVERY_TIMEOUT_S:-1800}
RUN_TIMEOUT_S=${SC_R3_PREFIX_RECOVERY_RUN_TIMEOUT_S:-1800}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS * NUM_GENERATIONS))
POLICY_MAX_NEW_TOKENS=0
for output_tokens in "${OUTPUT_TOKENS[@]}"; do
    if (( output_tokens > POLICY_MAX_NEW_TOKENS )); then
        POLICY_MAX_NEW_TOKENS=$output_tokens
    fi
done

rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# The model is rebuilt on every run, but Megatron's HF conversion is cached by
# model path: convert into this run's directory so a cache from an earlier run
# (another transformers version, other weights) is never reused.
export NRL_MEGATRON_CHECKPOINT_DIR="${TEST_DIR}/megatron-conversion"

# Two MoE layers of eight top-2 experts in Qwen3-30B-A3B's checkpoint layout,
# with the Qwen3 tokenizer. Seeded so its routing is the same on every run.
MODEL_DIR="${MODEL_DIR}" uv run --no-sync python - <<'PY'
import os

import torch
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

torch.manual_seed(1234)
config = Qwen3MoeConfig(
    num_hidden_layers=2,
    hidden_size=128,
    intermediate_size=128,
    moe_intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    num_experts=8,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    norm_topk_prob=True,
    vocab_size=151936,
    tie_word_embeddings=False,
    max_position_embeddings=8192,
)
model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
model.save_pretrained(os.environ["MODEL_DIR"])
AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(os.environ["MODEL_DIR"])
PY

TEST_DATA="${TEST_DATA}" MAX_STEPS="${MAX_STEPS}" OUTPUT_TOKENS="${OUTPUT_TOKENS[*]}" \
uv run --no-sync python - <<'PY'
import json
import os

budgets = [int(value) for value in os.environ["OUTPUT_TOKENS"].split()]
with open(os.environ["TEST_DATA"], "w") as output:
    for step in range(int(os.environ["MAX_STEPS"])):
        for slot, budget in enumerate(budgets):
            row = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": f"Step {step} item {slot}: count upward from 1.",
                        }
                    ],
                    "max_output_tokens": budget,
                },
                "expected_answer": "unused",
                "extraction_mode": "boxed",
                "case_sensitive": False,
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": "string_match_simple_agent",
                },
            }
            output.write(json.dumps(row) + "\n")
PY

install_phase_cleanup "${CHECKPOINT_DIR}" "${SELECTED_BACKUP}"
prefix_recovery_overrides

# One Megatron training GPU and one async vLLM GPU, router replay on, plus the
# text prefix-recovery settings.
COMMON_OVERRIDES=(
    policy.model_name="${MODEL_DIR}"
    policy.tokenizer.name="${MODEL_DIR}"
    ++policy.router_replay.enabled=true
    cluster.num_nodes=1
    cluster.gpus_per_node=2
    policy.megatron_cfg.tensor_model_parallel_size=1
    policy.megatron_cfg.expert_model_parallel_size=1
    policy.megatron_cfg.expert_tensor_parallel_size=1
    policy.megatron_cfg.pipeline_model_parallel_size=1
    policy.megatron_cfg.context_parallel_size=1
    policy.megatron_cfg.sequence_parallel=false
    policy.generation.colocated.enabled=false
    policy.generation.colocated.resources.num_nodes=1
    policy.generation.colocated.resources.gpus_per_node=1
    policy.generation.max_new_tokens="${POLICY_MAX_NEW_TOKENS}"
    policy.generation.vllm_cfg.tensor_parallel_size=1
    policy.generation.vllm_cfg.async_engine=true
    policy.generation.vllm_cfg.max_model_len="${MAX_MODEL_LEN}"
    policy.generation.vllm_cfg.gpu_memory_utilization=0.3
    # Matches the R3 recipes: every routed position is computed, none reused.
    ++policy.generation.vllm_cfg.enable_prefix_caching=false
    # Plain sampling: training applies the same top-p to its recomputed
    # logprobs, so a near-greedy top-p would make gen_kl_error compare vLLM's
    # raw logprobs with filtered ones. This test needs no predictable decode.
    policy.generation.temperature=1.0
    policy.generation.top_p=1.0
    '~env.nemo_gym.code_gen'
    '~env.nemo_gym.workplace_assistant_environment_server'
    "env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/string_match/configs/string_match.yaml]"
    ++env.nemo_gym.string_match_simple_agent.responses_api_agents.simple_agent.max_steps=1
    env.should_log_nemo_gym_responses=false
    ++data_plane.enabled=true
    ++data_plane.impl=transfer_queue
    ++data_plane.backend=simple
    ++data_plane.simple.storage_capacity=1000000
    ++data_plane.simple.num_storage_units=2
    ++data_plane.claim_meta_poll_interval_s=0.5
    grpo.async_grpo=null
    ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=300
    ++async_rl.stall_watchdog.stall_timeout_s=600
    "${PREFIX_RECOVERY_OVERRIDES[@]}"
)

# Fail on a partially missing route row instead of silently falling back to
# the training model's own router for it.
R3_CMD=(
    env NRL_ROUTER_REPLAY_VALIDATE=1
    uv run --no-sync python "${RECOVERY_HOOK}"
    --config "${TEST_CONFIG}"
    "${COMMON_OVERRIDES[@]}"
)

run_prefix_recovery_phases \
    "publish a checkpoint holding an active router-replay generation prefix" \
    "restore each prefix and its routes, then finish each suffix" \
    R3_CMD "$@"

# A missing route row would otherwise fall back to Megatron's own router.
if grep -q "R3 router replay fallback:" "${PHASE2_LOG}"; then
    echo "[ERROR] router replay fell back to Megatron's router for some tokens"
    exit 1
fi

uv run --no-sync tests/json_dump_tb_logs.py "${RECOVERY_LOG_DIR}" --output_path "${PHASE2_METRICS}"
uv run --no-sync tests/check_metrics.py "${PHASE2_METRICS}" \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0' \
    'all_finite(data["train/gen_kl_error"])'

# With --require-routes, each restored call must also have staged its
# restored routes for training.
verify_prefix_recovery --require-routes

echo "Router-replay generation-prefix recovery test passed"
