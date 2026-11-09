#!/bin/bash
# nano-3.5 30B-A3B (mamba-hybrid MoE) GRPO with non-colocated Megatron generation
# whose inference instance SPANS TWO NODES, exercising the NVLS MoE token
# dispatcher over multi-node NVLink (GB200 NVL72 / MNNVL).
#
# Layout on 4 GB200 nodes x 4 GPUs: generation takes 2 whole nodes (EP=8 across 8
# GPUs), training takes the remaining 2 (EP=8). Because the generation EP group
# (8) is larger than a node (4), MegatronGeneration.nvlink_domain_span puts the
# inference cluster in a single unified placement group with topology-sorted
# ranks, so the NVLS symmetric-memory all-gather runs across the NVLink fabric
# rather than within one node. cluster.segment_size=2 pins each 2-node group to
# one NVLink domain; without it node selection is arbitrary and the two
# generation nodes can land in different domains, where symmetric memory cannot
# be established.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=90
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    policy.generation.backend=megatron \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# The NVLS dispatcher is the only inference MoE dispatcher that permits CUDA
# graphs for non-decode steps: the 'nccl' dispatcher and the transformer_engine
# training path both force use_cuda_graphs_for_non_decode_steps off
# (megatron/core/inference/contexts/dynamic_context.py), and the warmup graph
# list is generated from that flag. So a captured warmup graph with a nonzero
# prefill request count ("<N> P + <M> D") is unreachable unless the NVLS path
# engaged. This matters because inference_moe_token_dispatcher_type is only read
# when transformer_impl=inference_optimized, and mcore's own default for it is
# already 'nvls' -- a run can therefore silently exercise neither the override
# nor the dispatcher without a positive signal.
NVLS_PREFILL_GRAPH_PATTERN='(cuda graph warmup - |\[graph [0-9]+/[0-9]+\] )\[[0-9]+\]: [1-9][0-9]* P \+'
echo "Verifying the NVLS inference dispatcher engaged in $RUN_LOG ..."
if grep -aE "$NVLS_PREFILL_GRAPH_PATTERN" "$RUN_LOG" > "$EXP_DIR/nvls_prefill_graphs.txt"; then
    echo "PASS: prefill CUDA graphs captured (last line):"
    tail -1 "$EXP_DIR/nvls_prefill_graphs.txt"
else
    echo "FAIL: no prefill CUDA graph captured, so the NVLS dispatcher did not engage."
    echo "      Expected a warmup line matching: $NVLS_PREFILL_GRAPH_PATTERN"
    exit 1
fi

# Guard the silent fallbacks that would leave the two generation nodes in
# different NVLink domains while the run still succeeds.
grep -q "Topology-aware allocation" $RUN_LOG || {
    echo "[ERROR] topology-aware allocation did not engage (no segment selection logged)" >&2
    exit 1
}
assert_not_grep "no NVLink domain info" $RUN_LOG \
    "segment_size fell back to unordered allocation; generation nodes may span NVLink domains"

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # No token_mult_prob_error check: generation runs transformer_impl=inference_optimized
    # while the logprobs come from the transformer_engine training model, so the two
    # disagree by design (the 2n8g recipe this inherits from omits it for the same reason).
    uv run tests/check_metrics.py $JSON_METRICS \
        'max(data["train/reward"]) > 0.0'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
