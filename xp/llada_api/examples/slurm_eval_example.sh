#!/bin/bash

# Example: SLURM-based Evaluation Workflow
# This example shows how to use the SLURM orchestration script

# STEP 1: Set your SLURM account (REQUIRED)
export ACCOUNT=your_slurm_account  # <-- CHANGE THIS

# STEP 2: Choose your configuration

# Example 1: Quick test with HuggingFace model (RECOMMENDED FOR FIRST RUN)
echo "Example 1: Quick test"
./xp/llada_api/scripts/slurm_launch_and_eval.sh \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --quick-test \
  --steps 64

# Example 2: Full evaluation on GSM8K
# echo "Example 2: Full GSM8K evaluation"
# ./xp/llada_api/scripts/slurm_launch_and_eval.sh \
#   --model-path GSAI-ML/LLaDA-8B-Instruct \
#   --benchmark gsm8k:4 \
#   --algorithm dinfer_hierarchy

# Example 3: Multi-GPU server with DCP checkpoint
# echo "Example 3: Multi-GPU with DCP"
# ./xp/llada_api/scripts/slurm_launch_and_eval.sh \
#   --dcp-path /path/to/your/checkpoint.dcp \
#   --base-model GSAI-ML/LLaDA-8B-Instruct \
#   --server-gpus 4 \
#   --algorithm dinfer_hierarchy \
#   --benchmark gsm8k:4

# Example 4: Custom SLURM resources
# echo "Example 4: Custom resources"
# ./xp/llada_api/scripts/slurm_launch_and_eval.sh \
#   --model-path GSAI-ML/LLaDA-8B-Instruct \
#   --benchmark math:2 \
#   --server-gpus 2 \
#   --server-mem 256G \
#   --server-partition batch \
#   --eval-partition interactive

# Example 5: Nemotron model
# echo "Example 5: Nemotron evaluation"
# ./xp/llada_api/scripts/slurm_launch_and_eval.sh \
#   --model-path nvidia/Nemotron-Diffusion-Research-4B-v0 \
#   --eval-model nemotron-4b \
#   --generation-algorithm nemotron \
#   --benchmark gsm8k:4

echo ""
echo "=========================================="
echo "Orchestration script completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check logs in: logs/slurm_launch_*/"
echo "2. View results in the output directory"
echo "3. Modify this script to use different configurations"
echo ""
echo "For more examples, see:"
echo "  xp/llada_api/scripts/SLURM_ORCHESTRATION.md"

