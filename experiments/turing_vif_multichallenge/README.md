# Turing VIF MultiChallenge GRPO Training

GRPO training on the MultiChallenge benchmark using the Turing VIF environment from NeMo-Gym.
Each task carries per-task LLM judge rubrics; the judge evaluates the model's response against
each rubric criterion and returns PASS/FAIL.

## Quick Start

```bash
# 1. Clone and checkout
git clone ssh://git@gitlab-master.nvidia.com:12051/terryk/nemo-rl-internal.git
cd nemo-rl-internal
git checkout mfathi/super-v3-turing-envs

# 2. Initialize submodules (requires SSH access to github.com for the Gym submodule)
git submodule sync
git submodule update --init --recursive

# Verify the Gym submodule has turing_vif:
ls 3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/app.py

# 3. Preprocess data
python3 experiments/turing_vif_multichallenge/data_preprocessing_multichallenge.py \
    --data-dir /lustre/fsw/portfolios/llmservice/users/mfathi/data/multichallenge \
    --output-dir 3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data \
    --splits advanced vanilla

# Verify: expect 1068 advanced + 1050 vanilla = 2118 total
wc -l 3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data/multichallenge_*.jsonl

# 4. Filter data to fit within max_total_sequence_length (8192 tokens)
#    Uses the actual model tokenizer + chat template for accurate measurement.
#    Samples exceeding the limit would cause vLLM to reject the request.
uv run experiments/filter_data_by_length.py \
    --tokenizer /lustre/fsw/portfolios/llmservice/users/mfathi/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --max-tokens 8192 \
    --inputs 3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data/multichallenge_vanilla.jsonl \
    --suffix _8k

# 5. Launch training
bash experiments/turing_vif_multichallenge/launch_turing_vif_multichallenge_training.sh
```

## Prerequisites

- **uv** package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- SSH access to GitHub (for the Gym submodule at `github.com:abubakaria56/Gym.git`)
- SLURM cluster with GPU nodes and a squashfs container image
- Raw MultiChallenge JSON data on the cluster filesystem

## Pre-flight Checks

Before launching, clear any stale HuggingFace module caches to avoid `AutoConfig` errors
with custom model architectures (e.g., NemotronH):

```bash
rm -rf ${HF_HOME:-~/.cache/huggingface}/modules/transformers_modules/
```

If you see `vllm/_C.abi3.so: file too short` errors, the vLLM shared object may not have
fully synced across Lustre. Force a read to warm the cache:

```bash
md5sum 3rdparty/vllm/vllm/_C.abi3.so
```

## Data Preprocessing

The preprocessing script converts raw per-task JSON files into the Turing VIF JSONL format.
Each output record contains the conversation input, per-rubric LLM judge items with embedded
context, and an agent reference for the turing_vif simple agent.

```bash
python3 experiments/turing_vif_multichallenge/data_preprocessing_multichallenge.py \
    --data-dir <path-to-raw-multichallenge-data> \
    --output-dir 3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data \
    --splits advanced vanilla
```

The `--data-dir` should contain subdirectories named after each split (e.g., `advanced/`, `vanilla/`),
each with `*.json` task files.

### Length Filtering

Some multichallenge samples have multi-turn conversation inputs that exceed `max_total_sequence_length`.
vLLM rejects requests that exceed `max_model_len` with a 400 error, and the `simple_agent` raises
on model errors, crashing the training run.

Filter the data to keep only samples that fit within the token budget:

```bash
uv run experiments/filter_data_by_length.py \
    --tokenizer /path/to/policy-model \
    --max-tokens 8192 \
    --inputs 3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data/multichallenge_vanilla.jsonl \
    --suffix _8k
```

This produces `multichallenge_vanilla_8k.jsonl` alongside the original file. The script uses the
actual model tokenizer with `apply_chat_template` for accurate token counting. With the default
Nemotron-3-Nano-30B tokenizer at 8192 tokens, ~84% of vanilla samples are kept (165 filtered).

To use a different token budget (e.g., 16384 for longer sequences):

```bash
uv run experiments/filter_data_by_length.py \
    --tokenizer /path/to/policy-model \
    --max-tokens 16384 \
    --inputs .../multichallenge_vanilla.jsonl \
    --suffix _16k
```

## Configuration

The launch script auto-detects the cluster (LAX vs DFW) and sets container/model paths accordingly.
Override any default via environment variables:

| Variable | Default | Description |
|---|---|---|
| `NUM_NODES` | 16 | Policy nodes (colocated training + vLLM inference) |
| `POLICY_MODEL` | Nemotron-3-Nano-30B-A3B-BF16 | HF model path for the policy |
| `TRAIN_DATA` | `multichallenge_vanilla_8k.jsonl` | Training data JSONL (filtered to 8K tokens) |
| `VAL_DATA` | `multichallenge_vanilla_8k.jsonl` | Validation data JSONL (filtered to 8K tokens) |
| `CONTAINER_IMAGE` | auto-detected per cluster | Squashfs container image |
| `SLURM_ACCOUNT` | `llmservice_modelalignment_ppo` | SLURM billing account |
| `SLURM_PARTITION` | `batch` | SLURM partition |
| `SLURM_TIME` | `4:00:00` | Job time limit |
| `NUM_JOBS` | 1 | Number of chained jobs (each resumes from previous checkpoint) |
| `EXP_NAME` | `turing_vif_multichallenge_grpo` | Experiment name for logs and W&B |

### Judge Configuration

A separate vLLM instance is spun up for the LLM judge (default: Qwen3-235B-A22B-Instruct FP8).

| Variable | Default | Description |
|---|---|---|
| `USE_SEPARATE_JUDGE` | `true` | Use a dedicated judge model (set `false` to use the policy) |
| `JUDGE_MODEL` | Qwen3-235B-A22B-Instruct-2507-FP8 | Judge model path |
| `JUDGE_NUM_NODES` | 4 | Nodes for the judge |
| `JUDGE_TP` | 8 | Tensor parallelism for the judge |
| `JUDGE_ROUTER_DP_SIZE` | 4 | Number of judge replicas |
| `JUDGE_MAX_LEN` | 65536 | Max context length for judge |

Smaller judge configurations for faster iteration:

```bash
# Qwen3-8B (1 node, fast, lower quality)
JUDGE_MODEL="/path/to/Qwen3-8B" \
JUDGE_TP=4 JUDGE_NUM_NODES=1 JUDGE_ROUTER_DP_SIZE=1 \
JUDGE_GPU_UTIL=0.85 JUDGE_MAX_LEN=16384 \
JUDGE_ENABLE_EP=false JUDGE_MULTITHREAD_LOAD=false \
bash experiments/turing_vif_multichallenge/launch_turing_vif_multichallenge_training.sh

# Qwen3-30B-A3B (1 node, balanced)
JUDGE_MODEL="/path/to/Qwen3-30B-A3B-Thinking-2507" \
JUDGE_TP=4 JUDGE_NUM_NODES=1 JUDGE_ROUTER_DP_SIZE=1 \
bash experiments/turing_vif_multichallenge/launch_turing_vif_multichallenge_training.sh
```

## Resource Summary

| Config | Policy Nodes | Judge Nodes | Total Nodes | Total GPUs |
|---|---|---|---|---|
| Default (235B judge) | 16 | 4 | 20 | 160 |
| Medium judge (30B) | 16 | 1 | 17 | 136 |
| Small judge (8B) | 16 | 1 | 17 | 136 |
| No separate judge | 16 | 0 | 16 | 128 |

## Monitoring

```bash
# Check job status
squeue -u $USER --name=turing_vif_multichallenge_grpo

# Watch driver logs
tail -f logs/turing_vif_multichallenge_grpo/<job_id>-logs/ray-driver.log

# W&B dashboard
# https://wandb.ai/nvidia/nemo-rl-turing-vif-multichallenge
```

## Job Chaining

For long training runs, chain multiple jobs so each resumes from the previous checkpoint:

```bash
NUM_JOBS=3 bash experiments/turing_vif_multichallenge/launch_turing_vif_multichallenge_training.sh
```

All jobs share the same checkpoint directory. Each subsequent job has an `afterok` dependency
on the previous one.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NemotronHConfig` not found | Stale HF transformers module cache | `rm -rf $HF_HOME/modules/transformers_modules/` |
| `_C.abi3.so: file too short` | Lustre file sync delay | Run `md5sum 3rdparty/vllm/vllm/_C.abi3.so` before launching |
| `Address already in use` (gRPC port) | Stale processes on a compute node | Relaunch (random node assignment) |
| `No module named 'gprof2dot'` | Missing dependency in container | Make the import lazy in `nemo_gym/profiling.py` |
| `exclude-dependencies` uv warning | Container uv version too old | Remove the field from Gym's `pyproject.toml` `[tool.uv]` |
