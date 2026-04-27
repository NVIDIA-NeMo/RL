# Nemotron Nano v3 VL: RL Post-Training with NeMo-RL

RL post-training for Nemotron Nano v3 VL (30B MoE vision-language model) using [NeMo-RL](README.nemo-rl.md). Supports GRPO and MPO algorithms with the Megatron training backend and vLLM generation backend.

## NeMo-RL Specifics

### Checkpoint Preparation

The `policy.model_name` parameter expects a path to an HF checkpoint. If you have an mcore checkpoint, convert it first using the [moe_mcore_to_hf.sh](https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/vlm2/examples/multimodal/v3/moe_mcore_to_hf.sh) script from Megatron-LM.

### Monitoring Training

Monitor training progress on WandB (org `adlr`, projects `mpo-nanov3vl` / `grpo-nano3vl`). The key metric to watch is **training reward** — it should trend upward over the course of training.

Ray driver logs go to `$JOBID-logs/ray-driver.log`. If Ray fails to start, check the Slurm logs at `slurm-$JOBID.out`.

### Resuming Training

There is no automatic resubmission. To resume after the Slurm time limit, submit the same `sbatch` command multiple times — the `--dependency=singleton` flag ensures they run in sequence, and each job resumes from the latest checkpoint.

## MPO Training

MPO (Mixed Preference Optimization) is a supervised learning method that trains from a static dataset 
using a pairwise loss. It doesn't use RL environments or vLLM generations.

### Multi-node (16 nodes)

```sh
NEMORL=/lustre/.../nemo-rl
NUM_NODES=16
JOB_NAME=mpo-nanov3omni-32k
SEED=$(echo -n train:${JOB_NAME} | openssl dgst -md5 -binary | od -An -tu4 -N4 | xargs)

export CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/rl/images/nemo-rl-hanrong_20260226.sqsh
export NRL_FORCE_REBUILD_VENVS=true
export COMMAND="\
rsync -a --include='*/' --include='*.py' --exclude='*' ${NEMORL}/3rdparty/ /opt/nemo-rl/3rdparty/ && \
uv run examples/run_vlm_mpo.py --config examples/configs/vlm_mpo_mmpr_nanov3vl.yaml \
cluster.num_nodes=$NUM_NODES \
policy.model_name=/lustre/.../mcore_to_hf \
mpo.seed=$SEED \
checkpointing.checkpoint_dir='results/${JOB_NAME}' \
logger.wandb_enabled=True \
logger.wandb.name='${JOB_NAME}'"
export NCCL_DEBUG=INFO
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NEMO_RL_LOG_GPU_MEMORY=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NRL_IGNORE_VERSION_MISMATCH=true

MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=llmservice_fm_vision \
    --job-name=nemo-rl-${JOB_NAME} \
    --partition=batch_block1 \
    --dependency=singleton \
    --time=4:00:00 \
    --gres=gpu:8 \
    ray.sub
```

## GRPO Training

GRPO (Group Relative Policy Optimization) is a reinforcement learning method that trains against a dataset and verifier-based rewards. It needs vLLM to generate on-policy responses.

### Multi-node (8 nodes)

```sh
NEMORL=/lustre/.../nemo-rl
NUM_NODES=8
JOB_NAME=grpo-nanov3omni-8k
SEED=$(echo -n train:${JOB_NAME} | openssl dgst -md5 -binary | od -An -tu4 -N4 | xargs)

export CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/rl/images/nemo-rl-hanrong_20260226.sqsh
export NRL_FORCE_REBUILD_VENVS=true
export COMMAND="\
rsync -a --include='*/' --include='*.py' --exclude='*' ${NEMORL}/3rdparty/ /opt/nemo-rl/3rdparty/ && \
uv run examples/run_vlm_grpo.py --config examples/configs/vlm_grpo_nanov3vl_blend.yaml \
cluster.num_nodes=$NUM_NODES \
policy.model_name=/lustre/.../mcore_to_hf \
grpo.seed=$SEED \
checkpointing.checkpoint_dir='results/${JOB_NAME}' \
logger.wandb_enabled=True \
logger.wandb.name='${JOB_NAME}'"
export NCCL_DEBUG=INFO
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NEMO_RL_LOG_GPU_MEMORY=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NRL_IGNORE_VERSION_MISMATCH=true

MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=llmservice_fm_vision \
    --job-name=nemo-rl-${JOB_NAME} \
    --partition=batch_block1 \
    --dependency=singleton \
    --time=4:00:00 \
    --gres=gpu:8 \
    ray.sub
```

## Single-node interactive testing

Start an interactive job and wait for it to start. After it starts, it writes a `$JOBID-attach.sh` script that you can run to open a terminal on the head node.

```sh
CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/rl/images/nemo-rl-hanrong_20260226.sqsh \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=1 \
    --account=llmservice_fm_vision \
    --job-name=nemo-rl-dev:interactive \
    --partition=interactive \
    --time=4:00:00 \
    --gres=gpu:8 \
    ray.sub
```


Use a truncated model (8 layers, 8 experts) for quick iteration:

```sh
NRL_FORCE_REBUILD_VENVS=true uv run examples/run_vlm_grpo.py \
  --config examples/configs/vlm_grpo_nanov3vl_blend.yaml \
  cluster.num_nodes=1 \
  policy.model_name=/lustre/fsw/portfolios/llmservice/users/smohsenitahe/checkpoints/sft_omni_300k_rebalanced_0301_trunc/ \
  policy.megatron_cfg.expert_model_parallel_size=2 \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  grpo.num_prompts_per_step=32 \
  grpo.num_generations_per_prompt=8 \
  policy.train_global_batch_size=32 \
  policy.generation.vllm_cfg.tensor_parallel_size=2 \
  checkpointing.checkpoint_dir=debug_results/single_node_debug_image_grpo_conv3d \
  checkpointing.save_period=2
```
