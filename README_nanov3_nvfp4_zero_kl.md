# Nano v3 NVFP4 zero-KL run (NeMo-RL PR #4791)

Needs GB200, 4 GPUs/node, Slurm + enroot/pyxis, and an HF token with access to Nemotron-3-Nano-30B-A3B.

## 1. Code

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git && cd RL
git fetch origin pull/4791/head:pr-4791 && git checkout pr-4791
git submodule sync --recursive && git submodule update --init --recursive

# Must print abde8697 and ace931f8b:
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge log -1 --oneline
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM log -1 --oneline
```

## 2. Container

```bash
enroot import -o nemo-rl.sqsh docker://nvcr.io#nvidia/nemo-rl:latest
export CONTAINER=$PWD/nemo-rl.sqsh
export MOUNTS="/lustre:/lustre,$PWD:/opt/nemo-rl"   # the checkout must be mounted at /opt/nemo-rl
```

## 3. Build the env (once, ~40 min)

```bash
srun -N1 --gpus-per-node=4 --container-image=$CONTAINER --container-mounts=$MOUNTS --no-container-mount-home \
  bash -c 'export NEMO_RL_VENV_DIR=/opt/nemo-rl/venvs NRL_FORCE_REBUILD_VENVS=true UV_LINK_MODE=copy TORCH_CUDA_ARCH_LIST=10.0
           git config --global --add safe.directory "*"
           cd /opt/nemo-rl && uv run --locked python -m nemo_rl.utils.prefetch_venvs MegatronPolicyWorker'
```

Rebuild only if `pyproject.toml` or `uv.lock` changes.

## 4. Run (3 nodes, 5 steps)

```bash
RECIPE=examples/configs/recipes/llm/grpo-nanov3-30BA3B-3n4g-megatron_generation-noncolocated-zero-kl-nvfp4-flashinfer-cp4.yaml
RUN=$PWD/runs/nanov3-nvfp4-zero-kl && mkdir -p $RUN/slurm

CONTAINER=$CONTAINER MOUNTS=$MOUNTS GPUS_PER_NODE=4 BASE_LOG_DIR=$RUN/slurm \
HF_HOME=<hf_cache> HF_TOKEN=<token> WANDB_API_KEY=<key> \
SETUP_COMMAND="cd /opt/nemo-rl && uv sync --inexact --frozen" \
COMMAND="export NEMO_RL_VENV_DIR=/opt/nemo-rl/venvs && cd /opt/nemo-rl && \
uv run --no-sync examples/run_grpo.py --config $RECIPE grpo.max_num_steps=5 logger.log_dir=$RUN/logs" \
sbatch --nodes=3 --gres=gpu:4 --account=<account> --partition=<partition> --time=2:00:00 \
  --output=$RUN/slurm/slurm-%j.out ray.sub
```

Log: `$RUN/slurm/<jobid>-logs/ray-driver.log`. Every step should print `Generation KL Error: 0.0000`.

Other recipes in the PR, run the same way:
- `grpo-nanov3-30BA3B-2n4g-...-async-gym-single-controller-token_capture-zero-kl-nvfp4-flashinfer.yaml`: 2 nodes; use `examples/run_grpo_single_controller.py`. It needs the workplace_assistant data from its `tests/test_suites/llm/` script.
- `grpo-nanov3-30BA3B-8n4g-dapo-...-zero-kl-nvfp4-flashinfer.yaml`: 8 nodes with `--segment=2`. Replace the hardcoded model, tokenizer and checkpoint paths first.
