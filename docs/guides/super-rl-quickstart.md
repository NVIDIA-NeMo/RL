# Super RL: replace the model and data, then run

For **ordinary RL, reasoning ON, W&B only** on a prepared AWS-CMH runtime.
This is the training-driver command, **not** an `sbatch` submission command.

## 1. One-time setup

Use a pinned [PR #4136](https://github.com/NVIDIA-NeMo/RL/pull/4136) checkout
inside a prepared **64-node / 256-GPU CMH Ray allocation**, not a login node.
Your site launcher supplies the image, patched dependencies/helpers, mounts,
tool services, **local DeepSeek judge**, private W&B authentication and
[required environment variables](super-rl-launch.md#regular-smoke-environment-contract).
PR #4136 does not yet include a complete site launcher; its fixed Gym commit
also requires an authorized dependency source. This is not a fresh-cluster
bootstrap or the hosted-judge recipe from PR #4195.

## 2. Replace your two inputs

In the prepared driver environment, from the NeMo-RL repository root:

```bash
export SUPER_RL_MODEL=/absolute/path/to/your/hf_checkpoint
export SUPER_RL_DATA=/absolute/path/to/your/train.jsonl
```

Use a compatible **Super 3.5 HF checkpoint with tokenizer and MTP1**, not a
Megatron `step_N` directory. JSONL must use the NeMo Gym schema (`agent_ref`,
`responses_create_params`, resource-specific answer/test fields) with configured
agents. Supply at least **768 rows** for this three-step smoke.

## 3. Run a three-step smoke

Choose a **new output root and W&B identity** for each fresh experiment:

```bash
export SUPER_RL_ROOT=/absolute/path/to/your/new_run
export SUPER_RL_RUN_NAME=my-super-rl-smoke
export WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME WANDB_FORK_ON_RESUME

uv run --no-sync python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config training_configs/super_rl/experiments/regular_s120_cp4_ep16.yaml \
  grpo.max_num_steps=3 \
  ++env.nemo_gym.global_aiohttp_connector_limit_per_host=16384
```

Defaults: TP4/CP4/EP16, 16 learner + 48 rollout nodes, segment16, GBS4096,
age2, 102400 output / 131072 context. `--no-sync` requires an existing environment.
Verify updates/refits, checkpoint save/reload and **W&B loss/reward**.

For **100 steps**, use a fresh root/identity, change the cap to `100`, provide
at least **25,600 rows**, and set `checkpointing.save_period=10` (FT stays 1/1).
Match `checkpointing.checkpoint_must_save_by` to Slurm walltime, e.g.
`00:03:45:00` for four hours; continuation jobs may be needed.

Outputs: `$SUPER_RL_ROOT/smoke/checkpoints` and `smoke/logs`. These training
checkpoints require conversion before HF evaluation.
