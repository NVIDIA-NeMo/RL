# PPO

[Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) is an actor-critic reinforcement learning algorithm that jointly trains a **policy** (actor) and a **value function** (critic). The value function estimates per-token state values, enabling [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) for lower-variance advantage signals compared to reward-only baselines.

## Key Differences from GRPO

- **Value Model (Critic)**: PPO trains a separate value model alongside the policy. GRPO uses a leave-one-out baseline and has no value model.
- **GAE Advantage Estimation**: PPO uses temporal-difference bootstrapping via GAE. GRPO normalizes group rewards.
- **Critic Warmup**: PPO supports training the value model for a configurable number of steps before starting policy training (`policy_training_start_step`).
- **VAPO Decoupled GAE**: Supports separate lambda parameters for policy advantages and value returns (`gae_lambda_policy`, `gae_lambda_value`).

## PPO Single Node

```sh
uv run examples/run_ppo.py \
  --config examples/configs/ppo_math_1B_megatron.yaml \
  policy.model_name="Qwen/Qwen2.5-1.5B" \
  cluster.gpus_per_node=8 \
  checkpointing.checkpoint_dir="results/ppo_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="ppo-math"
```

For Megatron-Core backend:

```sh
uv run examples/run_ppo.py \
  --config examples/configs/ppo_math_1B_megatron.yaml \
  policy.model_name="Qwen/Qwen2.5-1.5B" \
  cluster.gpus_per_node=8 \
  checkpointing.checkpoint_dir="results/ppo_megatron" \
  logger.wandb_enabled=True \
  logger.wandb.name="ppo-megatron"
```

## PPO Multi-node

```sh
NUM_ACTOR_NODES=8

COMMAND="uv run ./examples/run_ppo.py \
  --config examples/configs/ppo_math_1B_megatron.yaml \
  cluster.num_nodes=8 \
  cluster.gpus_per_node=8 \
  checkpointing.checkpoint_dir='results/ppo_8nodes' \
  logger.wandb_enabled=True \
  logger.wandb.name='ppo-multinode'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

> [!NOTE]
> For GB200 systems with 4 GPUs per node, use `--gres=gpu:4` instead.

## Configuration

PPO uses two base configurations:

- Megatron-Core backend: [examples/configs/ppo_math_1B_megatron.yaml](../../../examples/configs/ppo_math_1B_megatron.yaml)
- DTensor backend is not yet supported for the value model.

Key PPO-specific parameters:

```yaml
ppo:
  adv_estimator:
    name: "gae"
    gae_lambda: 0.95
    gae_gamma: 1.0
  steps_per_epoch: 4
  policy_training_start_step: 0

loss_fn:
  value_loss_scale: 0.4
  value_cliprange: 0.2

value:
  model_name: "Qwen/Qwen2.5-1.5B"
```

## Additional Resources

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [PPO In-Depth Guide](../../guides/ppo.md)
- [GAE Paper](https://arxiv.org/abs/1506.02438)
- [GRPO Documentation](grpo.md)
