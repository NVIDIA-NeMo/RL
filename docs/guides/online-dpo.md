# Online DPO in NeMo RL

Online DPO samples multiple responses from the current policy, scores those responses with a NeMo RL environment, converts the scored responses into chosen/rejected preference pairs, and trains the policy with the existing DPO loss against a frozen reference policy.

The first implementation is intentionally narrow:

- text-only LLM prompts
- one rollout turn
- exactly two sampled responses per prompt
- scalar environment rewards
- colocated generation
- no sequence packing or dynamic batching
- no NeMo-Gym rollouts or multiple prompt dataloaders

These constraints keep pair construction and DPO masking predictable. The rollout helpers append the environment observation after scoring; Online DPO strips that trailing observation and trains only on the sampled assistant response.

## Launch an Online DPO Run

Use [examples/run_online_dpo.py](../../examples/run_online_dpo.py):

```bash
uv run examples/run_online_dpo.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```

If `--config` is omitted, the script uses [examples/configs/online_dpo.yaml](../../examples/configs/online_dpo.yaml).

For example, to run a short local smoke test:

```bash
uv run examples/run_online_dpo.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    cluster.gpus_per_node=2 \
    online_dpo.num_prompts_per_step=2 \
    policy.train_global_batch_size=2 \
    policy.train_micro_batch_size=1 \
    online_dpo.max_num_steps=2 \
    checkpointing.enabled=false
```

## Data and Environments

Online DPO uses response-style prompt datasets, the same data surface used by GRPO. Each prompt is rolled out twice, and both responses are scored by the configured environment. The higher-reward response becomes `chosen`; the lower-reward response becomes `rejected`.

Pairs are dropped when:

- the absolute reward difference is less than or equal to `online_dpo.min_reward_margin`
- either response is truncated and `online_dpo.drop_truncated_pairs` is `true`

The trainer collects enough usable pairs to fill `policy.train_global_batch_size`. If the dataloader ends before a full usable batch is available, the partial batch is discarded rather than carried across epochs.

## Important Configuration

- `online_dpo.num_generations_per_prompt` must be `2`.
- `online_dpo.max_rollout_turns` must be `1`.
- `online_dpo.num_prompts_per_step` must match `policy.train_global_batch_size`.
- `policy.sequence_packing.enabled` and `policy.dynamic_batching.enabled` must be `false`.
- `checkpointing.metric_name` can use `val:reward` for environment rollout validation.

Online DPO reuses the DPO loss keys under `online_dpo`:

- `reference_policy_kl_penalty`
- `preference_average_log_probs`
- `sft_average_log_probs`
- `preference_loss_weight`
- `sft_loss_weight`

## Validation

Validation performs normal environment rollouts with one response per prompt and reports reward and average generated length. It does not build online preference pairs or run a validation DPO loss.
