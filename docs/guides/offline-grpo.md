# Offline GRPO in NeMo RL

Offline GRPO trains on fixed, rewarded teacher trajectories instead of generating
rollouts from the policy during every step. NeMo RL's implementation follows the
[KRAFTON Offline-GRPO recipe](https://github.com/krafton-ai/Offline-GRPO): it uses
group-relative rewards, learns from both positive and negative trajectories, and
assigns a configurable positive bias to groups in which every teacher trajectory
is correct.

No generation worker or reward environment is allocated. The only model workers
are the trainable policy and, when KL regularization is enabled, its frozen initial
reference policy.

## Objective

For each prompt, let the selected teacher trajectories have rewards
\(r_1, \ldots, r_G\). The default advantage is

\[
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j.
\]

When every reward in a group is greater than `positive_reward_threshold`, all
advantages in that group are replaced by `all_positive_bias`. This keeps fully
correct teacher groups useful even when the student cannot yet solve the prompt.

The teacher behavior-policy likelihood is unavailable, so the KRAFTON
approximation treats it as one. The per-token actor loss is therefore

\[
\mathcal{L}_{actor} = -A_i\exp(\log \pi_\theta(y_t \mid x, y_{<t})).
\]

An optional KL penalty anchors training to the frozen initial policy. This loss is
not SFT: negative advantages decrease the probability of incorrect trajectories,
and the positive term is probability-weighted rather than token NLL. It is also
not DPO: it consumes scalar rewards for a group of responses rather than a chosen
and rejected pair with a logistic preference objective.

## Dataset format

Each Parquet, JSON/JSONL, Arrow, or Hugging Face row must represent one
prompt group and contain:

- one prompt, as a string or a list of chat-message mappings;
- a list of teacher responses, where each response is a string or chat-message
  list ending in an assistant message; and
- a same-length list of finite scalar rewards.

For example:

```json
{
  "prompt": "Solve 2x + 3 = 11.",
  "responses": [
    "Subtract 3, then divide by 2, so x = 4.",
    "Dividing by 2 immediately gives x = 5.5."
  ],
  "rewards": [1.0, 0.0]
}
```

Column names are configurable. KRAFTON's prepared OpenThoughts Parquet uses
`prompt`, `target_lst`, and `target_rewards`; those mappings are already set in
[the example config](../../examples/configs/offline_grpo.yaml). Each prompt must
have at least `offline_grpo.num_responses_per_prompt` responses. Extra responses
are selected either deterministically from the front or by a reproducible random
selection that changes with the training step.

Only the final assistant response is included in the token loss. A trajectory
that exceeds `data.max_input_seq_length` is masked out rather than partially
trained.

## Launch training

Copy the example config, set the model and train/validation data paths, and run:

```bash
uv run examples/run_offline_grpo.py \
  --config examples/configs/offline_grpo.yaml \
  policy.model_name=<MODEL_NAME> \
  data.train.data_path=<TRAIN_PARQUET> \
  data.validation.data_path=<VALIDATION_PARQUET>
```

The example mirrors the blog's experiment settings: eight responses per prompt,
an all-positive bias of `0.1`, unnormalized rewards, learning rate `1e-7`, and
reference KL coefficient `0.1`. The policy global batch size is the number of
prompt groups multiplied by the responses selected per prompt:

```text
policy.train_global_batch_size
  = offline_grpo.num_prompts_per_step
  * offline_grpo.num_responses_per_prompt
```

Set `loss_fn.reference_policy_kl_penalty=0` to avoid loading a frozen reference
model. Set `offline_grpo.val_period=0`,
`offline_grpo.val_at_start=false`, and `data.validation=null` when no held-out
offline dataset is available. Offline validation measures the held-out training
objective; use NeMo RL's evaluation workflows for task-level benchmark scores.
