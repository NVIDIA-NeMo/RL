# Multi-Teacher On-Policy Distillation (MOPD)

Multi-Teacher On-Policy Distillation (MOPD) distills one or more teacher models
into the policy by replacing GRPO's reward-based advantage with a token-level
teacher-minus-student log-probability gap. The student generates on-policy
rollouts (optionally through NeMo Gym), each token is scored by a teacher, and
the policy is updated to close the gap with the teacher.

Unlike the teacher-logit knowledge distillation in
[On-policy Distillation](on-policy-distillation.md) (`run_distillation.py`), MOPD
runs on top of the GRPO trainer: it is selected with `adv_estimator: opd` and
serves teachers from dedicated, non-colocated worker groups during async
collection.

## Advantage

For each token `t`, the advantage is the stop-gradient teacher-minus-student
gap:

```
Â_t = sg[ log π_teacher(t) − log π_student(t) ]
```

`log π_student` is the policy's `prev_logprobs` and `log π_teacher` is computed
by the teacher worker group at collection time. Because the advantage subtracts
a real `prev_logprobs`, MOPD requires the student log-probabilities to actually
be computed — see [Configuration](#configuration).

## Configuration

Enable MOPD in two places: select the advantage estimator and add the
`on_policy_distillation` block.

```yaml
grpo:
  adv_estimator:
    name: opd
  # OPD subtracts a real prev_logprobs, so it must not be skipped.
  seq_logprob_error_threshold: 2.0

loss_fn:
  # REINFORCE-style update recommended by the MOPD paper.
  disable_ppo_ratio: true
  use_importance_sampling_correction: true
  truncated_importance_sampling_type: icepop

on_policy_distillation:
  enabled: true
  # Map each NeMo Gym agent name to a teacher checkpoint.
  teacher_model_by_agent_name:
    default_teacher: Qwen/Qwen3-1.7B
  # Agents not present in the map fall back to this alias (must be a mapped key).
  default_teacher_alias: default_teacher
  # If true, an unmapped agent raises instead of falling back.
  strict_agent_name_match: false
  # Aliases that share one checkpoint reuse a single teacher worker group.
  deduplicate_shared_teacher_checkpoints: true
  non_colocated_teachers:
    enabled: true
    # Resourcing for each teacher worker group.
    default_teacher_cfg:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      context_parallel_size: 1
      num_nodes: 1
      gpus_per_node: 8
      precision: bf16
      micro_batch_size: 1
    # Optional per-alias overrides on top of default_teacher_cfg.
    teacher_overrides: {}
```

> [!NOTE]
> Teachers run the Megatron backend in inference-only mode. A DTensor-configured
> policy is rejected for the teacher, and PEFT / draft modules are stripped from
> the teacher config so adapters are never attached to the frozen teacher.

> [!NOTE]
> `adv_estimator: opd` fails fast at setup if the config would zero
> `prev_logprobs` (`loss_fn.force_on_policy_ratio: true` with no
> `grpo.seq_logprob_error_threshold`), because the advantage would silently
> degrade to `teacher_logprobs − 0`.

### Resourcing

Non-colocated teachers run on their own nodes, reserved from the policy's node
budget: with a total of `total_nodes`, the teacher worker groups take
`sum(num_nodes)` nodes and the policy uses the remainder. Size the cluster so
the policy, generation, and teacher groups all fit.

## Running MOPD

MOPD collects rollouts through NeMo Gym, so use the NeMo Gym GRPO entrypoint
with an MOPD recipe. The checked-in recipe uses placeholder dataset paths;
override them for your local data:

```sh
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml \
  data.train.data_path=/path/to/train.jsonl \
  data.val.data_path=/path/to/val.jsonl
```

The reference recipe self-distills `Qwen/Qwen3-1.7B` (student == teacher) across
3 nodes (1 policy + 1 vLLM + 1 teacher) with sequence packing enabled. Because
student and teacher are identical, the OPD loss stays near zero — it is a
correctness smoke test, not a demonstration of distillation gains.
