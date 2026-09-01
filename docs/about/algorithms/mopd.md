# Multi-Teacher On-Policy Distillation (MOPD)

Multi-Teacher On-Policy Distillation (MOPD) distills one or more teacher models
into the policy by replacing GRPO's reward-based advantage with a token-level
distillation advantage ([MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)).
MOPD runs on async GRPO and collects rollouts through NeMo Gym, so the agent
loop drives multi-turn / multi-step interaction. Each token of the resulting
student rollout is scored by a teacher, and the policy is updated to close the
gap with the teacher.

Unlike the teacher-logit knowledge distillation in
[On-policy Distillation](on-policy-distillation.md) (`run_distillation.py`), MOPD
runs on top of the GRPO trainer: it is selected with `adv_estimator: opd` and
serves teachers from dedicated, non-colocated worker groups during async
collection.

## Advantage

For each token `t`, the distillation advantage is the stop-gradient
teacher-minus-student log-probability gap:

```
Â_t = sg[ log π_teacher(t) − log π_student(t) ]
```

`log π_student` is the policy's `prev_logprobs` and `log π_teacher` is computed
by the teacher worker group at collection time. Maximizing this advantage is
reverse-KL minimization — it pushes the student toward the teacher's token
distribution — but, unlike forward-KL logit distillation, it needs only the
teacher's log-probability for the *sampled* token rather than the full
vocabulary distribution.

The advantage is applied only to trained (assistant) tokens via the loss mask;
tool / environment tokens contribute zero. Because the advantage subtracts a
real `prev_logprobs`, MOPD requires the student log-probabilities to actually be
computed — see [Configuration](#configuration).

### Top-k reverse KL

Setting either `on_policy_distillation.student_topk` or `teacher_topk` replaces
the sampled-token MOPD loss with a lower-variance reverse-KL estimator. For
every token, the selected model supplies its highest-probability `k` vocabulary
entries. NeMo RL evaluates both models exactly on that support and uses the
sampled rollout token to estimate the contribution from the remaining
vocabulary. The two selectors are mutually exclusive.

The sampled tail uses the score-function coefficient
`1 + log π_student − log π_teacher`. The `1` is required because the derivative
also acts on the student probability multiplying the log-probability gap.
Support indices and teacher probabilities are treated as constants; gradients
flow only through the current student probabilities.

## Configuration

Enable MOPD in two places: select the advantage estimator and add the
`on_policy_distillation` block.

```yaml
grpo:
  # MOPD runs on async GRPO with NeMo Gym rollouts.
  async_grpo:
    enabled: true
  adv_estimator:
    name: opd
  # OPD subtracts a real prev_logprobs, so it must not be skipped.
  seq_logprob_error_threshold: 2.0

loss_fn:
  # REINFORCE form (drop the PPO probability-ratio clipping); on-policy
  # correction is handled by the ICE-POP gate below instead.
  disable_ppo_ratio: true
  # ICE-POP hard gate: zero tokens whose train/inference importance-sampling
  # weight falls outside bounds, correcting async off-policy drift.
  use_importance_sampling_correction: true
  truncated_importance_sampling_type: icepop
  # Teacher distillation is the entire learning signal — no reference-policy KL.
  reference_policy_kl_penalty: 0.0

on_policy_distillation:
  enabled: true
  # Optional: use the student's top-k vocabulary entries for an exact KL head
  # and the sampled rollout token for the remaining tail.
  # student_topk: 64
  # Alternatively, let the teacher select the support. This is mutually
  # exclusive with student_topk.
  # teacher_topk: 64
  # Map each NeMo Gym agent name to a teacher checkpoint.
  teacher_model_by_agent_name:
    default_teacher: Qwen/Qwen3-1.7B
    large_teacher: /checkpoints/large-teacher
  # Agents not present in the map fall back to this alias (must be a mapped key).
  default_teacher_alias: default_teacher
  # If true, an unmapped agent raises instead of falling back.
  strict_agent_name_match: false
  # Aliases that share one checkpoint and the same effective resource config
  # reuse a single teacher worker group. Conflicting configs fail validation.
  deduplicate_shared_teacher_checkpoints: true
  non_colocated_teachers:
    enabled: true
    # Resourcing for each teacher worker group.
    default_teacher_cfg:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      context_parallel_size: 1
      expert_tensor_parallel_size: 1
      expert_model_parallel_size: 1
      num_nodes: 1
      gpus_per_node: 8
      precision: bfloat16
      micro_batch_size: 1
      # Additional Megatron settings inherited by every teacher.
      megatron_cfg_overrides:
        moe_token_dispatcher_type: alltoall
    # Optional sparse per-alias overrides on top of default_teacher_cfg.
    # Nested megatron_cfg_overrides merge by key with the default map.
    teacher_overrides:
      large_teacher:
        tensor_model_parallel_size: 8
        expert_model_parallel_size: 4
        num_nodes: 2
        megatron_cfg_overrides:
          moe_token_dispatcher_type: flex
          moe_enable_deepep: true
          moe_shared_expert_overlap: false
          moe_flex_dispatcher_backend: deepep
```

An empty per-alias `megatron_cfg_overrides` map inherits all default keys; it
does not clear them. There is currently no configuration syntax for deleting an
inherited Megatron override. When the same setting appears in more than one
place, precedence is: default field, default Megatron override, alias field,
then alias Megatron override.

Teacher parallelism and precision are independent of the policy configuration.
In particular, `expert_tensor_parallel_size` defaults to 1; setup warns when
that differs from the policy because reducing ETP can increase per-rank expert
memory. Precision accepts `float32`, `bfloat16`, or `float16`; the legacy
spelling `bf16` is normalized to `bfloat16`.

> [!NOTE]
> Teachers run the Megatron backend in inference-only mode. A DTensor-configured
> policy is rejected for the teacher; PEFT / draft modules are stripped so
> adapters are never attached to the frozen teacher; and teachers run
> unquantized (a policy `quant_cfg` is ignored, with a warning).

> [!NOTE]
> `adv_estimator: opd` fails fast at setup if the config would zero
> `prev_logprobs` (`loss_fn.force_on_policy_ratio: true` with no
> `grpo.seq_logprob_error_threshold`), because the advantage would silently
> degrade to `teacher_logprobs − 0`.

> [!NOTE]
> Top-k mode currently requires async GRPO, a Megatron policy backend,
> token-level loss, `loss_fn.disable_ppo_ratio: true`, fused linear
> log-probabilities disabled, importance-sampling correction disabled
> (`loss_fn.use_importance_sampling_correction: false` and
> `loss_fn.truncated_importance_sampling_type: null`), and an unfiltered
> training distribution
> (`generation.top_k: null`, `top_p: 1.0`). Student-selected top-k additionally
> requires sequence packing disabled and context parallel size 1.
> Teacher-selected top-k supports packing and context parallelism when
> `policy.sequence_packing.fuse_loss: true`; context parallelism requires packing.
> CISPO, dual PPO clipping, and sequence-level importance ratios are unsupported.

Top-k training reports four additional metrics:

- `opd_topk_head_loss`: exact reverse-KL contribution on the selected support.
- `opd_topk_tail_loss`: sampled score-function surrogate for the tail gradient.
- `opd_topk_student_mass`: student probability mass captured by the support.
- `opd_topk_target_outside_fraction`: fraction of sampled targets outside it.

### Teacher routing

Each rollout sample carries its NeMo Gym `agent_ref`. At collection time the
agent name is resolved to a teacher alias (`teacher_model_by_agent_name`, falling
back to `default_teacher_alias`), samples are grouped by teacher, and each group
is scored by exactly one teacher — there is no ensemble averaging across
teachers. When several aliases map to the same checkpoint,
`deduplicate_shared_teacher_checkpoints` collapses them onto a single worker
group so they share GPUs. Those aliases must have identical effective resource
and Megatron overrides; conflicting overrides fail during setup. Align the
overrides to share one group, or set deduplication to `false` to create a
separate group—and reserve separate nodes—for every alias. Configurations that
previously relied on a later alias override being silently discarded must make
that choice explicitly.

### Resourcing

Non-colocated teachers each get their own Ray cluster on dedicated GPUs (they
are queried every rollout group, so time-sharing with the policy/generation
would serialize and destroy the async overlap). Their nodes are reserved from
the policy's budget: with `total_nodes` total, the teacher groups take
`sum(num_nodes)` and the policy uses the remainder (setup fails if nothing is
left for the policy). Deduplicated teachers share one group's nodes.

For example, the reference 3-node recipe lays out: 1 node policy (student,
trainable) + 1 node vLLM generation (frozen) + 1 node teacher (frozen). Ten
distinct teachers at 1 node each would instead add 10 nodes on top of the
policy and generation nodes.

## Running MOPD

MOPD collects rollouts through NeMo Gym and supports both the legacy async GRPO
runtime and the Single-Controller runtime. The checked-in recipes use
placeholder dataset paths; override them for your local data.

### Single-Controller text path

The Single-Controller path moves rollout and teacher-logprob tensors through
TransferQueue. It currently supports text-only MOPD rollouts:

```sh
uv run examples/run_grpo_single_controller.py \
  --config examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack-single-controller.yaml \
  data.train.data_path=/path/to/train.jsonl \
  data.validation.data_path=/path/to/val.jsonl
```

See [Train with Single-Controller](../../guides/single-controller.md) for the
runtime's configuration and architecture.

### Legacy async GRPO path

```sh
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml \
  data.train.data_path=/path/to/train.jsonl \
  data.validation.data_path=/path/to/val.jsonl
```

Both reference recipes self-distill `Qwen/Qwen3-1.7B` (student == teacher)
across 3 nodes (1 policy + 1 vLLM + 1 teacher) with sequence packing enabled.
Because student and teacher are identical, the OPD loss stays near zero — it is
a correctness smoke test, not a demonstration of distillation gains.

To exercise the student-top-k path with that recipe, disable its inherited
sequence packing and set the support size explicitly:

```sh
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml \
  policy.sequence_packing.enabled=false \
  on_policy_distillation.student_topk=64 \
  loss_fn.use_importance_sampling_correction=false \
  loss_fn.truncated_importance_sampling_type=null \
  data.train.data_path=/path/to/train.jsonl \
  data.validation.data_path=/path/to/val.jsonl
```

To run the teacher-selected variant with the recipe's inherited sequence
packing, keep packing enabled and select the teacher support explicitly:

```sh
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml \
  on_policy_distillation.teacher_topk=64 \
  loss_fn.use_importance_sampling_correction=false \
  loss_fn.truncated_importance_sampling_type=null \
  data.train.data_path=/path/to/train.jsonl \
  data.validation.data_path=/path/to/val.jsonl
```

The recipe sets `policy.sequence_packing.fuse_loss=true`, as required for
packed teacher-top-k. Teacher selection is performed inside each teacher worker
in the same forward that computes sampled-token probabilities; the selected
indices and normalized log-probabilities are stored with the trajectory and
reused by the trainer.

This remains a smoke-test configuration. Measure throughput, peak memory, and
reward convergence against sampled-token MOPD before using it as a performance
or quality baseline. Monitor `opd_topk_head_loss` for the exact-support
contribution; raw loss scales are not directly comparable between the two
estimators.

## References

- LLM-Core Xiaomi, *MiMo-V2-Flash Technical Report*, which introduces the
  multi-teacher on-policy distillation paradigm:
  [arxiv.org/abs/2601.02780](https://arxiv.org/abs/2601.02780)
