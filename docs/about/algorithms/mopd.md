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
distribution — but, in this default (top-k) form, it needs only the teacher's
log-probability for the *sampled* token rather than the full vocabulary
distribution. See [Full-vocabulary MOPD](#full-vocabulary-mopd) for the exact
K=V variant, which trades that property for an unbiased objective.

The advantage is applied only to trained (assistant) tokens via the loss mask;
tool / environment tokens contribute zero. Because the advantage subtracts a
real `prev_logprobs`, MOPD requires the student log-probabilities to actually be
computed — see [Configuration](#configuration).

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
> policy is rejected for the teacher; PEFT / draft modules are stripped so
> adapters are never attached to the frozen teacher; and teachers run
> unquantized (a policy `quant_cfg` is ignored, with a warning).

> [!NOTE]
> `adv_estimator: opd` fails fast at setup if the config would zero
> `prev_logprobs` (`loss_fn.force_on_policy_ratio: true` with no
> `grpo.seq_logprob_error_threshold`), because the advantage would silently
> degrade to `teacher_logprobs − 0`.

### Teacher routing

Each rollout sample carries its NeMo Gym `agent_ref`. At collection time the
agent name is resolved to a teacher alias (`teacher_model_by_agent_name`, falling
back to `default_teacher_alias`), samples are grouped by teacher, and each group
is scored by exactly one teacher — there is no ensemble averaging across
teachers. When several aliases map to the same checkpoint,
`deduplicate_shared_teacher_checkpoints` collapses them onto a single worker
group so they share GPUs.

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

## Full-vocabulary MOPD

`on_policy_distillation.full` replaces the sampled-token log-probability gap
with the exact reverse KL over the whole vocabulary:

```
L_t = Σ_v p_student(v) · [ log p_student(v) − log p_teacher(v) ]
```

This is the K=V limit of the top-k estimator: with the support spanning the
whole vocabulary the score-function tail term vanishes, so the objective is
exact, deterministic, and free of that estimator's off-policy bias. It replaces
the policy-gradient objective entirely — the OPD advantage estimator still runs
(`advantages` is a required training column and its stage supplies the
teacher/student gap diagnostic), but this loss ignores its output.

```yaml
on_policy_distillation:
  full:
    enabled: true
    teacher_payload: hidden_states  # or: logits
    divergence: reverse_kl
    payload_dtype: bfloat16
    teacher_lm_head_lifecycle: offload  # none | offload | evict
    chunk_size: 1024
    validate_decomposition: false
```

`teacher_payload` selects what crosses the teacher/student boundary:

| | width | notes |
|---|---|---|
| `hidden_states` (default) | `hidden_size` | Teacher ships pre-LM-head hidden states; the student projects them with an output-layer shard loaded from the teacher checkpoint. Teacher and student parallelism stay decoupled. |
| `logits` | `vocab_size` | Teacher ships full-vocabulary logits; no student-side teacher LM head. Roughly 74× larger for a 2k-hidden / 152k-vocab model — a numerical reference and fallback, not a production configuration. |

`chunk_size` bounds the live fp32 vocabulary working set in the divergence
kernels; unchunked, one 8K-token row materializes several GB of fp32
log-softmax. `teacher_lm_head_lifecycle` controls whether the teacher LM-head
shard stays resident on GPU, is parked on CPU between steps, or is freed and
reloaded each step.

Both payloads support more than one teacher checkpoint. On the `hidden_states`
path the student loads one LM-head shard per distinct teacher and every payload
row is tagged with the teacher that produced it, so a single microbatch may mix
teachers. Two consequences worth planning for: the resident LM-head cost grows
linearly with the number of distinct teachers (`[vocab_size / TP, hidden_size]`
each), which makes `teacher_lm_head_lifecycle` more important the more teachers
a run has; and because one payload column carries them all, every teacher on
this path must share the student's tokenizer and the same `hidden_size`. The
`logits` path ships an already-projected distribution and needs neither a
student-side LM head nor per-row tagging.

Student pipeline parallelism works on both payloads. Megatron builds
`output_layer` — and runs the loss — only on the last pipeline stage, so that is
the only stage that projects the teacher's hidden states; the earlier stages
still join the LM-head load collective, requesting nothing.

`validate_decomposition` additionally reports the reverse KL against its
entropy / cross-entropy decomposition. Note that this residual is an algebraic
identity — all three kernels read the same logits, so a corrupted teacher
cancels out of it. It pins the kernels' own arithmetic and nothing upstream of
them, at the cost of a second full-vocabulary log-softmax, so it is off by
default. The assertion that actually catches a broken payload, gather, LM-head
shard, or token shift is the divergence itself staying near zero under
self-distillation.

### Current restrictions

Rejected at construction rather than silently ignored:

- Megatron backend and the Single-Controller runtime only.
- `teacher_payload: hidden_states` additionally requires
  `policy.generation.temperature: 1.0`. The `logits` path has no such
  restriction.
- `teacher_payload: hidden_states` also requires a teacher whose logits are
  exactly `output_layer(h)`. Models that transform the logits after that linear
  — Gemma2 and Gemma4 (`final_logit_softcapping`), MuseGlimmer
  (`output_multiplier`), and any MuP model (`use_mup`) — are rejected on the
  teacher worker, because the student's reconstruction cannot reproduce the
  post-transform and would silently distill toward a distribution the teacher
  never emits. Use `teacher_payload: logits`, which is exact for these models.
- `policy.megatron_cfg.use_fused_linear_logprobs: false` and
  `policy.sequence_packing.fuse_loss: false`.
- The policy-gradient and reward-side KL knobs have no code path under this
  objective and are rejected: `disable_ppo_ratio: false`, `ratio_clip_c`,
  `use_cispo`, `force_on_policy_ratio`, `sequence_level_importance_ratios`,
  `use_importance_sampling_correction`, `truncated_importance_sampling_type`,
  `positive_example_nll_weight`, `use_kl_in_reward`, and
  `use_on_policy_kl_approximation` (the base MOPD recipe sets this one to
  `true`, so a derived full-vocabulary recipe must override it to `false`).

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

The full-vocabulary variants of that recipe are
`mopd-qwen3-1.7b-3n8g-megatron-pack-single-controller-fullvocab.yaml` (H100) and
`mopd-qwen3-1.7b-3n4g-megatron-pack-single-controller-fullvocab.yaml` (GB200),
run the same way.

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

## References

- LLM-Core Xiaomi, *MiMo-V2-Flash Technical Report*, which introduces the
  multi-teacher on-policy distillation paradigm:
  [arxiv.org/abs/2601.02780](https://arxiv.org/abs/2601.02780)
