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

When student and teacher tokenizers differ, the teacher scores its own rendered
transcript and the resulting log-probabilities are projected onto student token
positions. For each valid alignment pair `(s0, s1, t0, t1)`, every student token
in `[s0, s1)` receives:

```
sum(teacher_logprobs[t0:t1]) / (s1 - s0)
```

The projected student-span sum therefore equals the joint teacher
log-probability for the aligned teacher span. Invalid or structurally ambiguous
positions are zeroed in the advantage through a separate teacher-validity mask;
the training loss mask and its denominator are not changed.

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
    # Optional TOP-D scalar reward shaping; null preserves ordinary MOPD.
    proximal_reward_alpha: null
    proximal_reward_scale: 1.0
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
      # This is resolved for each teacher rather than inherited from the policy.
      use_fused_linear_logprobs: false
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

### Cross-tokenizer configuration

Add `cross_tokenizer` to a teacher resource when that teacher does not share the
student tokenizer. A null or omitted block preserves same-token MOPD behavior.

```yaml
on_policy_distillation:
  non_colocated_teachers:
    default_teacher_cfg:
      cross_tokenizer:
        tokenizer:
          name: /path/to/teacher-tokenizer
          chat_template: default
          chat_template_kwargs: {}
          tokenizer_kwargs: {}
        alignment_method: offset_cluster_decode_fix
        mask_first_teacher_prefix_chunk: false
        exclude_proven_template_only_teacher_tokens: true
        missing_think_close_policy: preserve_open_if_proven
```

`offset_cluster_decode_fix` is the only supported alignment method. The schema
defaults `mask_first_teacher_prefix_chunk` and
`exclude_proven_template_only_teacher_tokens` to `false`, and defaults
`missing_think_close_policy` to `mask`. The corrected parity recipe explicitly
enables template-only exclusion and uses `preserve_open_if_proven`.

Old flat teacher-resource keys are rejected instead of being forwarded as
Megatron overrides:

| Rejected key | Migration |
|---|---|
| `tokenizer_name` | Set `cross_tokenizer.tokenizer.name`. |
| `alignment_method` | Set `cross_tokenizer.alignment_method`. |
| `chunk_size` | Remove it; the supported offset aligner has no replacement setting. |

The collector loads independent, plain Hugging Face copies of both the student
and teacher alignment tokenizers. The runtime student tokenizer remains
authoritative for sampled IDs, while its plain copy supplies canonical character
offsets even when Fastokens is enabled; the teacher copy is separate from the
model worker's tokenizer. Tokenizer paths, revisions, chat templates, template
arguments, and tokenizer arguments must therefore be pinned for reproducibility.
`use_fused_linear_logprobs` is likewise resolved explicitly for each teacher;
set it in the teacher resource rather than relying on the student policy value.

### Cross-token transcript semantics

Cross-token scoring reconstructs the exact sampled student token stream from
`message_log`, then renders the full multi-turn transcript with the teacher's
native chat template. This includes developer/system normalization and tool
loops. Only generated assistant messages that carry generation log-probabilities
are aligned and trained; user, tool, environment, and template-only regions do
not become distillation targets.

The scorer preserves native or proven-open Qwen thinking state. An ambiguous
thinking/tool structure fails closed by masking the causally affected suffix.
`missing_think_close_policy: preserve_open_if_proven` preserves an open thinking
turn only when the reconstructed transcript proves that state; otherwise it is
masked. When template provenance can be proved,
`exclude_proven_template_only_teacher_tokens: true` excludes teacher tokens made
entirely from template text while retaining tokens that mix template and model
content. The scorer pairs a sampled end-of-turn token only when it is present in
the sampled stream and never synthesizes an EOS token. It also supports manual
offset recovery for noncanonical generated token streams.

Alignment errors, orphan spans, configured teacher-prefix chunks, sequence
overflow, out-of-bounds spans, and structurally unsafe regions are invalidated.
Every scored rollout entering replay contains both
`teacher_reference_logprobs` and `teacher_reference_logprobs_mask`; same-token
teachers produce an all-valid mask. At advantage time the effective mask is the
intersection of token, sample, and teacher masks, and masked selection is used
so invalid `NaN` or infinite scores cannot leak through `0 * value` arithmetic.
The mask affects only OPD advantages, not the training token mask, loss
denominator, or global valid-token count.

Older same-token replay entries without the mask are loaded with an all-ones
mask. A cross-token run rejects an older replay entry that lacks the mask; start
with a fresh replay buffer or disable replay loading rather than guessing which
alignments were valid. Both the score and mask are saved for new checkpoints.

### Optional TOP-D scalar reward shaping

MOPD can optionally transform the teacher/student log-probability gap `g` with:

```
proximal_reward_scale * log(
    proximal_reward_alpha * exp(g) + 1 - proximal_reward_alpha
)
```

Set `grpo.adv_estimator.proximal_reward_alpha` in `(0, 1]` and a finite positive
`proximal_reward_scale` to enable it. `proximal_reward_alpha: null` and
`proximal_reward_scale: 1.0` preserve ordinary MOPD and are used by the parity
recipe. The numerically stable transform runs after teacher-to-student
projection and before the effective alignment mask; malformed-thinking and
invalid-tool-call overrides are then applied, followed by advantage clipping.

This option is only **TOP-D scalar reward shaping**. It does not implement full
TOP-D future returns, group normalization, or PPO minibatch reuse, and it is
unrelated to [Full-vocabulary MOPD](#full-vocabulary-mopd).

### Observability

Cross-token scoring reports per-teacher latency and alignment coverage together
with counters for incorrect alignments, orphans, template-only and prefix masks,
structural failures, overflow, and out-of-bounds spans. The advantage stage
continues to report the raw
`on_policy_distillation/teacher_student_logprob_gap_mean` plus advantage
statistics; when scalar shaping is enabled it also reports transformed-reward
statistics. Treat falling coverage or rising structural/overflow counters as a
data or template regression rather than silently accepting a smaller training
signal.

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

### Runtime support matrix

Cross-tokenizer support is deliberately narrower than same-token MOPD v1.
Unsupported combinations fail during setup, before teacher workers are
allocated.

| Mode | Legacy async GRPO + NeMo Gym | Single-Controller |
|---|---:|---:|
| Sampled-token, same tokenizer | Supported | Supported (text only) |
| Sampled-token, cross tokenizer, text | **Supported** | Rejected; use legacy async GRPO |
| Full-vocabulary, same tokenizer | Rejected | Supported with the restrictions below |
| Full-vocabulary, cross tokenizer | Rejected | Rejected |
| Multimodal, cross tokenizer (v1) | Rejected | Rejected |

Cross-token MOPD requires `on_policy_distillation.enabled: true`, non-colocated
teachers, `grpo.adv_estimator.name: opd`, `grpo.async_grpo.enabled: true`, and
`env.should_use_nemo_gym: true`. The existing same-token teacher batching and
multimodal path are unchanged; cross-token v1 does not add cross-prompt teacher
batching.

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
- Exactly one teacher checkpoint.
- `teacher_payload: hidden_states` additionally requires student
  `policy.megatron_cfg.pipeline_model_parallel_size: 1` and
  `policy.generation.temperature: 1.0`. The `logits` path has neither
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

MOPD collects rollouts through NeMo Gym. Same-token sampled MOPD supports the
legacy async GRPO and Single-Controller runtimes; cross-tokenizer MOPD uses only
the legacy async entrypoint. The checked-in recipes use placeholder dataset and
checkpoint paths; replace them before launching.

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

For a cross-token run, start from the corrected parity recipe and replace its
checkpoint, tokenizer, and public/synthetic dataset placeholders:

```sh
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack-xtoken.yaml
```

That recipe pins seed 42, one generation per prompt, 256 prompts/global batch,
65,536 total tokens, 32,768 generated tokens, constant `1e-6` learning rate,
the recorded clipping and ICE-POP bounds, and `force_on_policy_ratio: true`.
The forced on-policy ratio takes precedence over ratio clipping and is required
to reproduce the recorded numeric loss. TOP-D scalar reward shaping remains
disabled. The corresponding internal reproduction manifest—not the public
recipe—pins exact checkpoint/tokenizer revisions, chat-template hashes, dataset
SHA, container digest, and NeMo Gym patch state. Private prompts, tool traces,
and the 192-record differential corpus are not included.

The two same-token reference recipes self-distill `Qwen/Qwen3-1.7B`
(student == teacher) across 3 nodes (1 policy + 1 vLLM + 1 teacher) with
sequence packing enabled. Because student and teacher are identical, the OPD
loss stays near zero — it is a correctness smoke test, not a demonstration of
distillation gains. The cross-token recipe instead contains explicit model and
tokenizer placeholders and is not runnable until they are replaced.

## References

- LLM-Core Xiaomi, *MiMo-V2-Flash Technical Report*, which introduces the
  multi-teacher on-policy distillation paradigm:
  [arxiv.org/abs/2601.02780](https://arxiv.org/abs/2601.02780)
