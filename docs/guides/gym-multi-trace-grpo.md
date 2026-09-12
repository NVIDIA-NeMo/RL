# Train multiple Gym traces per rollout

`grpo.gym_multi_trace: true` trains all independently conditioned sequences
returned by Gym's `training_traces` v1 envelope. This supports agent forks,
interleaved conversations, and context compaction without flattening unrelated
chats into one causal sequence. The default remains the existing single-response
Gym path.

## Configuration

Apply these overrides to a working synchronous NeMo Gym Megatron recipe:

```yaml
grpo:
  gym_multi_trace: true
  num_prompts_per_step: 2
  num_generations_per_prompt: 4
  async_grpo:
    enabled: false
data_plane: null
policy:
  train_global_batch_size: 8  # logical rollouts, before trace expansion
  train_micro_batch_size: 1
  megatron_cfg:
    enabled: true
loss_fn:
  token_level_loss: true
  sequence_level_importance_ratios: false
env:
  should_use_nemo_gym: true
  should_mask_flagged_samples: true
  nemo_gym:
    token_id_capture:
      enabled: true
      all_agents: true
      delivery: all_traces
      builder: per_request  # or prefix_merging
      dir: /shared/gym-capture
```

The Gym source must support `delivery: all_traces` and the v1 trace envelope.
Capture writers and the Gym actor must see the same capture directory.
RL calls Gym's finalizer after each completed rollout and retains the capture
files for inspection; it does not retire them before durable training handoff.

## Identity, reward, and loss

A task group is the original input example together with its independent
rollouts. Group membership is assigned before invoking the agent and survives
prompt rewriting. Gym's unique `rollout_id` identifies one such rollout;
`trace_id` identifies a physical training row within it.

GRPO computes a scalar advantage once per logical rollout using its task group
and outcome reward, then broadcasts that advantage to its eligible sampled
tokens. Trace count does not create extra reward observations. A masked rollout
or a rollout with no in-limit eligible trace is excluded from its group's
baseline and standard deviation. An overlong trace is individually replaced by
an inert row; valid sibling traces keep training. An entirely ineligible batch
fails before an optimizer update.

The objective is the mean over unique eligible sampled tokens in the complete
logical batch. It does not apply inverse-trace-count or equal-session weights.
Consequently longer sampled responses carry more token weight. Outcome reward
broadcast is not a process reward or a claim of causal credit assignment.

Each trace carries full `token_ids`, aligned `generation_logprobs` and a binary
`loss_mask`. `sampled_spans` identifies the model call owning each trainable
span. RL checks vector lengths, finiteness, masked initial conditioning tokens,
span coverage, and unique ownership. It never retokenizes training IDs. Shared
ancestor copies and intervening context remain masked. Prefixes and masks are
not inferred from rendered text or from a terminal `response.output`.

All physical rows form one optimizer update. Zero-loss rows pad the batch to
the data-parallel size times microbatch size; no trailing traces are dropped.
The Megatron learning-rate scheduler advances by the logical rollout count,
while checkpoint progress retains its existing count of original input prompts.
Metrics record logical rollouts, physical rows, padding, overlong traces, and
invalid rollouts. Training JSONL includes task-group, rollout, trace, and logical
row identifiers when training-data logging is enabled.

## Supported scope

This initial path requires synchronous GRPO, a text-only Megatron policy, and
token-level loss and importance ratios. Setup rejects asynchronous/TQ/staged
capture, custom capture sinks, router replay, multimodal models, alternative
advantage estimators, dynamic sampling, legacy episode-length filtering/shaping,
message-level penalties, active `seq-mask-tis` importance correction, and
post-advantage sequence-logprob-error masking. Sequence-level correction masks
depend on physical trace boundaries and cannot preserve split/merged equivalence.
`train_global_batch_size` must equal the logical rollout count per step.
Use `data_plane: null` or a complete disabled data-plane configuration.

The regression suite tests exact token custody, logical grouping and baseline
masking, unequal trace lengths/counts, individual overlength masking, physical
padding, configuration guards, and equivalent split/merged loss and gradients.
The Gym-marked tests also exercise real capture-store records through Gym's
finalizer and the RL actor for both builders.
