<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput { max-width: none !important; width: 98% !important; }
table { width: 100% !important; }
</style>

# Buffered Omega rollouts

`grpo.rollout_batch_multiplier` is a positive integer, default1. With synchronous
Omega and `omega_loss.form=per_sequence`, setting it to M collects up to M
consecutive prompt batches in one generation phase. The trainer consumes each
batch once, in order. It generates again only when those batches are exhausted.

`num_prompts_per_step=P` still counts prompts **per optimizer update**.
`num_generations_per_prompt=K` and `policy.train_global_batch_size=P*K` remain
unchanged. A full collection has M*P*K responses; final collections can be shorter
at an epoch or step limit. Dataloader drop-last behavior is unchanged.

Before **each** update, the trainer recomputes current-policy log probabilities,
advantages and detached Omega coefficients for that update's complete prompt
groups. Packing and microbatching then use the existing per-sequence loss. The
buffer stores generated responses, rewards and generation-time log probabilities,
never cached current-policy log probabilities or coefficients. Thus generation
logprob mismatch can grow with response age; a configured sequence-error mask
continues to apply and is not silently disabled.

The default1 preserves the existing generation/update path. Values above1 reject
clipped-PG/group-Omega, asynchronous GRPO, the data plane, dynamic sampling,
multiple dataloaders and on-policy distillation. Existing Omega restrictions still
apply, including no reward normalization and no force-on-policy shortcut.

Native checkpoints include `rollout_buffer.pt`, committed in the same temporary
checkpoint directory as model, optimizer and dataloader state. It contains only
unconsumed responses plus collection age and epoch-exhaustion state. Resume
requires the same buffer multiplier, P and K, matching checkpoint step and schema.
Missing/incompatible sidecars fail rather than regenerating or skipping responses.
A legacy checkpoint cannot be resumed directly with buffering enabled.

Training metrics `rollout/generated_sequences`, `rollout/generated_tokens`,
`rollout/collection_step`, `rollout/age_updates` and `rollout/remaining_updates`
separate generation work from cached updates. Cached updates do not wake or sleep
the generator. Validation retains the normal current-weight refit lifecycle.

See `smoke_tests/polaris_buffered_rollouts/` for the native Qwen3-1.7B/POLARIS
five-update test, including a process restart after update1. This smoke verifies
execution and resume correctness; it does not establish convergence or throughput.
