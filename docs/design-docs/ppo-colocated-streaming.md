# Colocated PPO streaming

## Scope and implementation plan

Policy and value share the training GPUs; generation uses separate GPUs. Enable
streaming by setting `async_rl.min_groups_for_streaming_train` below
`ppo.num_prompts_per_step`. This is a readiness threshold, not a fixed chunk count:
the in-order sampler may return more groups when they are already ready. Chunk
sizes align to both models' data-parallel layouts (and static microbatch sizes
when packing is disabled). Any unaligned tail stays queued for the next chunk;
no samples are duplicated or dropped. The full batch must also be divisible by
that common alignment.

1. Validate streaming before creating workers: require one policy epoch, a
   Megatron policy with classic DDP. Keep existing
   full-batch configurations working, including multiple policy epochs.
2. For each ready chunk, run value forward, policy/reference forward when needed,
   GAE, and policy backward. All trajectories in a selected group are complete.
3. Begin the policy step on the first valid chunk. Accumulate raw gradient sums
   and valid-token/sequence counts across chunks. Save gradient buffers to CPU
   before loading the critic, then restore them before further policy work.
4. Finish once after assembling the original RL batch. The existing split API
   normalizes by the actual full-step valid count and applies one optimizer and
   scheduler update. Publish/refit this policy version before critic training.
5. Concatenate processed metadata and run every critic epoch over the full
   original batch. Pre-update value predictions and GAE returns remain frozen.
   Clear consumed data-plane rows only after both model updates finish.
6. Validate config guards, phase ordering, gradient preservation across switches,
   masked chunks, warmup, and a multi-step run using the provided nightly image.

## Update and residency boundaries

Each streamed chunk follows:

```text
park policy (preserving pending gradients) -> value forward -> park value
-> restore policy -> policy/reference forward -> GAE -> policy backward
```

After the final chunk:

```text
policy optimizer step -> policy refit -> park policy
-> value epoch 1 -> value epoch 2 ... -> park value -> clear consumed rows
```

No critic backward or optimizer update occurs before the entire batch is ready.
The policy weights remain fixed throughout chunk processing; later chunks do not
observe an intermediate policy update. One early refit advances the policy
version, while completed-step counters advance after critic training. Periodic
snapshots remain excluded throughout this optimizer-commit interval.

During critic warmup, forwards and GAE can still stream, but policy backward,
optimizer update, and refit are skipped. The complete batch trains the critic.
An entirely masked chunk contributes no policy gradients; a completely masked
step fails before either model performs an optimizer update.

## Configuration and limits

```yaml
ppo:
  num_prompts_per_step: 256
  ppo_epochs: 1
  critic_ppo_epochs: 2
  adv_estimator:
    normalize_advantages: true
async_rl:
  min_groups_for_streaming_train: 32
  sampler:
    name: in_order
    max_lookahead_versions: 1
```

The full-batch size remains `ppo.num_prompts_per_step *
ppo.num_generations_per_prompt` for both policy and value. No new value-training
streaming option is introduced. Setting the readiness threshold equal to
`ppo.num_prompts_per_step` retains the existing critic-before-policy ordering and
permits multiple policy epochs. A smaller threshold selects the streaming path
even if the sampler happens to find an entire batch ready at once.

`ppo.adv_estimator.normalize_advantages` remains a boolean. With streaming
active, `true` normalizes advantages independently within each selected chunk;
`false` leaves them unnormalized. Full-batch execution continues to normalize
over the complete batch when enabled. Thus changing the chunk boundaries can
change normalized advantages. GAE returns used by the critic are unaffected by
this option. Gradient/loss normalization over the actual valid tokens or
sequences still occurs once for the full policy update.

Gradient preservation initially supports classic Megatron DDP, including expert
gradient buffers. Megatron FSDP and MXFP8 parameter gathering that shares gradient
storage are rejected. Generation colocation remains incompatible with streaming;
the recipe's `noncolocated` suffix describes generation placement, not a separate
policy/value GPU split. Existing in-order sampling and no-short-batch PPO guards
remain in effect.

## Verifying streaming

Inspect `train_pump: step ... chunk ...` and `closing on ... chunk(s)` records to
confirm that the run actually processed multiple chunks. Job completion alone
does not establish streaming: a ready queue can yield one complete batch.
