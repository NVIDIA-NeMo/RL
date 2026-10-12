# Colocated PPO streaming

In async PPO setup policy and value share the training GPUs; generation uses
separate GPUs. Streaming runs value forward, policy/reference forward, GAE, and
policy backward on completed rollout chunks while the rest of the batch finishes.
After all chunks, one policy optimizer update precedes full-batch value (critic)
training; critic training is not streamed.

## Configuration

To enable streaming set the readiness threshold below the number of prompts per step:

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

The threshold is a minimum, not a fixed chunk size: the sampler can select more
ready groups, including a complete batch. Setting it equal to
`ppo.num_prompts_per_step` selects full-batch execution and permits multiple policy
epochs. Both modes train the critic on
`ppo.num_prompts_per_step * ppo.num_generations_per_prompt` samples per epoch.
A minimum above about two thirds of the batch cannot leave the tail floor, so
every step runs as one chunk even though streaming requirements still apply.
For a batch of 128, this occurs with a minimum of 86 through 127.

## Update lifecycle

1. For each chunk: value forward, policy/reference forward as needed, GAE, then
   policy backward. Neither model takes an optimizer step between chunks. Each
   sample's value prediction is computed with the unchanged critic and retained
   as the old value for all critic epochs.
2. Keep accumulated policy gradients on GPU across policy/value switches while
   offloading policy parameters and optimizer state. Normalize gradients by the
   full step's actual valid-token or sequence count, then perform one policy
   optimizer/scheduler update.
3. Refit the policy before full-batch critic training. Every critic epoch
   uses the assembled original batch and frozen GAE returns.
4. Release consumed replay rows after both updates finish. Checkpoint exclusion
   covers the entire update; completed-step counters advance after critic training.

The controller prepares the value model before waiting for the next chunk and
reuses that residency across empty selections. Preparation can overlap generation
but may still lie on the critical path. Full-batch PPO also uses value-forward-first
ordering and early refit. If full-batch PPO places generation on the same GPUs as
policy and value, generation stays paused during critic training and refit follows
critic training.

During critic warmup, policy backward, update and refit are skipped. Masked chunks
contribute no policy gradients; an entirely masked step fails before either update.

## Tail-aware selection

The sampler caps selection at the batch remainder, aligns it, then defers a
nonfinal chunk that would leave fewer than
`ceil(async_rl.min_groups_for_streaming_train / 2)` prompt groups. Deferred groups
remain ready and unclaimed. An aligned final remainder can flush below the minimum;
leaving exactly the tail floor is allowed.

For batch 128 and minimum 32, selecting 88 then 32 would leave 8. The gate waits
for the remaining 40, producing `88 + 40` instead of `88 + 32 + 8`. This can avoid
a model switch: it is a heuristic that can be intuitively better, at the cost of
delaying policy work.

The heuristic defers the largest aligned selection rather than trimming it.
With batch 128, minimum 32, and group alignment 1, 112 ready groups can run, but
113–127 ready groups wait for all 128. A nearly complete batch can therefore run
as one chunk.
Other algorithms and full-batch PPO retain their selection behavior.

## Advantage normalization and support limits

**Warning:** with `normalize_advantages: true`, streaming normalizes advantages
independently within each chunk, while full-batch execution normalizes over the
complete batch. Changing chunk boundaries can therefore change policy gradients.
Critic returns are unaffected; gradient/loss normalization still uses the full
update's valid count.

Streaming requires `ppo_epochs=1`, an in-order or ready-first sampler, separate generation GPUs,
and classic Megatron DDP (including expert gradient buffers). Setup rejects
Megatron FSDP and MXFP8 parameters with
`policy.megatron_cfg.optimizer.use_distributed_optimizer: true`, regardless of
`overlap_param_gather`: MXFP8 shares parameter and gradient storage in that
configuration, so parameter offload cannot preserve resident gradients.
Keeping policy gradients on GPU avoids their CPU transfers between chunks but
leaves less GPU memory for the value-inference batch.
Chunks and the full batch must satisfy both models' data-parallel alignment and,
without packing, static microbatch alignment. No samples are duplicated or dropped;
the existing no-short-batch PPO guard remains.

Verify actual chunking in `train_pump: step ... chunk ...` and
`closing on ... chunk(s)` logs: successful completion alone does not show that
more than one chunk was processed.
