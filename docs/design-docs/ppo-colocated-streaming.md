# Colocated PPO streaming

Policy and value share the training GPUs; generation uses separate GPUs.
Streaming processes completed rollout groups while the rest of the batch finishes,
with one policy update and full-batch critic training.

## Configuration

Set the readiness threshold below the number of prompts per step:

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

## Update lifecycle

1. For each chunk: value forward, policy/reference forward as needed, GAE, then
   policy backward. Policy weights and pre-update value predictions stay fixed.
2. Keep accumulated policy gradients on GPU across policy/value switches while
   offloading policy parameters and optimizer state. Normalize gradients by the
   full step's actual valid-token or sequence count, then perform one policy
   optimizer/scheduler update.
3. Publish/refit the policy before full-batch critic training. Every critic epoch
   uses the assembled original batch and frozen GAE returns.
4. Release consumed replay rows after both updates finish. Checkpoint exclusion
   covers the entire update; completed-step counters advance after critic training.

The controller prepares the value model before waiting for the next chunk and
reuses that residency across empty selections. Preparation can overlap generation
but may still lie on the critical path. Full-batch PPO also uses value-forward-first
ordering and early refit; when generation shares training GPUs, refit instead waits
until critic training finishes.

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
a model switch but also delays policy work; it does not guarantee a speedup.
Other algorithms and full-batch PPO retain their selection behavior.

## Advantage normalization and support limits

**Warning:** with `normalize_advantages: true`, streaming normalizes advantages
independently within each chunk, while full-batch execution normalizes over the
complete batch. Changing chunk boundaries can therefore change policy gradients.
Startup emits a warning for this configuration. Setting the boolean to `false`
leaves advantages unnormalized in either mode. Critic returns are unaffected;
gradient/loss normalization still uses the full update's valid count.

Streaming requires one policy epoch, in-order sampling, separate generation GPUs,
and classic Megatron DDP (including expert gradient buffers). Megatron FSDP and
MXFP8 parameter gathering that shares gradient storage are rejected during setup.
GPU memory must fit resident policy gradients alongside value inference.
Chunks and the full batch must satisfy both models' data-parallel alignment and,
without packing, static microbatch alignment. No samples are duplicated or dropped;
the existing no-short-batch PPO guard remains. The recipe's `noncolocated` suffix
refers to generation placement, not separate policy/value GPUs.

Verify actual chunking in `train_pump: step ... chunk ...` and
`closing on ... chunk(s)` logs: successful completion alone does not show that
more than one chunk was processed.
