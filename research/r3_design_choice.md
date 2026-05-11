# R3 Design Choices for NeMo-RL + TQ

Date: 2026-05-08

Status: design aligned; first implementation smoke-passed on 2026-05-08.

This note summarizes the design choices for the first NeMo-RL R3 routing-replay PR. It is intentionally focused on review decisions, not a full implementation guide.

## Goal

Use routing decisions produced during vLLM rollout to replay MoE router choices in Megatron policy evaluation/training.

The first PR should support the NeMo-RL data-plane path:

```text
vLLM rollout -> NeMo-RL rollout data -> TQ jagged payload -> Megatron prev-logprob/train
```

It should not copy Peter's block-store design. Peter needed block-store/range-read support; this PR should use normal TQ tensor fields.

## Scope

In scope:

- vLLM rollout as producer.
- TQ/data-plane as transport.
- Megatron as consumer.
- CP-compatible token slicing by reusing Megatron/MCore token-aligned packing and CP slicing.
- Prev-logprob and current-policy train as replay consumers.

Out of scope for the first PR:

- Reference-policy replay.
- Virtual pipeline parallelism.
- Peter-style block-store/range-read replay.
- Optimizing routed-expert storage beyond jagged TQ tensors.

## Decision 1: Producer Payload

Use vLLM returned routed experts as the source of truth.

The rollout side should enable routed-expert return in vLLM and collect one routed-experts tensor per generated sample. The tensor should be normalized into a NeMo-RL batch field named `routed_experts`.

Chosen NeMo-RL pre-TQ shape:

```text
routed_experts: [B, S, num_moe_layers, topk]
dtype: int32
```

where `S == input_ids.shape[1]` for the padded batch.

The field is full-sequence aligned with `input_ids`. For a sample of sequence length `L`, NeMo-RL should only trust vLLM rows `[0:L-1)`, because those positions produce next-token logprobs. The final token row has no next-token target, so NeMo-RL fills it with a valid dummy top-k route. Padding rows use the same dummy route. The dummy route is currently `[0, 1, ..., topk - 1]` so dropless MoE dispatchers never see duplicate expert ids.

For terminal environment-observation messages that are appended after the last assistant message, there is no later vLLM forward to provide real routes for those observation tokens. These tokens are masked out of RL loss. When R3 routing data is already present in the message log, NeMo-RL fills such observation-message `routed_experts` rows with the same valid dummy route so message-log flattening stays token-aligned. In continuing multi-turn samples, the next vLLM forward overwrites prompt-message route slices with captured routes for the full prompt.

Why `L` instead of `L - 1`:

- TQ's current jagged write path uses `input_lengths` for all token-aligned tensor fields.
- Full-sequence alignment means `routed_experts[t]` corresponds to `input_ids[t]`.
- It avoids a special `routed_experts_lengths = input_lengths - 1` path.
- The final token row is carried only to keep the tensor shape token-aligned; replay validation should not depend on vLLM producing a meaningful route for that row.

Producer prefix-cache policy:

- Disable vLLM prefix caching when router replay is enabled.
- The current vLLM routed-experts return path reads captured routes by KV slot.
- Prefix-cached prompt tokens may not be forwarded again for a request, so their routes may not be freshly captured unless vLLM also has a route-aware prefix-cache/block-store path.
- A route-aware prefix-cache optimization is out of scope for this NeMo-RL PR.

## Decision 2: TQ Representation

Write `routed_experts` as a normal token-aligned tensor field and let it use
the same jagged TQ wire path as `input_ids`.

After Zhiyu's jagged update, `kv_first_write` calls
`maybe_pack_jagged(..., input_lengths)` for tensor fields shaped
`[B, max(input_lengths), ...]`. `routed_experts` intentionally matches that
shape contract:

```text
[B, S, num_moe_layers, topk]
```

so TQ strips rows to `input_lengths[i]` before the put, preserving the trailing
`[num_moe_layers, topk]` layout.

On fetch, NeMo-RL materializes TQ data as padded tensors before Megatron data
prep. So the consumer-side shape is:

```text
[B, S, num_moe_layers, topk]
```

When `layout="padded"` materializes jagged data, scalar padding can create
invalid repeated-zero top-k rows outside `input_lengths`. The packed Megatron
path only copies valid rows and fills packed padding with a valid dummy route;
the non-packed path also repairs padding rows to `[0, 1, ..., topk-1]` before
router replay.

## Decision 3: Consumers

Use replay for:

- prev-logprobs, also called old-policy logprobs;
- current-policy train.

Do not use replay for:

- reference-policy logprobs.

Rationale:

- Prev-logprobs and current-policy train should align with rollout routing for R3.
- Reference-policy logprobs are KL/reference evaluation. Replaying rollout-policy routing into the reference model changes the reference semantics.

Implementation implication:

- Do not add `routed_experts` unconditionally to global seed field constants.
- When router replay is enabled:
  - prev-logprob fetches `LP_SEED_FIELDS + ["routed_experts"]`;
  - train fetches `DP_SEED_FIELDS + ["routed_experts"]`;
  - reference logprob fetches only `LP_SEED_FIELDS`.

## Decision 4: CP Slicing

Do not invent a NeMo-RL CP topology for R3.

`routed_experts` is token-aligned, so it should be packed and CP-sliced with the same token-aligned path as `input_ids`.

For packed THD data, NeMo-RL should continue delegating CP token ownership to MCore's `get_thd_batch_on_this_cp_rank(...)`. This keeps R3 aligned with Megatron's CP topology, including packed token order.

## Decision 5: MCore Replay Contract

Local MCore source establishes this consumer contract:

- `moe_enable_routing_replay=True` makes each MoE `TopKRouter` create a `RouterReplay` instance.
- `RouterReplay.set_replay_data(...)` expects a list of per-router tensors.
- List length must equal the number of local `RouterReplay` instances.
- List order is local router instance order.
- Each list element must be shaped:

```text
[T_local, topk]
```

- Each list element is later used as `scores.gather(1, top_indices)`, where `scores` is `[T_local, num_experts]`.

Implementation implications:

- Store `int32` in TQ. This matches the vLLM routed-experts dtype observed in
  local smoke tests and is supported by NCCL broadcast; `int16` is rejected by
  this stack's NCCL process group.
- Cast to `torch.long` before calling MCore.
- Ensure token order matches local packed/CP-sliced `input_ids`.
- Do not assume MCore accepts a `moe_topk_routing_replay_indices` model-forward kwarg; local public MCore does not expose that API.

## Decision 6: Layer Mapping

Use explicit layer mapping. Do not rely on blind list order.

vLLM's routed-expert layer axis should be treated as a compressed MoE-layer ordinal:

```text
routed_experts[:, moe_ordinal, :]
```

This is not always the same as `decoder_layer_number - 1`. It is identity-like only for all-MoE models.

Megatron/MCore routers carry global decoder layer numbers via `router.layer_number`. The bridge should:

1. Walk local Megatron model modules.
2. Find MoE routers that have `router.router_replay`.
3. Read each router's `layer_number`.
4. Derive global MoE layer order from the model/config, for example from `moe_layer_freq` or actual MoE modules.
5. Map each global decoder layer number to vLLM's compressed MoE ordinal.
6. Build the MCore replay list in `RouterReplay.global_router_replay_instances` order.

Conceptually:

```python
per_layer = []
for replay_instance in RouterReplay.global_router_replay_instances:
    global_layer_number = replay_instance_to_global_layer[replay_instance]
    moe_ordinal = global_moe_layers.index(global_layer_number)
    per_layer.append(routed_experts_local[:, moe_ordinal, :].to(torch.long).contiguous())

RouterReplay.set_replay_data(per_layer)
```

Why this matters:

If the layer mapping is wrong, Megatron may silently replay layer N's experts into layer M. That can run without crashing while breaking R3 semantics.

## Decision 7: VPP Scope

Gate off virtual pipeline parallelism in the first PR.

Virtual pipeline parallelism can put multiple virtual chunks on the same rank. MCore's global router replay list is process-local, while a forward may operate chunk-by-chunk. Supporting that safely requires chunk-aware router selection similar to VERL's helper logic.

First PR behavior:

- support PP/TP/EP/CP without VPP;
- if R3 is enabled and `virtual_pipeline_model_parallel_size` is set, raise a clear config error;
- leave VPP support for a follow-up.

## Decision 8: Replay State Lifetime

Set replay state close to the Megatron model forward, not in data prep and not in loss post-processing.

Prev-logprob path:

```text
set replay data
set action = REPLAY_FORWARD
model(...)
clear replay data/action
```

Train path:

```text
set replay data
set action = REPLAY_FORWARD
model forward
set action = REPLAY_BACKWARD
backward / recompute
clear replay data/action
```

Reason:

- Setting replay in data prep risks stale replay state leaking into unrelated forwards.
- Setting replay in loss code is too late.
- Training may recompute MoE routers during backward; MCore has `REPLAY_BACKWARD` specifically for this queued replay case.

## Validation Plan

Required before PR is considered ready:

1. Unit test field gating:
   - prev-logprob fetch includes `routed_experts` only when replay is enabled;
   - train fetch includes `routed_experts` only when replay is enabled;
   - reference fetch never includes `routed_experts`.

2. Unit test token alignment:
   - synthetic `input_ids` and `routed_experts`;
   - pack and CP-slice both;
   - assert local routed-expert rows correspond to local token rows.

3. Unit/probe test MCore bridge:
   - fake or small local router/replay instances;
   - assert per-layer list shape `[T_local, topk]`;
   - assert dtype cast to `torch.long`;
   - assert explicit global-layer mapping is used.

4. Slurm smoke:
   - vLLM emits `routed_experts`;
   - TQ carries the field;
   - Megatron prev-logprob and train consume it;
   - reference logprob skips it;
   - run through both prev-logprob and train replay with CP enabled.

## Open Follow-Ups

- VPP-aware replay list selection.
- Whether to add a captured-batch equivalence test for stricter logprob matching.
- Longer R3 run after smoke passes.
- Potential future optimization that stores less than full `[B, S, layers, topk]` in TQ. That would likely require Peter-style token identity or an equivalent range-read map.
