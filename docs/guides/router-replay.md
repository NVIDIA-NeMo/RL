# Router Replay

Router Replay, or R3, records MoE router choices made during rollout generation
and replays those choices in Megatron forward passes. This keeps each generated
token's expert assignment consistent across rollout, logprob, and training
stages. Without replay, two valid router implementations can choose different
experts for the same token, which introduces train-vs-rollout logprob mismatch
that is unrelated to the policy update.

Router Replay is disabled by default. It is not needed for dense models. In
the current NeMo RL integration, Router Replay is wired and tested for
Megatron MoE policy training with vLLM rollout generation. Other
inference/generation backends are not wired into this path and have not been
tested with Router Replay.

## Configuration

Set `policy.router_replay.enabled=true` in the training config:

```yaml
policy:
  router_replay:
    enabled: true
```

When Router Replay is enabled, NeMo RL configures vLLM rollout generation to
return routed expert indices by setting `enable_return_routed_experts=True` in
the vLLM kwargs. The generation payload is then carried through the normal
rollout and policy data path as the `routed_experts` field.

For models that also train MoE-based MTP heads, Router Replay skips MTP
routers by default. This keeps MTP routers on their native routing decisions
while replaying vLLM routes only in the decoder layers. Set
`NRL_ROUTER_REPLAY_EXCLUDE_MTP=0` only when intentionally debugging the legacy
behavior that replays MTP routers too.

An example recipe is available at:

```text
examples/configs/recipes/llm/grpo-qwen3-30ba3b-8n8g-megatron-cp2-r3.yaml
```

The native async TransferQueue path uses the SingleController entrypoint with:

```text
examples/configs/recipes/llm/grpo-qwen3-30ba3b-10n8g-megatron-cp2-r3-async-single-controller.yaml
```

## Multi-turn rollouts with token capture

This section applies when both `policy.router_replay.enabled=true` and
`token_capture.enabled=true`.

### Why turn boundaries need special handling

In a multi-turn rollout, each turn is a separate vLLM call. The last token a
call generates is sampled but never run back through the model in that call,
so vLLM returns only a placeholder route for it. The next turn's prompt
includes that token, and vLLM computes its real route during prefill.

NeMo RL uses the next turn's route to replace the placeholder, so every token
except the very last one in the trajectory is replayed with a real routing
decision.

### How it works

- When a follow-up (`token_in`) call is captured, it stages its usual
  per-token routes (`[delta_len, layers, topk]`) plus one extra route for the
  previous turn's last token: `routed_experts_boundary`, shape
  `[1, layers, topk]`. That token's position in the trajectory,
  `routed_experts_boundary_index`, is stored in the extras metadata. The
  extras digest covers both.
- During route assembly, this route replaces the placeholder in the assembled
  tensor. The previous turn's staged routes are not modified.
- Only the follow-up call on the selected path, as recorded in the route
  plan, does the replacement. Abandoned retries and the order in which
  responses arrive have no effect.
- The replacement only happens when both turns generated tokens.
- Token IDs, masks, logprobs, and delta lengths are unchanged.
- The last token of the final turn keeps its placeholder route, because no
  later call runs prefill on it.
- If the replacement route is all `-1`, that token falls back to Megatron's
  native router (see [Fallback for Missing Routes](#fallback-for-missing-routes)).

This works both when routes are assembled up front and when
`token_capture.defer_routed_experts_to_policy=true`.

If a boundary route is missing or fails verification, the existing failure
policy applies. When routes are assembled up front, the rollout is rejected.
With `defer_routed_experts_to_policy=true`, that rollout uses native routing.

### Upgrading and compatibility

- **Upgrade NeMo RL and Gym together.** With Router Replay enabled, the vLLM
  worker fails at startup if the installed Gym adapter does not support
  boundary routes. Token capture without Router Replay is not affected.
- **Route plans** are now written as schema v3. Schema v2 plans can still be
  read and simply have no boundary routes.
- **Previously staged data** can still be read. Follow-up turns without a
  boundary route keep the placeholder and log a warning
  ("... has no boundary route ...").
- **Native TQ checkpoints:** resuming with token capture and Router Replay
  enabled requires a checkpoint written by this version (metadata
  `routed_experts_boundary_schema_version: 1`). Older checkpoints, and those
  saved with Router Replay disabled, are rejected at startup because the
  staging schema cannot be migrated. Resume from a compatible checkpoint or
  start with fresh rollout/staging state.

## Validation

Router Replay validation covers two end-to-end questions:

1. whether rollout routes are carried through TransferQueue, packing, context
   parallel slicing, and Megatron replay without changing token identity;
2. whether matched R3-on runs reduce train-vs-rollout mismatch relative to
   matched R3-off controls.

### Validation and Trace Debugging

Router Replay can emit JSONL traces for a small number of training steps. This
is intended for correctness debugging, not long training runs.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `NRL_ROUTER_REPLAY_EXCLUDE_MTP` | `1` | Skip routers under MCore MTP layers. Set to `0` to include MTP routers in replay. |
| `NRL_ROUTER_REPLAY_VALIDATE` | `0` | Validate replay tensors before Megatron installs them, rejecting partially missing routes, duplicate top-k expert IDs, and out-of-range expert IDs. |
| `NRL_R3_TRACE` | `0` | Master switch for R3 JSONL trace emission. |
| `NRL_R3_TRACE_STEPS` | `1` | Number of training steps to trace. |
| `NRL_R3_TRACE_SAMPLES` | `2` | Number of samples with full tensor previews. |
| `NRL_R3_TRACE_DIR` | `logs/r3_trace` | Trace output directory. |
| `NRL_R3_TRACE_MICROBATCHES` | `2` | Number of microbatches to trace per stage. |
| `NRL_R3_TRACE_VERIFY_FORWARD` | `0` | Verifies replayed top-k indices against the installed replay tensor during forward. |

Example:

```bash
export NRL_R3_TRACE=1
export NRL_R3_TRACE_VERIFY_FORWARD=1
export NRL_R3_TRACE_STEPS=1
export NRL_R3_TRACE_SAMPLES=1
export NRL_R3_TRACE_MICROBATCHES=1
export NRL_R3_TRACE_DIR=/path/to/run/r3_trace
```

After the run, validate the emitted trace:

```bash
python tools/check_r3_trace.py /path/to/run/r3_trace \
  --require-forward-verify \
  --require-cp-identity
```

The checker verifies that:

- rollout payload samples include both `input_ids` and `routed_experts`;
- TransferQueue fetches match the rollout payload;
- context-parallel slicing preserves token identity for routed experts;
- Router Replay assignments are installed for prev-logprob and train stages;
- forward verification reports that replayed routes match the installed tensor.

### Effectiveness Check

1. Run matched R3-off controls to check that the PR does not regress existing
   packed-sequence and context-parallel Megatron training paths.
2. Run matched R3-on/R3-off pairs to measure whether Router Replay reduces
   train-vs-rollout mismatch under the intended rollout settings.

The main metrics to inspect are:

- `train/token_mult_prob_error`
- `train/js_divergence_error`

Validation report: <https://api.wandb.ai/links/nvidia-nemo-fw-public/lxoovk60>

## Other Notes

### Fallback for Missing Routes

In rare cases, vLLM can return fewer routed-expert entries than expected for a
sample. NeMo RL represents each missing token route with an all-`-1` sentinel.
Megatron then uses its normal router only for those missing token routes, while
all returned vLLM routes are still replayed exactly.

The fallback is intentionally route-local: it does not disable Router Replay for
the whole batch or sample.

When fallback is used, the vLLM worker emits a `R3 router replay fallback:` warning
to the run log naming the affected sample count and missing token-route count.
Fallback should normally be absent or rare; frequent warnings mean a meaningful
share of token routes used Megatron's normal router instead of replay.

The generation backend also computes
`r3/routed_experts_fallback_token_route_fraction`, but no training loop currently
forwards it to the metric logger, so do not rely on it in dashboards or gates.
