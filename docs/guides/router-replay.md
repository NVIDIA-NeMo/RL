# Router Replay

Router Replay, or R3, records MoE router choices made during rollout generation
and replays those choices in Megatron forward passes. This keeps each generated
token's expert assignment consistent across rollout, logprob, and training
stages. Without replay, two valid router implementations can choose different
experts for the same token, which introduces train-vs-rollout logprob mismatch
that is unrelated to the policy update.

Router Replay is disabled by default. It is not needed for dense models. The
current NeMo RL integration supports Megatron MoE policy training with either
vLLM rollout generation or Megatron Inference (MInf) generation through the
SingleController token-capture path. Other inference/generation backends are
not wired into Router Replay.

## Configuration

Set `policy.router_replay.enabled=true` in the training config:

```yaml
policy:
  router_replay:
    enabled: true
```

### vLLM generation

When Router Replay is enabled with vLLM, NeMo RL configures rollout generation
to return routed expert indices by setting `enable_return_routed_experts=True`
in the vLLM kwargs. The generation payload is then carried through the normal
rollout and policy data path as the `routed_experts` field.

### Megatron Inference generation

MInf Router Replay is supported only by the SingleController NeMo-Gym path:
launch with `examples/run_grpo_single_controller.py` (`examples/run_grpo.py`
rejects this combination) on top of a config that already meets the
[Single-Controller requirements](single-controller.md), such as
`data_plane.enabled: true`. Then set:

```yaml
env:
  should_use_nemo_gym: true
token_capture:
  enabled: true
  defer_routed_experts_to_policy: false
policy:
  generation:
    backend: megatron
    mcore_generation_config:
      expose_http_server: true
      async_sched_mode: "legacy"
      pipeline_model_parallel_size: 1
```

Two MInf constraints are enforced at config validation. Megatron-Core's async
scheduler does not support routing replay, so `async_sched_mode` must be
`legacy` (the `grpo_math_1B.yaml` template sets `async`). MInf records routes
per pipeline stage and never gathers them across stages, so the generation
model must run with `pipeline_model_parallel_size: 1`; the generation layout
inherits the training layout unless overridden in `mcore_generation_config`.

MInf produces routes with shape `[T - 1, L, K]`. Its serving-side canonical
stager appends the terminal fallback row, delta-aligns the routes using Gym's
capture admission, and commits them to TransferQueue with the token IDs and
logprobs. Deferred route materialization in the policy worker is not supported
for MInf. See the [token-capture ledger design](../design-docs/token-capture-ledger.md#minf-router-replay)
for the component-level data flow and multi-turn boundary behavior.

For models that also train MoE-based MTP heads, Router Replay skips MTP
routers by default. This keeps MTP routers on their native routing decisions
while replaying rollout routes only in the decoder layers. Set
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
