# Router Replay

Router Replay, or R3, records MoE router choices made during rollout generation
and replays those choices in the policy's forward passes (Megatron or automodel). This keeps each generated
token's expert assignment consistent across rollout, logprob, and training
stages. Without replay, two valid router implementations can choose different
experts for the same token, which introduces train-vs-rollout logprob mismatch
that is unrelated to the policy update.

Router Replay is disabled by default. It is not needed for dense models. In
the current NeMo RL integration, Router Replay is wired for MoE policy training
on the Megatron and automodel (DTensor v2) backends with vLLM rollout
generation. Other inference/generation backends are not wired into this path
and have not been tested with Router Replay.

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

### Automodel (DTensor v2) Policy Backend

With the automodel backend, enabling Router Replay builds every Automodel MoE gate
with a replay hook (`moe_overrides.enable_routing_replay`) and replays the rollout
routes in the `get_logprobs` forward and in the training forward and backward.
Reference-policy logprobs never replay rollout routes. Current requirements:

- The model must use an Automodel MoE implementation; the HF / `force_hf` path has
  no replay hooks and fails at worker setup.
- Sequence packing, `dtensor_cfg.context_parallel_size > 1`, and
  `dtensor_cfg.sequence_parallel` are not supported yet and fail at setup, as does
  `token_capture.defer_routed_experts_to_policy`.
- Set `policy.generation.vllm_cfg.enable_prefix_caching=false` so vLLM computes, and
  returns routes for, every prompt token instead of reusing cached prefixes.
- vLLM may emit one route slot per MoE layer or one per decoder layer (hybrid
  models such as NemotronH); both layouts are mapped to the gates automatically.
- Batch padding is never replayed. Rows other than all-`-1` missing routes that are
  partially negative, out of range, or repeat an expert are rejected as corrupt.

Example: `examples/configs/recipes/llm/dapo-nanov3.5-30BA3B-4n8g-automodel-r3.yaml`
(Nemotron-3.5 Lightning) keeps `train/token_mult_prob_error` near 1.010 for all steps
and halves `train/gen_kl_error` (0.00037 vs 0.00086). Without Router Replay
(`dapo-nanov3.5-30BA3B-4n8g-automodel.yaml`), 0-2 tokens per step disagreed by 10-31
nats between vLLM and training, spiking the mean to 1e6-3e9 on about 20% of steps.

## Validation

Router Replay validation covers two end-to-end questions:

1. whether rollout routes are carried through TransferQueue, packing, context
   parallel slicing, and Megatron replay without changing token identity;
2. whether matched R3-on runs reduce train-vs-rollout mismatch relative to
   matched R3-off controls.

### Validation and Trace Debugging

Router Replay on the Megatron policy can emit JSONL traces for a small number of
training steps. This is intended for correctness debugging, not long training runs.
The variables below apply to the Megatron policy only; the automodel policy always
excludes MTP routers, always validates routes, and emits no traces.

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
The policy (Megatron or automodel) then uses its normal router only for those
missing token routes, while all returned vLLM routes are still replayed exactly.

The fallback is intentionally route-local: it does not disable Router Replay for
the whole batch or sample.

When fallback is used, the vLLM worker emits a `R3 router replay fallback:` warning
to the run log naming the affected sample count and missing token-route count.
Fallback should normally be absent or rare; frequent warnings mean a meaningful
share of token routes used the policy's normal router instead of replay.

The generation backend also computes
`r3/routed_experts_fallback_token_route_fraction`, but no training loop currently
forwards it to the metric logger, so do not rely on it in dashboards or gates.
