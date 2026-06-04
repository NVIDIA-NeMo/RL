# Router Replay

Router Replay, or R3, records MoE router choices made by the rollout generation
backend and replays those choices in Megatron forward passes. This is useful
when training an MoE policy with vLLM rollout generation and Megatron policy
training: without replay, the vLLM rollout path and Megatron logprob/train path
can choose different experts for the same token, which introduces
train-vs-rollout logprob mismatch.

## When to Use Router Replay

Enable Router Replay when all of the following are true:

- rollout generation uses vLLM;
- the policy backend is Megatron;
- the model is an MoE model;
- vLLM can return routed experts for the generated sequences.

Router Replay is disabled by default. It is not needed for dense models, and it
is not currently supported for non-vLLM generation backends.

## Configuration

Set `policy.router_replay.enabled=true` in the training config:

```yaml
policy:
  router_replay:
    enabled: true
```

When Router Replay is enabled, NeMo RL configures vLLM to return routed expert
indices by setting `enable_return_routed_experts=True` in the vLLM kwargs. The
generation payload is then carried through the normal rollout and policy data
path as the `routed_experts` field.

An example recipe is available at:

```text
examples/configs/recipes/llm/grpo-qwen3-30ba3b-8n8g-megatron-cp2-r3.yaml
```

## Fallback for Missing Routes

In rare cases, vLLM can return fewer routed-expert entries than expected for a
sample. NeMo RL represents each missing token route with an all-`-1` sentinel.
Megatron then uses its normal router only for those missing token routes, while
all returned vLLM routes are still replayed exactly.

The fallback is intentionally route-local: it does not disable Router Replay for
the whole batch or sample.

To check a vLLM build directly, run:

```bash
uv run --extra vllm python tools/model_diagnostics/6.vllm_routed_experts_completeness.py \
  Qwen/Qwen3-30B-A3B \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --llm-kwarg moe_backend=triton
```

The following metrics are logged when generation returns Router Replay fallback
stats:

```text
r3/routed_experts_fallback_token_routes
r3/routed_experts_expected_token_routes
r3/routed_experts_actual_token_routes
r3/routed_experts_fallback_token_route_fraction
r3/routed_experts_fallback_samples
```

## Runtime Validation

Set `NRL_ROUTER_REPLAY_VALIDATE=1` to validate replay tensors before they are
installed on Megatron `RouterReplay` instances. This is useful when debugging
routing mismatches or suspected data corruption.

The validation checks:

- partial sentinel corruption: fallback entries must be all `-1`, not partially
  negative;
- duplicate experts: a token's top-k route must not contain the same expert id
  twice;
- expert range: expert ids must be in `[0, num_moe_experts)`.

This validation is disabled by default because it runs on every replay tensor.

## Trace Debugging

Router Replay can emit JSONL traces for a small number of training steps. This
is intended for correctness debugging, not long training runs.

| Environment variable | Default | Meaning |
| --- | --- | --- |
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
