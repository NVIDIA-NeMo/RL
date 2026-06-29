# Refit Debug Logging Design

## Goal

Add opt-in diagnostics that identify where a policy weight first diverges during
refit: policy source state, transfer payload, vLLM reconstruction, or vLLM model
loading. The 5-layer CP smoke test enables the diagnostics with
`NRL_REFIT_DEBUG=1`.

The diagnostics must not change refit values, tensor order, synchronization, or
the normal behavior when the environment variable is unset.

## Scope

The implementation covers both refit transports:

- colocated IPC/ZMQ, used by `exp/grpo-m3-5layers.yaml`;
- non-colocated collective broadcast, so the same evidence can be collected
  from the original 32-node recipe.

It adds instrumentation to the DTensor policy producer and the vLLM consumer,
plus small unit tests and the smoke-test configuration change.

## Activation and Log Format

Diagnostics are enabled only when `NRL_REFIT_DEBUG=1`. The smoke-test YAML sets
this variable in both the policy DTensor worker environment and the generation
vLLM worker environment.

All messages use a stable `[REFIT_DEBUG]` prefix and identify the refit phase,
worker/rank, parameter name, shape, dtype, device, and compact fingerprint. No
full tensors are printed.

A shared helper centralizes:

- environment-variable parsing;
- representative-parameter selection;
- bounded tensor fingerprints;
- dtype/count/byte summaries.

Fingerprints use a deterministic, bounded sample so debug logging does not copy
an entire large tensor to CPU. They include finite/non-finite counts and small
numeric statistics sufficient to compare adjacent boundaries.

## Representative Parameters

The selector samples parameters that exercise MiniMax-M3's important mapping
and dtype paths:

- token embedding and language-model head;
- one attention QKV/indexer parameter;
- one router gate weight;
- `e_score_correction_bias`;
- one expert parameter.

The selector is deterministic and limits repeated matches, keeping log volume
bounded regardless of model depth.

## Instrumentation Boundaries

### Policy producer

At refit metadata preparation, log a summary of parameter count, dtype count,
and bytes, plus for selected parameters:

- source dtype and shape;
- resolved transfer dtype;
- transfer metadata dtype.

Immediately before yielding an IPC payload or broadcasting a collective
payload, log the selected tensor's actual dtype and fingerprint. This boundary
must inspect the tensor that will really be sent, after any flattening,
reshaping, or dtype conversion.

Policy-side detailed messages are emitted only by the source rank for a payload;
summaries state the relevant mesh rank so a missing producer can be identified.

### vLLM consumer

Immediately after IPC reconstruction or collective reception, but before
`load_weights`, log the selected incoming tensor's dtype and fingerprint.

Capture the return value of `load_weights` and log a per-refit load summary.
Because vLLM can map or pack incoming Hugging Face names into different model
parameter names, incoming and returned names are reported separately; a raw
one-to-one count difference is not treated as an error.

After loading, inspect resolvable representative destination parameters and log
their destination dtype and fingerprint. Where name packing makes an exact
source-to-destination comparison invalid, the log explicitly labels the entry
as mapped/packed rather than claiming a mismatch.

vLLM details include the tensor-parallel rank because each rank holds a local
shard.

## Diagnostic Interpretation

The resulting boundary sequence distinguishes these failure classes:

- source is fp32 but transfer metadata/payload is bf16: policy transfer-dtype
  regression, matching the previous refit bug family;
- policy payload and vLLM incoming fingerprints differ: IPC/collective transport
  or reconstruction bug;
- vLLM incoming is correct but the destination is missing or inconsistent:
  vLLM name mapping, packing, or model loading bug;
- all refit boundaries agree but generation KL remains high: refit is unlikely
  to be the cause, so the next investigation should compare CP and non-CP
  forward/logprob behavior.

In the expected mixed-dtype case, `e_score_correction_bias` remains fp32 through
transport, while a bf16 policy gate may legitimately load into an fp32 vLLM
destination. Logs therefore record both source and destination dtype instead of
requiring them to be equal.

## Error Handling and Overhead

Debug helpers do not add collectives or alter generator consumption. A failure
to compute a fingerprint produces a diagnostic marker and does not mask the
underlying refit operation. When disabled, the path performs only the environment
check and skips tensor sampling.

## Expected Files

- `nemo_rl/utils/refit_debug.py`: shared gating, selection, fingerprint, and
  summary helpers;
- `nemo_rl/models/policy/dtensor_policy_worker_v2.py`: producer metadata and
  payload boundaries;
- `nemo_rl/models/generation/vllm/vllm_backend.py`: receive, load, and
  post-load boundaries;
- `exp/grpo-m3-5layers.yaml`: enable diagnostics for the smoke test;
- focused unit tests for gating, selection, bounded fingerprints, dtype
  reporting, and non-mutation.

## Verification

Run focused unit tests for the helper and both integration points, then validate
the edited YAML parses. Inspect one short local/synthetic refit trace to confirm:

- disabled logging is silent;
- enabled logging contains all four boundaries;
- selected fingerprints are stable across an unchanged payload;
- logging does not change tensors or loaded-weight results;
- no full parameter value dump is emitted.
