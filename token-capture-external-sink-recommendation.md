# PR 2278 Review: External Token Staging with `LineageStore`

## Recommendation

Use the existing `LineageStore` directly for parent resolution and cumulative
token tracking. Supporting an external staging sink does not require a separate
`RolloutCaptureGate` or `GateStateStore`.

The current gate does not provide a second lineage algorithm. It resolves the
parent through `LineageStore`, duplicates the resulting call state in the gate,
and publishes successful commits back to `LineageStore`. This creates two
sources of truth for the same rollout ancestry.

The recommended initial implementation is framework-authoritative:

- Gym owns request correlation and lineage resolution.
- The inference worker owns durable staging through `StagingSink`.
- The external framework owns rollout completeness, terminal selection,
  finalization, and cleanup.
- The finalizer verifies the staged records with `verify_and_linearize()` before
  publishing a training row.

## Why `LineageStore` Is Sufficient for Lineage

`LineageStore` already provides the two operations needed by the serving path:

1. `resolve(rollout_id, request_items)` identifies the verified parent and
   returns its cumulative token IDs.
2. `record(...)` publishes a completed call so a later request, including one
   handled by another model-server worker, can continue it.

`FileLineageStore` supplies cross-process read-after-write visibility for
multi-worker model servers. The gate currently delegates lineage publication
back to this interface after it receives worker commit coordinates.

Removing the gate does not change the lineage algorithm. It removes the
duplicated lifecycle database around it.

## Simplified External-Sink Flow

1. Gym assigns `rollout_id` and `model_call_id` in the request-scoped capture
   context.
2. Gym calls `LineageStore.resolve()` before dispatch.
3. Gym sends the call identity, parent identity, and required prefix to the
   inference worker.
4. The worker creates a `StagedCallRecord` and makes it durable through the
   configured `StagingSink` before acknowledging success.
5. After successful staging, Gym computes the cumulative token sequence and
   calls `LineageStore.record()`.
6. Gym removes token IDs, log probabilities, routed-expert values, and internal
   coordinates before returning the model response to the agent.
7. The external framework reads staged records through `StagingSource`, selects
   the terminal ancestry, and calls `verify_and_linearize()`.

The ordering requirement is important: a call must not become a lineage parent
until its external staged record is durable.

## What the Gate Adds

The gate adds lifecycle and authorization behavior, not lineage behavior:

- rollout registration and per-rollout data capabilities;
- admitted-but-not-committed call tracking;
- owner-bound and operation-bound retries;
- logical-request-to-model-call indexing;
- seal and failure transitions;
- receipt construction;
- TTL expiry, tombstones, cleanup queues, and metrics.

These behaviors may be useful, but they should not be treated as prerequisites
for an external sink. If NeMo RL already owns rollout lifecycle and finalization,
duplicating those responsibilities inside Gym increases coupling without
improving lineage resolution.

## Important Caveat: Root Versus Unresolved Lineage

The existing lineage result uses `None` for several different cases:

- the request is a genuine first call with no assistant-authored history;
- the assistant fingerprint is ambiguous;
- the request context digest no longer matches;
- a prior call cannot be found.

The local capture builder can sometimes recover a missing parent through strict
token-prefix matching. The external finalizer follows explicit parent links and
cannot rely on that recovery.

The external path should therefore distinguish:

```text
ROOT         -> admit as a text-mode root
MATCH        -> admit as a token-in child
UNRESOLVED   -> reject or poison token capture
```

Silently converting `UNRESOLVED` into a new root can turn earlier
policy-generated tokens into prompt tokens with mask zero and produce an
incorrect training trajectory.

This is a lineage API correction and remains necessary whether or not a gate is
present.

## Responsibilities Outside `LineageStore`

`LineageStore` as currently defined does not store:

- external staging keys;
- weight versions and complete commit manifests;
- admitted calls that never committed;
- terminal model-call identity;
- failed or sealed rollout state.

For the framework-authoritative design, those responsibilities remain with the
external sink, source, controller, and finalizer. The staged record already
contains the parent link, lengths, weight version, and integrity digests needed
for final verification.

If Gym must eventually own fail-closed receipt construction, extend the lineage
store into a per-rollout append-only capture ledger. Do not introduce a separate
global gate state file. A ledger could add `CallStarted`, `CallCommitted`,
`CallFailed`, and `RolloutSealed` events alongside the existing lineage data.

## External Sink Contract Recommendations

Keep the public integration surface small:

- `StagingSink.stage(record) -> StageResult`
- `StagingSource.fetch(staging_keys) -> snapshots`
- versioned `StagedCallRecord`, `CallRecord`, and receipt types where needed;
- digest and conformance helpers;
- `verify_and_linearize()` as the final trust boundary;
- engine-specific capture adapters behind the generic capture protocol.

For cleanup after a lost worker acknowledgment, prefer a deterministic staging
key derived from `rollout_id` and `model_call_id`, or allocate the key before
dispatch. An opaque key returned only after staging cannot be recovered when the
acknowledgment is lost.

Authorization, if required for untrusted agent traffic, can be implemented as a
small stateless signed capability check. It does not require a complete rollout
state machine.

## Suggested PR Scope

The core Gym PR should contain:

- staging protocols and versioned records;
- integrity and conformance helpers;
- the worker capture hook;
- the generic lineage integration;
- the external finalization verifier;
- a vLLM adapter behind the engine-neutral interface.

Move framework- or harness-specific behavior into companion changes:

- NeMo RL TransferQueue sink/source and finalizer wiring;
- SWE-specific credential-file routing and terminal discovery;
- rollout-controller policy, cleanup, and metrics;
- unrelated packaging exclusions or runtime-directory cleanup.

## Decision Summary

Both behaviors are required for a fully fail-closed system, but both current
components are not:

| Concern | Recommended owner |
| --- | --- |
| Parent resolution and exact prefix | Existing `LineageStore` |
| Durable token/logprob/route staging | External `StagingSink` |
| Snapshot retrieval | External `StagingSource` |
| Terminal selection and rollout completeness | External framework |
| Final integrity verification | `verify_and_linearize()` |
| Separate Gym gate database | Remove from the initial external-sink path |

The smallest useful change is therefore to keep `LineageStore` and the staging
contract, remove the independent gate state system, and leave lifecycle policy
with the framework that already owns the rollout.

## Reviewed Change

This recommendation is based on the five-commit delta in
[NVIDIA-NeMo/Gym PR 2278](https://github.com/NVIDIA-NeMo/Gym/pull/2278), from
base `d2123272` through head `1b342084`.
