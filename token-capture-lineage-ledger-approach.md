# Token Capture Lineage Ledger — Approach Plan

This plan replaces the `RolloutCaptureGate` / `GateStateStore` pair in NeMo Gym's
token-capture path with a small extension to the existing `LineageStore`, turning it
into a per-rollout append-only capture ledger. The external staging contract
(`StagingSink` / `StagingSource`), the worker capture path, and the
`verify_and_linearize()` trust boundary are unchanged.

**Scope note:** when this change lands, the gate is **removed completely from the Gym
codebase** — `gate.py`, `gate_store.py`, the gate control routes, gate configuration,
and gate tests are deleted in the same change. The gate is not retained behind a
config flag and no backward-compatibility path is kept. Both components ship in the
same PR (NVIDIA-NeMo/Gym PR 2278), so there are no external deployments to migrate.

## Motivation

The gate does not provide a second lineage algorithm. Parent resolution runs upstream
through `LineageStore.resolve()`; the gate cross-checks that result against its own
copy of the call state and republishes successful commits back to
`LineageStore.record()`. This creates two sources of truth for the same ancestry:
each call's cumulative token IDs are stored both in `GateCallState` and in the
lineage JSONL.

The gate's state store is also a scalability liability. `FileGateStateStore`
serializes the **entire global gate state** — every live rollout's cumulative token
arrays and stored request items — and atomically rewrites it under one exclusive
file lock. `commit_coords()` performs three such transactions per model call. For
long-context rollouts this is megabytes of fsynced rewrite per call, serialized
across all serving workers.

What the gate legitimately provides — admission, retry idempotency, rollout
completeness, terminal selection, cleanup — is either a pure function of the lineage
result or belongs to the framework that already owns the rollout. This design moves
each responsibility to its natural owner and deletes the redundant state machine.

## Design

### 1. `LineageStore` becomes the capture ledger

`FileLineageStore` already writes one append-only JSONL row per committed call
(`model_call_id`, fingerprint, context digest, cumulative token IDs, digest) with
per-rollout locked appends and cross-worker read-after-write visibility. Three
additions make it the single record of rollout capture state:

1. **`record(...)` gains the token-free `CallRecord` fields.** The commit hook
   already holds the worker's `CommitCoords`, so each row additionally stores
   `parent_call_id`, `staging_key`, `weight_version`, `prev_len` / `delta_len` /
   `cum_len`, `extras_digest`, `mode`, and `logical_request_id` (the request header
   when present, else the response id — the same binding the gate's commit uses
   today). Additive change; existing fields are untouched.
2. **`record_failure(rollout_id, model_call_id, reason)`** appends a failure row when
   a call's coordinates come back `capture_failed` or the request dies after
   admission. Failure rows carry no fingerprint, so `resolve()` never returns them
   as parents — no filtering logic is needed.
3. **`manifest(rollout_id) -> list[CallRecord]`** reads the rows back token-free
   (cumulative token IDs are stripped; token arrays stay off Gym's HTTP surface).

The only duplication that remains between ledger rows and staged records (digest,
lengths, parent link) is intentional cross-attestation — it is exactly what
`verify_and_linearize()` checks one against the other.

### 2. Gate-free admission in `resolve_parent()`

When external staging is enabled, the model server builds the `CaptureAdmission`
directly from the lineage result. Admission is a pure function; no shared state is
required. The lineage outcome is a strict tri-state:

| Lineage outcome | Admission |
| --- | --- |
| `ROOT` — empty assistant fingerprint, or unmatched fingerprint on a rollout with no ledger rows (seeded assistant history in the task prompt) | `text` mode, no parent |
| `MATCH` — unique fingerprint match with verified context digest | `token_in` mode, `required_prefix_token_ids` = parent's cumulative tokens |
| `UNRESOLVED` — non-empty fingerprint with no match, ambiguity, or digest mismatch | no admission; `record_failure()` poisons the call |

`UNRESOLVED` is never silently converted into a new root: doing so would turn earlier
policy-generated tokens into mask-zero prompt tokens and corrupt the training row.
The seeded-history carve-out (unmatched fingerprint on an empty rollout is a `ROOT`)
requires a cheap "has any rows" check on the ledger.

### 3. Gate-free commit

Where the model server today hands `ng_commit_coords` to `gate.commit_coords()`,
it instead:

- on `disposition == "staged"`: computes
  `cumulative = context.parent_tokens + coords.token_ids_delta` and calls
  `lineage_store.record(...)` with the extended row. This is the lineage-publication
  block currently inside the gate, relocated — minus the state machine around it.
- on `disposition == "capture_failed"`, missing coordinates, or a request error after
  admission: calls `record_failure(...)`.

The ordering invariant the external sink requires — *a call must not become a
lineage parent until its staged record is durable* — holds structurally: the worker
stages through `StagingSink.stage()` before acknowledging, coordinates exist only
after the bytes are durable, and the ledger row (which is what makes a call
resolvable as a parent) is written only after the coordinates arrive.

### 4. One read-only control route

The register / seal / fail control routes are deleted (see removal section). They
are replaced by a single stateless read:

```
GET /training-token-capture/rollouts/{rollout_id}/manifest
```

which returns `manifest(rollout_id)`. The framework must not read the ledger JSONL
files directly — the route keeps `LineageStore` implementations swappable.

### 5. Retry handling (fail-closed now, idempotent as a follow-up)

For the initial change, `model_call_id` stays a per-request `uuid4` minted by the
capture middleware, and a ledger row's `logical_request_id` field is filled the way
the gate's commit binding fills it today: the client header when present, else the
vLLM response id. A retried HTTP request (Gym's `ServerClient` retries with backoff)
therefore lands as a *distinct* call. The ledger handles this fail-closed rather
than idempotently:

- a retry whose first attempt never committed leaves a failure row (the request
  died after admission) → the rollout is poisoned;
- a retry whose first attempt committed becomes a sibling row. If the regenerated
  text differs, later calls resolve uniquely to the survivor and the losing sibling
  is a dead branch in the manifest (never on the terminal chain); if the text is
  identical, resolution is ambiguous → `UNRESOLVED` → poisoned. Terminal selection
  still works because the agent reports the response id of the response it actually
  kept, which matches exactly one row's `logical_request_id`.

No retry outcome is silently wrong — the current gate's worst case (ambiguous
resolution silently falling back to a text-mode root) is closed by the `UNRESOLVED`
rule. What is deferred is *recovery*: making identical retries collapse into the
same row instead of poisoning. That follow-up is two touch points — the harness
mints a per-call `x-nemo-gym-logical-request-id` (reused verbatim on retry), and the
middleware derives `model_call_id` deterministically from it — after which
`record()`'s existing semantics absorb retries: an identical payload is a no-op, a
conflicting payload raises and the call is poisoned, and staging keys
(`{rollout_id}/{model_call_id}`) overwrite/no-op rather than orphaning a record.

### 6. Framework-owned receipt, lifecycle, and cleanup

NeMo RL (the rollout owner) assembles the `RolloutReceipt` locally at rollout end:

- `manifest` = the fetched `CallRecord` list, deduped by `model_call_id`;
- `terminal_model_call_id` = the row whose `logical_request_id` matches the
  rollout's terminal logical request;
- `capture_poisoned` = any failure row present, or no row for the terminal request.

`verify_and_linearize(receipt, snapshots)` runs unchanged — it consumes a plain
value and does not care who built it. Cleanup is framework-driven: on publish or on
rollout failure/abandonment, NeMo RL prefix-clears `{rollout_id}/` in the staging
backend and drops the ledger file. No registration TTLs, tombstones, or
cleanup-manifest queues.

## Complete gate removal (no backward compatibility)

The same change that introduces the ledger **deletes the gate entirely** from Gym.
Nothing is kept behind a flag, deprecated, or retained for compatibility:

- `nemo_gym/token_id_capture/gate.py` — deleted.
- `nemo_gym/token_id_capture/gate_store.py` (including `FileGateStateStore`,
  `InMemoryGateStateStore`, `SharedGateState`, `GateRolloutState`, `GateCallState`,
  `GateTombstone`, `CleanupManifest`, `RolloutRegistration`) — deleted.
  `CallRecord` and `RolloutReceipt` survive: they live in `staging/records.py` and
  are consumed by `verify_and_linearize()`.
- Control routes `PUT /rollouts/{id}`, `POST /rollouts/{id}/seal`,
  `POST /rollouts/{id}/fail`, and the gate metrics route — deleted, replaced by the
  single manifest route.
- `token_id_capture.gate.*` configuration (`enabled`, `state_store_path`,
  `registration_ttl_s`, `tombstone_ttl_s`) — deleted, not deprecated. A config that
  still sets these keys fails validation loudly rather than silently ignoring them.
- Gate wiring in `base_responses_api_model.py` (`RolloutCaptureGate` construction,
  `_fail_uncommitted_gate_call`, `_reject_gate_streaming`, capability header
  handling) — deleted. Streaming rejection is retained where external capture
  requires it, keyed off the capture context rather than the gate.
- `CaptureContext.staging_gate` and `data_capability` fields — deleted.
- Gate unit tests (`test_token_capture_gate.py`,
  `test_token_capture_gate_multiworker.py`) — deleted; their invariants that still
  apply (admission tri-state, commit ordering, retry idempotency, poisoning) are
  re-expressed as ledger and admission tests.

Rationale for not keeping a compatibility path: the gate ships in the same unmerged
PR as the external sink, so there is no deployed consumer of the gate API. Retaining
it would preserve the dual-source-of-truth problem this design exists to remove and
double the surface that must be tested.

Data-capability authorization is dropped with the gate. The NeMo RL deployment is a
trusted, framework-owned serving path. If untrusted agent traffic later needs
per-rollout authorization, a stateless signed capability check can be added without
any state store.

## NeMo RL companion changes

- `nemo_rl/environments/nemo_gym.py`: remove the register / seal / fail control-route
  calls and capability plumbing; fetch the manifest at rollout end; keep configuring
  `FileLineageStore` for the policy model server.
- `nemo_rl/experience/blackbox_finalizer.py`: build the `RolloutReceipt` from the
  fetched manifest as described above; verification and linearization are unchanged.
- Rollout failure paths (`rollout_manager.py`): prefix-clear staged rows for the
  rollout instead of draining gate cleanup manifests.
- `TQTokenSink` / `TQTokenSource` and the vLLM worker capture path
  (`vllm_worker_async.py`): unchanged.

## Example walkthrough

A SWE-agent rollout `r42_g0` with three model calls, served by a Gym model server
with the ledger and a vLLM worker staging into TransferQueue.

**Call 1 (root).** The agent server sends the task prompt; the server mints
`model_call_id=c1`, and commit binds `logical_request_id=lr-1` (the response id)
onto the ledger row. The request has no
assistant-authored turns → `ROOT` → `text` admission. The worker generates
(prompt 700 tokens, generation 200), stages
`StagedCallRecord(staging_key="r42_g0/c1", prev_len=0, delta_len=900, cum_len=900,
weight_version=17, ...)` durably into TransferQueue, then returns token-free
coordinates. The server appends ledger row 1 (lineage fields + `CallRecord` fields +
`lr-1`). The agent receives text only.

**Call 2 (continuation).** The agent appends a 150-token tool result and sends
`lr-2`. The fingerprint uniquely matches row 1 and the context digest verifies →
`MATCH` → `token_in` admission with the 900-token required prefix. The worker
asserts the prompt begins with exactly that prefix, generates 120 tokens, stages
`r42_g0/c2` (`prev_len=900`, `delta_len=270` — 150 carry tokens mask 0 + 120
generated mask 1, `cum_len=1170`). Ledger row 2 records `parent_call_id=c1`.

**Call 3 (terminal).** Same shape via `lr-3` → `c3`, `cum_len=1290`, ledger row 3.
The rollout result returns to NeMo RL with the reward and
`terminal_logical_request_id=lr-3`.

**Finalization.**
1. `GET .../rollouts/r42_g0/manifest` → three `CallRecord`s.
2. Build the receipt: terminal row is `lr-3` → `terminal_model_call_id=c3`; no
   failure rows → not poisoned.
3. `StagingSource.fetch(["r42_g0/c1", "r42_g0/c2", "r42_g0/c3"])`, then
   `verify_and_linearize(receipt, snapshots)`: digest recomputation, parent-graph
   walk `c3 → c2 → c1 → root`, length chaining `900 + 270 = 1170`,
   `1170 + 120 = 1290`, mask-order checks. Output: one contiguous 1290-token
   training row, mask 1 over the 320 policy tokens.
4. Publish the row with the reward; prefix-clear `r42_g0/*` from TransferQueue and
   drop the ledger file.

## Failure semantics

- **Capture fails mid-rollout** (e.g. call 2): the model call still succeeds for the
  agent, but a failure row is written instead of a lineage row. Call 3 then misses
  resolution with a non-empty assistant history → `UNRESOLVED` → failure row.
  Finalization sees failure rows → poisoned → masked placeholder row (the group
  still publishes exactly N rows); staging is prefix-cleared.
- **Terminal response lost, harness retries:** the retry is admitted as a new call
  (uuid identity) and becomes a sibling of the lost attempt. The harness keeps the
  retry's response, so its response id is reported as the terminal logical request
  and receipt assembly selects the retry's row; the lost attempt is a dead branch
  that never joins the terminal chain and is prefix-cleared with the rollout. Two
  versions of the same call can never *both* train, and an ambiguous mid-rollout
  sibling (identical regenerated text) poisons via `UNRESOLVED` instead of silently
  becoming a root. (The deterministic-identity follow-up upgrades this to full
  idempotent collapse.)
- **Crash after staging but before the ledger append:** the staged record exists but
  the ledger has no row. If the call has descendants, they resolve `UNRESOLVED` and
  poison; if it was terminal, receipt assembly finds no terminal row and poisons.
  The orphaned staged record is removed by the prefix-clear. Fail-closed with zero
  admitted-call bookkeeping.
- **Abandoned rollout:** NeMo RL's failure path prefix-clears TransferQueue and
  drops the ledger file. No TTL machinery.

## Constraints and caveats

- **`InMemoryLineageStore` cannot serve the ledger role.** Its backing index evicts
  rollouts under memory bounds, which is acceptable for a resolution cache but not
  for a completeness record. External-sink mode requires a non-evicting store
  (`FileLineageStore` or a framework-provided implementation); multi-worker serving
  already requires this. The in-memory store remains for single-worker development
  and unit tests only.
- **Receipt independence is reduced.** The manifest is derived from the same commit
  stream that produced the staged records rather than from an independent state
  machine. Per-record integrity is unaffected (digests are recomputed at
  finalization); completeness rests on lineage poisoning plus the terminal-row
  check, which the failure analysis above shows is fail-closed at the same points
  the gate was.
- **Ledger row growth.** Rows store full cumulative token ID arrays (as the lineage
  JSONL does today), so a rollout's ledger grows quadratically with call count.
  Acceptable at current rollout lengths; if it becomes a problem, rows can store a
  prefix reference instead of the full array without changing the contract.

## PR scope

Gym (PR 2278):

- `LineageStore` protocol + `FileLineageStore`: extended `record()`,
  `record_failure()`, `manifest()`, has-rows check.
- `sink.py`: gate-free admission tri-state in `resolve_parent()`.
- Commit hook: relocated lineage publication + failure recording.
- New manifest control route; `logical_request_id` stored on ledger rows (header or
  response-id fallback). Deterministic `model_call_id` derivation is **out of scope**
  (follow-up; see Implementation plan).
- **Full deletion of the gate** (files, routes, config, wiring, tests) as itemized
  above.

NeMo RL (companion):

- Receipt assembly in the finalizer; manifest fetch in `nemo_gym.py`; removal of
  register/seal/fail calls; prefix-clear cleanup on failure paths.

## Implementation plan

File-by-file changes, in landing order. Retry *idempotency* (client-minted logical
ids + deterministic `model_call_id`) is explicitly deferred — see the follow-up
section at the end. Gym paths are relative to the `3rdparty/Gym-workspace/Gym`
checkout; NeMo RL paths are relative to the repo root. Line references are against
the current state of both trees.

### Gym (PR 2278)

**1. `nemo_gym/token_id_capture/protocols.py` — widen the `LineageStore` protocol.**
`record()` (`protocols.py:74-82`) gains the token-free `CallRecord` fields:
`parent_call_id`, `staging_key`, `weight_version`, `prev_len`, `delta_len`,
`cum_len`, `extras_digest`, `mode`, `logical_request_id`. New protocol methods:
`record_failure(rollout_id, model_call_id, reason)`,
`manifest(rollout_id) -> list[CallRecord]`, and `has_rows(rollout_id) -> bool`.
Additive, so it lands standalone.

**2. `nemo_gym/token_id_capture/lineage.py` — implement in both stores.**
- `FileLineageStore._record` (`lineage.py:520-527`) extends the six-key JSONL row
  with the new fields. The existing same-id idempotency scan (`lineage.py:528-535`
  — identical payload no-op, conflicting payload `ValueError`) is kept unchanged.
- `record_failure()` appends a row with `reason` and **no `fingerprint`**; since
  `_resolve` (`lineage.py:472-490`) filters by fingerprint, failure rows can never
  be returned as parents — no filtering logic needed.
- `manifest()` reads rows under the per-rollout lock, strips
  `cumulative_token_ids`, returns validated `CallRecord`s. `has_rows()` is a locked
  existence check.
- `InMemoryLineageStore` implements the same methods but is rejected in
  external-sink mode (its `LineageIndex` evicts rollouts, `lineage.py:296-344`,
  which breaks completeness); it remains for unit tests and single-worker dev.

**3. `nemo_gym/base_responses_api_model.py` — de-wiring (no identity change).**
- `model_call_id = uuid4().hex` (`:1254`) **stays as-is** for now.
- Delete gate construction and control-route install (`:1481-1498`), the
  `GateError` handler (`:1500-1510`), and capability-header handling in the
  middleware (`:1263-1278`).
- Replace `_fail_uncommitted_gate_call` (`:1161-1176`, `finally` call site `:1286`)
  with a `record_failure(reason="request_finished_without_staged_coordinates")` in
  the same `finally` — a request that dies after admission must still poison.
- `CaptureContext` (`token_id_capture/sink.py:53-87`) drops `staging_gate` and
  `data_capability`.
- `_reject_gate_streaming` (`:104-111`) is re-keyed off the capture context instead
  of the gate; its three dialect call sites (`:235`, `:280`, `:322`) stay.

**4. `nemo_gym/token_id_capture/sink.py` — tri-state admission in
`resolve_parent()`.** Replace the gate admission step (`sink.py:153-159`) with the
pure function of the lineage result described in Design §2: unique verified match →
`token_in`; empty fingerprint or unmatched fingerprint with `has_rows() == False`
(seeded history) → `text` root; anything else → no admission +
`record_failure(reason="unresolved_parent")`. This closes the gate's silent
root-fallback hole and is load-bearing for the deferred-retry story.

**5. `responses_api_models/vllm_model/app.py` — gate-free commit hook.**
`_finalize_gate_capture` (`app.py:946-987`) keeps its shape; the gate call
(`:962-966`) becomes:
- `disposition == "staged"` → `cumulative = context.parent_tokens +
  coords.token_ids_delta`, then the extended `lineage_store.record(...)` — the
  publication block relocated from `gate.py:356-384`;
- `capture_failed`, missing coords, or any exception (including `record()`'s
  conflict `ValueError`) → `record_failure(...)`.
The `logical_request_id` binding keeps today's fallback:
`context.logical_request_id or str(payload["id"])`. The existing catch-all that
never turns a valid completion into a harness failure (`:972-985`) stays.

**6. `nemo_gym/token_id_capture/control_routes.py` — one read-only route.**
Delete `PUT /rollouts/{id}`, `POST .../seal`, `POST .../fail`, `GET /cleanup`,
`GET /metrics`, and the TTL sweeper task (`control_routes.py:74-164`). Add
`GET /training-token-capture/rollouts/{rollout_id}/manifest` returning
`manifest(rollout_id)`. `RolloutControlClient` (`:167-266`) shrinks to `manifest()`.

**7. Deletions.** `gate.py`, `gate_store.py`,
`test_token_capture_gate*.py`; `token_id_capture.gate.*` config keys removed with
**loud validation failure** on leftovers; the `rebuild_response`-with-gate guard
(`token_id_capture/config.py:162-166`) re-keyed to external capture.
`CallRecord` / `RolloutReceipt` / `CommitCoords` stay in `staging/records.py`;
`verify_and_linearize()` is untouched. Gate test invariants that still apply
(tri-state admission, commit ordering, same-call commit idempotency, poisoning) are
re-expressed as ledger and admission tests.

### NeMo RL (companion)

**8. `nemo_rl/environments/nemo_gym.py`.** Delete `register_rollouts`
(`:600-618`), `fail_rollouts` (`:620-637`), `gate_metrics` (`:639-640`), capability
stamping in `run_rollouts` (`:664-672`), and the `gate:` config block (`:521-527`),
keeping the `FileLineageStore` configuration (`:519-520`).
`_postprocess_receipt_mode()` (`:753-812`) replaces the seal call (`:779-789`) with
the manifest `GET`, then builds the `RolloutReceipt` locally: terminal row = the
manifest row whose `logical_request_id` equals the reported
`terminal_logical_request_id` (a response id, unchanged agent behavior);
`capture_poisoned` = any failure row or no terminal row. **Open check:** retry
duplicates now appear as dead-branch rows in the manifest (distinct call ids, same
parent); receipt assembly must either prune to the terminal parent-chain or we must
confirm `_validate_manifest_graph` (`staging/rebuild.py:159`) tolerates rows
unreferenced by the terminal chain.

**9. `nemo_rl/experience/blackbox_finalizer.py`.** Near-zero delta: the receipt
already arrives by value into `finalize_rollout` (`:241`); every downstream check,
`verify_and_linearize` (`:411-424`), placeholder masking (`:566-590`), and
`_clear_staging` (`:776-786`) are unchanged. Dead-branch rows are in the manifest,
so their staging keys are enumerated and cleaned — no new orphan class.

**10. `nemo_rl/experience/rollout_manager.py`.** The abandonment path (`:929-935`)
drops `fail_rollouts.remote(...)`; cleanup becomes staging clear + ledger drop. The
`gate_metrics` proxy (`:799-811`) is removed or repointed at ledger-derived counters.

### Landing order

1→2 (additive ledger, standalone-testable) → 3→7 as one Gym change → 8→10 as the
NeMo RL companion → end-to-end via `docs/guides/nano-swe-token-capture.md`.

### Open item to settle before step 10

**No prefix-clear primitive exists.** `TQTokenSink.clear()` is explicit-keys-only
(`nemo_rl/data_plane/tq_token_sink.py:280-290`); the layer below
(`data_plane/interfaces.py:415-440`) only offers a producer-local clear-all that may
silently no-op for non-producers. Manifest-enumerated keys cover every row the
ledger knows about, including dead branches. The remaining gap is the
crash-between-stage-and-ledger-append orphan (a staged row no manifest names).
Options: (a) add a prefix/scan delete to the TransferQueue data plane, (b) accept
the leak until partition teardown, (c) have the worker report the staging key in
failure coords so the orphan lands in a failure row and stays enumerable.

### Follow-up (explicitly out of scope for this change)

**Retry idempotency via deterministic identity.** The harness mints a unique
`x-nemo-gym-logical-request-id` per model call before dispatch, reuses it verbatim
on retry, and reports the terminal call's logical id (replacing the post-hoc
response-id derivation at `responses_api_agents/swe_agents/app.py:3119`); the
middleware then derives `model_call_id` deterministically from it (`:1254`).
Identical retries collapse into the same row and staging key instead of poisoning
or dead-branching; divergent retries poison via `record()`'s conflict rule.
Optionally layered on top: store the response payload on the ledger row and replay
it on retry (idempotency cache), eliminating divergent regeneration entirely. The
deferred surface is exactly two touch points, so nothing in this plan is redone.

## Related documents

- `token-capture-external-sink-recommendation.md` (repo root) — the review that
  motivated this design; this document promotes its "per-rollout append-only capture
  ledger" option from a future alternative to the recommended mechanism.
- `docs/guides/nano-swe-token-capture.md` — the recipe exercising this path end to
  end.
