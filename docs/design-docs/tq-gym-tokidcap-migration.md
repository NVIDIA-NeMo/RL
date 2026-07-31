# Token Capture v4: Migration onto the Gym tokidcap Stack

Consolidated plan for re-basing the gate-authoritative token-capture work
(`tq-gym-gate-authoritative.md`, MVP S1–S5 complete and signed off) **on top
of** the upstream NVIDIA-NeMo/Gym token-id-capture PR stack, instead of
carrying it as a parallel fork off old `main`. The TransferQueue (TQ) save
becomes NeMo-RL's implementation of the stack's `TokenSink`/`TokenSource`
protocols; the stack owns capture, lineage resolution, prefix supply, and the
rebuild engine.

This supersedes the fork strategy in `tq-gym-gate-authoritative.md` § 9.2 and
§ 10. The **invariants** of that design (tokens rest at two places, one heavy
hop, fail-closed custody, always-N accounting) are preserved; only their
**home** changes — from re-created modules to upstream seams.

---

## 1. Why migrate

The gate-authoritative MVP was built on a fork whose base (Gym `32b555f04`)
predated the upstream capture stack. Since then upstream shipped the same
primitives we re-created, plus two capabilities we did not build:

- **We re-created what the stack now owns.** Our `records`/`sink`/`store`/
  `config`/`routes`/`reader` capture core is byte-replaced by #2124/#2182;
  our `rebuild.py` happy path is subsumed by the stack's
  `run_builder(prefix_merging)`; our `memory_store.py` delta buffer is
  redundant with the stack's `LineageIndex` nodes (which already hold
  `cum_tokens`).
- **The stack has request-time lineage + prefix supply we lack cleanly.**
  #2180 resolves each call's parent at request time by fingerprinting the
  model-authored turns across all three dialects (chat / Anthropic /
  Responses); #2181 supplies the engine the parent's exact tokens on a unique
  verified match. This is dialect-portable — it works for Anthropic-dialect
  harnesses (Claude Code) that strip unknown fields, which our explicit
  `ng_call_id` marker does not (no `anthropic_converter.py` plumbing).
- **Divergence is now cheap to price.** The remaining gap between our design
  and the stack is a small set of **integration seams** we have requested as
  upstream asks (§ 4). Each has a workaround, so declining an ask changes fork
  *size*, not feasibility.

The strategic driver is external-agent-harness training (SWE): the stack was
built for it, and migrating puts us on the maintained, dialect-portable base
instead of a fork that must chase it.

## 2. What we keep, replace, and delete

Cherry-pick, do **not** rebase the old branch (its base is ~50 commits behind
and the capture core is a verbatim collision):

| Our module | Fate | Rationale |
|---|---|---|
| Re-created capture core (`records`/`sink`/`store`/`config`/`routes`/`reader`) | **Delete** | #2124/#2182 replace verbatim |
| `staging/records.py` (`StagedCallRecord`, `CommitCoords`, `RolloutReceipt`, `SCHEMA_VERSION`) | **Keep** | worker↔TQ and gate↔SC wire schema; no stack equivalent |
| `staging/digest.py` (row-integrity digest over ids+mask+logprob bits) | **Keep** | complementary to the stack's parent-*identity* digest (different domain, different purpose) |
| `staging/capture.py` + `adapters/vllm.py` (engine-blind capture, splice, extraction) | **Keep, rekey** | worker-locus core; rekey identity onto the stack's `model_call_id` |
| `gate.py` + `control_routes.py` (register/seal/fail, receipts, TTLs) | **Keep, rewire** | host over the stack's `LineageIndex` + a sink observer instead of our own store |
| `staging/lineage.py` (pure admit/commit/fail/seal state machine) | **Keep** | the gate registry; its inputs become stack capture events |
| `staging/rebuild.py` | **Mostly delete** | `run_builder(prefix_merging)` is the rebuild engine; keep only a thin terminal-aware chain-selection wrapper (§ 6, finding T) |
| `memory_store.py` | **Delete** | `LineageIndex` nodes already hold `cum_tokens` (§ 6, finding M — with a caveat) |
| `staging/conformance/` + S5 byte-counter middleware | **Keep, retarget** | retarget fixtures to stack identity |

RL-side modules (`tq_token_sink.py`, `blackbox_finalizer.py`, replay-buffer
surgery, SC wiring, advantage validity, S5 tooling) are **largely reusable
as-is** — they need a re-pin and an API/identity alignment, not a rewrite.

## 3. Identity: the single carrier switch

The one cross-cutting change. Drop the `responses_create_params.metadata`
side-channel; adopt the stack's correlation:

- `RolloutManager` mints `{group_id}_g{i}` and places the rollout id into the
  **run body** (via #2126-c2's opaque `rollout_id` key if it lands; else set
  `task_index = <global dispatch counter>`, `rollout_index = i`).
- Agents stamp `/ng-rollout/<id>` on every model call via the stack's existing
  `url_path_for_run` / `base_url_for_run` helpers — **27 agent impls
  unmodified**.
- The middleware's `model_call_id` becomes our `call_id`. Gate registration
  stays **create-only, pre-dispatch**.

`{group_id}_g{i}` is deliberately identical to the TQ sample ids
(`payload.py:115`), so the finalizer's publish keys line up unchanged.

## 4. The integration seams (upstream asks) and their workarounds

Each of our kept components plugs into a specific stack seam. The workaround
column is the cost if the ask is declined — it sets the fork size, not
feasibility.

| Seam (ask) | Status | Our component | If declined |
|---|---|---|---|
| `install_token_sink` honored + activation without a capture dir (**#2124-c1**) | posted | gate's manifest-recording sink at startup; `TQTokenSink` for server-locus fallback | throwaway capture dir + monkeypatch the `CaptureContext` path — ugly, contained |
| `mark_incomplete` on the `TokenSink` protocol (**#2124-c2**) | posted | capture-poison → gate registry → receipt → finalizer placeholder | implement `mark_incomplete` on our sinks anyway (duck-typed) — fragile, works |
| `commit_entry` split from `capture_tokens` (**#2124-c3**) | posted | coords-ingestion entry point: worker `ng_commit_coords` + delta ids → lineage record + manifest | synthesize a token-bearing fake response to feed `capture_tokens` — works, pollutes audit fields |
| `schema_version` on the record (**#2124-c4**) | posted | registration-time version negotiation | carry version in our TQ tags only; Gym records stay unversioned |
| External delivery mode (**#2126-c1**) | posted | not on our hot path; needed for `gym eval` A/B tooling with a sink installed | run A/B with delivery warnings log-filtered; accept masked-sample noise |
| Opaque `rollout_id` key (**#2126-c2**) | posted | `RolloutManager` passes `{group_id}_g{i}` directly | global dispatch counter as `task_index` (must persist across SC restarts) |
| `set_lineage_index` (**#2180-c1**) | to post | gate constructs the index with training capacity, `drop()` at seal, counters as metrics | monkeypatch `sink._LINEAGE` before first request; pin the attr name in a bump checklist |
| Capacity config + eviction metric (**#2180-c2**) | to post | gate config; `token_in_rate` diagnostics | comes free with the setter workaround (we construct the instance) |
| Resolver seam for a custom parent resolver (**#2180-c3**) | to post | Phase-2 marker plugs in here | fork `sink.py`'s resolve path |
| `ng_capture` hook on the outbound body (**#2181-c1**) | to draft | carries `{rollout_id, call_id, parent_call_id, prev_len}` to the worker — the identity `adapters/vllm.py` keys on | patch `_apply_prefix_supply` in the fork — highest-churn file, worst to carry |
| `required_prefix_token_ids` contract (**#2181-c2**) | to draft | worker splice consumes as-is (proven code) | no code impact; rename-risk on bumps |

**The two seams that shape the fork most** are `set_lineage_index` (#2180-c1)
and the `ng_capture` hook (#2181-c1): their declined-workarounds are the most
invasive (patching a private attr; forking the highest-churn file). Their
review outcomes should be resolved before Stage 1 commits to a locus.

## 5. Staged plan

Same one-PR, sign-off-gated shape as the MVP; each stage keeps the tree green
(capture dormant behind `token_capture.enabled=false`).

### Stage 0 — land the asks, restructure the base

1. Finish the review pass: post #2180 (**with** the resolver-seam comment, per
   § 6 finding A) and draft+post #2181's two asks. Track outcomes — the
   workaround matrix (§ 4) is the fork-size ledger.
2. Cut a new Gym branch from the stack top (pin to an **exact rev**, recorded;
   this stack has already squashed/rebased once). Apply the § 2 cherry-picks.
3. Re-pin the RL submodule to the new branch; run the capture L1 test with
   `enabled=false` to flush all API drift in one sweep.
4. Switch the identity carrier (§ 3).

**Gate:** submodule branch + gitlink/CI decided; capture-disabled functional
test green (pin-bump regression); leaf package importable in the worker venv.

### Stage 1 — worker-locus MVP

Wire each kept component to its seam (§ 4). RL-side is mostly alignment:

- rekey `TQTokenSink` rows and staging keys to `{rollout_id}/{model_call_id}`;
- point `install_capture` at the stack-provided identity from `ng_capture`;
- keep the weight-version fan-out (worker stamps at generation start);
- finalizer fetches by receipt manifest, verifies (staging digest,
  `prev_len + delta_len == cum_len`, wv tags), rebuilds via `run_builder` over
  reconstructed entries, publishes always-N with placeholders,
  `commit_finalized`;
- **add** (§ 6): control-plane bearer auth; ambiguity-cause metric; terminal-
  aware chain selection; consume the builder's `unresolved_retries` flag.

**Gate (S5 tooling, retargeted):** locus-equivalence (same seed through
server-locus JSONL and worker-locus TQ → byte-identical canonical rows);
fixed-seed A/B vs legacy echo with offline row diff; chaos smoke (kill gate
mid-step → placeholders + TTL, no leaks); per-call HTTP byte measurement;
capture-enabled L1 functional test.

### Stage 2 — marker (optional, data-driven)

Land the explicit `ng_call_id` marker **behind the #2180-c3 resolver seam**,
triggered when Stage-1 `fallback_rate`-by-cause data (esp. the *ambiguity*
cause, § 6 finding A) justifies it. Marker names the parent exactly where
fingerprinting misses (identical retries, dialect-stripping harnesses).

### Stage 3 — hardening

Order unchanged from the MVP: **H1** chaos (bound control-plane retries — the
S5 gate-death silent-stall finding) → **H2** hash layer (`chain_hash`/
`cum_hash` fill reserved fields) → **H3** staleness modes → **H4** upstream the
gate/control-plane/TQ layers as the next PRs of this same stack → **H5** perf/
scale (delta-linked `LineageIndex` nodes if SWE memory demands it, § 6 finding
M; SGLang adapter).

## 6. Review findings folded into the plan

Five findings from the integration review, folded into the stages above:

- **Finding M — `LineageIndex` memory regresses; #2180-c2 is a Stage-1
  blocker, not a nice-to-have.** The index stores the full cumulative sequence
  per call (O(calls × context) per rollout — the delta forest we delete was
  O(total tokens)) and evicts oldest-rollout-first under pressure, **including
  live rollouts**. Its default ~290 MiB budget was sized for eval collection,
  not training with hundreds of in-flight long-context rollouts. Size capacity
  from the training config (in-flight × max context), `drop()` at seal, add an
  eviction counter to § 8 metrics. Keep the deleted delta-forest code as the
  H5 donor for a delta-linked node representation upstream.
- **Finding A — fingerprint-only lineage hits ambiguity on SWE.** Byte-
  identical assistant turns (the model retrying the same command) resolve to a
  miss → fallback root → split chain. Instrument the *ambiguity* fallback
  cause separately from day 1; expect it to fire on SWE. This is why Stage 0
  posts #2180 **with** the resolver seam (c3): it makes Stage 2 a config flip,
  not a `sink.py` fork.
- **Finding T — token-mass main-chain selection mispicks on forks; make the
  terminal-aware wrapper day-1.** Upstream picks the chain with the most
  generated tokens; a sub-agent fork that out-generates the main conversation
  wins, and SWE spawns sub-agents — silent training on the wrong chain, which
  the row diff catches only if the A/B workload forks. The gate registry knows
  the terminal call at seal; keep a ~20-line terminal-aware selection override
  in the finalizer's `run_builder` path, and consume the builder's
  `unresolved_retries` → placeholder.
- **Finding S — control-plane auth is missing.** `register/seal/fail` sit on
  the sandbox-reachable model-server app; ids are guessable (`{group_id}_g{i}`)
  and this plan makes them *more* visible (they ride the run body + URL
  prefix). A sandboxed harness can forge a sibling seal, and the forged seal
  *wins* (legit seal 404s into the placeholder path). Add the #2182 bearer-
  token pattern (default-required) to the control routes in Stage 1 — our
  code, no ask needed. Sandboxed external harnesses are the point of the
  reorientation.
- **Finding G — two flow nits.** (a) The middleware rejects only *prefixed-
  but-unknown* ids; an uncorrelated call (no `/ng-rollout/` prefix — eval
  traffic, pre-registration side calls) passes through untouched, or the dual-
  mode "no coords → stack fallback unchanged" path breaks. (b) Pin the Gym
  branch to exact revs and record them.

## 7. Detailed rollout walkthrough

Ownership legend: **[stack]** upstream as merged · **[ask]** an upstream seam
we requested · **[ours]** our layers (Gym cherry-picks + NeMo-RL).

### 7.1 Dispatch (SingleController Ray actor)

`_rollout_pump → RolloutManager._generate_and_finalize`:

1. mint rollout ids `{group_id}_g{i}` for the group **[ours]**
2. `tq_buffer.reserve(group, rollout_ids, weight_version=v_start)` — records
   the rollout ids on the slot for cleanup **[ours]**
3. `PUT /ng-control/rollouts/<id>` — create-only registration,
   `schema_version` negotiated here **[ask 2124-4]**; a duplicate id is a loud
   `DuplicateRolloutError` (rerun protection)
4. `run_rollouts.remote(rows + rollout_id in the run body)` **[ask 2126-2]** —
   token-free dispatch

### 7.2 Agent server (27 impls, unmodified)

`rollout_id_from_run(body)` derives the id and stamps `/ng-rollout/<id>` on
every model call via `url_path_for_run` / `base_url_for_run` **[stack]**. The
agent sees only messages — never token arrays, ever.

### 7.3 Gate = Gym model server, admission (request path)

Per model call the agent makes:

- middleware strips the `/ng-rollout/` prefix, **rejects a prefixed-but-
  unknown id** (uncorrelated calls pass through, § 6 finding G) **[ours]**,
  mints `model_call_id`, sets `CaptureContext` **[stack + ask 2124-1]**
- the **sampling pin** is applied last on every engine-bound path
  (`temperature`/`top_p`/`top_k=-1` from `policy.generation`) **[stack, #2190]**
- `_apply_prefix_supply`: fingerprint the model-authored turns →
  `LineageIndex.resolve` → a **unique digest-verified parent** →
  `required_prefix_token_ids = parent.cum_tokens` **[stack, #2180/#2181]**. A
  miss or ambiguity forwards the request untouched (a new root) — **never
  wrong**, only a cold cache. The ambiguity cause is counted (§ 6 finding A).
- attach `ng_capture{rollout_id, call_id, parent_call_id, prev_len}` to the
  outbound body **[ask 2181-1]** — the identity `adapters/vllm.py` keys on

Wire to the worker: prefix ids + suffix messages + identity (~4 B/token).

### 7.4 vLLM worker (NeMo-RL, hosting the Gym capture core)

- `begin_call`: splice the supplied prefix via `_replace_prefix_tokens`;
  record the **post-splice engine prompt ids** (the exact conditioning bytes)
  **[ours]**
- generate; extract ids + logprobs **natively** — no `token_id:NNN` string
  parsing, no `/tokenize` round trip **[ours]**
- `complete_call`: build `StagedCallRecord` — delta = `ids[prev_len:]`, mask
  (0.0 carried / 1.0 generated), logprobs, staging digest, and
  `weight_version` (stamped at generation start from the
  `_rollout_weight_version` fan-out) **[ours]**
- `TQTokenSink.stage(record)` → **TQ `rollout_staging`** — ★ **the only heavy
  hop**, durable **before** the response releases **[ours]**
- response back to the gate: text + delta ids + `CommitCoords{staged |
  capture_failed}` — **no logprob block** **[ours]**

### 7.5 Gate, commit (response path)

- coords present → skip the stack's extraction; validate (known `call_id`,
  `prev_len` match, dedupe) → `commit_entry` **[ask 2124-3]**:
  - `lineage.record(call_id, request + reconstructed turn,
    parent.cum_tokens + delta_ids, digest)` **[stack]** — this is what indexes
    the call so the *next* request's fingerprint resolves it
  - the manifest-recording sink observes → appends to the registry manifest
    **[ask 2124-1]**
  - failure → `mark_incomplete` → rollout capture-poisoned **[ask 2124-2]**
- **no coords** (unpatched backend / server-locus fallback) → the stack's
  extraction path runs unchanged, writing JSONL **[stack]**. This is the dual-
  mode property that keeps `enabled=false` byte-identical.
- strip coords + delta ids → **text-only completion** to the agent **[ours]**

A child call cannot exist before its parent's bytes are durable: the
completion whose history the child echoes is released only after coords
ingestion.

Then the agent loops — tool calls, sub-agent forks — and each new call
re-enters at 7.3. A sub-agent forking from turn-1 history fingerprints to the
turn-1 call and resolves to that interior node.

### 7.6 Seal + finalize

- environment `verify()` → reward → `POST /ng-control/rollouts/<id>/seal`
  with the reward **and an explicit `terminal_call_id`** (§ 6 finding T)
  **[ours]** → `RolloutReceipt{manifest, terminal_call_id}`; the registry
  entry is dropped and `LineageIndex.drop(id)` releases the in-flight tokens
- Ray return: text + receipt (~100 B/call) — token-free **[ours]**
- `BlackboxFinalizer`: `TQTokenSource.fetch(manifest keys)` → verify rows
  (digest, lengths, wv) → `run_builder` over the reconstructed entries
  (verified links = O(1) path) → **terminal-aware** main-chain selection →
  always-N rows (placeholders for poisoned / `unresolved_retries`) →
  `put_samples(rollout_data)` → `commit_finalized(group_min_wv)` → clear
  staging keys **[ours + stack]**

### 7.7 Invariants and where each is enforced

- **Tokens rest at exactly two places** — `LineageIndex` nodes (in-flight) and
  TQ (at rest); enforced by the text-only completion (7.5) and token-free Ray
  return (7.6).
- **Tokens move on exactly one heavy hop** — worker → TQ; enforced by the
  stage-before-respond ordering in `complete_call` (7.4).
- **Wrong-prefix service is structurally impossible** — the gate serves only
  `cum_tokens` of a unique verified resolution, else text mode (7.3).
- **Every rollout produces exactly one accounting outcome** — receipt-with-
  manifest, or poison → placeholder, or TTL sweep — so the published group is
  always size N (7.6).

## 8. Metrics (retargeted)

Carried from the MVP § 8, plus stack-derived: `token_in_rate` (unique-resolve
hit); `fallback_rate` **by cause** — `no_prefix` / `no_match` / **`ambiguity`**
/ `multi_worker` (§ 6 finding A); `LineageIndex` eviction count (§ 6 finding
M); `capture_failure_rate`; `digest_verify_failures`; `invalid_row_rate`;
`delivered_fraction` (sampled tokens on the delivered chain ÷ total sampled —
surfaces fork/compaction loss); `unresolved_retries`; finalize p50/p99;
`wv_spread`; per-call HTTP bytes (the headline number vs. the echo path).

## 9b. Implementation strategy — the clean Gym PR

The outcome is a **single Gym PR based on the top of the tokidcap stack**,
carrying only our additions, pre-shaped so H4 upstreaming is "merge these
commits as the stack's next PRs" rather than a fork reconciliation. Today's
#2208 is a 27-file / ~5k-line diff against the *old* base `32b555f04`; on the
new base most of that either disappears (the base now owns it) or shrinks to a
seam call (the asks collapse it). The target diff is the gate + staging +
adapter layers plus a handful of hook installations.

### 9b.1 Branch topology

- **Base**: the stack tip, pinned to an **exact rev** (§6 finding G), with the
  PR base branch set to that tip so GitHub shows only our commits. The stack is
  itself a chain (#2190→#2124→#2125→#2126→#2180→#2181→#2182); we base on #2182
  (or wherever it stabilizes) so all seams below us are present in one base.
- **Rebase discipline**: our commits touch base files **only through public
  seams**, so a stack rebase is mechanical — re-point the base, replay. Gate
  logic lives in its own modules; base-file edits are hook *installation*, not
  logic.
- **Companion**: a separate NeMo-RL PR pins the submodule to this Gym branch's
  head, reviewed side-by-side (the current #2208 ↔ RL#3421 relationship).

### 9b.1a NeMo-RL companion base

The RL companion PR is built on top of the **nano SWE recipe branch**
(`3fcc69666cfab3014a1dfdb6b316fe18dcd1e912`, "fix(swe): drop the short QOS
from the nano SWE recipe"), **not** bare `main` and **not** the legacy
`yukih/sc-entrypoint` branch. That branch is `main` + two commits — the nano
SWE recipe with an honoured TransferQueue data plane (`87866eee`) plus the QOS
fix — and it is the SWE workload this migration targets. Established facts:

- The async SingleController our design depends on is **already on `main`**
  (`single_controller.py` blob `9effa5168ca2` is identical on `main` and this
  branch), so the base is `main`-lineage, not a divergent controller line.
- The branch already wires the TQ data plane for the recipe
  (`nano_swe_teacher_sync_tq.yaml`), so `StagingSink`/`StagingSource` +
  `BlackboxFinalizer` layer onto the recipe that exercises them.
- `tq_token_sink.py` is **not** present on the branch — our capture work is
  purely additive.

**Coordination caveat (Stage 0 gate item):** the branch is not linked to an
open PR and is ~24+ commits behind `main`. Before the companion PR layers on
it, decide its trajectory: rebase the SWE branch onto current `main` first
(preferred — avoids inheriting staleness), or layer now and rebase later.
Because the branch owner is Zhiyu Li, this is a coordination point, not a
unilateral rebase. Pin the exact base rev in the PR description (§6 finding G).

### 9b.2 Commit series (each commit = a future stack PR)

1. **`feat(token-id-capture): staging wire schema + digest`** — `staging/
   records.py` (`StagedCallRecord`/`CommitCoords`/`RolloutReceipt`,
   `SCHEMA_VERSION`), `staging/digest.py`, `staging/protocols.py` with our sink
   renamed **`StagingSink`/`StagingSource`** to avoid colliding with the base's
   `TokenSink`/`TokenSource` (§D name hazard). Pure-additive leaf; purity test.
2. **`feat(token-id-capture): terminal-aware linearize over run_builder`** —
   the thin `rebuild.py` wrapper (terminal-hint chain selection + placeholder
   hooks, §6 finding T) delegating to the base `run_builder(prefix_merging)`.
   The rest of our old `rebuild.py` is deleted.
3. **`feat(token-id-capture): engine-blind capture core + vLLM adapter`** —
   `staging/capture.py`, `adapters/vllm.py`, rekeyed onto the base's
   `model_call_id`. Standalone against a mock adapter+sink.
4. **`feat(token-id-capture): gate hosting, prefix serving, control plane`** —
   `gate.py` (hosting `lineage.py` over the base's `LineageIndex` via
   `set_lineage_index`, coords ingestion via `commit_entry`, serving rule),
   `control_routes.py` (+ bearer auth, §6 finding S). **The seam-touching
   commit**; its base-file edits (`app.py`, `responses_converter.py`,
   `openai_utils.py`) are the diff that the asks shrink.
5. **`feat(token-id-capture): observability`** — `server_utils.py` byte-counter
   middleware, the `unattributed_calls` counter + first-occurrence warning
   (Limitation 3), fallback-by-cause counters incl. `ambiguity` (§6 finding A).

`memory_store.py` and the old `__init__.py` lazy-export block do not appear —
deleted per §2 (the base's leaf `__init__` and `LineageIndex` replace them).

### 9b.3 File manifest on the new base

| File | On new base | Note |
|---|---|---|
| `staging/records.py`, `digest.py`, `protocols.py` | **new, additive** | sink renamed `StagingSink`/`StagingSource` |
| `staging/capture.py`, `adapters/vllm.py` | **new, additive** | rekeyed to `model_call_id` |
| `staging/lineage.py`, `gate.py`, `control_routes.py` | **new** | host over base `LineageIndex`; not a second buffer |
| `staging/rebuild.py` | **new, thin** | terminal-aware wrapper only |
| `staging/conformance/` + fixtures | **new, retargeted** | fixtures rekeyed to stack identity |
| `memory_store.py` | **deleted** | `LineageIndex` holds `cum_tokens` |
| `token_id_capture/__init__.py` | **base wins** | adopt base's leaf `__init__`; append staging exports only |
| `responses_api_models/vllm_model/app.py` | **seam edits** | gate install + `commit_entry` ingestion; shrinks with #2124-c1/c3, #2181-c1 |
| `responses_converter.py`, `openai_utils.py` | **shrink/vanish** | marker plumbing is Stage-2-only; MVP uses fingerprints (no marker fields) |
| `server_utils.py` | **new middleware** | byte-counter + unattributed counter |

The single most important consequence: **the MVP touches `responses_converter.py`
and `openai_utils.py` far less than #2208 does** — those files existed only to
carry the `ng_call_id` marker, and the MVP has no marker (lineage is the base's
content fingerprint). They return only in Stage 2, behind the resolver seam.

### 9b.4 What keeps it "clean"

- **Every ask reduces a base-file diff.** `install_token_sink` (2124-c1) →
  sink install is one call, no `make_token_store` patch. `commit_entry` (2124-c3)
  → coords ingestion is a public entry, not a synthesized fake response.
  `ng_capture` hook (2181-c1) → identity rides a supported field, not a patch to
  `_apply_prefix_supply` (the highest-churn base file). If an ask is declined,
  the workaround is isolated to *our* modules where possible (§4), so the base
  diff stays small even in the degraded case.
- **Purity preserved.** `staging/` stays dependency-free (no fastapi/ray/torch),
  same subprocess-import test as the base's leaf rule; `gate.py`/
  `control_routes.py`/`adapters/` sit outside that scope, as they do today.
- **Base suite stays green** at every commit — the stack's capture suite + full
  Gym unit suite run as the base sanity check (the S3 log already does this at
  1507 passed).

### 9b.5 Gym-PR test plan

- base stack capture suite + full Gym unit suite green (regression);
- our conformance kit **retargeted to stack identity** — the S1 golden call
  sequences replayed **through the gate** produce receipts byte-identical to a
  direct `RolloutLineage` drive (the S3 test, rekeyed to `model_call_id`);
- `staging/` purity test;
- flag-off byte-identity: with capture disabled the server exposes nothing and
  the legacy path is unchanged (the S3 flag-off functional evidence, re-run on
  the new pin);
- server e2e: register → multi-turn with exact prefix service → seal receipt;
  missing-coords poison; edited-history fallback; unattributed-call counter
  fires. These are the existing `test_token_capture_gate_app.py` tests,
  rehosted on the stack's request path.

### 9b.6 Upstreaming shape (H4)

Because the base *is* the stack and our commits touch it only through seams,
H4 is not a reconciliation: the five commits above become the stack's next
PRs (gate → control-plane → TQ-facing observability), and the conformance kit
publishes as the multi-framework contract. The NeMo-RL companion PR is the
reference `StagingSink`/`StagingSource` implementation cited by that contract.

## 9. Risks specific to the migration

- **Stack instability.** The upstream stack has already squashed/rebased once;
  pin to exact revs (§ 6 finding G) and keep our cherry-picks touching the
  base only through public seams so a bump is mechanical.
- **Locus regression under decline.** If both #2180-c1 and #2181-c1 are
  declined, the worst-case fork is a private-attr monkeypatch plus a patch to
  the highest-churn file — carryable but noisy. Resolve these two before Stage
  1 commits.
- **Memory at SWE scale** (§ 6 finding M) — the `LineageIndex` full-cumulative
  representation is the sharpest one; the delta-forest donor is the H5 exit.
- **Fingerprint ambiguity on agentic retries** (§ 6 finding A) — the marker
  (Stage 2) is the exit; keep it seam-ready.
