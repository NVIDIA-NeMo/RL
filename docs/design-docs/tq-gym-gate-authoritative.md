# Token Capture v3: Gate Token Custody (Token-In / Token-Out)

Design and implementation plan for running NeMo-Gym rollouts through a
**token-in/token-out** capture pipeline in the async SingleController (SC)
GRPO pipeline. The Gym gate (the model server every agent calls) becomes the
**custodian of token lineage**: it holds each rollout's cumulative token
buffer, serves exact token prefixes to vLLM workers, and returns token-free
receipts. Workers stage per-call token deltas + logprobs directly to the
TransferQueue (TQ); tokens make exactly one heavy hop, at generation time.

This supersedes two earlier drafts:

- the **v2 doc** (`tq-gym-async-single-controller.md`): two trackers (gate
  message tree + RL-side registry actor), two Ray RPCs per model call;
- the **hash-lineage draft** of this doc: single gate tracker, but lineage
  verified after the fact by worker-side hash confirmation. Review found its
  claim-confirm was not computable for non-root parents and its candidate
  set excluded forks; more importantly, a code survey (§ 2) showed the
  hash machinery solves a problem the codebase does not have.

The v2 doc remains the reference for inherited material where cited
(weight-version design § 6, parts of the failure model).

---

## 1. Goal and design principle

Replace the async SC's Gym rollout path — where every generated token
transits Gym HTTP responses (twice per turn, growing with history), a Ray
return, and an SC-side tensorize — with a pipeline where:

- **Tokens rest at exactly two places**: the gate's per-rollout buffer
  (in-flight custody) and TQ (durable staging + canonical rows).
- **Tokens move on exactly one heavy hop**: worker → TQ, once per model
  call, carrying delta ids + logprobs + extras.
- **Every other hop is token-light**: gate→worker carries prefix ids
  (replacing the text prompt, comparable size); worker→gate carries delta
  ids only (~4 B/token, no logprobs); agent-facing messages and the Ray
  return are token-free.
- **The integration is framework- and backend-portable.** Gym defines the
  protocols and owns lineage (§ 3.0); an RL framework integrates by
  implementing four small contracts (sink, source, weight-version provider,
  wiring); an inference engine integrates via one adapter module.
  NeMo-RL/TQ and vLLM are the first providers, not the design.

### 1.1 The core principle: relocation, not invention

The survey of this branch (2026-07-27) established that **token-in/token-out
already runs in production here** — routed through the most expensive
possible path:

1. Gym's model server forces `logprobs=True, return_tokens_as_token_ids`,
   string-parses generated ids from logprob entries, recovers prompt ids via
   a second `/tokenize` HTTP call, and attaches
   `prompt_token_ids`/`generation_token_ids`/`generation_log_probs` to the
   response message (`responses_api_models/vllm_model/app.py:497-551`,
   `nemo_gym/responses_converter.py:362-374`).
2. The agent echoes those arrays back inside its message history every turn.
3. The vLLM worker scrapes them (`model_post_init`,
   `vllm_worker_async.py:505-517`) and splices the model's exact sampled ids
   in front of the freshly rendered suffix (`_replace_prefix_tokens`,
   `vllm_worker_async.py:52`).
4. The Ray return then carries all tokens again (`message_log` tensors +
   decoded strings in `full_result`, `environments/nemo_gym.py:319-422`).

Exact prefix conditioning, lineage identification, and template splicing are
therefore **proven code**. This design changes only custody: the cumulative
buffer moves from "echoed through the agent" to "held at the gate"; the
delta+logprobs move from "attached to HTTP responses and the Ray return" to
"staged once to TQ". Correctness properties are preserved *by construction*
(the model is conditioned on the same bytes as today), not re-verified after
the fact.

### 1.2 Design alternatives

| Design | Lineage mechanism | Token custody | Verdict |
|---|---|---|---|
| v2 doc | RL registry actor, hash-verified | worker→TQ | Works; two trackers; 2 RPCs/call |
| Hash-lineage draft | gate tree of hashes, worker hash-confirm | worker→TQ | Confirm not computable at depth ≥ 2 without extra wire fields; forks misresolve; verifies what token-in guarantees |
| **This doc** | **gate token buffer + message marker** | **gate (in-flight) + TQ** | Chosen: lineage explicit, zero verification on hot path |
| Bytes-through-gate | gate holds tokens + extras | gate | Rejected: logprobs/`routed_experts` (~KB/token) transit Python HTTP |
| Status quo | token echo through agent | agent messages + Ray return | The measured bytes problem |

The hash machinery is not deleted from the universe — it returns in the
hardening phase (§ 10, H2) as finalize-time tamper evidence and as the
mechanism that later relaxes the strict serving rule (§ 3.3).

## 2. Current state (what the survey established)

Facts the plan depends on, with sources:

- **SC path**: `_rollout_pump` → `RolloutManager.generate_and_push`
  (`rollout_manager.py:644`) → `reserve` slot → `run_rollouts.remote`
  (tokens ride the return) → `TQReplayBuffer.commit` tensorizes + puts N
  rows (`replay_buffer.py:612-660`). `RolloutManager` runs *inside* the SC
  actor; the only Ray boundary is `run_rollouts`.
- **No weight-version signal reaches vLLM workers** today; the only
  fan-out is `RolloutManager.set_weight_version`
  (`single_controller.py:591`). The `_sync_weights` drain is commented out
  (`single_controller.py:571-584`).
- **vLLM workers are data-plane-unaware**: zero `data_plane` imports under
  `models/generation/`. Template to copy: `sync_rollout_actor.py:128-130`.
- **`rollout_data` is never `register_partition`-ed**; lazy field
  registration under concurrent puts is a documented TQ controller race
  (`adapters/transfer_queue.py:449-461`).
- **Gym pin**: submodule `3rdparty/Gym-workspace/Gym` @ `610a08a`
  (editable uv workspace member). It has **no gate and no capture
  package** — in particular it predates upstream PR
  [#2124](https://github.com/NVIDIA-NeMo/Gym/pull/2124)
  (`nemo_gym/token_id_capture/`), which is a **required base** for the
  Gym-side work (§ 9.2). The gate donor code (admission middleware,
  `RolloutRegistry`, control router) lives in the prototype checkout's Gym
  pin (`/lustre/fsw/.../gym/RL`, Gym @ `6ea5810`,
  `nemo_gym/observability/capture_gate.py`).
- **Rollout identity today**: NeMo-RL passes no id into Gym (`_rowidx`
  re-sort only); Gym↔vLLM affinity is a session cookie → sticky
  round-robin (`vllm_model/app.py:576-584`).
- **Validity hook exists**: `calculate_baseline_and_std_per_prompt`
  accepts a `valid_mask` currently hardwired to ones
  (`advantage_estimator.py:69`).
- **Prototype donors** (`/lustre/fsw/.../gym/RL`): staging delta builder +
  three-column TensorDict write, `compute_staging_digest`
  (float32-bit-pattern scheme), hash-free mask-driven rebuild
  (`StagedSnapshotTokenSource.entries()`), finalizer reconcile/verify
  skeleton, gate admission + call-id mint + register/seal control API.

## 3. The contract

### 3.0 Protocol architecture: Gym defines, frameworks provide

The integration varies along two axes (RL framework's storage/training;
inference backend) and is invariant along one (lineage custody, wire
schema, serving rule). The code mirrors that split:

- **Gym owns the invariant, concretely**: the gate, the wire records
  (single definition), the digest, the rebuild/linearize semantics, the
  control routes, and the protocol definitions below. All of it lives in
  the `nemo_gym/token_id_capture/` **leaf package** (grown from #2124),
  under a hard purity rule — **no fastapi, no ray, no torch, no TQ
  imports** in the core modules, enforced by an import-linter test — so
  the package is importable inside any framework's worker process.
- **Gym also ships the backend adapters** (`adapters/vllm.py`, later
  `adapters/sglang.py`) implementing a `CaptureAdapter` protocol: how to
  enter prefix ids into an engine, splice the suffix, and extract
  generated ids + logprobs. The gate itself is engine-blind.
- **The RL framework provides four small implementations** against Gym's
  `protocols.py`:

```python
# Defined in GYM (token_id_capture/staging/protocols.py); frameworks implement.
class TokenSink(Protocol):        # WHERE deltas go (NeMo-RL impl: TQTokenSink)
    def stage(self, record: StagedCallRecord) -> StageResult: ...

class TokenSource(Protocol):      # finalizer's read-back (NeMo-RL: TQ get_samples)
    def fetch(self, staging_keys: list[str]) -> list[StagedCallSnapshot]: ...

class WeightVersionProvider(Protocol):   # trainer state, framework-owned
    def __call__(self) -> int: ...

# plus one wiring call at worker startup:
install_capture(serving_layer, sink=..., weight_version_fn=...)
```

Everything training-side (finalizer orchestration, placeholders, group
staleness, advantages) remains framework code — Gym has no opinion there.
Two rules make multi-framework work in practice: **staging keys are opaque
to Gym** (TQ keys, file paths, redis keys are all valid — the receipt
manifest is the only join between lineage and storage), and **conformance
is tested, not trusted** — Gym ships golden fixtures (call sequences →
byte-exact records/digests/manifests/linearized rows) that every framework
and adapter runs in its CI. `SCHEMA_VERSION` is negotiated at rollout
registration, so version skew fails at register time, not at finalize.

**The whole contract at a glance** — everything above the line ships in
Gym; everything below is the entire surface an RL framework writes:

```
GYM  nemo_gym/token_id_capture/            (leaf package; purity rule)
     records.py    StagedCallRecord, CommitCoords, RolloutReceipt,
                   CallRecord, staging_key(), SCHEMA_VERSION
     protocols.py  TokenSink, TokenSource, WeightVersionProvider,
                   CaptureAdapter; install_capture(...)
     digest.py     compute_staging_digest, encoders
     lineage.py    RolloutLineage — pure per-rollout state machine:
                   admit/commit/fail/seal -> manifest   (no HTTP, no store;
                   separately tested; gate hosting stays thin)
     capture.py    RolloutTokenCapture.begin_call / complete_call —
                   engine-blind: record + digest build, fail-closed
                   stage->respond ordering, coords assembly
     rebuild.py    snapshots -> entries -> linearize(policy, terminal_hint)
     adapters/vllm.py   engine-specific ONLY: suffix splice, id+logprob
                   extraction, serving-layer hookup    (sglang.py later)
     store.py / memory_store.py    #2124 store iface; in-memory buffer
     conformance/  golden fixtures + installable test kit
     gate          thin hosting of lineage.py: admission, call_id mint,
                   marker resolve + fingerprint, serving rule, prefix
                   serving, coords ingestion, seal -> receipt;
                   control routes + RolloutControlClient
──────────────────────────────────────────────────────────────────────────
RL   TQTokenSink / TQTokenSource           # TQ impls of the protocols
RL   blackbox_finalizer.py                 # orchestration over Gym rebuild
RL   worker setup (once): install_capture(serving_layer, sink=…,
                          weight_version_fn=lambda: self._rollout_weight_version)
RL   weight_version value (trainer state) + set_rollout_weight_version fan-out
RL   control-plane calls via Gym's RolloutControlClient; buffer/setup wiring
```

### 3.1 Identity: rollout ids and the call marker

- The SC mints rollout ids `{group_id}_g{i}` (matching existing TQ sample
  ids, `payload.py:115`) and registers them at the gate (create-only)
  before dispatch. Ids ride `responses_create_params.metadata` — the
  zero-agent-change carrier already used for side-channel params
  (`vllm_model/app.py:288-300`).
- The gate mints a `call_id` per admitted model call.
- **The marker.** When the gate forwards a completion to the agent, it
  attaches `ng_call_id: <call_id>` to the assistant message — replacing
  today's token-array attachment, same carrier, ~10 B instead of KBs. The
  agent echoes history verbatim (this is how the token echo works today),
  so the next request's messages carry the parent pointer **explicitly**.
  Lineage is a dictionary lookup, not a content-matching tree walk: forks
  (a sub-agent inheriting turn-1 history carries turn-1's marker) and
  identical siblings (distinct markers) are resolved exactly.

### 3.2 Wire changes per hop

| Hop | Today | This design |
|---|---|---|
| agent → gate | messages + echoed token arrays (grows/turn) | messages + markers (token-free) |
| gate → worker | text prompt (+ echoed ids in messages) | `prefix_ids` + suffix messages via `extra_body`; or text mode (fallback) |
| worker → TQ | — | **the only heavy hop**: delta ids + logprobs + masks + extras, once/call |
| worker → gate | text + logprob block (ids string-parsed) + 2nd `/tokenize` call | text + delta ids (~4 B/token) + `CommitCoords` (~100 B); no `/tokenize` |
| gate → agent | text + token arrays | text + marker |
| gate → SC (`run_rollouts` return) | full token tensors + decoded strings | `RolloutReceipt` (~100 B/call, token-free) |
| finalizer ↔ TQ | — | staged rows in, canonical rows out (off hot path) |

### 3.3 The serving rule (what makes zero-hash correct)

> **Serve token-in only when the incoming request carries a unique, known
> marker AND the message history up to the marker matches the gate's
> recorded fingerprint for that call. Anything else — no marker, unknown
> marker, edited history — falls back to text mode: the worker renders the
> full conversation from scratch and the call is captured as a new root
> (full ids staged, `parent=None`).**

The gate keeps a compact fingerprint (hash of normalized messages) per
committed call to detect history edits above the marker. A fallback is
wasteful (duplicated prefix storage, cold KV cache) but perfectly correct:
the model trains on exactly the bytes it saw. Silent wrong-prefix service is
structurally impossible — the gate serves only bytes whose provenance it
verified, or serves nothing. `token_in_rate` (§ 8) measures how often the
happy path holds.

### 3.4 Records

**`StagedCallRecord`** (worker → TQ, key `"{rollout_id}/{call_id}"`):
`token_ids_delta`, `token_mask_delta` (0.0 carried prompt / 1.0 generated),
`generation_logprobs_delta`, optional extras (`routed_experts`); tags:
`rollout_id, call_id, parent_call_id, prev_len, new_len, weight_version,
digest, schema_version`.

**`CommitCoords`** (worker → gate, rides the response): `call_id,
parent_call_id, delta_len, cum_len, digest, staging_key, weight_version,
disposition: staged | capture_failed`, plus the delta ids for the gate's
buffer.

**`RolloutReceipt`** (gate → SC at seal, token-free): `rollout_id, reward,
terminal_call_id, manifest: list[(call_id, parent_call_id|None, delta_len,
cum_len, digest, staging_key, weight_version, mode: token_in|text)],
schema_version`.

All record shapes are defined **once**, in Gym's
`nemo_gym/token_id_capture/staging/records.py` (§ 3.0); frameworks import them
from the leaf package. Wire shapes freeze at the S1 gate. Hash fields
(`chain_hash`, `cum_hash`) are **reserved optional fields** so the
hardening layer (H2) is additive.

### 3.5 Fail-closed ordering (per call)

```
gate: admit (marker → parent | fallback), mint call_id,
      forward prefix_ids + suffix (or text)
worker: splice (or render), generate, extract ids+logprobs natively
worker: sink.stage(record)            # bytes durable BEFORE ack
        ok   -> coords{staged} + delta ids ride the response
        fail -> coords{capture_failed}
gate: ingest coords = authoritative commit; extend token buffer;
      attach marker; forward text to agent
```

A child request cannot exist before its parent's bytes are durable and
committed, because the marker the child needs rides the response the gate
releases only after ingesting coords. Capture failure does not break the
agent: the completion still returns; the gate marks the rollout
capture-poisoned; the finalizer produces a placeholder.
`token_capture.on_capture_failure: continue | abort` (default `continue`).

## 4. Data flow

```
SC _rollout_pump
  └─ generate_and_push(prompt, target_step)
       ├─ tq_buffer.reserve(weight_version=v_start)                (unchanged)
       ├─ mint N rollout_ids, register at gate (create-only)       (new)
       ├─ NemoGym.run_rollouts.remote(rows + rollout_ids)          (token-free)
       │
       │   per model call:
       │     agent ─(messages + markers)─► GATE
       │       marker → parent lookup; fingerprint check;
       │       mint call_id; prefix_ids from token buffer
       │     GATE ─(prefix_ids + suffix | text)─► vLLM WORKER
       │       splice (_replace_prefix_tokens) | full render
       │       generate; extract ids+logprobs natively
       │       sink.stage(record) ──► TQ rollout_staging     ◄── the only
       │       delta ids + coords ride the response              heavy hop
       │     GATE ingests coords = COMMIT; buffer += delta;
       │       marker on assistant message; text to agent
       │
       ├─ seal each rollout → RolloutReceipt; gate drops buffer + state
       ├─ finalize group: fetch rows by manifest keys → digest check →
       │       mask-driven rebuild → linearize(main_chain_only,
       │       terminal_hint) → N rows (placeholders as needed) →
       │       put_samples(rollout_data)
       └─ tq_buffer.commit_finalized(group_id, meta, group_min_wv, group_max_wv)

SC _train_pump: unchanged; baseline over valid rows (§ 7)
SC _sync_weights: rotates worker weight_version (§ 6)
```

Two TQ partitions: `rollout_staging` (delta rows, cleared by finalizer /
eviction / TTL) and the existing `rollout_data` (canonical rows). Both are
pre-registered at setup (§ 2, controller race).

### 4.1 Worked example

Group `g7`, rollout `g7_r0`, wv 4 throughout. **c1**: no marker → text mode,
new root; worker renders `[10..14]` (3 prompt + 2 generated), stages
`g7_r0/c1`, coords + delta ids back; gate buffers, marks the assistant
message `ng_call_id=c1`. **c2** (tool result): messages carry `c1`'s marker
→ fingerprint ok → token-in: `prefix_ids=[10..14]` + suffix `[tool:"391"]`;
worker splices `[20,21,22]`, generates `[23,24]`; stages delta of 5;
`parent=c1`. **c3** (sub-agent from turn-1 history): carries `c1`'s marker →
token-in from an *interior* node — a fork, resolved exactly. **c4**
(framework rewrote a message): fingerprint miss → text mode, new root; 9 ids
staged, `parent=None`. Seal → receipt manifest `[(c1,∅,5),(c2,c1,+5),
(c3,c1,+4),(c4,∅,9)]`, terminal `c2`, reward 1.0. Finalize: fetch 4 rows,
digest-check, rebuild main chain c1→c2 → one canonical row (10 ids, 4
trainable), c3/c4 verified-untrained; publish N rows; clear staging.

## 5. What the finalizer verifies (and what it doesn't)

Per row: digest recomputation (`compute_staging_digest` over ids + mask +
logprob bit patterns — catches TQ corruption and key mixups), shape/mask/
finite-logprob checks, `prev_len + delta_len == cum_len`, weight-version tag
equality. Rebuild is the prototype's **hash-free, mask-driven**
`StagedSnapshotTokenSource.entries()`; linearization
`main_chain_only` with `terminal_hint`. Any rejection → masked placeholder
(always N rows; `prompt_ids_for_adv` copied from a valid sibling;
`min_valid_fraction_per_group` optionally drops the group).

Not verified in the MVP: cross-row chain integrity (an adversarial reorder
of manifest entries that also fixes up lengths). The gate is inside the
trust boundary (network-isolated gate→worker hop, as v2 § 4.6); chain
hashes return as tamper evidence in H2.

## 6. Weight versions, refit, staleness

Inherited from v2 § 4.4: per-call tagging via the worker's
`_rollout_weight_version` (new attribute; set at the end of both refit
paths and via a new `set_rollout_weight_version` fan-out from
`_sync_weights`); group staleness = oldest call version (`group_min_wv`),
stored as the slot's effective version (`commit_finalized`);
`mixed_weight_version_policy: allow | reject`; the T1 atomic-rotation TODO
unchanged and still prerequisite for strict modes; verl's work-preserving
`wait` option on the hardening list. MVP matrix: `staleness_window` +
`allow` only; strict modes raise `NotImplementedError`. Note the SC drain
is currently commented out (`single_controller.py:571-584`) — straddles are
normal and absorbed by `group_min_wv` conservatism.

## 7. Failure model

| Failure | Consequence | Recovery |
|---|---|---|
| Worker stages, dies before responding | Gate call-timeout → call failed → no marker released → rollout cannot continue → placeholder | Staging TTL sweeps the orphan row |
| Worker responds, gate dies | Rollout dies (gate mediates it); no receipt | Staging TTL; SC dispatch timeout |
| Gate dies between seal and receipt delivery | Receipt rides the `run_rollouts` return; if lost, rollout unrecoverable → placeholder + TTL | Optional receipt persistence (H1) |
| Coords lost (response dropped) | Call-timeout → failed; agent never got completion → no child exists | TTL |
| Agent strips/edits markers | Fingerprint or marker miss → text fallback, new root | Correct but wasteful; visible as `token_in_rate` drop |
| SC dispatch cancelled / shutdown | `try/finally`: `fail_rollout` + clear staged keys by prefix + release buffer permit | Registration TTL backstop |
| NaN-logprob batch retry | Would re-register create-only ids | `rollout_max_attempts_to_avoid_lp_nan == 1` enforced at setup |

Cleanup is two stores: the SC slot (+ its TQ rows, staging + canonical) and
the gate's per-rollout state (self-clears at seal / `fail_rollout` /
registration TTL). Slots record their rollout_ids at reserve time;
`TQReplayBuffer.remove(remove_in_dp=True)` clears both partitions.

## 8. Metrics

`token_in_rate` (marker hit), `fallback_rate` by cause (no-marker /
fingerprint-miss / unknown-marker), `capture_failure_rate`,
`digest_verify_failures`, `invalid_row_rate`, finalize p50/p99, `wv_spread`,
gate admission→commit latency, receipts lost, staging partition size,
per-call HTTP bytes (the headline number vs. the echo path).

## 9. Component changes

### 9.0 Runtime placement at a glance

Who runs where, and the one-line ownership rule for each home:

| Runtime home | Components | Owns |
|---|---|---|
| **SingleController Ray actor** (NeMo-RL) | `_rollout_pump`/`_train_pump`/`_sync_weights`; `RolloutManager` (mints rollout_ids); `TQReplayBuffer` (`reserve`/`commit_finalized`/`remove`); `BlackboxFinalizer` (runs in the dispatch task via `asyncio.to_thread` for the MVP); `StalenessSampler`; advantage pump | **Training assembly**: group semantics, slots, receipts → N canonical rows, staleness, cleanup |
| **NemoGym Ray actor** (NeMo-RL file wrapping Gym) | server spin-up; token-free `run_rollouts` + receipt unpacking; gate control-plane client (register/seal/`fail_rollout`) | The Ray↔HTTP boundary; no token logic |
| **Gym model server = the gate** (submodule fork = main + #2124 + gate work) | thin hosting of the pure **`lineage.py`** state machine; admission + call_id mint + TTLs; per-rollout **token buffer** (in-memory `token_id_capture` store impl); `ng_call_id` marker attach/resolve + history fingerprint; `prefix_ids` serving + rollout→worker affinity; coords ingestion = commit; seal → receipt | **Lineage custody**: everything that requires understanding *messages*. Holds token ids in flight; never logprobs; never writes TQ |
| **Gym agent + resources servers** | — | **Nothing new** (design goal: 27 agent impls untouched; markers ride messages opaquely; rewards via `/verify` as today) |
| **vLLM worker Ray actor** (NeMo-RL, hosting Gym's capture) | in-process HTTP server; **Gym `capture.py`** (engine-blind: record/digest build, fail-closed ordering, coords) + **Gym `adapters/vllm.py`** (splice, extraction) via `install_capture`; NeMo-RL provides `TQTokenSink.stage()` (the only heavy hop), the DP client, and `_rollout_weight_version` | **Token production + durability**: capture logic is Gym's; storage and hosting are NeMo-RL's; bytes durable before ack |
| **TransferQueue** | `rollout_staging` (finalizer is the only reader) + `rollout_data` (train pump is the only reader) | Bytes at rest |

The wire records, digest, rebuild semantics, and protocols are defined
**once**, in Gym's dependency-free `token_id_capture` leaf package
(§ 3.0), and imported by both the gate and the framework's worker/finalizer
— importable anywhere because the core modules carry no heavy dependencies
(enforced in Gym CI).

### 9.1 NeMo-RL

There is **no** `nemo_rl/experience/token_capture/` package — records,
digest, protocols, rebuild, and capture logic live in Gym (§ 3.0, § 9.2).
NeMo-RL writes only the provider implementations and training assembly:

| Component | Change |
|---|---|
| `nemo_rl/data_plane/tq_token_sink.py` (new) | implements Gym's `TokenSink` and `TokenSource` protocols over TQ (`put_samples` / `get_samples`); the only hot-path file that knows TQ |
| `nemo_rl/experience/blackbox_finalizer.py` (new) | orchestration only: fetch via `TokenSource`, call Gym's verify/`rebuild`/`linearize`, always-N placeholders, `group_min_wv`, publish to `rollout_data` |
| `algorithms/async_utils/replay_buffer.py` | `commit_finalized`; slots record rollout_ids; `remove` clears staging; `abort(group_id)`; fix `commit` on evicted slots (`:657`) |
| `algorithms/single_controller.py` | release `_buffer_capacity` in dispatch `finally` (`:319`); `set_rollout_weight_version` fan-out at `:591`; teardown sweep; metrics |
| `algorithms/single_controller_utils/setup.py` | pre-register `rollout_staging` + `rollout_data`; gate config into `spinup_nemo_gym_actor`; sink factory + `dp_cfg` to generation setup |
| `algorithms/single_controller_utils/config.py` | `TokenCaptureConfig` on `MasterConfig`; `AsyncRLConfig` additions |
| `models/generation/vllm/vllm_worker_async.py` | **hosting only**: one `install_capture(serving_layer, sink=TQTokenSink(...), weight_version_fn=...)` call at startup; `setup_token_capture(dp_cfg)` fan-out target building the in-worker DP client; `_rollout_weight_version` attribute. Capture logic moves to Gym: fail-closed ordering + record build in `capture.py`; prefix-in, splice (`_replace_prefix_tokens` relocates), extraction in `adapters/vllm.py` |
| `models/generation/vllm/vllm_generation.py` | `setup_token_capture` / `set_rollout_weight_version` fan-outs (existing `run_all_workers_single_data` pattern) |
| `experience/rollout_manager.py` | mint rollout_ids (thread `group_id` into `run_rollout`); receipt mode in `_result_to_completion` / `_compute_rollout_metrics` (incl. removing tokens from the wandb table, `:575`) |
| `environments/nemo_gym.py` | register/seal/fail control-plane helpers; receipt-mode `_postprocess` (drop token walk + contiguity assert `:329-388` — the gate owns that guarantee now); fix `run_rollouts` return annotation; NaN-retry hard error |
| `algorithms/advantage_estimator.py` | pass real validity into `calculate_baseline_and_std_per_prompt` (`:69`); validity folds into `sample_mask` (no new train field) |

### 9.2 Gym (vendored submodule, fork branch **based on #2124**)

The Gym implementation builds **on top of upstream PR #2124** (token id
capture core, `nemo_gym/token_id_capture/`), not beside it. The fork branch
is upstream main + #2124 (pinned to a specific rev of that PR) + the gate
work. What #2124 supplies and how it is used:

- `records.py` `TokenEntry` (rollout id + server call id + prompt/gen ids +
  logprobs + message content) → the gate's per-call buffer entry; this
  design adopts its rollout/call-id grammar rather than minting a parallel
  one.
- `sink.py` (records a `TokenEntry` from the finished model-server
  response, streaming-safe) → the coords-ingestion point: extended to read
  the worker's delta ids + `CommitCoords` and extend the rollout buffer.
- `store.py` (`TokenCaptureStore`, per-rollout JSONL, per-file locking) →
  becomes the store interface behind the gate buffer: a new in-memory
  implementation serves the hot path; the JSONL store is retained as an
  optional debug/persistence backend and for H1 receipt persistence.
- `config.py` (`token_id_capture_enabled` + directory) and the
  `base_responses_api_model.py` integration (record + install routes +
  re-stream) → the on/off switch and wiring the gate extends.
- `routes.py` (`GET /ng-capture/tokens/{rollout_id}`) → retained; useful
  as a debug read path beside TQ staging.

The package is grown into the **integration SDK** under the § 3.0 purity
rule (core modules dependency-free, import-linter-enforced), so any
framework's worker can import it:

| Component | Change |
|---|---|
| `nemo_gym/token_id_capture/` (from #2124) | base package: records/store/sink/config/routes as above; in-memory store impl added |
| `token_id_capture/staging/records.py` (new) | **single definition** of all wire shapes (§ 3.4), beside #2124's `TokenEntry` (whose `records.py` is untouched); `SCHEMA_VERSION` |
| `token_id_capture/staging/protocols.py` (new) | `TokenSink`, `TokenSource`, `WeightVersionProvider`, `CaptureAdapter`; `install_capture` entrypoint |
| `token_id_capture/staging/digest.py` (new) | `compute_staging_digest` + encoders (prototype `rollout_writer.py:1014` port; golden vectors) |
| `token_id_capture/staging/rebuild.py` (new) | pure functions: snapshots → entries → `linearize(policy, terminal_hint)` — identical training-row semantics for every framework (prototype `staged_token_source.py` core) |
| `token_id_capture/staging/lineage.py` (new) | `RolloutLineage` — pure per-rollout state machine (admit/commit/fail/seal → manifest); no HTTP, no store; unit-tested standalone at S1 |
| `token_id_capture/staging/capture.py` (new) | `RolloutTokenCapture.begin_call`/`complete_call` — engine-blind: record + digest build, fail-closed stage→respond ordering, coords assembly; tested against mock adapter + mock sink |
| `token_id_capture/adapters/vllm.py` (new) | engine-specific only: prefix-in entry, suffix splice (relocated `_replace_prefix_tokens`), native id+logprob extraction, serving-layer hookup. `adapters/sglang.py` is a later drop-in that inherits `capture.py`'s ordering for free |
| `token_id_capture/staging/conformance/` (new) | golden fixtures + installable test kit run by every framework/adapter CI |
| gate hosting (port from prototype `capture_gate.py`, `rollout_registry.py`) | **thin hosting of `lineage.py`**: admission + call-id mint, marker resolve + fingerprint, serving rule, coords ingestion, seal → receipt, `fail_rollout`/TTL; control router; Gym also ships the `RolloutControlClient` frameworks call |
| `responses_api_models/vllm_model/app.py` | per-rollout **token buffer**; marker → parent lookup + fingerprint check in `_preprocess_chat_completion_create_params` (`:259`); `prefix_ids` into `extra_body`; coords ingestion replacing the logprob-scrape + `/tokenize` block (`:497-551`); rollout affinity in `_resolve_client` (`:576`) |
| `nemo_gym/responses_converter.py` | attach `ng_call_id` marker instead of token arrays (`:362-374`) |
| rollout id plumbing | `responses_create_params.metadata` carrier end-to-end; no agent changes |

**Gym dependency:** a fork branch of the submodule = upstream main +
**#2124 (pinned rev)** + the gate work; the gitlink must point at a rev
fetchable by CI (decision at the S1 gate: fork remote vs. NVIDIA-NeMo/Gym
branch). Because #2124 is adopted as the base, H4 upstreaming reduces to
contributing the gate/TQ layers and reconciling with the *rest* of the
stack (#2125–#2128).

### 9.3 File manifest

New files, **Gym fork** (base = main + #2124; all under the § 3.0 purity
rule except `adapters/` and the gate/server wiring):

| File | Defines | Stage |
|---|---|---|
| `nemo_gym/token_id_capture/staging/records.py` | single wire schema (§ 3.4): `StagedCallRecord`/`CommitCoords`/receipt/manifest; `SCHEMA_VERSION` (#2124's `records.py` untouched) | S1 |
| `nemo_gym/token_id_capture/staging/protocols.py` | `TokenSink`, `TokenSource`, `WeightVersionProvider`, `CaptureAdapter`, `install_capture` | S1 |
| `nemo_gym/token_id_capture/staging/digest.py` | `compute_staging_digest` + encoders (prototype port; golden vectors) | S1 |
| `nemo_gym/token_id_capture/staging/lineage.py` | `RolloutLineage` pure state machine: admit/commit/fail/seal → manifest (no HTTP, no store) | S1 |
| `nemo_gym/token_id_capture/staging/rebuild.py` | snapshots → entries → `linearize(policy, terminal_hint)` (pure; prototype `staged_token_source.py` core) | S1 |
| `nemo_gym/token_id_capture/staging/conformance/` | golden fixtures + installable conformance kit | S1 |
| `nemo_gym/token_id_capture/staging/capture.py` | `RolloutTokenCapture` — engine-blind capture: record/digest build, fail-closed stage→respond ordering, coords assembly | S2 |
| `nemo_gym/token_id_capture/adapters/vllm.py` | engine-specific: prefix-in entry, suffix splice (relocated `_replace_prefix_tokens`), id+logprob extraction, serving-layer hookup | S2 |
| `nemo_gym/token_id_capture/memory_store.py` | in-memory rollout token buffer behind #2124's store interface | S3 |
| `nemo_gym/token_id_capture/gate.py` | thin hosting of `lineage.py`: admission, call_id mint, marker resolution + fingerprint, serving rule, receipt assembly | S3 |
| `nemo_gym/token_id_capture/control_routes.py` | register/seal/`fail_rollout` control API + `RolloutControlClient` | S3 |

Modified, Gym fork: `responses_api_models/vllm_model/app.py`
(marker lookup + `prefix_ids` in `_preprocess…`; coords ingestion replacing
the logprob-scrape + `/tokenize`; affinity in `_resolve_client`; S3),
`nemo_gym/responses_converter.py` (marker instead of token arrays; S3),
`nemo_gym/base_responses_api_model.py` (install gate + routes; S3),
`nemo_gym/global_config.py` (config keys; S3) — details in § 9.2.

New files, **NeMo-RL** (provider implementations + training assembly only):

| File | Defines | Stage |
|---|---|---|
| `nemo_rl/data_plane/tq_token_sink.py` | `TQTokenSink` / `TQTokenSource` implementing Gym's protocols over `put_samples`/`get_samples` | S1 |
| `nemo_rl/experience/blackbox_finalizer.py` | orchestration over Gym verify/`rebuild`/`linearize`; `finalize_group` (always N rows, placeholders, `group_min_wv`) | S4 |

Modified, NeMo-RL: `algorithms/async_utils/replay_buffer.py` (S1),
`algorithms/single_controller_utils/config.py` (S1), `.../setup.py`
(S1+S4), `algorithms/single_controller.py` (S1 fix + S4),
`models/generation/vllm/vllm_worker_async.py` (S2, hosting only),
`models/generation/vllm/vllm_generation.py` (S2, fan-outs),
`environments/nemo_gym.py` (S3-RL+S4), `experience/rollout_manager.py`
(S4), `algorithms/advantage_estimator.py` (S4), exemplar YAML
`examples/configs/grpo_math_1B_single_controller.yaml` (S4) — details in
§ 9.1.

Placement rule: protocols, records, digest, rebuild, and capture logic are
defined once in Gym's leaf package and imported everywhere (venv check at
the S1 gate); NeMo-RL contributes storage implementations, hosting, and
training assembly. Another RL framework integrates by re-implementing only
the two NeMo-RL "new files" rows; another backend by one `adapters/` file.

## 10. Implementation plan

### MVP — one PR, five sign-off-gated stages

Single branch, one PR; each stage a signed-off commit series keeping the
tree green (S1–S3 dormant behind `token_capture.enabled=false`); each ends
at a review gate (stage diff, test results, deviations presented for
explicit sign-off). **Wire shapes (§ 3.4) and the serving rule (§ 3.3)
freeze at the S1 gate.**

- **S1 — primitives (Gym fork) + buffer surgery (RL repo).** Gym fork:
  `token_id_capture` records (single wire schema), `protocols.py`,
  `digest.py` (golden vectors vs. prototype), **`lineage.py`** (pure state
  machine), `rebuild.py`, conformance kit, import-linter purity test. RL:
  `TQTokenSink`/`TQTokenSource` implementing the protocols;
  `TokenCaptureConfig`; `TQReplayBuffer` `commit_finalized` / rollout_ids
  on slots / staging-aware `remove` / `abort` / evicted-slot fix;
  `_buffer_capacity` leak fix; partition pre-registration. Unit tests:
  records/digest vectors; **lineage state machine standalone** (admit/
  commit/fail/seal transitions, manifest ordering, TTL/fail paths, fork
  topologies) — lineage logic is fully tested two stages before it
  touches HTTP; conformance kit green on the TQ implementations; buffer
  ops incl. eviction/cleanup.
  **S1-gate checklist:** submodule fork logistics decided; the leaf
  package importable in the vLLM worker venv; vLLM prefix-ids path +
  splice validated for the functional-test templates.
- **S2 — capture core + vLLM adapter (Gym fork) + worker hosting (RL
  repo).** Gym, two layers: **`capture.py`** — engine-blind
  `RolloutTokenCapture` (record + digest build, fail-closed stage→respond
  ordering, coords + delta ids on the response, non-streaming assert),
  tested against a **mock adapter + mock sink** so the ordering matrix is
  backend-independent; **`adapters/vllm.py`** — engine-specific only:
  `extra_body` prefix-in feeding the relocated `_replace_prefix_tokens`
  splice, native id+logprob extraction, serving-layer hookup, with its own
  per-template splice goldens. RL: `install_capture` call at worker
  startup, DP client + `setup_token_capture` fan-out,
  `_rollout_weight_version` + `set_rollout_weight_version` fan-out;
  hosting/fan-out/version-stamping tests.
- **S3 — Gym gate (submodule fork, based on #2124).** Rebase the fork onto
  upstream main + #2124 (pinned rev; run its 20-test capture suite as the
  base sanity check). **Host the S1 `lineage.py` machine in the gate** —
  no new lineage logic lands here, only hosting: in-memory store impl,
  admission + call-id mint ports (prototype donors), marker attach/resolve
  + fingerprint, the serving rule, prefix serving, coords ingestion via
  the #2124 sink seam, rollout affinity, token-free receipts, TTLs,
  control routes + `RolloutControlClient`. Conformance tests: the S1
  golden call sequences replayed **through the gate** → manifests
  byte-identical to S1's direct-drive results; marker-stripped and
  history-edited fallbacks; duplicate-coords / wrong-rollout rejection;
  timeout; #2124 suite stays green.
- **S4 — receipts, finalizer, SC integration.** Receipt-mode
  `run_rollouts` (tokens removed from message_log, `full_result`, wandb
  table); `blackbox_finalizer` + `finalize_group` (always N rows,
  placeholders, `group_min_wv`/`group_max_wv`,
  `min_valid_fraction_per_group`); validity-aware baseline (unit-tested);
  config validation (MVP matrix only); setup wiring; `commit_finalized` +
  cleanup; teardown sweep; NaN-retry hard error; metrics (§ 8). Gate
  evidence: 2-GPU manual run with a sub-agent fork → two-root manifest →
  trained main chain.
- **S5 — verification.** Fixed-seed job on legacy and capture paths,
  training rows dumped from TQ and diffed offline (token-in should be
  byte-identical where the legacy echo path drifts); reward-curve
  comparison; per-call HTTP bytes measured vs. the echo path; chaos smoke
  (kill gate mid-step → placeholders + TTL, no leaks); capture-enabled
  functional test in `L1_Functional_Tests_SingleController.sh`.
  S5 sign-off = MVP acceptance.

### Hardening (ordered)

- **H1 — failure sweep + chaos.** Gate death mid-step, coords loss,
  receipt loss; optional receipt persistence if S5 measures meaningful
  loss; kill-gate CI test. First, because bring-up debugging kills actors.
- **H2 — hash layer.** `cum_hash`/`chain_hash` fill the reserved record
  fields: chain re-verification at finalize (tamper evidence) and
  worker-side prefix confirm — which relaxes the strict serving rule
  (uncertain marker → one hash check instead of a full-render new root).
- **H3 — atomic weight-version rotation** (v2 § 4.4 TODO) then **strict
  modes**: `strict_on_policy` (drain re-enabled), `reject`, `wait`
  (verl's dropless pattern), `force_in_order`.
- **H4 — upstream into Gym proper.** With the § 3.0 structure the code is
  already in its final home: upstreaming is merging the fork branch onto
  the merged #2124 and reconciling with the remainder of the stack (#2125
  trajectory builder, #2126 delivery/scoping, #2127 on-policy pin, #2128
  example); publish the conformance kit as the multi-framework contract.
- **H5 — perf + scale.** Incremental suffix tokenization (drop the double
  render); finalizer isolation/pool; admission caps (calls/rollout, buffer
  bytes/rollout — the design's unbounded resources); gate scale-out with
  rollout affinity; SGLang adapter.

**Post-MVP validation:** 1-off async nightly in capture mode; perf report
reproducing the sync prototype measurements (HTTP bytes/token, step time)
plus `token_in_rate` and gate latency.

## 11. Risks

- **Gym submodule fork on the MVP critical path.** Mitigations: donor gate
  code is proven (prototype `6ea5810`); the fork is ours; the gitlink/CI
  question is forced at the S1 gate, not discovered late.
- **#2124 is an open PR.** Basing the fork on it means rebase churn if it
  changes before merging. Mitigations: pin the fork to a specific #2124
  rev; keep gate code in separate modules touching the base package only
  through its public seams (records/store/sink/config), so a rebase is
  mechanical; track the PR during S3.
- **Protocols in the fork raise iteration friction**: every schema/protocol
  change during bring-up touches the Gym fork even for RL-only work.
  Mitigated by the fork being a local editable workspace member (no
  publish cycle) and by freezing shapes at the S1 gate.
- **Leaf-package availability in worker venvs.** The purity rule makes the
  import safe, but each framework's worker environment must actually
  contain `nemo_gym` (or a separately published leaf distribution).
  Checked at the S1 gate for NeMo-RL's vLLM `py_executable`.
- **Marker survival across agent frameworks.** The marker uses the exact
  carrier today's token echo uses, so it survives wherever the current
  pipeline works; an agent that strips unknown fields degrades to text
  fallback — correct but wasteful. `token_in_rate` is first-class from S4.
- **Gate as stateful hot-path service** (per-rollout token buffers, KBs–MBs
  × in-flight rollouts). Buffer bytes are bounded by H5 caps; state
  self-clears at seal/TTL; crash blast radius measured in S5 chaos; and
  the lineage state machine is a pure, separately-tested class
  (`lineage.py`, S1), so gate hosting stays thin.
- **No cross-row chain integrity in the MVP** (§ 5): accepted; the gate is
  inside the existing trust boundary; H2 restores tamper evidence.
- **Fingerprint definition** (message normalization for equality) must be
  pinned in S3 conformance tests — too strict inflates fallbacks, too loose
  misses edits. Fallback-on-mismatch keeps both errors safe.
- **Finalize latency, placeholder-heavy groups, checkpointing**: unchanged
  from v2 § 8 (in-flight rollouts abandoned on restore; TTL + create-only
  registration make this safe).

## Appendix A — prior art

- **verl**: token-in/token-out rationale ("apply_chat_template to final
  history makes PPO not converge"); TQ meta-passing; `global_steps` spans +
  `drop|wait` staleness policies. No forest — linear trajectories only.
- **slime** (commit `ea9819f`): message-tree lineage with prefix-match +
  rewrite-merge; consistent-hash session→engine affinity
  (`X-SMG-Routing-Key`) mirrored by this design's rollout affinity. The
  earlier drafts adopted its mount-point walk as a lineage hint; this
  design replaces content inference entirely with the explicit marker —
  the failure modes slime papers over (ambiguity, rewrites) become
  explicit fallbacks here.
- **Gym #2124–#2128**: token-id capture stack for external-agent training
  (issue #1824). **#2124 (capture core: records/store/sink/config/routes)
  is the adopted base for this design's Gym work** (§ 9.2); it has no
  gate/lineage/prefix-serving surface — that is what this design adds on
  top. #2125–#2128 (trajectory builder, delivery/scoping, on-policy pin,
  example) are reconciled at H4.
- **The sync prototype** (`/lustre/fsw/.../gym/RL`, branch
  `pranav/tq_gym_prototype`): proved the staging dataflow and the perf
  numbers (−46.9 % HTTP bytes/token, −55.7 % exchanges, −4.3 % step time
  p50); donor for the digest scheme, staging sink, finalizer skeleton, and
  the gate's admission/control plane.
